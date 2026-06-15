"""
Claw Bot — Wan 2.2 S2V (Sound-to-Video) Adapter  [P4 of character-dialogue]

Audio-DRIVEN talking-head video. Given a reference frame + a spoken-audio WAV,
Wan 2.2 S2V animates the character's mouth/head to match the speech (real
lip-sync) instead of the silent motion-only I2V path.

Routing lives in clip_generator: NARRATOR shots use the normal I2V backend
(no lip-sync needed for voiceover); CHARACTER shots are sent here with the
character's spoken WAV so the on-screen speaker's mouth moves.

This mirrors comfyui_wan22_14B.py. The ONE structural difference is an audio
input (LoadAudio) injected alongside the reference image. The workflow JSON is
configurable (`workflow_file` in models.json) — export it from ComfyUI once the
Wan2.2-S2V nodes + weights are installed (Workflow → Export (API Format)).

INERT until configured: the adapter only loads when lip-sync is enabled and the
registry entry + workflow file exist; clip_generator falls back to I2V on any
failure, so the pipeline never breaks.
"""

import json
import logging
import random
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import requests

_AGENT_DIR = Path(__file__).parent.parent.parent.resolve()
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

from modules.video_backend import VideoBackend
from modules import runtime_settings as rs

log = logging.getLogger("claw_bot.video_backend.wan_s2v")

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
WORKFLOWS_DIR = PROJECT_ROOT / "05_Config" / "workflows"
OUTPUT_DIR = PROJECT_ROOT / "04_Outputs" / "videos"

DEFAULT_NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


class Backend(VideoBackend):
    """Wan 2.2 Sound-to-Video (lip-sync) via ComfyUI API."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.server_url = config.get("server_url", "http://127.0.0.1:8188").rstrip("/")
        self.unet_name = config.get("unet_name", "")
        self.clip_name = config.get("clip_name", "")
        self.vae_name = config.get("vae_name", "")
        # S2V-specific audio encoder (e.g. wav2vec). Optional — injected if present.
        self.audio_encoder_name = config.get("audio_encoder_name", "")
        self.cfg = float(config.get("cfg", 4.0))
        self.steps = int(config.get("steps", 20))
        self.default_width = int(config.get("default_width", 832))
        self.default_height = int(config.get("default_height", 480))
        self.default_fps = int(config.get("default_fps", 16))
        # Optional: direct path to ComfyUI's input dir for audio staging. When
        # set we copy the WAV there; otherwise we POST it via /upload/image.
        self.comfyui_input_dir = config.get("comfyui_input_dir", "")

        wf_name = config.get("workflow_file", "video_wan2_2_s2v_api.json")
        self.workflow_path = WORKFLOWS_DIR / wf_name
        if not self.workflow_path.exists():
            raise FileNotFoundError(
                f"Wan S2V API workflow not found: {self.workflow_path}. "
                f"Install the Wan2.2-S2V nodes+weights in ComfyUI, build the "
                f"graph, then Workflow → Export (API Format) and save it as "
                f"'{wf_name}' under 05_Config/workflows/."
            )
        self._workflow_template = json.loads(self.workflow_path.read_text(encoding="utf-8"))
        self._last_seed: int = -1
        log.info(f"Wan S2V adapter initialized — workflow: {self.workflow_path.name}")

    # ---------------- VideoBackend contract ----------------

    def health_check(self) -> tuple[bool, str]:
        try:
            r = requests.get(f"{self.server_url}/system_stats", timeout=5)
            if r.status_code == 200:
                return True, f"ComfyUI alive at {self.server_url}"
            return False, f"ComfyUI returned HTTP {r.status_code}"
        except requests.exceptions.ConnectionError:
            return False, f"Cannot connect to ComfyUI at {self.server_url}. Is it running?"
        except Exception as e:
            return False, f"Health check failed: {e}"

    def format_prompt_for_backend(self, raw_prompt: str) -> str:
        return raw_prompt.strip()

    def round_frame_count(self, frames: int) -> int:
        """Wan latent temporal stride = 4 → frames must be 4N+1. Round UP."""
        if (frames - 1) % 4 == 0:
            return frames
        return ((frames - 1) // 4 + 1) * 4 + 1

    def generate(
        self,
        prompt: str,
        input_image: Path,
        negative_prompt: Optional[str] = None,
        frame_count: int = None,
        fps: int = None,
        aspect_ratio: str = "16:9",
        output_filename: Optional[str] = None,
        seed: Optional[int] = None,
        audio_path: Optional[Path] = None,
        cfg_override: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        **_ignored,
    ) -> Path:
        """Generate a lip-synced clip. `audio_path` is REQUIRED — it drives the
        mouth. Falls back to the I2V negative/dims conventions otherwise."""
        if audio_path is None or not Path(audio_path).exists():
            raise ValueError(
                "Wan S2V requires an audio_path (the spoken WAV) to drive lip-sync."
            )
        if not Path(input_image).exists():
            raise FileNotFoundError(f"Input image not found: {input_image}")

        if seed is None:
            seed = random.randint(1, 2**31 - 1)
        self._last_seed = int(seed)
        fps = fps or self.default_fps
        eff_width = int(width) if width else self.default_width
        eff_height = int(height) if height else self.default_height
        effective_cfg = float(cfg_override) if cfg_override is not None else self.cfg
        negative = negative_prompt or DEFAULT_NEGATIVE

        uploaded_image = self._upload_image(Path(input_image))
        uploaded_audio = self._stage_audio(Path(audio_path))
        log.info(
            f"S2V generate — seed={seed}, image={uploaded_image}, "
            f"audio={uploaded_audio}, {eff_width}x{eff_height}@{fps}fps, "
            f"cfg={effective_cfg}, prompt='{prompt[:60]}...'"
        )

        workflow = self._build_workflow(
            prompt=self.format_prompt_for_backend(prompt),
            negative_prompt=negative,
            input_image_name=uploaded_image,
            audio_name=uploaded_audio,
            fps=fps, seed=seed, cfg=effective_cfg,
            width=eff_width, height=eff_height,
        )

        client_id = str(uuid.uuid4())
        prompt_id = self._submit_prompt(workflow, client_id)
        log.info(f"S2V submitted prompt_id={prompt_id}")
        history = self._wait_for_completion(prompt_id, timeout=2400)
        video_path = self._download_output(history, prompt_id, output_filename)
        log.info(f"S2V video saved: {video_path}")
        return video_path

    # ---------------- internals ----------------

    def _upload_image(self, image_path: Path) -> str:
        url = f"{self.server_url}/upload/image"
        with open(image_path, "rb") as f:
            files = {"image": (image_path.name, f, "image/png")}
            r = requests.post(url, files=files, data={"overwrite": "true"}, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"Image upload failed (HTTP {r.status_code}): {r.text[:300]}")
        return r.json().get("name", image_path.name)

    def _stage_audio(self, audio_path: Path) -> str:
        """Make the WAV available to ComfyUI's LoadAudio. Prefer copying into a
        configured input dir; else POST through the upload endpoint."""
        if self.comfyui_input_dir:
            dest_dir = Path(self.comfyui_input_dir)
            try:
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / audio_path.name
                shutil.copy2(audio_path, dest)
                return audio_path.name
            except Exception as e:
                log.warning(f"Audio copy to input dir failed ({e}); trying upload.")
        url = f"{self.server_url}/upload/image"  # ComfyUI stores arbitrary files in input/
        with open(audio_path, "rb") as f:
            files = {"image": (audio_path.name, f, "audio/wav")}
            r = requests.post(url, files=files, data={"overwrite": "true"}, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"Audio upload failed (HTTP {r.status_code}): {r.text[:300]}")
        return r.json().get("name", audio_path.name)

    def _build_workflow(self, prompt, negative_prompt, input_image_name,
                        audio_name, fps, seed, cfg, width, height) -> dict:
        """Inject dynamic values by class_type + title (IDs ignored)."""
        wf = json.loads(json.dumps(self._workflow_template))
        injected = {k: False for k in
                    ("positive", "negative", "image", "audio", "seed", "dims", "fps")}

        for node in wf.values():
            if not isinstance(node, dict):
                continue
            ctype = node.get("class_type", "")
            title = node.get("_meta", {}).get("title", "").lower()
            inputs = node.setdefault("inputs", {})

            if ctype == "UNETLoader" and self.unet_name:
                inputs["unet_name"] = self.unet_name
            elif ctype == "CLIPLoader" and self.clip_name:
                inputs["clip_name"] = self.clip_name
                inputs.setdefault("type", "wan")
            elif ctype == "VAELoader" and self.vae_name:
                inputs["vae_name"] = self.vae_name
            elif "audioencoder" in ctype.lower() and self.audio_encoder_name:
                # e.g. AudioEncoderLoader for the wav2vec model
                for k in ("audio_encoder_name", "model_name", "name"):
                    if k in inputs:
                        inputs[k] = self.audio_encoder_name
            elif ctype == "LoadImage":
                inputs["image"] = input_image_name
                injected["image"] = True
            elif ctype in ("LoadAudio", "VHS_LoadAudioUpload") or "audio" in ctype.lower():
                for k in ("audio", "audio_file", "file"):
                    if k in inputs:
                        inputs[k] = audio_name
                        injected["audio"] = True
                        break
                else:
                    if ctype == "LoadAudio":
                        inputs["audio"] = audio_name
                        injected["audio"] = True
            elif ctype in ("KSampler", "KSamplerAdvanced"):
                if "noise_seed" in inputs:
                    inputs["noise_seed"] = seed
                    injected["seed"] = True
                elif "seed" in inputs:
                    inputs["seed"] = seed
                    injected["seed"] = True
                if "cfg" in inputs:
                    inputs["cfg"] = float(cfg)
            elif "width" in inputs and "height" in inputs:
                inputs["width"] = int(width)
                inputs["height"] = int(height)
                injected["dims"] = True
            elif ctype == "PrimitiveFloat" and "fps" in title:
                inputs["value"] = float(fps)
                injected["fps"] = True

            if ctype == "CLIPTextEncode":
                if "negative" in title:
                    inputs["text"] = negative_prompt
                    injected["negative"] = True
                elif "positive" in title or not injected["positive"]:
                    inputs["text"] = prompt
                    injected["positive"] = True

        log.info(f"S2V workflow injection: {injected}")
        critical = ("positive", "image", "audio", "seed")
        missing = [k for k in critical if not injected[k]]
        if missing:
            raise RuntimeError(
                f"Wan S2V workflow injection failed for critical fields: {missing}. "
                f"Workflow '{self.workflow_path.name}' node titles likely differ — "
                f"re-export (API Format) and ensure a LoadImage, an audio loader, "
                f"a positive CLIPTextEncode, and a KSampler are present."
            )
        return wf

    def _submit_prompt(self, workflow: dict, client_id: str) -> str:
        r = requests.post(f"{self.server_url}/prompt",
                          json={"prompt": workflow, "client_id": client_id}, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"ComfyUI rejected prompt (HTTP {r.status_code}): {r.text[:300]}")
        data = r.json()
        if "prompt_id" not in data:
            raise RuntimeError(f"ComfyUI response missing prompt_id: {data}")
        return data["prompt_id"]

    def _wait_for_completion(self, prompt_id: str, timeout: int = 2400) -> dict:
        history_url = f"{self.server_url}/history/{prompt_id}"
        start = time.time()
        fails = 0
        while time.time() - start < timeout:
            try:
                r = requests.get(history_url, timeout=60)
                fails = 0
                if r.status_code == 200:
                    data = r.json()
                    if prompt_id in data:
                        entry = data[prompt_id]
                        status = entry.get("status", {})
                        if status.get("completed") is True:
                            log.info(f"S2V {prompt_id[:8]} done in {int(time.time()-start)}s")
                            return entry
                        if status.get("status_str") == "error":
                            raise RuntimeError(f"ComfyUI error: {status.get('messages', [])}")
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                fails += 1
                if fails >= 5:
                    raise RuntimeError(f"Lost ComfyUI connection for {prompt_id[:8]}") from e
                time.sleep(10)
                continue
            time.sleep(2)
        raise TimeoutError(f"Timed out after {timeout}s waiting for {prompt_id}")

    def _download_output(self, history_entry: dict, prompt_id: str,
                         output_filename: Optional[str]) -> Path:
        outputs = history_entry.get("outputs", {})
        if not outputs:
            raise RuntimeError("ComfyUI finished but produced no outputs.")
        video_info = None
        for output in outputs.values():
            for key in ("videos", "gifs", "images"):
                items = output.get(key)
                if items:
                    video_info = next(
                        (i for i in items if str(i.get("filename", "")).lower().endswith((".mp4", ".webm"))),
                        items[0],
                    )
                    break
            if video_info:
                break
        if video_info is None:
            raise RuntimeError("No video in S2V outputs.")
        view_url = (f"{self.server_url}/view?filename={video_info['filename']}"
                    f"&subfolder={video_info.get('subfolder','')}&type={video_info.get('type','output')}")
        r = requests.get(view_url, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"Could not download S2V video: HTTP {r.status_code}")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        target = (OUTPUT_DIR / output_filename if output_filename
                  else OUTPUT_DIR / f"vid_s2v_{prompt_id[:12]}.mp4")
        target.write_bytes(r.content)
        return target
