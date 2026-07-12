"""
Claw Bot — LTX-2 Video Adapter (via ComfyUI API)

Talks to ComfyUI's HTTP API to run an LTX-2 image-to-video workflow.
Reads the exported API workflow from 05_Config/workflows/, uploads the
starting frame, injects the prompt, submits, polls, downloads the MP4.
"""

import json
import logging
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import requests

# Ensure the 02_Agent folder is on sys.path so imports resolve consistently
# regardless of how this module is invoked.
_AGENT_DIR = Path(__file__).parent.parent.parent.resolve()
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

from modules.video_backend import VideoBackend
from modules import model_registry
from modules import runtime_settings as rs

log = logging.getLogger("claw_bot.video_backend.ltx2")


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
WORKFLOWS_DIR = PROJECT_ROOT / "05_Config" / "workflows"
DEFAULT_WORKFLOW_FILE = "ltx2_video_only_v1_api.json"
OUTPUT_DIR = PROJECT_ROOT / "04_Outputs" / "videos"
DEFAULT_NEGATIVE = (
    "blurry, oversaturated, pixelated, low resolution, grainy, distorted, "
    "noise, compression artifacts, jpg artifacts, glitches, watermark, text, "
    "logo, signature, copyright, subtitles, extra limbs, deformed, scary, "
    "static, frozen, no motion"
)


class Backend(VideoBackend):
    """LTX-2 19B Dev (GGUF) image-to-video via ComfyUI API."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.server_url = config.get("server_url", "http://127.0.0.1:8188").rstrip("/")
        self.unet_name      = config["unet_name"]
        self.clip_name1     = config["clip_name1"]
        self.clip_name2     = config["clip_name2"]
        self.vae_name       = config["vae_name"]
        self.steps          = config.get("steps", 20)
        self.cfg            = config.get("cfg", 3.0)
        self.sampler        = config.get("sampler", "euler_ancestral")
        self.sigmas         = config.get("sigmas")        # comma-separated string or None
        self.img_compression = config.get("img_compression", 33)
        # The saved workflow halves the frame after the resize node.
        self.scale_by = float(config.get("scale_by", 1.0))
        self.default_width  = config.get("default_width", 832)
        self.default_height = config.get("default_height", 480)
        self.default_frames = config.get("default_frames", 81)
        self.default_fps    = config.get("default_fps", 24)

        # Workflow file is configurable so a new LTX entry (e.g. LTX-2.3) can point
        # at its own API-format workflow without a new adapter — provided the node
        # shape matches what _build_workflow injects.
        wf_name = config.get("workflow_file", DEFAULT_WORKFLOW_FILE)
        workflow_path = WORKFLOWS_DIR / wf_name
        if not workflow_path.exists():
            raise FileNotFoundError(
                f"API workflow not found: {workflow_path}. "
                f"Export it from ComfyUI GUI (Menu → Save (API Format))."
            )
        self._workflow_template = json.loads(workflow_path.read_text(encoding="utf-8"))
        log.info(f"LTX adapter initialized. Workflow loaded from {workflow_path.name}")

    # --------------------------------------------------------------
    # VideoBackend contract
    # --------------------------------------------------------------

    def health_check(self) -> tuple[bool, str]:
        """Ping ComfyUI's API root."""
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
        """
        LTX-2 likes paragraph-style natural language describing actions over time.
        Tip from LTX-2 docs:
          1. Core Actions: describe events as they occur over time
          2. Reference Image: do not repeat details already present in the image
          3. Consistency: avoid instructions that don't match the reference image
        """
        return raw_prompt.strip()

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
        width: Optional[int] = None,
        height: Optional[int] = None,
        cfg_override: Optional[float] = None,
        lora_4step_override: Optional[bool] = None,  # ignored (LTX has no 4-step LoRA)
    ) -> Path:
        if seed is None:
            seed = random.randint(1, 2**31 - 1)
        if frame_count is None:
            frame_count = self.default_frames
        if fps is None:
            fps = self.default_fps
        # Resolution: explicit override > ASPECT RATIO > backend config default.
        # The aspect argument used to be ignored entirely, so a 9:16 request came
        # back landscape at the config's 1280x720. These dims are multiples of 32
        # (LTX's latent stride) and 9:16 matches the Qwen still exactly, so the
        # frame is never rescaled on the way in.
        _AR_DIMS = {
            "16:9": (1280, 720),
            "9:16": (768, 1344),
            "1:1":  (1024, 1024),
        }
        _ar = (aspect_ratio or "16:9").replace("x", ":").strip()
        _aw, _ah = _AR_DIMS.get(_ar, (self.default_width, self.default_height))
        eff_width  = int(width)  if width  else int(_aw)
        eff_height = int(height) if height else int(_ah)

        # Resolve runtime overrides (explicit param > runtime_settings > config)
        effective_steps = rs.get_steps_override() if rs.get_steps_override() is not None else self.steps
        if cfg_override is not None:
            effective_cfg = float(cfg_override)
        else:
            effective_cfg = rs.get_cfg_override() if rs.get_cfg_override() is not None else self.cfg
        formatted_prompt = self.format_prompt_for_backend(prompt)
        negative = negative_prompt or DEFAULT_NEGATIVE

        if not input_image.exists():
            raise FileNotFoundError(f"Input image not found: {input_image}")

        log.info(
            f"Generating video — seed={seed}, frames={frame_count}@{fps}fps, "
            f"res={eff_width}x{eff_height}, cfg={effective_cfg}, "
            f"image={input_image.name}, prompt='{prompt[:80]}...'"
        )

        # 1. Upload the input image to ComfyUI's input folder
        uploaded_name = self._upload_image(input_image)
        log.info(f"Uploaded starting frame: {uploaded_name}")

        # 2. Build the workflow with all dynamic values injected
        workflow = self._build_workflow(
            prompt=formatted_prompt,
            negative_prompt=negative,
            input_image_name=uploaded_name,
            frame_count=frame_count,
            fps=fps,
            seed=seed,
            steps_override=effective_steps,
            cfg_override=effective_cfg,
            width=eff_width,
            height=eff_height,
        )

        # 3. Submit to ComfyUI
        client_id = str(uuid.uuid4())
        prompt_id = self._submit_prompt(workflow, client_id)
        log.info(f"Submitted prompt_id={prompt_id}")

        # 4. Wait for it to finish (videos take longer than images)
        history = self._wait_for_completion(prompt_id, timeout=1800)  # 30 min ceiling

        # 5. Download the MP4
        video_path = self._download_output(history, prompt_id, output_filename)
        log.info(f"Video saved: {video_path}")
        return video_path

    # --------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------

    def _upload_image(self, image_path: Path) -> str:
        """
        Upload the starting frame to ComfyUI's input folder via /upload/image.
        Returns the filename ComfyUI assigned (usually the same as input).
        """
        url = f"{self.server_url}/upload/image"
        with open(image_path, "rb") as f:
            files = {"image": (image_path.name, f, "image/png")}
            data = {"overwrite": "true"}
            r = requests.post(url, files=files, data=data, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"Image upload failed (HTTP {r.status_code}): {r.text[:400]}")
        result = r.json()
        return result.get("name", image_path.name)

    def _build_workflow(self, prompt: str, negative_prompt: str,
                        input_image_name: str,
                        frame_count: int, fps: int, seed: int,
                        steps_override: int = None,
                        cfg_override: float = None,
                        width: int = None, height: int = None) -> dict:
        """
        Take the exported workflow template and fill in dynamic values.
        Targets nodes by class_type + title — exact match against the saved
        ltx2_video_only_v1_api.json structure.
        """
        wf = json.loads(json.dumps(self._workflow_template))   # deep copy
        effective_cfg = cfg_override if cfg_override is not None else self.cfg
        # Note: this LTX-2 workflow uses ManualSigmas (fixed schedule) — steps
        # is implicit in the sigma list length. steps_override is ignored here.

        injected = {
            "prompt": False, "image": False, "frames": False,
            "fps": False, "seed": False, "cfg": False,
            "sigmas": False, "compression": False, "dims": False,
        }

        for node_id, node in wf.items():
            if not isinstance(node, dict):
                continue
            ctype = node.get("class_type", "")
            title = node.get("_meta", {}).get("title", "").lower()
            inputs = node.setdefault("inputs", {})

            # --- Loaders (model swap support) ---
            if ctype == "UnetLoaderGGUF":
                inputs["unet_name"] = self.unet_name
            elif ctype == "DualCLIPLoaderGGUF":
                inputs["clip_name1"] = self.clip_name1
                inputs["clip_name2"] = self.clip_name2
                inputs.setdefault("type", "ltxv")
            elif ctype == "VAELoader":
                inputs["vae_name"] = self.vae_name

            # --- Starting frame ---
            elif ctype == "LoadImage":
                inputs["image"] = input_image_name
                injected["image"] = True

            # --- Sampler / seed / cfg ---
            elif ctype == "RandomNoise":
                inputs["noise_seed"] = seed
                injected["seed"] = True
            elif ctype == "KSamplerSelect":
                inputs["sampler_name"] = self.sampler
            elif ctype == "CFGGuider":
                inputs["cfg"] = effective_cfg
                injected["cfg"] = True

            elif ctype == "ManualSigmas":
                if self.sigmas:
                    inputs["sigmas"] = self.sigmas
                    injected["sigmas"] = True

            elif ctype == "LTXVPreprocess":
                inputs["img_compression"] = self.img_compression
                injected["compression"] = True

            # --- Working resolution (the resize node sets actual gen dims) ---
            elif ctype == "ResizeImageMaskNode":
                if width and height:
                    # API-format flattens nested widget keys with dotted names.
                    if "resize_type.width" in inputs:
                        inputs["resize_type.width"] = int(width)
                        inputs["resize_type.height"] = int(height)
                    else:
                        inputs["width"] = int(width)
                        inputs["height"] = int(height)
                    injected["dims"] = True

            # --- The hidden halver ---
            # The saved workflow scales the frame by 0.5 AFTER the resize node, so
            # every dimension we injected was silently cut in half: a 768x1344
            # request came out 384x672, which is exactly why LTX looked soft and
            # dissolving. Pin it to 1.0 — the resize node already set the dims.
            elif ctype == "ImageScaleBy":
                inputs["scale_by"] = float(self.scale_by)
                injected["scale_by"] = True

            # --- Title-targeted primitives ---
            elif ctype == "PrimitiveStringMultiline" and "positive prompt" in title:
                inputs["value"] = prompt
                injected["prompt"] = True

            elif ctype == "PrimitiveInt" and title == "frame count":
                inputs["value"] = int(frame_count)
                injected["frames"] = True

            elif ctype == "PrimitiveFloat" and "frame rate" in title:
                inputs["value"] = float(fps)
                injected["fps"] = True
            elif ctype == "PrimitiveInt" and "frame rate" in title:
                inputs["value"] = int(fps)
                # don't set injected["fps"]=True yet — float is the canonical one

        log.info(
            f"Workflow injection: prompt={injected['prompt']} image={injected['image']} "
            f"frames={injected['frames']} fps={injected['fps']} "
            f"seed={injected['seed']} cfg={injected['cfg']} "
            f"sigmas={injected['sigmas']} compression={injected['compression']}"
        )
        for k, v in injected.items():
            # 'sigmas' is optional — skip warning if no sigmas configured
            if not v and not (k == "sigmas" and not self.sigmas):
                log.warning(f"Failed to inject '{k}' — workflow structure may have changed.")
        return wf

    def _submit_prompt(self, workflow: dict, client_id: str) -> str:
        url = f"{self.server_url}/prompt"
        payload = {"prompt": workflow, "client_id": client_id}
        r = requests.post(url, json=payload, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"ComfyUI rejected prompt (HTTP {r.status_code}): {r.text[:400]}")
        data = r.json()
        if "prompt_id" not in data:
            raise RuntimeError(f"ComfyUI response missing prompt_id: {data}")
        return data["prompt_id"]

    def _wait_for_completion(self, prompt_id: str, timeout: int = 1800) -> dict:
        """Poll /history until complete. Heartbeat every 30s."""
        history_url = f"{self.server_url}/history/{prompt_id}"
        queue_url = f"{self.server_url}/queue"
        start = time.time()
        last_heartbeat = start

        while time.time() - start < timeout:
            r = requests.get(history_url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if prompt_id in data:
                    entry = data[prompt_id]
                    status = entry.get("status", {})
                    if status.get("completed") is True:
                        elapsed = int(time.time() - start)
                        log.info(f"Prompt {prompt_id[:8]} completed in {elapsed}s")
                        return entry
                    if status.get("status_str") == "error":
                        messages = entry.get("status", {}).get("messages", [])
                        raise RuntimeError(f"ComfyUI reported error: {messages}")

            now = time.time()
            if now - last_heartbeat >= 30:
                last_heartbeat = now
                elapsed = int(now - start)
                try:
                    q = requests.get(queue_url, timeout=5).json()
                    running = len(q.get("queue_running", []))
                    pending = len(q.get("queue_pending", []))
                    log.info(
                        f"Waiting for {prompt_id[:8]}: {elapsed}s elapsed, "
                        f"queue_running={running}, queue_pending={pending}"
                    )
                except Exception:
                    log.info(f"Waiting for {prompt_id[:8]}: {elapsed}s elapsed (queue check failed)")
            time.sleep(2)

        raise TimeoutError(f"Timed out after {timeout}s waiting for prompt {prompt_id}")

    def _download_output(self, history_entry: dict, prompt_id: str,
                         output_filename: Optional[str]) -> Path:
        """
        Download the MP4 produced by VHS_VideoCombine.
        Outputs section can have 'gifs' (yes — VHS uses 'gifs' key for any
        animated output incl. mp4) or 'images' depending on save format.
        """
        outputs = history_entry.get("outputs", {})
        if not outputs:
            raise RuntimeError("ComfyUI finished but produced no outputs.")

        # Find the first node that produced a video file
        video_info = None
        for node_id, output in outputs.items():
            for key in ("gifs", "videos", "images"):
                items = output.get(key)
                if items:
                    # Pick the .mp4/.webm if present, else first item
                    mp4 = next(
                        (i for i in items if str(i.get("filename", "")).lower().endswith((".mp4", ".webm"))),
                        None,
                    )
                    video_info = mp4 or items[0]
                    break
            if video_info:
                break
        if video_info is None:
            raise RuntimeError("No video in outputs.")

        filename = video_info["filename"]
        subfolder = video_info.get("subfolder", "")
        ftype = video_info.get("type", "output")

        view_url = (
            f"{self.server_url}/view?"
            f"filename={filename}&subfolder={subfolder}&type={ftype}"
        )
        r = requests.get(view_url, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"Could not download video: HTTP {r.status_code}")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if output_filename:
            target = OUTPUT_DIR / output_filename
        else:
            ext = Path(filename).suffix or ".mp4"
            target = OUTPUT_DIR / f"vid_{prompt_id[:12]}{ext}"

        target.write_bytes(r.content)
        return target


# ==============================================================================
# Standalone test
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from modules import video_backend as vb
    backend = vb.get_active_backend()
    print(f"Backend: {backend.backend_id}")
    ok, msg = backend.health_check()
    print(f"Health: {msg}")
    if not ok:
        raise SystemExit("Backend unhealthy — start ComfyUI first.")

    test_image = PROJECT_ROOT / "04_Outputs" / "storyboards" / "shot1_first.png"
    if not test_image.exists():
        raise SystemExit(f"Test image missing: {test_image}")

    test_prompt = (
        "The young boy gently picks up a piece of trash from the street and places "
        "it into the recycling bin. The turtle nods approvingly beside him. Soft "
        "afternoon sunlight, gentle camera push-in, warm storybook colors."
    )

    path = backend.generate(
        prompt=test_prompt,
        input_image=test_image,
        frame_count=81,
        fps=24,
        output_filename="test_ltx2_1.mp4",
    )
    print(f"Generated: {path}")
