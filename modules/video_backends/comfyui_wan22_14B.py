"""
Claw Bot — Wan 2.2 14B I2V Video Adapter (via ComfyUI API)

Production video backend. Wan 2.2 i2v 14B fp8 dual-UNet (high-noise +
low-noise paths). Optional 4-step lightx2v LoRA for ~4× speedup.

Mirrors comfyui_ltx_video.py / comfyui_wan22.py patterns.
Workflow is a flattened subgraph (~30 nodes), targeted by class_type+title.

Quality tradeoff:
  enable_4steps_lora=True  → ~2-3 min/clip on 5080 (LoRA path)
  enable_4steps_lora=False → ~10-15 min/clip on 5080 (full 20 steps)
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

# Ensure 02_Agent on sys.path
_AGENT_DIR = Path(__file__).parent.parent.parent.resolve()
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

from modules.video_backend import VideoBackend
from modules import runtime_settings as rs

log = logging.getLogger("claw_bot.video_backend.wan22_14b")


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
# Locked to the validated fast workflow (~7min first / ~4min rest, 4-step LoRA).
# This is the ONLY video workflow the bot uses. API-format export.
WORKFLOW_PATH = PROJECT_ROOT / "05_Config" / "workflows" / "video_wan2_2_14B_i2v(7min-4min).json"
OUTPUT_DIR = PROJECT_ROOT / "04_Outputs" / "videos"

# Default negative is the Chinese-language negative bundled with the workflow,
# which is what the model was prompt-trained against. Keep it.
DEFAULT_NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


class Backend(VideoBackend):
    """Wan 2.2 14B I2V via ComfyUI API. Dual-UNet + optional 4-step LoRA."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.server_url = config.get("server_url", "http://127.0.0.1:8188").rstrip("/")
        self.unet_high_name   = config["unet_high_name"]
        self.unet_low_name    = config["unet_low_name"]
        self.lora_high_name   = config["lora_high_name"]
        self.lora_low_name    = config["lora_low_name"]
        self.clip_name        = config["clip_name"]
        self.vae_name         = config["vae_name"]
        self.enable_4steps_lora = bool(config.get("enable_4steps_lora", True))
        self.cfg              = float(config.get("cfg", 3.5))
        self.steps_full       = int(config.get("steps_full", 20))
        self.steps_lora       = int(config.get("steps_lora", 4))
        self.default_width    = int(config.get("default_width", 832))
        self.default_height   = int(config.get("default_height", 480))
        self.default_duration = float(config.get("default_duration", 5.0))
        self.default_fps      = int(config.get("default_fps", 16))

        if not WORKFLOW_PATH.exists():
            raise FileNotFoundError(
                f"API workflow not found: {WORKFLOW_PATH}. "
                f"Export it from ComfyUI GUI (Workflow → Export (API Format))."
            )
        self._workflow_template = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
        self._last_seed: int = -1  # exposed for manifest after generate()
        log.info(
            f"Wan 2.2 14B adapter initialized — workflow: {WORKFLOW_PATH.name}, "
            f"4-step LoRA default: {self.enable_4steps_lora}"
        )

    # --------------------------------------------------------------
    # VideoBackend contract
    # --------------------------------------------------------------

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
        """
        Wan 2.2 prefers concise scene/motion descriptions in natural English.
        Trim leading/trailing whitespace; otherwise pass through.
        """
        return raw_prompt.strip()

    def round_frame_count(self, frames: int) -> int:
        """Wan 2.2 latent temporal stride = 4. Frames must be 4N+1. Round UP."""
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
        cfg_override: Optional[float] = None,
        lora_4step_override: Optional[bool] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> Path:
        """
        Note: Wan 2.2 workflow is duration-based (sec * fps = frames).
        We accept frame_count for VideoBackend interface compatibility, but
        convert it into duration for injection.
        """
        if seed is None:
            seed = random.randint(1, 2**31 - 1)
        self._last_seed = int(seed)
        if fps is None:
            fps = self.default_fps
        # Resolution: explicit width/height > aspect-ratio mapping > backend default.
        # The aspect_ratio arg was previously ignored (always landscape 832x480);
        # map it so a 9:16 request renders NATIVE vertical (same pixel budget,
        # just rotated — no extra VRAM/time) instead of landscape.
        # Wan 2.2 14B is trained for BOTH tiers. This used to be pinned to 480p,
        # which meant a 768x1344 mascot still was GENERATED at 480x832: the fur,
        # the petals and the chest logo were gone before any upscaler saw the
        # frame, and an upscaler cannot restore detail that was never rendered.
        # 720p costs ~2.2x the time (526 s vs 240 s a clip, measured) and it shows.
        _TIERS = {
            "720p": {"16:9": (1280, 720), "9:16": (720, 1280), "1:1": (960, 960)},
            "480p": {"16:9": (832, 480),  "9:16": (480, 832),  "1:1": (640, 640)},
        }
        _AR_DIMS = _TIERS.get(rs.get_video_tier(), _TIERS["720p"])
        _ar = (aspect_ratio or "16:9").replace("x", ":").strip()
        _aw, _ah = _AR_DIMS.get(_ar, (self.default_width, self.default_height))
        eff_width  = int(width)  if width  else _aw
        eff_height = int(height) if height else _ah
        if frame_count is None:
            duration = self.default_duration
        else:
            # Wan workflow's math: frames = floor(duration * fps + 1)
            # Reverse: duration = (frames - 1) / fps
            duration = max((frame_count - 1) / fps, 1.0)

        formatted_prompt = self.format_prompt_for_backend(prompt)
        negative = negative_prompt or DEFAULT_NEGATIVE

        # Runtime overrides (CFG): explicit param > runtime_settings > backend default
        if cfg_override is not None:
            effective_cfg = float(cfg_override)
        else:
            rs_cfg = rs.get_cfg_override()
            effective_cfg = rs_cfg if rs_cfg is not None else self.cfg
        # Per-call LoRA toggle override
        effective_lora = (
            bool(lora_4step_override)
            if lora_4step_override is not None
            else self.enable_4steps_lora
        )

        if not input_image.exists():
            raise FileNotFoundError(f"Input image not found: {input_image}")

        log.info(
            f"Generating video — seed={seed}, duration={duration:.2f}s @{fps}fps "
            f"(~{int(duration*fps+1)} frames), cfg={effective_cfg}, "
            f"4step_lora={effective_lora}, image={input_image.name}, "
            f"prompt='{prompt[:80]}...'"
        )

        # 1. Upload starting frame
        uploaded_name = self._upload_image(input_image)
        log.info(f"Uploaded starting frame: {uploaded_name}")

        # 2. Build workflow
        workflow = self._build_workflow(
            prompt=formatted_prompt,
            negative_prompt=negative,
            input_image_name=uploaded_name,
            duration=duration,
            fps=fps,
            seed=seed,
            cfg=effective_cfg,
            lora_enabled=effective_lora,
            width=eff_width,
            height=eff_height,
        )

        # 3. Submit
        client_id = str(uuid.uuid4())
        prompt_id = self._submit_prompt(workflow, client_id)
        log.info(f"Submitted prompt_id={prompt_id}")

        # 4. Wait — generous timeout. The FIRST gen is a cold load of the 14B
        # dual-UNet fp8 (high+low) + lora + vae + clip off disk, which alone can
        # eat ~7-10min before sampling even starts. 600s killed it mid-load. Give
        # the LoRA path real headroom too; full path stays longer.
        timeout = 1800 if effective_lora else 2400
        history = self._wait_for_completion(prompt_id, timeout=timeout)

        # 5. Download
        video_path = self._download_output(history, prompt_id, output_filename)
        log.info(f"Video saved: {video_path}")
        return video_path

    # --------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------

    def _upload_image(self, image_path: Path) -> str:
        url = f"{self.server_url}/upload/image"
        with open(image_path, "rb") as f:
            files = {"image": (image_path.name, f, "image/png")}
            data = {"overwrite": "true"}
            r = requests.post(url, files=files, data=data, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"Image upload failed (HTTP {r.status_code}): {r.text[:400]}")
        return r.json().get("name", image_path.name)

    def _build_workflow(self, prompt: str, negative_prompt: str,
                        input_image_name: str,
                        duration: float, fps: int, seed: int,
                        cfg: float, lora_enabled: Optional[bool] = None,
                        width: Optional[int] = None,
                        height: Optional[int] = None) -> dict:
        """
        Inject dynamic values into Wan 2.2 14B workflow.
        Subgraph-flattened workflow has node IDs like '129:93' — we target
        by class_type + _meta.title, ignoring IDs.
        """
        wf = json.loads(json.dumps(self._workflow_template))

        injected = {
            "positive": False, "negative": False, "image": False,
            "duration": False, "fps": False, "seed": False,
            "cfg": False, "lora_toggle": False,
            "wan_dims": False,
        }
        ksampler_seeded = False  # we only inject seed into the FIRST KSamplerAdvanced

        for node_id, node in wf.items():
            if not isinstance(node, dict):
                continue
            ctype = node.get("class_type", "")
            title = node.get("_meta", {}).get("title", "").lower()
            inputs = node.setdefault("inputs", {})

            # --- Loaders ---
            if ctype == "UNETLoader":
                # Two of them — high noise vs low noise. Match by current value.
                current = inputs.get("unet_name", "")
                if "high_noise" in current.lower():
                    inputs["unet_name"] = self.unet_high_name
                elif "low_noise" in current.lower():
                    inputs["unet_name"] = self.unet_low_name

            elif ctype == "LoraLoaderModelOnly":
                current = inputs.get("lora_name", "")
                if "high_noise" in current.lower():
                    inputs["lora_name"] = self.lora_high_name
                elif "low_noise" in current.lower():
                    inputs["lora_name"] = self.lora_low_name

            elif ctype == "CLIPLoader":
                inputs["clip_name"] = self.clip_name
                inputs.setdefault("type", "wan")
            elif ctype == "VAELoader":
                inputs["vae_name"] = self.vae_name

            # --- Starting frame ---
            elif ctype == "LoadImage":
                inputs["image"] = input_image_name
                injected["image"] = True

            # --- Wan latent dims ---
            elif ctype == "WanImageToVideo":
                inputs["width"] = int(width) if width else self.default_width
                inputs["height"] = int(height) if height else self.default_height
                injected["wan_dims"] = True

            # --- Sampling: seed only on FIRST KSamplerAdvanced (with add_noise=enable) ---
            elif ctype == "KSamplerAdvanced":
                if inputs.get("add_noise") == "enable" and not ksampler_seeded:
                    inputs["noise_seed"] = seed
                    injected["seed"] = True
                    ksampler_seeded = True

            # --- 4-step LoRA toggle ---
            elif ctype == "PrimitiveBoolean" and "4steps" in title.lower():
                inputs["value"] = (
                    bool(lora_enabled)
                    if lora_enabled is not None
                    else self.enable_4steps_lora
                )
                injected["lora_toggle"] = True

            # --- Title-targeted primitives ---
            elif ctype == "PrimitiveFloat" and "duration" in title:
                inputs["value"] = float(duration)
                injected["duration"] = True

            elif ctype == "PrimitiveFloat" and "fps" in title:
                inputs["value"] = float(fps)
                injected["fps"] = True

            elif ctype == "PrimitiveFloat" and "cfg" in title:
                # TWO cfg primitives feed a switch (full-quality path vs 4-step
                # LoRA path). The LoRA path is locked to cfg=1 — lightx2v 4-step
                # distill REQUIRES cfg=1 or the output burns. Never overwrite that
                # node; only inject the user/beat cfg into the full-quality path.
                if abs(float(inputs.get("value", 0.0)) - 1.0) < 1e-6:
                    continue
                inputs["value"] = float(cfg)
                injected["cfg"] = True

            # --- Prompts (positive vs negative by title) ---
            elif ctype == "CLIPTextEncode":
                if "negative" in title:
                    inputs["text"] = negative_prompt
                    injected["negative"] = True
                elif "positive" in title:
                    inputs["text"] = prompt
                    injected["positive"] = True

        log.info(
            f"Workflow injection: pos={injected['positive']} neg={injected['negative']} "
            f"image={injected['image']} dims={injected['wan_dims']} "
            f"duration={injected['duration']} fps={injected['fps']} "
            f"seed={injected['seed']} cfg={injected['cfg']} "
            f"lora_toggle={injected['lora_toggle']}"
        )
        for k, v in injected.items():
            if not v:
                log.warning(f"Failed to inject '{k}' — workflow structure may have changed.")
        # FAIL FAST on critical fields. Soft-warn on the rest.
        critical = ("positive", "image", "wan_dims", "seed")
        missing_critical = [k for k in critical if not injected[k]]
        if missing_critical:
            raise RuntimeError(
                f"Wan 2.2 workflow injection failed for critical fields: "
                f"{missing_critical}. Workflow file ({WORKFLOW_PATH.name}) "
                f"likely changed shape. Re-export from ComfyUI GUI "
                f"(Workflow → Export (API Format)) and verify node titles."
            )
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
        history_url = f"{self.server_url}/history/{prompt_id}"
        queue_url = f"{self.server_url}/queue"
        start = time.time()
        last_heartbeat = start

        consecutive_poll_failures = 0
        while time.time() - start < timeout:
            try:
                r = requests.get(history_url, timeout=60)
                consecutive_poll_failures = 0  # reset on success
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
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                consecutive_poll_failures += 1
                log.warning(
                    f"Poll {consecutive_poll_failures}/5 failed for {prompt_id[:8]} "
                    f"(ComfyUI likely GPU-busy): {type(e).__name__}"
                )
                if consecutive_poll_failures >= 5:
                    raise RuntimeError(
                        f"Lost connection to ComfyUI for {prompt_id[:8]} "
                        f"after 5 consecutive timeouts. Is ComfyUI still running?"
                    ) from e
                # Back off — give ComfyUI breathing room
                time.sleep(10)
                continue

            now = time.time()
            if now - last_heartbeat >= 30:
                last_heartbeat = now
                elapsed = int(now - start)
                try:
                    q = requests.get(queue_url, timeout=10).json()
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
        outputs = history_entry.get("outputs", {})
        if not outputs:
            raise RuntimeError("ComfyUI finished but produced no outputs.")

        video_info = None
        for node_id, output in outputs.items():
            for key in ("videos", "gifs", "images"):
                items = output.get(key)
                if items:
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
            target = OUTPUT_DIR / f"vid_wan22_14B_{prompt_id[:12]}{ext}"

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
