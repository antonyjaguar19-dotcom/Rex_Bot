"""
Claw Bot — Wan 2.2 5B Video Adapter (via ComfyUI API)

Fallback video backend. Wan 2.2 ti2v 5B model. Produces higher quality but
much slower than LTX-2 (~45 min per 81-frame clip on RTX 3070).
Kept around as a quality reference and architectural fallback.

Mirrors comfyui_ltx_video.py patterns. Workflow is simpler (12 nodes).
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

# Ensure 02_Agent is on sys.path
_AGENT_DIR = Path(__file__).parent.parent.parent.resolve()
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

from modules.video_backend import VideoBackend
from modules import runtime_settings as rs

log = logging.getLogger("claw_bot.video_backend.wan22")


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
WORKFLOW_PATH = PROJECT_ROOT / "05_Config" / "workflows" / "wan22_5B_api.json"
OUTPUT_DIR = PROJECT_ROOT / "04_Outputs" / "videos"
DEFAULT_NEGATIVE = (
    "distorted, fast motion, jitter, low quality, blurry, oversaturated, "
    "pixelated, grainy, watermark, text, logo, signature, deformed, "
    "extra limbs, scary, frozen, static"
)


class Backend(VideoBackend):
    """Wan 2.2 ti2v 5B image-to-video via ComfyUI API."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.server_url = config.get("server_url", "http://127.0.0.1:8188").rstrip("/")
        self.unet_name      = config["unet_name"]
        self.clip_name      = config["clip_name"]
        self.vae_name       = config["vae_name"]
        self.steps          = config.get("steps", 20)
        self.cfg            = config.get("cfg", 5.0)
        self.sampler        = config.get("sampler", "euler")
        self.scheduler      = config.get("scheduler", "simple")
        self.default_width  = config.get("default_width", 832)
        self.default_height = config.get("default_height", 480)
        self.default_frames = config.get("default_frames", 81)
        self.default_fps    = config.get("default_fps", 24)

        if not WORKFLOW_PATH.exists():
            raise FileNotFoundError(
                f"API workflow not found: {WORKFLOW_PATH}. "
                f"Export it from ComfyUI GUI (Menu → Save (API Format))."
            )
        self._workflow_template = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
        log.info(f"Wan 2.2 adapter initialized. Workflow loaded from {WORKFLOW_PATH.name}")

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
        Wan 2.2 prefers concise, scene-focused prompts. The model handles
        natural language but tends to do better with shorter descriptions
        than LTX-2's paragraph style.
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
    ) -> Path:
        if seed is None:
            seed = random.randint(1, 2**31 - 1)
        if frame_count is None:
            frame_count = self.default_frames
        if fps is None:
            fps = self.default_fps

        effective_steps = rs.get_steps_override() if rs.get_steps_override() is not None else self.steps
        effective_cfg = rs.get_cfg_override() if rs.get_cfg_override() is not None else self.cfg
        formatted_prompt = self.format_prompt_for_backend(prompt)
        negative = negative_prompt or DEFAULT_NEGATIVE

        if not input_image.exists():
            raise FileNotFoundError(f"Input image not found: {input_image}")

        log.info(
            f"Generating video — seed={seed}, frames={frame_count}@{fps}fps, "
            f"image={input_image.name}, prompt='{prompt[:80]}...'"
        )
        log.warning("Wan 2.2 5B is SLOW on 8GB VRAM (~45 min per clip). "
                    "Consider switching to LTX-2 for production.")

        # 1. Upload starting frame
        uploaded_name = self._upload_image(input_image)
        log.info(f"Uploaded starting frame: {uploaded_name}")

        # 2. Build workflow
        workflow = self._build_workflow(
            prompt=formatted_prompt,
            negative_prompt=negative,
            input_image_name=uploaded_name,
            frame_count=frame_count,
            fps=fps,
            seed=seed,
            steps=effective_steps,
            cfg=effective_cfg,
        )

        # 3. Submit
        client_id = str(uuid.uuid4())
        prompt_id = self._submit_prompt(workflow, client_id)
        log.info(f"Submitted prompt_id={prompt_id}")

        # 4. Wait — Wan 2.2 is slow, ceiling needs to be generous
        history = self._wait_for_completion(prompt_id, timeout=4800)  # 80 min ceiling

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
                        frame_count: int, fps: int, seed: int,
                        steps: int, cfg: float) -> dict:
        """Inject dynamic values into Wan 2.2 workflow nodes."""
        wf = json.loads(json.dumps(self._workflow_template))

        injected = {
            "positive": False, "negative": False, "image": False,
            "frames": False, "fps": False, "seed": False,
        }

        for node_id, node in wf.items():
            if not isinstance(node, dict):
                continue
            ctype = node.get("class_type", "")
            title = node.get("_meta", {}).get("title", "").lower()
            inputs = node.setdefault("inputs", {})

            # --- Loaders ---
            if ctype == "UNETLoader":
                inputs["unet_name"] = self.unet_name
            elif ctype == "CLIPLoader":
                inputs["clip_name"] = self.clip_name
                inputs.setdefault("type", "wan")
            elif ctype == "VAELoader":
                inputs["vae_name"] = self.vae_name

            # --- Starting frame ---
            elif ctype == "LoadImage":
                inputs["image"] = input_image_name
                injected["image"] = True

            # --- Sampler (single node — much simpler than LTX-2) ---
            elif ctype == "KSampler":
                inputs["seed"] = seed
                inputs["steps"] = steps
                inputs["cfg"] = cfg
                inputs["sampler_name"] = self.sampler
                inputs["scheduler"] = self.scheduler
                injected["seed"] = True

            # --- Wan22ImageToVideoLatent: width, height, length ---
            elif ctype == "Wan22ImageToVideoLatent":
                inputs["width"] = self.default_width
                inputs["height"] = self.default_height
                inputs["length"] = frame_count
                injected["frames"] = True

            # --- CreateVideo: fps lives here ---
            elif ctype == "CreateVideo":
                inputs["fps"] = fps
                injected["fps"] = True

            # --- Prompts (positive vs negative by title) ---
            elif ctype == "CLIPTextEncode":
                if "negative" in title:
                    inputs["text"] = negative_prompt
                    injected["negative"] = True
                elif "positive" in title:
                    inputs["text"] = prompt
                    injected["positive"] = True
                else:
                    # Unknown — assume positive if not yet set
                    if not injected["positive"]:
                        inputs["text"] = prompt
                        injected["positive"] = True
                    else:
                        inputs["text"] = negative_prompt
                        injected["negative"] = True

        log.info(
            f"Workflow injection: pos={injected['positive']} neg={injected['negative']} "
            f"image={injected['image']} frames={injected['frames']} "
            f"fps={injected['fps']} seed={injected['seed']}"
        )
        for k, v in injected.items():
            if not v:
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

    def _wait_for_completion(self, prompt_id: str, timeout: int = 4800) -> dict:
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
        Wan 2.2 uses SaveVideo node — output is in 'videos' or 'gifs' key.
        """
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
            target = OUTPUT_DIR / f"vid_wan22_{prompt_id[:12]}{ext}"

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
