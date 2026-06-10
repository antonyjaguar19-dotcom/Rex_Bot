"""
Claw Bot — Z-Image Base Adapter (via ComfyUI API)

Talks to ComfyUI's HTTP API to run a Z-Image workflow.
Reads the exported API workflow from 05_Config/workflows/,
injects the prompt, submits, polls, downloads the result.
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

from modules.image_backend import ImageBackend
from modules import model_registry
from modules import runtime_settings as rs

log = logging.getLogger("claw_bot.image_backend.zimage")


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
WORKFLOW_PATH = PROJECT_ROOT / "05_Config" / "workflows" / "zimage_base_storyboard_v1_api.json"
OUTPUT_DIR = PROJECT_ROOT / "04_Outputs" / "storyboards"
DEFAULT_NEGATIVE = (
    "blurry, distorted, low quality, deformed, disfigured, extra limbs, "
    "bad anatomy, ugly, watermark, text, realistic photograph, 3d render, scary"
)


class Backend(ImageBackend):
    """Z-Image Base via ComfyUI API."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.server_url = config.get("server_url", "http://127.0.0.1:8188").rstrip("/")
        self.unet_name  = config["unet_name"]
        self.clip_name  = config["clip_name"]
        self.vae_name   = config["vae_name"]
        self.steps      = config.get("steps", 25)
        self.cfg        = config.get("cfg", 4.0)
        self.sampler    = config.get("sampler", "euler")
        self.scheduler  = config.get("scheduler", "simple")

        if not WORKFLOW_PATH.exists():
            raise FileNotFoundError(
                f"API workflow not found: {WORKFLOW_PATH}. "
                f"Export it from ComfyUI GUI (Menu → Save (API Format))."
            )
        self._workflow_template = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
        log.info(f"Z-Image adapter initialized. Workflow loaded from {WORKFLOW_PATH.name}")

    # --------------------------------------------------------------
    # ImageBackend contract
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
        Z-Image likes paragraph-style natural language. If raw_prompt looks like
        SDXL comma-tag style, wrap it lightly — still works.
        For now, just pass through. Phase 3 update will emit proper paragraphs.
        """
        return raw_prompt.strip()

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        aspect_ratio: str = "16:9",
        output_filename: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Path:
        if seed is None:
            seed = random.randint(1, 2**31 - 1)

        # Apply runtime overrides — user settings beat model defaults
        aspect_ratio = rs.get_resolution_override() or aspect_ratio
        width, height = model_registry.get_resolution(aspect_ratio)
        # Per-generation steps / cfg overrides
        effective_steps = rs.get_steps_override() if rs.get_steps_override() is not None else self.steps
        effective_cfg = rs.get_cfg_override() if rs.get_cfg_override() is not None else self.cfg
        formatted_prompt = self.format_prompt_for_backend(prompt)
        negative = negative_prompt or DEFAULT_NEGATIVE

        log.info(f"Generating image — seed={seed}, {width}x{height}, prompt='{prompt[:80]}...'")

        workflow = self._build_workflow(
            prompt=formatted_prompt,
            negative_prompt=negative,
            width=width,
            height=height,
            seed=seed,
            steps_override=effective_steps,
            cfg_override=effective_cfg,
        )

        # 1. Submit to ComfyUI
        client_id = str(uuid.uuid4())
        prompt_id = self._submit_prompt(workflow, client_id)
        log.info(f"Submitted prompt_id={prompt_id}")

        # 2. Wait for it to finish
        history = self._wait_for_completion(prompt_id, timeout=900)  # 15 min for lowvram mode

        # 3. Download the image
        image_path = self._download_output(history, prompt_id, output_filename)
        log.info(f"Image saved: {image_path}")
        return image_path

    # --------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------

    def _build_workflow(self, prompt: str, negative_prompt: str,
                        width: int, height: int, seed: int,
                        steps_override: int = None, cfg_override: float = None) -> dict:
        """
        Take the exported workflow template and fill in dynamic values.
        We duplicate the template so multiple generations don't interfere.

        We walk through the workflow looking for nodes by class_type and inject
        values — this is more robust than hardcoding node IDs (which change
        when the workflow is re-exported).
        """
        wf = json.loads(json.dumps(self._workflow_template))   # deep copy
        effective_steps = steps_override if steps_override is not None else self.steps
        effective_cfg = cfg_override if cfg_override is not None else self.cfg

        # Some workflows nest the subgraph as a single node; others expand it
        # into many nodes. Handle both gracefully by walking all nodes.
        positive_set = False
        negative_set = False
        dims_set = False
        sampler_set = False
        unet_set = False
        clip_set = False
        vae_set = False

        for node_id, node in wf.items():
            if not isinstance(node, dict):
                continue
            class_type = node.get("class_type", "")
            inputs = node.setdefault("inputs", {})

            # Subgraph-style Z-Image "Text to Image" node — has many inputs inline
            if class_type.startswith("Text to Image") or "Z-Image" in str(node.get("_meta", {})):
                if "unet_name" in inputs:
                    inputs["unet_name"] = self.unet_name; unet_set = True
                if "clip_name" in inputs:
                    inputs["clip_name"] = self.clip_name; clip_set = True
                if "vae_name" in inputs:
                    inputs["vae_name"] = self.vae_name; vae_set = True
                if "width" in inputs:
                    inputs["width"] = width
                if "height" in inputs:
                    inputs["height"] = height
                    dims_set = True
                if "steps" in inputs:
                    inputs["steps"] = effective_steps
                if "cfg" in inputs:
                    inputs["cfg"] = effective_cfg
                # Subgraph often has a single giant "prompt" / "text" field
                for k in ("text", "prompt", "positive"):
                    if k in inputs and isinstance(inputs[k], str) and not positive_set:
                        inputs[k] = prompt
                        positive_set = True
                        break

            # Standard node types — handle individually
            elif class_type == "UNETLoader":
                inputs["unet_name"] = self.unet_name; unet_set = True
            elif class_type in ("CLIPLoader", "DualCLIPLoader"):
                # Prefer clip_name1 if it exists, else clip_name
                if "clip_name1" in inputs:
                    inputs["clip_name1"] = self.clip_name
                else:
                    inputs["clip_name"] = self.clip_name
                clip_set = True
            elif class_type == "VAELoader":
                inputs["vae_name"] = self.vae_name; vae_set = True
            elif class_type in ("EmptySD3LatentImage", "EmptyLatentImage"):
                inputs["width"] = width
                inputs["height"] = height
                dims_set = True
            elif class_type == "KSampler":
                inputs["seed"] = seed
                inputs["steps"] = effective_steps
                inputs["cfg"] = effective_cfg
                inputs["sampler_name"] = self.sampler
                inputs["scheduler"] = self.scheduler
                inputs["denoise"] = 1.0
                sampler_set = True
            elif class_type == "CLIPTextEncode":
                # Heuristic: first one is positive, second is negative
                title = node.get("_meta", {}).get("title", "").lower()
                current_text = inputs.get("text", "").lower()
                if ("negative" in title) or ("negative" in current_text[:50]):
                    inputs["text"] = negative_prompt
                    negative_set = True
                elif not positive_set:
                    inputs["text"] = prompt
                    positive_set = True
                else:
                    inputs["text"] = negative_prompt
                    negative_set = True

        log.debug(f"Workflow injection complete: prompt={positive_set}, neg={negative_set}, "
                  f"dims={dims_set}, sampler={sampler_set}")
        if not positive_set:
            log.warning("Positive prompt was not injected — check workflow structure.")
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

    def _wait_for_completion(self, prompt_id: str, timeout: int = 900) -> dict:
        """Poll /history until complete. Heartbeat every 30s so we know ComfyUI isn't frozen."""
        import time
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
            time.sleep(1)

        raise TimeoutError(f"Timed out after {timeout}s waiting for prompt {prompt_id}")

    def _download_output(self, history_entry: dict, prompt_id: str,
                         output_filename: Optional[str]) -> Path:
        outputs = history_entry.get("outputs", {})
        if not outputs:
            raise RuntimeError("ComfyUI finished but produced no outputs.")

        # Find the first node that produced an image
        image_info = None
        for node_id, output in outputs.items():
            imgs = output.get("images")
            if imgs:
                image_info = imgs[0]
                break
        if image_info is None:
            raise RuntimeError("No image in outputs.")

        filename = image_info["filename"]
        subfolder = image_info.get("subfolder", "")
        ftype = image_info.get("type", "output")

        view_url = (
            f"{self.server_url}/view?"
            f"filename={filename}&subfolder={subfolder}&type={ftype}"
        )
        r = requests.get(view_url, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"Could not download image: HTTP {r.status_code}")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if output_filename:
            target = OUTPUT_DIR / output_filename
        else:
            # Default name: prompt_id + original extension
            ext = Path(filename).suffix or ".png"
            target = OUTPUT_DIR / f"img_{prompt_id[:12]}{ext}"

        target.write_bytes(r.content)
        return target


# ==============================================================================
# Standalone test
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from modules import image_backend as ib
    backend = ib.get_active_backend()
    print(f"Backend: {backend.backend_id}")
    ok, msg = backend.health_check()
    print(f"Health: {msg}")
    if not ok:
        raise SystemExit("Backend unhealthy — start ComfyUI first.")

    # Simple paragraph-style test prompt
    test_prompt = (
        "A warm, hand-painted storybook illustration of a happy 5-year-old Indian boy "
        "hugging a bright red toy fire truck to his chest. He sits cross-legged on a "
        "woven jute rug, wearing a mustard yellow kurta, with short black hair. Soft "
        "afternoon sunlight streams through a window with light blue curtains behind "
        "him. Warm color palette, soft pastel tones, Pixar-inspired character design, "
        "gentle rim lighting, cozy children's picture book aesthetic."
    )

    path = backend.generate(test_prompt, aspect_ratio="16:9", output_filename="test_zimage_1.png")
    print(f"Generated: {path}")