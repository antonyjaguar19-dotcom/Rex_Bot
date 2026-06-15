"""
Claw Bot — Flux.2 Klein 9B Image Adapter (via ComfyUI API)

Production storyboard image backend. Flux.2 Klein 9B fp8.
Mirrors comfyui_zimage_base.py patterns. Workflow has 15 nodes;
targeted by class_type + _meta.title.

Quality: best-in-class for storybook/anime/realism mixed content.
Speed on RTX 5080: ~3-6 sec per 1280x720 frame.
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

from modules.image_backend import ImageBackend
from modules import model_registry
from modules import runtime_settings as rs

log = logging.getLogger("claw_bot.image_backend.flux2")


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
WORKFLOW_PATH = PROJECT_ROOT / "05_Config" / "workflows" / "flux2_klein_api.json"
REF_WORKFLOW_PATH = PROJECT_ROOT / "05_Config" / "workflows" / "flux2_klein_ref_api.json"
OUTPUT_DIR = PROJECT_ROOT / "04_Outputs" / "storyboards"
DEFAULT_NEGATIVE = (
    "blurry, low quality, watermark, text, logo, signature, deformed, "
    "extra limbs, distorted faces, ugly, jpeg artifacts, oversaturated"
)


class Backend(ImageBackend):
    """Flux.2 Klein 9B fp8 via ComfyUI API."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.server_url = config.get("server_url", "http://127.0.0.1:8188").rstrip("/")
        self.unet_name  = config["unet_name"]
        self.clip_name  = config["clip_name"]
        self.vae_name   = config["vae_name"]
        self.steps      = int(config.get("steps", 20))
        self.cfg        = float(config.get("cfg", 5.0))
        self.sampler    = config.get("sampler", "euler")
        self.clip_type  = config.get("clip_type", "flux2")

        if not WORKFLOW_PATH.exists():
            raise FileNotFoundError(
                f"API workflow not found: {WORKFLOW_PATH}. "
                f"Export it from ComfyUI GUI (Workflow → Export (API Format))."
            )
        self._workflow_template = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8-sig"))
        self._ref_workflow_template = None
        if REF_WORKFLOW_PATH.exists():
            self._ref_workflow_template = json.loads(REF_WORKFLOW_PATH.read_text(encoding="utf-8-sig"))
            log.info(f"Flux.2 Klein adapter initialized — base + ref workflow loaded.")
        else:
            log.info(f"Flux.2 Klein adapter initialized — base workflow only (no ref workflow found).")

    # --------------------------------------------------------------
    # ImageBackend contract
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
        Flux.2 prefers natural-language paragraph prompts (similar to Z-Image).
        Pass through as-is, just trim whitespace.
        """
        return raw_prompt.strip()

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        aspect_ratio: str = "16:9",
        output_filename: Optional[str] = None,
        seed: Optional[int] = None,
        reference_image: Optional[Path] = None,
        environment_image: Optional[Path] = None,
        cfg_override: Optional[float] = None,
        **kwargs,
    ) -> Path:
        if seed is None:
            seed = random.randint(1, 2**31 - 1)

        # Decide whether to use the reference-latent workflow
        use_ref = (
            reference_image is not None
            and reference_image.exists()
            and self._ref_workflow_template is not None
            and rs.get_reference_mode_enabled()
        )
        ref_image_name = None
        if use_ref:
            try:
                ref_image_name = self._upload_image(reference_image)
                log.info(f"Reference image uploaded: {ref_image_name}")
            except Exception as e:
                log.warning(f"Reference upload failed, falling back to no-reference: {e}")
                use_ref = False

        env_image_name = None
        if environment_image is not None and environment_image.exists():
            try:
                env_image_name = self._upload_image(environment_image)
                log.info(f"Environment image uploaded: {env_image_name}")
            except Exception as e:
                log.warning(f"Environment upload failed: {e}")

        # Resolve dimensions from aspect ratio (registry has these)
        width, height = model_registry.get_resolution(aspect_ratio)

        # Runtime overrides
        rs_steps = rs.get_steps_override()
        rs_cfg = rs.get_cfg_override()
        effective_steps = rs_steps if rs_steps is not None else self.steps
        # Precedence: explicit per-shot cfg_override (beat bias) > runtime setting > model default.
        if cfg_override is not None:
            effective_cfg = float(cfg_override)
        elif rs_cfg is not None:
            effective_cfg = rs_cfg
        else:
            effective_cfg = self.cfg

        formatted_prompt = self.format_prompt_for_backend(prompt)
        negative = negative_prompt or DEFAULT_NEGATIVE

        log.info(
            f"Generating image — seed={seed}, {width}x{height}, "
            f"steps={effective_steps}, cfg={effective_cfg}, "
            f"prompt='{prompt[:80]}...'"
        )

        # 1. Build workflow (reference template if we have a reference image)
        template = self._ref_workflow_template if use_ref else self._workflow_template
        workflow = self._build_workflow(
            prompt=formatted_prompt,
            negative_prompt=negative,
            width=width,
            height=height,
            seed=seed,
            steps=effective_steps,
            cfg=effective_cfg,
            template=template,
            ref_image_name=ref_image_name if use_ref else None,
            env_image_name=env_image_name,
        )

        # 2. Submit
        client_id = str(uuid.uuid4())
        prompt_id = self._submit_prompt(workflow, client_id)
        log.info(f"Submitted prompt_id={prompt_id}")

        # 3. Wait
        history = self._wait_for_completion(prompt_id, timeout=900)

        # 4. Download
        image_path = self._download_output(history, prompt_id, output_filename)
        log.info(f"Image saved: {image_path}")
        return image_path

    # --------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------

    def _upload_image(self, image_path: Path) -> str:
        """Upload a reference image into ComfyUI's input folder. Returns its name."""
        url = f"{self.server_url}/upload/image"
        with open(image_path, "rb") as f:
            files = {"image": (image_path.name, f, "image/png")}
            data = {"overwrite": "true"}
            r = requests.post(url, files=files, data=data, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"Image upload failed (HTTP {r.status_code}): {r.text[:400]}")
        return r.json().get("name", image_path.name)

    def _build_workflow(self, prompt: str, negative_prompt: str,
                        width: int, height: int, seed: int,
                        steps: int, cfg: float,
                        template: Optional[dict] = None,
                        ref_image_name: Optional[str] = None,
                        env_image_name: Optional[str] = None) -> dict:
        """Inject dynamic values into Flux.2 workflow nodes."""
        wf = json.loads(json.dumps(template if template is not None else self._workflow_template))

        # Inject character reference into Node 100
        if ref_image_name:
            if "100" in wf and wf["100"].get("class_type") == "LoadImage":
                wf["100"].setdefault("inputs", {})["image"] = ref_image_name
                log.info(f"Injected character reference image into Node 100: {ref_image_name}")
            else:
                for node in wf.values():
                    if isinstance(node, dict) and node.get("class_type") == "LoadImage":
                        node.setdefault("inputs", {})["image"] = ref_image_name
                        log.info(f"Injected character reference image into LoadImage: {ref_image_name}")
                        break

        # Inject environment reference into Node 200
        if env_image_name and "200" in wf and wf["200"].get("class_type") == "LoadImage":
            wf["200"].setdefault("inputs", {})["image"] = env_image_name
            log.info(f"Injected environment reference image into Node 200: {env_image_name}")

        injected = {
            "positive": False, "negative": False,
            "width": False, "height": False,
            "steps": False, "cfg": False, "seed": False,
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
                inputs.setdefault("type", self.clip_type)
            elif ctype == "VAELoader":
                inputs["vae_name"] = self.vae_name

            # --- Sampling ---
            elif ctype == "RandomNoise":
                inputs["noise_seed"] = seed
                injected["seed"] = True
            elif ctype == "KSamplerSelect":
                inputs["sampler_name"] = self.sampler
            elif ctype == "CFGGuider":
                inputs["cfg"] = cfg
                injected["cfg"] = True
            elif ctype == "Flux2Scheduler":
                inputs["steps"] = steps
                injected["steps"] = True

            # --- Title-targeted primitives ---
            elif ctype == "PrimitiveInt" and title == "width":
                inputs["value"] = int(width)
                injected["width"] = True
            elif ctype == "PrimitiveInt" and title == "height":
                inputs["value"] = int(height)
                injected["height"] = True

            # --- Prompts ---
            elif ctype == "CLIPTextEncode":
                if "negative" in title:
                    inputs["text"] = negative_prompt
                    injected["negative"] = True
                elif "positive" in title:
                    inputs["text"] = prompt
                    injected["positive"] = True

        log.info(
            f"Workflow injection: pos={injected['positive']} neg={injected['negative']} "
            f"w={injected['width']} h={injected['height']} "
            f"steps={injected['steps']} cfg={injected['cfg']} seed={injected['seed']}"
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

    def _wait_for_completion(self, prompt_id: str, timeout: int = 900) -> dict:
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
                log.info(f"Waiting for {prompt_id[:8]}: {elapsed}s elapsed")
            time.sleep(1)

        raise TimeoutError(f"Timed out after {timeout}s waiting for prompt {prompt_id}")

    def _download_output(self, history_entry: dict, prompt_id: str,
                         output_filename: Optional[str]) -> Path:
        outputs = history_entry.get("outputs", {})
        if not outputs:
            raise RuntimeError("ComfyUI finished but produced no outputs.")

        image_info = None
        for node_id, output in outputs.items():
            items = output.get("images")
            if items:
                image_info = items[0]
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
        r = requests.get(view_url, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"Could not download image: HTTP {r.status_code}")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if output_filename:
            target = OUTPUT_DIR / output_filename
        else:
            ext = Path(filename).suffix or ".png"
            target = OUTPUT_DIR / f"img_flux2_{prompt_id[:12]}{ext}"

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
