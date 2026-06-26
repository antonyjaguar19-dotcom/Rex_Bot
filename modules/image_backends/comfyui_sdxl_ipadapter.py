"""
Claw Bot — SDXL + style-LoRA + IP-Adapter Image Adapter (via ComfyUI API)

The KIDS-mode CHARACTER CONSISTENCY backend. Three things stacked:
  1. SDXL checkpoint (sd_xl_base_1.0) — strong, well-supported base.
  2. A real per-STYLE SDXL LoRA (pixar / cartoon_saloon / claymation / stickman) —
     unlike flux, these LoRAs actually apply, so the look is on-style not
     prompt-only.
  3. IP-Adapter PLUS (cubiq ComfyUI_IPAdapter_plus) conditioned on the script's
     CAST SHEET — every shot is pulled toward the same character design, which
     kills the same-species drift the flux backends showed (e.g. two mice whose
     scarves changed colour shot-to-shot).

When NO reference image is supplied (e.g. while rendering the cast sheet itself)
the IP-Adapter nodes are dropped and the sampler reads straight from the LoRA —
plain SDXL+LoRA txt2img.

Workflow template: 05_Config/workflows/sdxl_ipadapter_api.json (node ids 1-11).
Speed on RTX 5080: ~10-20s per 1024x576 frame.
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

_AGENT_DIR = Path(__file__).parent.parent.parent.resolve()
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

from modules.image_backend import ImageBackend
from modules import model_registry
from modules import runtime_settings as rs

log = logging.getLogger("claw_bot.image_backend.sdxl_ipa")

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
WORKFLOW_PATH = PROJECT_ROOT / "05_Config" / "workflows" / "sdxl_ipadapter_api.json"
OUTPUT_DIR = PROJECT_ROOT / "04_Outputs" / "storyboards"

DEFAULT_NEGATIVE = (
    "photo, photorealistic, realistic human, deformed, extra limbs, extra fingers, "
    "fused fingers, blurry, low quality, jpeg artifacts, watermark, text, logo, "
    "signature, ugly, distorted face, oversaturated"
)

# Per-style SDXL LoRA. These are REAL SDXL style LoRAs (no-op on flux) so the
# kids look comes from the LoRA, not just the prompt. trigger = words to prepend
# so the LoRA fires; strength tuned per LoRA.
STYLE_LORAS = {
    "pixar":          {"file": "disneyPixarCartoon_v10.safetensors",       "trigger": "Disney Pixar style 3D animation", "strength": 0.85},
    "cartoon_saloon": {"file": "Cartoon_Saloon_Style_XL_kk-000009.safetensors", "trigger": "cartoon saloon style, hand-drawn 2D", "strength": 0.9},
    "claymation":     {"file": "CLAYMATE_V2.03_.safetensors",              "trigger": "claymation, stop-motion clay model", "strength": 0.9},
    "stickman":       {"file": "Stickman_stick_figure-000006.safetensors", "trigger": "stickman, simple stick figure drawing", "strength": 1.0},
}


class Backend(ImageBackend):
    """SDXL + style LoRA + IP-Adapter via ComfyUI API."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.server_url = config.get("server_url", "http://127.0.0.1:8188").rstrip("/")
        self.checkpoint = config.get("checkpoint", "sd_xl_base_1.0.safetensors")
        self.steps      = int(config.get("steps", 28))
        self.cfg        = float(config.get("cfg", 6.5))
        self.sampler    = config.get("sampler", "dpmpp_2m")
        self.scheduler  = config.get("scheduler", "karras")
        self.ipa_preset = config.get("ipadapter_preset", "PLUS (high strength)")
        self.ipa_weight = float(config.get("ipadapter_weight", 0.75))
        self.default_style = config.get("default_style", "pixar")
        self.style_loras = {**STYLE_LORAS, **(config.get("style_loras") or {})}

        if not WORKFLOW_PATH.exists():
            raise FileNotFoundError(f"SDXL+IPA workflow not found: {WORKFLOW_PATH}.")
        self._template = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8-sig"))
        self._mask_cache: dict = {}
        log.info("SDXL+IP-Adapter adapter initialized "
                 f"(ckpt={self.checkpoint}, preset={self.ipa_preset}, "
                 f"{len(self.style_loras)} style LoRAs).")

    # ---------------- ImageBackend contract ----------------

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

    def _resolve_style(self, style: Optional[str]) -> dict:
        s = (style or self.default_style or "pixar").strip().lower()
        return self.style_loras.get(s) or self.style_loras.get(self.default_style) \
            or next(iter(self.style_loras.values()))

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        aspect_ratio: str = "16:9",
        output_filename: Optional[str] = None,
        seed: Optional[int] = None,
        reference_image: Optional[Path] = None,
        environment_image: Optional[Path] = None,   # accepted, unused (one IP ref)
        cfg_override: Optional[float] = None,
        style: Optional[str] = None,
        **kwargs,
    ) -> Path:
        if seed is None:
            seed = random.randint(1, 2**31 - 1)

        # Accept either a single reference_image or a list reference_images (the
        # per-character solo sheets). Multiple refs are fed as an IMAGE BATCH into
        # IP-Adapter (combine_embeds=concat) — conditions on BOTH identities
        # without imposing a side-by-side layout (a composited diptych made the
        # output a split-screen).
        refs = kwargs.get("reference_images")
        if not refs:
            refs = [reference_image] if reference_image is not None else []
        ref_paths = [Path(r) for r in refs if r is not None and Path(r).exists()]
        ref_names = []
        for rp in ref_paths:
            try:
                ref_names.append(self._upload_image(rp))
            except Exception as e:
                log.warning(f"Reference upload failed for {rp.name}: {e}")
        use_ref = bool(ref_names)
        if use_ref:
            log.info(f"IP-Adapter references uploaded: {ref_names}")

        width, height = model_registry.get_resolution(aspect_ratio)
        rs_steps = rs.get_steps_override()
        rs_cfg = rs.get_cfg_override()
        effective_steps = rs_steps if rs_steps is not None else self.steps
        if cfg_override is not None:
            effective_cfg = float(cfg_override)
        elif rs_cfg is not None:
            effective_cfg = rs_cfg
        else:
            effective_cfg = self.cfg

        lora = self._resolve_style(style)
        # Prepend the LoRA trigger so the style fires; the incoming prompt already
        # carries scene + locked character tokens.
        pos = f"{lora['trigger']}. {self.format_prompt_for_backend(prompt)}"
        neg = negative_prompt or DEFAULT_NEGATIVE

        log.info(f"SDXL+IPA gen — style={style or self.default_style} "
                 f"lora={lora['file']} ref={'yes' if use_ref else 'no'} "
                 f"seed={seed} {width}x{height} steps={effective_steps} cfg={effective_cfg}")

        wf = self._build_workflow(
            pos=pos, neg=neg, width=width, height=height, seed=seed,
            steps=effective_steps, cfg=effective_cfg, lora=lora,
            ref_names=ref_names if use_ref else [],
            output_filename=output_filename,
        )

        client_id = str(uuid.uuid4())
        prompt_id = self._submit_prompt(wf, client_id)
        log.info(f"Submitted prompt_id={prompt_id}")
        history = self._wait_for_completion(prompt_id, timeout=900)
        image_path = self._download_output(history, prompt_id, output_filename)
        log.info(f"Image saved: {image_path}")
        return image_path

    # ---------------- internals ----------------

    def _upload_image(self, image_path: Path) -> str:
        url = f"{self.server_url}/upload/image"
        with open(image_path, "rb") as f:
            files = {"image": (image_path.name, f, "image/png")}
            r = requests.post(url, files=files, data={"overwrite": "true"}, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"Image upload failed (HTTP {r.status_code}): {r.text[:400]}")
        return r.json().get("name", image_path.name)

    def _build_workflow(self, pos: str, neg: str, width: int, height: int,
                        seed: int, steps: int, cfg: float, lora: dict,
                        ref_names: list, output_filename: Optional[str]) -> dict:
        wf = json.loads(json.dumps(self._template))

        wf["1"]["inputs"]["ckpt_name"] = self.checkpoint
        wf["2"]["inputs"]["lora_name"] = lora["file"]
        wf["2"]["inputs"]["strength_model"] = lora.get("strength", 0.85)
        wf["2"]["inputs"]["strength_clip"] = lora.get("strength", 0.85)
        wf["3"]["inputs"]["preset"] = self.ipa_preset
        wf["5"]["inputs"]["weight"] = self.ipa_weight
        wf["6"]["inputs"]["text"] = pos
        wf["7"]["inputs"]["text"] = neg
        wf["8"]["inputs"]["width"] = width
        wf["8"]["inputs"]["height"] = height
        wf["9"]["inputs"].update({"seed": seed, "steps": steps, "cfg": cfg,
                                  "sampler_name": self.sampler, "scheduler": self.scheduler})
        if output_filename:
            wf["11"]["inputs"]["filename_prefix"] = Path(output_filename).stem

        ref_names = ref_names or []
        if not ref_names:
            # No reference → drop the IP-Adapter chain (nodes 3,4,5) and feed the
            # sampler straight from the LoRA model output.
            wf["9"]["inputs"]["model"] = ["2", 0]
            for nid in ("3", "4", "5"):
                wf.pop(nid, None)
            return wf

        if len(ref_names) == 1:
            # Single character → the one IP-Adapter node (template node 5).
            wf["4"]["inputs"]["image"] = ref_names[0]
            wf["5"]["inputs"]["image"] = ["4", 0]
            return wf

        # MULTIPLE different characters → REGIONAL masking. One IPAdapterAdvanced
        # per character, each masked to its vertical slice of the frame (char 0 =
        # left, char 1 = right, ...). The KSampler still paints ONE coherent scene;
        # the masks only steer each character's IDENTITY to its region. This keeps
        # both species distinct and in one frame — a combined sheet collapses them
        # to one species, a batch/concat blends them into a hybrid, a side-by-side
        # composite reproduces a split-screen seam.
        masks = self._region_masks(len(ref_names), width, height)
        wf.pop("4", None)
        wf.pop("5", None)
        prev_model = ["3", 0]          # model out of the unified loader
        ipadapter = ["3", 1]
        for i, name in enumerate(ref_names):
            img_id = f"ipa_img_{i}"
            wf[img_id] = {"class_type": "LoadImage", "inputs": {"image": name}}
            mimg_id = f"ipa_mimg_{i}"
            wf[mimg_id] = {"class_type": "LoadImage", "inputs": {"image": masks[i]}}
            mask_id = f"ipa_mask_{i}"
            wf[mask_id] = {"class_type": "ImageToMask",
                           "inputs": {"image": [mimg_id, 0], "channel": "red"}}
            ipa_id = f"ipa_node_{i}"
            wf[ipa_id] = {"class_type": "IPAdapterAdvanced", "inputs": {
                "model": prev_model, "ipadapter": ipadapter, "image": [img_id, 0],
                "attn_mask": [mask_id, 0], "weight": self.ipa_weight,
                "weight_type": "linear", "combine_embeds": "concat",
                "start_at": 0.0, "end_at": 1.0, "embeds_scaling": "V only"}}
            prev_model = [ipa_id, 0]
        wf["9"]["inputs"]["model"] = prev_model
        return wf

    def _region_masks(self, n: int, width: int, height: int) -> list:
        """One uploaded mask image per character: a white vertical band over that
        character's slice of the frame, black elsewhere. ImageToMask(red) turns it
        into the IP-Adapter attn_mask. Cached per (n,width,height)."""
        key = (n, width, height)
        if key in self._mask_cache:
            return self._mask_cache[key]
        from PIL import Image
        import tempfile, os
        names = []
        tmpdir = Path(tempfile.gettempdir()) / "claw_ipa_masks"
        tmpdir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img = Image.new("RGB", (width, height), (0, 0, 0))
            x0 = int(round(width * i / n))
            x1 = int(round(width * (i + 1) / n))
            band = Image.new("RGB", (max(1, x1 - x0), height), (255, 255, 255))
            img.paste(band, (x0, 0))
            fp = tmpdir / f"mask_{n}_{i}_{width}x{height}.png"
            img.save(fp)
            names.append(self._upload_image(fp))
        self._mask_cache[key] = names
        return names

    def _submit_prompt(self, workflow: dict, client_id: str) -> str:
        r = requests.post(f"{self.server_url}/prompt",
                          json={"prompt": workflow, "client_id": client_id}, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"ComfyUI rejected prompt (HTTP {r.status_code}): {r.text[:500]}")
        data = r.json()
        if "prompt_id" not in data:
            raise RuntimeError(f"ComfyUI response missing prompt_id: {data}")
        return data["prompt_id"]

    def _wait_for_completion(self, prompt_id: str, timeout: int = 900) -> dict:
        url = f"{self.server_url}/history/{prompt_id}"
        start = last = time.time()
        while time.time() - start < timeout:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if prompt_id in data:
                    entry = data[prompt_id]
                    status = entry.get("status", {})
                    if status.get("completed") is True:
                        log.info(f"Prompt {prompt_id[:8]} completed in {int(time.time()-start)}s")
                        return entry
                    if status.get("status_str") == "error":
                        raise RuntimeError(f"ComfyUI error: {status.get('messages', [])}")
            if time.time() - last >= 30:
                last = time.time()
                log.info(f"Waiting for {prompt_id[:8]}: {int(last-start)}s elapsed")
            time.sleep(1)
        raise TimeoutError(f"Timed out after {timeout}s waiting for {prompt_id}")

    def _download_output(self, history_entry: dict, prompt_id: str,
                         output_filename: Optional[str]) -> Path:
        outputs = history_entry.get("outputs", {})
        image_info = None
        for _nid, output in outputs.items():
            if output.get("images"):
                image_info = output["images"][0]
                break
        if image_info is None:
            raise RuntimeError("ComfyUI finished but produced no image.")
        view_url = (f"{self.server_url}/view?filename={image_info['filename']}"
                    f"&subfolder={image_info.get('subfolder','')}&type={image_info.get('type','output')}")
        r = requests.get(view_url, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"Could not download image: HTTP {r.status_code}")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if output_filename:
            target = OUTPUT_DIR / output_filename
        else:
            target = OUTPUT_DIR / f"img_sdxl_{prompt_id[:12]}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(r.content)
        return target
