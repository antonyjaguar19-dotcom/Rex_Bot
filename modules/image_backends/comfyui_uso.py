"""
Claw Bot — ByteDance USO Backend (single-image character consistency)

USO = Unified Style & Subject-driven generation, built on FLUX.1-dev.
Give it ONE reference image of a character and a text prompt describing a new
scene; it renders the SAME character in that new scene/pose/lighting.

  Shot 1: text-to-image (no reference) — establishes character look
  Shot 2+: reference-driven — pass shot 1 (or a cast sheet) as reference_image

Pieces (all ComfyUI-native, downloaded to models/):
  checkpoints/   flux1-dev-fp8.safetensors
  loras/         uso-flux1-dit-lora-v1.safetensors        (the USO DiT LoRA)
  model_patches/ uso-flux1-projector-v1.safetensors       (SigLIP projector — style path)
  clip_vision/   sigclip_vision_patch14_384.safetensors   (style path)

Subject/identity path (what we use): reference image -> ImageScaleToMaxDimension
-> VAEEncode -> ReferenceLatent -> FluxKontextMultiReferenceLatentMethod("uxo/uno")
-> FluxGuidance. The SigLIP projector path is only needed for STYLE references
(not wired here; kept as an optional style_image hook for later).

Pattern mirrors comfyui_kontext_base.py.
"""

import json
import logging
import random
import shutil
import sys
import time as _t
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

log = logging.getLogger("claw_bot.image_backend.uso")

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
COMFY_ROOT = PROJECT_ROOT / "01_ComfyUI" / "ComfyUI_windows_portable" / "ComfyUI"
COMFY_INPUT = COMFY_ROOT / "input"
COMFY_OUTPUT = COMFY_ROOT / "output"

COMFY_URL = "http://127.0.0.1:8188"

# Model file names (must match what's installed in ComfyUI's models/ folders)
CKPT_NAME = "flux1-dev-fp8.safetensors"
USO_LORA_NAME = "uso-flux1-dit-lora-v1.safetensors"

# Generation defaults — USO / Flux-dev recommendations (from the ComfyUI template)
DEFAULT_STEPS = 20
DEFAULT_CFG = 1.0
DEFAULT_GUIDANCE = 3.5
DEFAULT_SAMPLER = "euler"
DEFAULT_SCHEDULER = "simple"
DEFAULT_LORA_STRENGTH = 1.0
# UXO team: 512 preserves MORE character features than 1024 (use 1024 only for
# head-only refs to avoid the subject swelling to fill the frame).
DEFAULT_REF_SIZE = 512

POLL_INTERVAL_SEC = 1.0
JOB_TIMEOUT_SEC = 300


@dataclass
class GenResult:
    success: bool
    image_path: Optional[Path] = None
    seed: int = 0
    error: Optional[str] = None
    backend_id: str = "comfyui_uso"


# ==============================================================================
# COMFY API HELPERS
# ==============================================================================

def _comfy_alive() -> bool:
    try:
        r = requests.get(f"{COMFY_URL}/system_stats", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _post_workflow(workflow: dict) -> str:
    client_id = str(uuid.uuid4())
    r = requests.post(
        f"{COMFY_URL}/prompt",
        json={"prompt": workflow, "client_id": client_id},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    pid = data.get("prompt_id")
    if not pid:
        raise RuntimeError(f"ComfyUI did not return prompt_id: {data}")
    return pid


def _poll_until_done(prompt_id: str, timeout: float) -> dict:
    start = _t.time()
    while _t.time() - start < timeout:
        try:
            r = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=10)
            if r.status_code == 200:
                data = r.json()
                if prompt_id in data:
                    return data[prompt_id]
        except Exception as e:
            log.debug(f"Poll error (will retry): {e}")
        _t.sleep(POLL_INTERVAL_SEC)
    raise TimeoutError(f"USO job {prompt_id} did not finish in {timeout}s")


def _extract_output_image(history_entry: dict) -> Path:
    outputs = history_entry.get("outputs", {})
    candidates = []
    for _node_id, node_out in outputs.items():
        if "images" in node_out:
            for item in node_out["images"]:
                fn = item.get("filename")
                sub = item.get("subfolder", "")
                folder_type = item.get("type", "output")
                if fn:
                    base = COMFY_OUTPUT if folder_type == "output" else COMFY_INPUT
                    candidates.append(base / sub / fn)
    if not candidates:
        raise RuntimeError(f"No image filename in history: {history_entry}")
    for _ in range(50):
        for c in candidates:
            if c.exists():
                return c
        _t.sleep(0.1)
    raise RuntimeError(f"Image not found on disk after 5s. Expected one of: {candidates}")


# ==============================================================================
# WORKFLOW BUILDERS
# ==============================================================================

def _build_workflow_t2i(prompt: str, width: int, height: int, seed: int,
                        steps: int, guidance: float, lora_strength: float) -> dict:
    """Text-to-image (no reference). flux1-dev + USO LoRA."""
    return {
        "11": {"class_type": "CheckpointLoaderSimple",
               "inputs": {"ckpt_name": CKPT_NAME}},
        "43": {"class_type": "LoraLoaderModelOnly",
               "inputs": {"model": ["11", 0], "lora_name": USO_LORA_NAME,
                          "strength_model": lora_strength}},
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["11", 1]}},
        "35": {"class_type": "FluxGuidance",
               "inputs": {"guidance": guidance, "conditioning": ["6", 0]}},
        "48": {"class_type": "ConditioningZeroOut",
               "inputs": {"conditioning": ["6", 0]}},
        "110": {"class_type": "EmptySD3LatentImage",
                "inputs": {"width": width, "height": height, "batch_size": 1}},
        "31": {"class_type": "KSampler",
               "inputs": {"seed": seed, "steps": steps, "cfg": DEFAULT_CFG,
                          "sampler_name": DEFAULT_SAMPLER, "scheduler": DEFAULT_SCHEDULER,
                          "denoise": 1.0, "model": ["43", 0], "positive": ["35", 0],
                          "negative": ["48", 0], "latent_image": ["110", 0]}},
        "8": {"class_type": "VAEDecode",
              "inputs": {"samples": ["31", 0], "vae": ["11", 2]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": "uso", "images": ["8", 0]}},
    }


def _build_workflow_with_refs(prompt: str, width: int, height: int, seed: int,
                              steps: int, guidance: float, lora_strength: float,
                              ref_filenames: list, ref_size: int) -> dict:
    """Reference-driven: keep the subject(s) from ref image(s), new scene from prompt.

    One reference -> one ReferenceLatent. Multiple references (e.g. two characters
    in one shot) -> a CHAIN of ReferenceLatent nodes (UNO-style multi-subject), so
    every subject's identity is carried without a side-by-side composite.
    """
    wf = _build_workflow_t2i(prompt, width, height, seed, steps, guidance, lora_strength)
    prev = ["6", 0]  # conditioning from CLIPTextEncode
    for i, fn in enumerate(ref_filenames):
        li, si, ei, ri = f"ld{i}", f"sc{i}", f"en{i}", f"rl{i}"
        wf[li] = {"class_type": "LoadImage", "inputs": {"image": fn}}
        wf[si] = {"class_type": "ImageScaleToMaxDimension",
                  "inputs": {"image": [li, 0], "upscale_method": "area",
                             "largest_size": ref_size}}
        wf[ei] = {"class_type": "VAEEncode",
                  "inputs": {"pixels": [si, 0], "vae": ["11", 2]}}
        wf[ri] = {"class_type": "ReferenceLatent",
                  "inputs": {"conditioning": prev, "latent": [ei, 0]}}
        prev = [ri, 0]
    wf["41"] = {"class_type": "FluxKontextMultiReferenceLatentMethod",
                "inputs": {"conditioning": prev, "reference_latents_method": "uxo/uno"}}
    # Re-wire FluxGuidance onto the ref-augmented conditioning
    wf["35"]["inputs"]["conditioning"] = ["41", 0]
    return wf


# ==============================================================================
# RESOLUTION
# ==============================================================================

_RESOLUTION_TABLE = {
    "1:1":  (1024, 1024),
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "4:3":  (1152, 896),
    "3:4":  (896, 1152),
}


def _resolve_dims(aspect_ratio: str) -> tuple[int, int]:
    return _RESOLUTION_TABLE.get(aspect_ratio, _RESOLUTION_TABLE["1:1"])


# ==============================================================================
# REFERENCE IMAGE STAGING
# ==============================================================================

def _stage_reference(ref_path: Path) -> str:
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference image not found: {ref_path}")
    COMFY_INPUT.mkdir(parents=True, exist_ok=True)
    staged_name = f"uso_ref_{int(_t.time() * 1000)}_{ref_path.name}"
    shutil.copy2(ref_path, COMFY_INPUT / staged_name)
    return staged_name


def _cleanup_staged(staged_name: str):
    try:
        (COMFY_INPUT / staged_name).unlink(missing_ok=True)
    except Exception as e:
        log.warning(f"Could not clean staged ref {staged_name}: {e}")


# ==============================================================================
# MAIN GENERATE (module-level)
# ==============================================================================

def generate(prompt: str, output_path: Path, aspect_ratio: str = "1:1",
             seed: int = -1, steps: int = DEFAULT_STEPS,
             guidance: float = DEFAULT_GUIDANCE,
             lora_strength: float = DEFAULT_LORA_STRENGTH,
             ref_size: int = DEFAULT_REF_SIZE,
             reference_image: Optional[Path] = None,
             reference_images: Optional[list] = None, **kwargs) -> GenResult:
    if not _comfy_alive():
        return GenResult(success=False,
                         error=f"ComfyUI not reachable at {COMFY_URL} — is it running?")

    if seed is None or seed < 0:
        seed = random.randint(1, 2**31 - 1)

    width, height = _resolve_dims(aspect_ratio)

    # Collect references: explicit list wins, else the single ref. De-dup, keep order.
    refs = []
    if reference_images:
        refs = [Path(r) for r in reference_images if r]
    elif reference_image is not None:
        refs = [Path(reference_image)]

    staged = []
    try:
        for r in refs:
            staged.append(_stage_reference(r))
        if staged:
            wf = _build_workflow_with_refs(prompt, width, height, seed, steps,
                                           guidance, lora_strength, staged, ref_size)
            mode = f"ref×{len(staged)}"
        else:
            wf = _build_workflow_t2i(prompt, width, height, seed, steps,
                                     guidance, lora_strength)
            mode = "t2i"

        log.info(f"USO generate: mode={mode}, dims={width}x{height}, seed={seed}, "
                 f"steps={steps}")

        pid = _post_workflow(wf)
        history = _poll_until_done(pid, JOB_TIMEOUT_SEC)
        comfy_output = _extract_output_image(history)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        # ComfyUI can still hold the output file open for a moment after finishing
        # the job → shutil.move raises PermissionError [WinError 32]. Retry a few
        # times, then fall back to copy (leaving the source for ComfyUI to clean).
        _moved = False
        for _i in range(10):
            try:
                shutil.move(str(comfy_output), str(output_path))
                _moved = True
                break
            except PermissionError:
                _t.sleep(0.4)
        if not _moved:
            shutil.copy2(str(comfy_output), str(output_path))
            try:
                Path(comfy_output).unlink()
            except Exception:
                pass
        log.info(f"USO image saved: {output_path}")
        return GenResult(success=True, image_path=output_path, seed=seed)

    except Exception as e:
        log.exception("USO generation failed")
        return GenResult(success=False, error=str(e), seed=seed)
    finally:
        for s in staged:
            _cleanup_staged(s)


# ==============================================================================
# STANDALONE TEST
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if len(sys.argv) < 2:
        print("Usage: python comfyui_uso.py <prompt> [reference_image_path]")
        sys.exit(1)
    _prompt = sys.argv[1]
    _ref = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    _out = PROJECT_ROOT / "04_Outputs" / "uso_test.png"
    _res = generate(prompt=_prompt, output_path=_out, aspect_ratio="1:1",
                    seed=-1, reference_image=_ref)
    if _res.success:
        print(f"OK Saved: {_res.image_path}  seed={_res.seed}")
    else:
        print(f"FAILED: {_res.error}")
        sys.exit(1)


# ==============================================================================
# BACKEND CLASS
# ==============================================================================

from modules.image_backend import ImageBackend


class Backend(ImageBackend):
    def __init__(self, config: dict):
        super().__init__(config)
        self.server_url = config.get("server_url", COMFY_URL).rstrip("/")
        self.steps = int(config.get("steps", DEFAULT_STEPS))
        self.guidance = float(config.get("guidance", DEFAULT_GUIDANCE))
        self.lora_strength = float(config.get("lora_strength", DEFAULT_LORA_STRENGTH))
        self.ref_size = int(config.get("ref_size", DEFAULT_REF_SIZE))

    def health_check(self):
        try:
            r = requests.get(f"{self.server_url}/system_stats", timeout=5)
            if r.status_code == 200:
                return True, f"ComfyUI alive at {self.server_url}"
            return False, f"ComfyUI returned HTTP {r.status_code}"
        except requests.exceptions.ConnectionError:
            return False, f"Cannot connect to ComfyUI at {self.server_url}. Is it running?"
        except Exception as e:
            return False, f"Health check failed: {e}"

    def format_prompt_for_backend(self, raw_prompt):
        return raw_prompt.strip()

    def generate(self, prompt, negative_prompt=None, aspect_ratio="1:1",
                 output_filename=None, seed=None, reference_image=None,
                 reference_images=None, **kwargs):
        # **kwargs absorbs adapter-agnostic extras the pipeline passes that USO
        # doesn't use (cfg_override, environment_image, style, ...).
        if seed is None:
            seed = random.randint(1, 2**31 - 1)
        self._last_seed = int(seed)  # storyboard_generator reads this for the manifest
        try:
            from modules import runtime_settings as rs
            aspect_ratio = rs.get_resolution_override() or aspect_ratio
            so = rs.get_steps_override()
            effective_steps = so if so is not None else self.steps
        except Exception:
            effective_steps = self.steps
        if output_filename:
            output_path = Path(output_filename)
            if not output_path.is_absolute():
                output_path = PROJECT_ROOT / "04_Outputs" / "storyboards" / output_path
        else:
            output_path = PROJECT_ROOT / "04_Outputs" / f"uso_{int(_t.time()*1000)}.png"
        result = generate(prompt=prompt, output_path=output_path,
                          aspect_ratio=aspect_ratio, seed=seed,
                          steps=effective_steps, guidance=self.guidance,
                          lora_strength=self.lora_strength, ref_size=self.ref_size,
                          reference_image=reference_image,
                          reference_images=reference_images)
        if not result.success:
            raise RuntimeError(f"USO generation failed: {result.error}")
        return result.image_path
