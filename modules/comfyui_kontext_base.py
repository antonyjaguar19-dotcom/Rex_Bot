"""
Claw Bot — Flux Kontext Backend (Phase 4.5)

Image generation via Flux.1 Kontext Dev fp8 with character-consistency
reference image support.

  Shot 1: text-to-image (no reference) — establishes character look
  Shot 2+: image-to-image (shot 1 as reference) — preserves character identity

Pattern mirrors comfyui_zimage_base.py:
  - POST workflow JSON to /prompt
  - Poll /history/{prompt_id}
  - Download result from ComfyUI output folder
"""

import base64
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

log = logging.getLogger("claw_bot.image_backend.kontext")

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
COMFY_ROOT = PROJECT_ROOT / "01_ComfyUI" / "ComfyUI_windows_portable" / "ComfyUI"
COMFY_INPUT = COMFY_ROOT / "input"
COMFY_OUTPUT = COMFY_ROOT / "output"

COMFY_URL = "http://127.0.0.1:8188"

# Model file names (must match what's installed in ComfyUI's models/ folders)
UNET_NAME = "flux1-dev-kontext_fp8_scaled.safetensors"
VAE_NAME = "ae.safetensors"
CLIP_L_NAME = "clip_l.safetensors"
T5_NAME = "t5xxl_fp8_e4m3fn_scaled.safetensors"

# Generation defaults — Kontext recommendations
DEFAULT_STEPS = 20
DEFAULT_CFG = 1.0
DEFAULT_GUIDANCE = 2.5
DEFAULT_SAMPLER = "euler"
DEFAULT_SCHEDULER = "simple"

POLL_INTERVAL_SEC = 1.0
JOB_TIMEOUT_SEC = 300


# ==============================================================================
# RESULT DATACLASS (matches z-image backend's shape)
# ==============================================================================

@dataclass
class GenResult:
    success: bool
    image_path: Optional[Path] = None
    seed: int = 0
    error: Optional[str] = None
    backend_id: str = "comfyui_kontext"


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
    raise TimeoutError(f"Kontext job {prompt_id} did not finish in {timeout}s")


def _extract_output_image(history_entry: dict) -> Path:
    outputs = history_entry.get("outputs", {})
    candidates = []
    for node_id, node_out in outputs.items():
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

    # Wait up to 5s for the file to materialize (ComfyUI may still be flushing)
    for _ in range(50):
        for c in candidates:
            if c.exists():
                return c
        _t.sleep(0.1)

    raise RuntimeError(f"Image not found on disk after 5s. Expected one of: {candidates}")


# ==============================================================================
# WORKFLOW BUILDERS
# ==============================================================================

def _build_workflow_t2i(prompt: str, width: int, height: int, seed: int, steps: int) -> dict:
    """Text-to-image workflow (shot 1, no reference)."""
    return {
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["38", 0]},
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["31", 0], "vae": ["39", 0]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "kontext", "images": ["8", 0]},
        },
        "31": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": DEFAULT_CFG,
                "sampler_name": DEFAULT_SAMPLER,
                "scheduler": DEFAULT_SCHEDULER,
                "denoise": 1.0,
                "model": ["37", 0],
                "positive": ["35", 0],
                "negative": ["135", 0],
                "latent_image": ["124", 0],
            },
        },
        "35": {
            "class_type": "FluxGuidance",
            "inputs": {"guidance": DEFAULT_GUIDANCE, "conditioning": ["6", 0]},
        },
        "37": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": UNET_NAME, "weight_dtype": "default"},
        },
        "38": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": CLIP_L_NAME,
                "clip_name2": T5_NAME,
                "type": "flux",
                "device": "default",
            },
        },
        "39": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": VAE_NAME},
        },
        "124": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "135": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["6", 0]},
        },
    }


def _build_workflow_with_ref(
    prompt: str,
    width: int,
    height: int,
    seed: int,
    steps: int,
    ref_image_filename: str,
) -> dict:
    """Image-to-image workflow with reference (shot 2+).

    Uses Kontext's ReferenceLatent: ref image is encoded via VAE, then mixed
    into the conditioning so the model preserves character identity.
    """
    wf = _build_workflow_t2i(prompt, width, height, seed, steps)
    # Add reference image loader + scaler + encoder + ReferenceLatent
    wf["142"] = {
        "class_type": "LoadImage",
        "inputs": {"image": ref_image_filename},
    }
    wf["173"] = {
        "class_type": "FluxKontextImageScale",
        "inputs": {"image": ["142", 0]},
    }
    wf["180"] = {
        "class_type": "VAEEncode",
        "inputs": {"pixels": ["173", 0], "vae": ["39", 0]},
    }
    wf["177"] = {
        "class_type": "ReferenceLatent",
        "inputs": {
            "conditioning": ["6", 0],
            "latent": ["180", 0],
        },
    }
    # Re-wire FluxGuidance to use the ref-augmented conditioning
    wf["35"]["inputs"]["conditioning"] = ["177", 0]
    return wf


# ==============================================================================
# RESOLUTION SUPPORT
# ==============================================================================

# Kontext native resolutions (Flux Kontext recommended buckets)
_RESOLUTION_TABLE = {
    "1:1":  (1024, 1024),
    "16:9": (1392, 752),
    "9:16": (752, 1392),
    "4:3":  (1184, 880),
    "3:4":  (880, 1184),
}


def _resolve_dims(aspect_ratio: str) -> tuple[int, int]:
    return _RESOLUTION_TABLE.get(aspect_ratio, _RESOLUTION_TABLE["9:16"])


# ==============================================================================
# REFERENCE IMAGE STAGING
# ==============================================================================

def _stage_reference(ref_path: Path) -> str:
    """Copy ref image into ComfyUI's input/ folder, return its filename."""
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference image not found: {ref_path}")
    COMFY_INPUT.mkdir(parents=True, exist_ok=True)
    staged_name = f"kontext_ref_{int(_t.time() * 1000)}_{ref_path.name}"
    shutil.copy2(ref_path, COMFY_INPUT / staged_name)
    return staged_name


def _cleanup_staged(staged_name: str):
    try:
        (COMFY_INPUT / staged_name).unlink(missing_ok=True)
    except Exception as e:
        log.warning(f"Could not clean staged ref {staged_name}: {e}")


# ==============================================================================
# MAIN GENERATE FUNCTION (public API — called by storyboard_generator)
# ==============================================================================

def generate(
    prompt: str,
    output_path: Path,
    aspect_ratio: str = "9:16",
    seed: int = -1,
    steps: int = DEFAULT_STEPS,
    reference_image: Optional[Path] = None,
    **kwargs,
) -> GenResult:
    """Generate one image via Kontext.

    If reference_image is provided, the result preserves the character look
    from that reference (image-to-image mode).
    """
    if not _comfy_alive():
        return GenResult(
            success=False,
            error=f"ComfyUI not reachable at {COMFY_URL} — is it running?",
        )

    if seed is None or seed < 0:
        seed = random.randint(1, 2**31 - 1)

    width, height = _resolve_dims(aspect_ratio)

    staged_ref = None
    try:
        if reference_image is not None:
            staged_ref = _stage_reference(Path(reference_image))
            wf = _build_workflow_with_ref(prompt, width, height, seed, steps, staged_ref)
            mode = "i2i"
        else:
            wf = _build_workflow_t2i(prompt, width, height, seed, steps)
            mode = "t2i"

        log.info(
            f"Kontext generate: mode={mode}, dims={width}x{height}, seed={seed}, "
            f"steps={steps}, ref={staged_ref or 'none'}"
        )

        pid = _post_workflow(wf)
        history = _poll_until_done(pid, JOB_TIMEOUT_SEC)
        comfy_output = _extract_output_image(history)

        # Move to caller-requested output path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(comfy_output), str(output_path))
        log.info(f"Kontext image saved: {output_path}")

        return GenResult(success=True, image_path=output_path, seed=seed)

    except Exception as e:
        log.exception("Kontext generation failed")
        return GenResult(success=False, error=str(e), seed=seed)
    finally:
        if staged_ref:
            _cleanup_staged(staged_ref)


# ==============================================================================
# BACKEND CLASS (matches ImageBackend interface used by storyboard_generator)
# ==============================================================================

from modules.image_backend import ImageBackend


# ==============================================================================
# BACKEND CLASS
# ==============================================================================

from modules.image_backend import ImageBackend


class Backend(ImageBackend):
    """Flux Kontext via ComfyUI API. Supports optional reference image."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.server_url = config.get("server_url", COMFY_URL).rstrip("/")
        self.unet_name = config.get("unet_name", UNET_NAME)
        self.steps = int(config.get("steps", DEFAULT_STEPS))
        self.guidance = float(config.get("guidance", DEFAULT_GUIDANCE))

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
        """Kontext likes paragraph-style natural language. Pass through."""
        return raw_prompt.strip()

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        aspect_ratio: str = "9:16",
        output_filename: Optional[str] = None,
        seed: Optional[int] = None,
        reference_image: Optional[Path] = None,
    ) -> Path:
        if seed is None:
            seed = random.randint(1, 2**31 - 1)

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
            output_path = PROJECT_ROOT / "04_Outputs" / f"kontext_{int(_t.time() * 1000)}.png"

        result = generate(
            prompt=prompt,
            output_path=output_path,
            aspect_ratio=aspect_ratio,
            seed=seed,
            steps=effective_steps,
            reference_image=reference_image,
        )
        if not result.success:
            raise RuntimeError(f"Kontext generation failed: {result.error}")
        return result.image_path


# ==============================================================================
# STANDALONE TEST
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if len(sys.argv) < 2:
        print("Usage: python comfyui_kontext_base.py <prompt> [reference_image_path]")
        sys.exit(1)
    prompt = sys.argv[1]
    ref = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    out = PROJECT_ROOT / "04_Outputs" / "kontext_test.png"
    result = generate(
        prompt=prompt, output_path=out, aspect_ratio="9:16",
        seed=-1, steps=DEFAULT_STEPS, reference_image=ref,
    )
    if result.success:
        print(f"Saved: {result.image_path}, seed: {result.seed}")
    else:
        print(f"Failed: {result.error}")
        sys.exit(1)
