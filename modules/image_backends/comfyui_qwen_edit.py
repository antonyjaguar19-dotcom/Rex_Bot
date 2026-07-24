"""
Claw Bot — Qwen-Image-Edit-2509 backend (reference-driven, Apache-2.0)

Give it 1-3 reference images and a prompt describing a NEW scene. It keeps the
subject's identity while freely changing pose, action and framing.

Why this exists (measured against USO on the same mascot, scene and seed):

    metric            USO (Flux.1-dev)         Qwen-Image-Edit-2509
    ------            ----------------         --------------------
    warm render       ~150 s                   15 s (4-step) / 93 s (20-step)
    pose              copied from the ref;     free at full strength
                      only breaks at lora
                      0.65, which degrades
                      identity
    text on clothing  garbled ("ROCKMME")      legible ("REXJAW")
    negative prompt   silently discarded       honoured
    licence           NON-COMMERCIAL           Apache-2.0

The licence is the decisive one: thumbnails are the most public artefact this
project makes, and Flux.1-dev forbids commercial use.

Two hard-won workflow details:
  * Use an EMPTY latent at the TARGET aspect. Feeding VAEEncode(reference) as
    the latent makes the output inherit the reference's aspect ratio and, in
    testing, hallucinated a fake red watermark into the corner.
  * The reference goes in through TextEncodeQwenImageEditPlus (image1..image3),
    which is the 2509 multi-image encoder. Pass separate images, never a
    collage — a 4-view grid destroys the subject (verified on USO; the same
    failure mode applies to any latent-encoded reference).

Memory: the fp8 model is ~20 GB on disk and ~13.5 GB resident. It cannot share
the GPU with Wan (13.6 GB) or a loaded Ollama model (12.6 GB) on a 16 GB card.
See modules/mascot.py for the residency policy — callers render a BATCH warm,
then release.
"""

import logging
import random
import shutil
import time as _t
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

log = logging.getLogger("claw_bot.image_backend.qwen_edit")

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
COMFY_ROOT = PROJECT_ROOT / "01_ComfyUI" / "ComfyUI_windows_portable" / "ComfyUI"
COMFY_INPUT = COMFY_ROOT / "input"
COMFY_OUTPUT = COMFY_ROOT / "output"
COMFY_URL = "http://127.0.0.1:8188"

# 2511 (Dec 2025) over 2509: better character consistency, less image drift, stronger
# geometric reasoning — chosen after an A/B on the same lesson beat + seed. Same CLIP/VAE,
# same ~20 GB fp8 footprint, Apache-2.0. The 2509 file stays on disk; revert by name.
UNET_NAME = "qwen_image_edit_2511_fp8mixed.safetensors"
CLIP_NAME = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
VAE_NAME = "qwen_image_vae.safetensors"
# Lessons render full-quality (use_lightning=False), so this 2509 Lightning LoRA is not on
# the lesson path; kept for any fast-preview caller until a 2511 Lightning LoRA is installed.
LIGHTNING_LORA = "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"

# The 4-step Lightning LoRA runs at cfg 1.0 (distilled guidance). Without it,
# 20 steps at cfg 2.5. Lightning was 10x faster than USO with no visible loss on
# cartoon subjects, so it is the default.
DEFAULT_USE_LIGHTNING = True
DEFAULT_STEPS_LIGHTNING = 4
DEFAULT_CFG_LIGHTNING = 1.0
DEFAULT_STEPS_FULL = 20
DEFAULT_CFG_FULL = 2.5
DEFAULT_SAMPLER = "euler"
DEFAULT_SCHEDULER = "simple"
DEFAULT_SHIFT = 3.0            # ModelSamplingAuraFlow, standard for Qwen

MAX_REFS = 3                   # TextEncodeQwenImageEditPlus takes image1..image3

_RESOLUTION_TABLE = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "4:3": (1152, 896),
    "3:4": (896, 1152),
}

POLL_INTERVAL_SEC = 1.0
JOB_TIMEOUT_SEC = 600           # a cold 20 GB load can take ~4 min


@dataclass
class GenResult:
    success: bool
    image_path: Optional[Path] = None
    seed: int = 0
    error: Optional[str] = None
    backend_id: str = "comfyui_qwen_edit"
    # True when the GPU/ComfyUI is now unusable, not just this one prompt.
    # A CUDA illegal-access kills the context: every later prompt fails too, so
    # a caller looping over many videos must STOP rather than quietly emit
    # fallback art for the rest of the batch.
    fatal: bool = False


_FATAL_SIGNS = (
    "illegal memory access",
    "cuda error",
    "cuda_error",
    "out of memory",
    "device-side assert",
    "no cuda gpus are available",
)


def _is_fatal(text: str) -> bool:
    t = (text or "").lower()
    return any(s in t for s in _FATAL_SIGNS)


# ==============================================================================
# COMFY API
# ==============================================================================

def _comfy_alive() -> bool:
    try:
        return requests.get(f"{COMFY_URL}/system_stats", timeout=5).status_code == 200
    except Exception:
        return False


def _post_workflow(workflow: dict) -> str:
    r = requests.post(f"{COMFY_URL}/prompt",
                      json={"prompt": workflow, "client_id": str(uuid.uuid4())},
                      timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"ComfyUI rejected the workflow ({r.status_code}): "
                           f"{r.text[:500]}")
    pid = r.json().get("prompt_id")
    if not pid:
        raise RuntimeError(f"ComfyUI returned no prompt_id: {r.text[:200]}")
    return pid


def _poll_until_done(prompt_id: str, timeout: float) -> dict:
    start = _t.time()
    while _t.time() - start < timeout:
        try:
            r = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=10)
            if r.status_code == 200 and prompt_id in r.json():
                return r.json()[prompt_id]
        except Exception as e:
            log.debug(f"poll retry: {e}")
        _t.sleep(POLL_INTERVAL_SEC)
    raise TimeoutError(f"Qwen job {prompt_id} did not finish in {timeout}s")


def _extract_output_image(history_entry: dict) -> Path:
    candidates = []
    for node_out in history_entry.get("outputs", {}).values():
        for item in node_out.get("images", []):
            fn = item.get("filename")
            if not fn:
                continue
            base = COMFY_OUTPUT if item.get("type", "output") == "output" else COMFY_INPUT
            candidates.append(base / item.get("subfolder", "") / fn)
    if not candidates:
        status = history_entry.get("status", {})
        raise RuntimeError(f"Qwen produced no image. status={status}")
    for _ in range(50):
        for c in candidates:
            if c.exists():
                return c
        _t.sleep(0.1)
    raise RuntimeError(f"Image never appeared on disk: {candidates}")


def _stage_reference(ref: Path) -> str:
    if not ref.exists():
        raise FileNotFoundError(f"Reference image not found: {ref}")
    COMFY_INPUT.mkdir(parents=True, exist_ok=True)
    name = f"qwen_ref_{int(_t.time() * 1000)}_{ref.name}"
    shutil.copy2(ref, COMFY_INPUT / name)
    return name


def _cleanup_staged(name: str) -> None:
    try:
        (COMFY_INPUT / name).unlink(missing_ok=True)
    except Exception as e:
        log.debug(f"could not remove staged ref {name}: {e}")


def _resolve_dims(aspect_ratio: str) -> tuple[int, int]:
    return _RESOLUTION_TABLE.get(aspect_ratio, _RESOLUTION_TABLE["1:1"])


# ==============================================================================
# WORKFLOW
# ==============================================================================

def _build_workflow(prompt: str, negative: str, width: int, height: int,
                    seed: int, steps: int, cfg: float, staged: list[str],
                    use_lightning: bool) -> dict:
    wf: dict = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": UNET_NAME, "weight_dtype": "fp8_e4m3fn"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": CLIP_NAME, "type": "qwen_image"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_NAME}},
        "5": {"class_type": "ModelSamplingAuraFlow",
              "inputs": {"model": ["1", 0], "shift": DEFAULT_SHIFT}},
        # Empty latent at the TARGET aspect — never VAEEncode(reference), which
        # forces the reference's aspect and invites a hallucinated watermark.
        "9": {"class_type": "EmptySD3LatentImage",
              "inputs": {"width": width, "height": height, "batch_size": 1}},
        "10": {"class_type": "KSampler",
               "inputs": {"model": ["5", 0], "seed": seed, "steps": steps,
                          "cfg": cfg, "sampler_name": DEFAULT_SAMPLER,
                          "scheduler": DEFAULT_SCHEDULER,
                          "positive": ["7", 0], "negative": ["8", 0],
                          "latent_image": ["9", 0], "denoise": 1.0}},
        "11": {"class_type": "VAEDecode",
               "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
        "12": {"class_type": "SaveImage",
               "inputs": {"images": ["11", 0], "filename_prefix": "qwen_edit"}},
    }

    pos = {"clip": ["2", 0], "prompt": prompt, "vae": ["3", 0]}
    neg = {"clip": ["2", 0], "prompt": negative, "vae": ["3", 0]}
    for i, name in enumerate(staged[:MAX_REFS], start=1):
        wf[f"4_{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        pos[f"image{i}"] = [f"4_{i}", 0]
        neg[f"image{i}"] = [f"4_{i}", 0]

    wf["7"] = {"class_type": "TextEncodeQwenImageEditPlus", "inputs": pos}
    wf["8"] = {"class_type": "TextEncodeQwenImageEditPlus", "inputs": neg}

    if use_lightning:
        wf["6"] = {"class_type": "LoraLoaderModelOnly",
                   "inputs": {"model": ["5", 0], "lora_name": LIGHTNING_LORA,
                              "strength_model": 1.0}}
        wf["10"]["inputs"]["model"] = ["6", 0]
    return wf


# ==============================================================================
# PUBLIC
# ==============================================================================

def generate(prompt: str, output_path: Path, aspect_ratio: str = "1:1",
             seed: int = -1, steps: Optional[int] = None,
             cfg: Optional[float] = None,
             use_lightning: bool = DEFAULT_USE_LIGHTNING,
             negative_prompt: str = "",
             reference_image: Optional[Path] = None,
             reference_images: Optional[list] = None, **kwargs) -> GenResult:
    if not _comfy_alive():
        return GenResult(success=False,
                         error=f"ComfyUI not reachable at {COMFY_URL} — is it running?")

    if seed is None or seed < 0:
        seed = random.randint(1, 2**31 - 1)
    if steps is None:
        steps = DEFAULT_STEPS_LIGHTNING if use_lightning else DEFAULT_STEPS_FULL
    if cfg is None:
        cfg = DEFAULT_CFG_LIGHTNING if use_lightning else DEFAULT_CFG_FULL

    width, height = _resolve_dims(aspect_ratio)

    refs = []
    if reference_images:
        refs = [Path(r) for r in reference_images if r]
    elif reference_image is not None:
        refs = [Path(reference_image)]
    if len(refs) > MAX_REFS:
        log.warning(f"{len(refs)} references given; Qwen accepts {MAX_REFS} — "
                    f"using the first {MAX_REFS}")
        refs = refs[:MAX_REFS]

    staged: list[str] = []
    try:
        for r in refs:
            staged.append(_stage_reference(r))
        wf = _build_workflow(prompt, negative_prompt, width, height, seed,
                             steps, cfg, staged, use_lightning)
        mode = f"ref×{len(staged)}" if staged else "t2i"
        log.info(f"Qwen-Edit generate: mode={mode}, dims={width}x{height}, "
                 f"seed={seed}, steps={steps}, cfg={cfg}, "
                 f"lightning={use_lightning}")

        t0 = _t.time()
        pid = _post_workflow(wf)
        history = _poll_until_done(pid, JOB_TIMEOUT_SEC)
        produced = _extract_output_image(history)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(produced, output_path)
        log.info(f"Qwen-Edit image saved: {output_path} ({_t.time() - t0:.0f}s)")
        return GenResult(success=True, image_path=output_path, seed=seed)
    except Exception as e:
        fatal = _is_fatal(str(e))
        if fatal:
            # Keep the log readable: ComfyUI dumps every tensor in the error.
            log.error("Qwen-Edit hit a FATAL GPU error (CUDA context is dead; "
                      "ComfyUI must be restarted before it can render again)")
        else:
            log.warning(f"Qwen-Edit generation failed: {e}")
        return GenResult(success=False, seed=seed, error=str(e)[:400], fatal=fatal)
    finally:
        for name in staged:
            _cleanup_staged(name)


# ==============================================================================
# BACKEND CLASS (pluggable-adapter contract)
# ==============================================================================

from modules.image_backend import ImageBackend  # noqa: E402


class Backend(ImageBackend):
    def __init__(self, config: dict):
        super().__init__(config)
        self.server_url = config.get("server_url", COMFY_URL).rstrip("/")
        self.use_lightning = bool(config.get("use_lightning", DEFAULT_USE_LIGHTNING))
        self.steps = config.get("steps")
        self.cfg = config.get("cfg")
        self._last_seed = 0

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

    def format_prompt_for_backend(self, raw_prompt: str) -> str:
        return raw_prompt

    def generate(self, prompt, negative_prompt=None, aspect_ratio="1:1",
                 output_filename=None, seed=None, reference_image=None,
                 reference_images=None, **kwargs):
        if seed is None:
            seed = random.randint(1, 2**31 - 1)
        self._last_seed = int(seed)
        if output_filename:
            output_path = Path(output_filename)
            if not output_path.is_absolute():
                output_path = PROJECT_ROOT / "04_Outputs" / "storyboards" / output_path
        else:
            output_path = PROJECT_ROOT / "04_Outputs" / f"qwen_{int(_t.time()*1000)}.png"

        result = generate(prompt=prompt, output_path=output_path,
                          aspect_ratio=aspect_ratio, seed=seed,
                          steps=self.steps, cfg=self.cfg,
                          use_lightning=self.use_lightning,
                          negative_prompt=negative_prompt or "",
                          reference_image=reference_image,
                          reference_images=reference_images)
        if not result.success:
            raise RuntimeError(f"Qwen-Edit generation failed: {result.error}")
        return result.image_path
