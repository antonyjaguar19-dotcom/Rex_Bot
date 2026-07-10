"""
Claw Bot — mascot thumbnails

Branded thumbnails: your mascot, in a scene that says what the video is about.
The goldfish reel gets "the mascot holding a small bowl tank with a goldfish",
the black-holes reel gets the mascot beside a swirling black hole, and so on.

How it works:
  1. An LLM turns the video's own content into ONE concrete scene description,
     always starring the mascot and one object drawn from the narration.
  2. USO (Flux.1-dev + the USO LoRA) renders that scene using the mascot image
     as a single-image identity reference — same character, new pose and scene.
  3. publish_kit paints the title over it and writes the 16:9 / 9:16 thumbnails.

Drop your mascot at `02_Agent/assets/mascot.png` (jpg/webp also accepted).
A clean, front-facing, full-body shot on a plain background works best — that is
what USO's identity transfer keys off.

Everything degrades: no mascot file, ComfyUI down, USO unhealthy, or a bad LLM
scene all fall back to the ordinary still-based thumbnail. A thumbnail must
never cost you a render.
"""

import logging
import random
import re
import sys
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

log = logging.getLogger("claw_bot.mascot")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
ASSETS_DIR = PROJECT_ROOT / "02_Agent" / "assets"

# Checked in priority order — the first that exists wins.
MASCOT_NAMES = ("mascot.png", "mascot.jpg", "mascot.jpeg", "mascot.webp")

BACKEND_ID = "comfyui_qwen_edit"     # Apache-2.0, pose-free, legible logo
FALLBACK_BACKEND_ID = "comfyui_uso"  # Flux.1-dev, NON-COMMERCIAL — last resort

# Aspects both backends render natively.
NATIVE_ASPECTS = {"16x9": "16:9", "9x16": "9:16", "1x1": "1:1"}

# --- Legacy USO knob -------------------------------------------------------
# USO transfers POSE together with identity, and the reference is a T-pose. At
# the default strength (1.0) every thumbnail came out as the same arms-out
# stance with a prop floating at the mouth. Measured on the cake scene, one seed:
#   1.00  identity perfect, T-pose locked, cake floats (no hand on it)
#   0.80  identity perfect, still T-pose
#   0.65  identity holds, real action pose — arm bent, cake gripped at the mouth
#   0.50  identity BREAKS: blue eyes, spots gone, wrong species
# Only used on the fallback path now; Qwen-Edit needs no such compromise.
POSE_LORA_STRENGTH = 0.65

# --- VRAM residency --------------------------------------------------------
# Qwen-Image-Edit fp8 is ~13.5 GB resident on a 16 GB card. It cannot coexist
# with Wan 2.2 14B (~13.6 GB) or a loaded Ollama model (~12.6 GB) — that
# collision is what once turned a 90 s Wan clip into 16 minutes of PCIe
# thrashing. The policy is one big model at a time:
#
#   1. write the scene with Ollama FIRST, then unload it
#   2. free whatever ComfyUI is holding (Wan / Z-Image / Flux)
#   3. render every aspect back-to-back while Qwen stays warm (15 s each)
#   4. release ComfyUI so the next pipeline stage starts on an empty card
#
# Cold load is ~4 min, warm render 15 s — so step 3 batches, and step 4 happens
# once per video, not once per image.
MODEL_VRAM_GB = 13.5
REQUIRED_FREE_GB = 14.0

# Qwen honours negative conditioning (USO discards it — its workflow wires
# ConditioningZeroOut into the sampler's negative input).
NEGATIVE = (
    "blurry, low quality, deformed hands, extra limbs, extra characters, "
    "text, letters, watermark, logo overlay, signature, frame, border, "
    "t-pose, arms spread wide, stiff symmetrical standing pose, "
    "cluttered background, busy background"
)

# Stated positively as well as in NEGATIVE, because the fallback backend (USO)
# discards negatives entirely — see the note further down.
STYLE_SUFFIX = (
    "dynamic expressive action pose, exaggerated cartoon body language, "
    "leaning into the action, elbows bent, both hands busy with the object, "
    "big readable facial expression, three-quarter view, "
    "subject large in frame, cropped at the knees, "
    "bold vivid solid color background, crisp studio lighting, high contrast, "
    "generous empty space at the bottom for a title, "
    "professional youtube thumbnail art, no text, no letters, no words"
)

# NOTE: the FALLBACK backend (USO) takes no negative prompt. Its workflow wires
# ConditioningZeroOut into the sampler's `negative` input (comfyui_uso.py node
# "48") and its generate() has no negative parameter — anything passed is
# discarded. Flux-dev at cfg 1.0 works that way. On that path the pose is fixed
# by POSE_LORA_STRENGTH instead. Qwen-Edit honours NEGATIVE properly.

_SCENE_SYS = (
    "You design a single funny YouTube thumbnail image for a short video.\n"
    'Output ONLY valid JSON: {"scene": "..."}\n'
    "Rules:\n"
    "- The scene ALWAYS stars 'the mascot character'. Name it exactly that.\n"
    "- Give it a STRONG PHYSICAL ACTION with ONE concrete object or creature "
    "from the video's content. Never 'standing next to' or 'holding' something "
    "passively — make it DO something: eating it, chasing it, hiding it, "
    "dodging it, hugging it, recoiling from it, being splashed by it.\n"
    "- Add a big comic REACTION: caught in the act, guilty, shocked, delighted, "
    "terrified, smug. The face must sell the joke.\n"
    "- Comedy beats accuracy. Mischief is good.\n"
    "- Under 20 words. Describe only what is visible: action, object, expression.\n"
    "- Do NOT describe the background, lighting, or the camera — those are set "
    "for you.\n"
    "- No text, captions, letters or numbers in the image.\n"
    "- No abstractions ('knowledge', 'curiosity'). Physical things only.\n"
    "Examples:\n"
    '{"scene": "the mascot character caught mid-bite stuffing a huge slice of '
    'chocolate cake into its mouth, cheeks bulging, eyes wide with guilt"}\n'
    '{"scene": "the mascot character recoiling as a goldfish leaps out of its '
    'bowl and splashes its face, mouth open in shock"}\n'
    '{"scene": "the mascot character clinging to a fire hose as it whips '
    'sideways, legs flying, panicked grin"}'
)

# The model keeps describing backgrounds anyway; strip them so STYLE_SUFFIX wins.
_BG_RE = re.compile(
    r",?\s*(on|against|with|in front of)?\s*a?\s*"
    r"(plain|solid|simple|clean|bold|blurred)?\s*[\w-]*\s*background\b[^,]*", re.I)


# ==============================================================================
# AVAILABILITY
# ==============================================================================

def mascot_path() -> Optional[Path]:
    """The mascot reference image, or None when it hasn't been added yet."""
    for name in MASCOT_NAMES:
        p = ASSETS_DIR / name
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def active_backend_id() -> str:
    """Qwen-Edit when it's usable, else the non-commercial USO fallback."""
    try:
        from modules import image_backend as ib
        ok, _ = ib.get_named_backend(BACKEND_ID).health_check()
        if ok:
            return BACKEND_ID
    except Exception as e:
        log.warning(f"{BACKEND_ID} unusable ({e}); falling back to {FALLBACK_BACKEND_ID} "
                    f"— NOTE: that model is non-commercial")
    return FALLBACK_BACKEND_ID


def backend_healthy() -> tuple[bool, str]:
    try:
        from modules import image_backend as ib
        bid = active_backend_id()
        ok, msg = ib.get_named_backend(bid).health_check()
        return ok, f"{bid}: {msg}"
    except Exception as e:
        return False, f"no usable image backend: {e}"


# ==============================================================================
# VRAM RESIDENCY — see the policy note at the top of the constants block
# ==============================================================================

def prepare_gpu() -> None:
    """Evict everything else before loading a ~13.5 GB model on a 16 GB card.

    Ollama first: the scene prompt has already been written by then, so its
    12.6 GB is dead weight. Then ComfyUI's current model (Wan / Flux / Z-Image).
    """
    try:
        from modules import gpu_utils
        gpu_utils.free_ollama_vram()
        gpu_utils.ensure_vram_free(min_gb=REQUIRED_FREE_GB, force_ollama_unload=True)
    except Exception as e:
        log.warning(f"VRAM pre-flight failed ({e}); rendering anyway")


def release() -> None:
    """Hand the card back. Call once a batch of thumbnails is done, never
    between aspects — a cold reload costs ~4 min, a warm render 15 s."""
    try:
        from modules import gpu_utils
        gpu_utils.free_comfyui_vram()
        log.info("mascot: released ComfyUI VRAM")
    except Exception as e:
        log.warning(f"could not release VRAM: {e}")


def is_available() -> tuple[bool, str]:
    """Can we render a mascot thumbnail right now? (file + backend)"""
    p = mascot_path()
    if p is None:
        return False, (f"no mascot image — add one at "
                       f"{ASSETS_DIR / 'mascot.png'}")
    ok, msg = backend_healthy()
    if not ok:
        return False, msg
    return True, f"mascot {p.name} + {BACKEND_ID}"


# ==============================================================================
# SCENE PROMPT
# ==============================================================================

_STOP = {
    "the", "and", "for", "are", "was", "were", "that", "this", "with", "from",
    "have", "has", "can", "could", "their", "them", "they", "you", "your",
    "about", "more", "than", "into", "over", "some", "most", "much", "many",
    "these", "those", "there", "here", "what", "when", "which", "while",
    "facts", "fact", "surprising", "amazing", "things", "thing", "actually",
    "really", "never", "knew", "know", "follow", "here's", "wild",
}


def _topic_noun(title: str, context: str, topic: str = "") -> str:
    """A concrete thing to put in the mascot's hands, without an LLM."""
    if topic.strip():
        return topic.strip()
    words = re.findall(r"[A-Za-z]{4,}", f"{title} {context}")
    for w in words:
        if w.lower() not in _STOP:
            return w.lower()
    return "a glowing question mark"


def fallback_scene(title: str, context: str = "", topic: str = "") -> str:
    """Deterministic scene when the LLM can't be reached. Deliberately plain —
    it never invents anything beyond a noun already in the video."""
    noun = _topic_noun(title, context, topic)
    return (f"the mascot character standing next to {noun}, "
            f"friendly pose, plain bold background")


def scene_prompt(title: str, context: str = "", topic: str = "") -> str:
    """Ask the LLM for one concrete mascot scene. Falls back, never raises."""
    fb = fallback_scene(title, context, topic)
    if not context.strip():
        return fb
    try:
        from modules.script_generator import _call_llm, _extract_json
        prompt = (
            f"Video title: {title}\n"
            f"Topic: {topic or '(from content)'}\n\n"
            f"Video content:\n{context[:900]}\n\n"
            f"Describe the thumbnail scene."
        )
        raw = _call_llm(prompt, _SCENE_SYS, role="creative")
        scene = (_extract_json(raw).get("scene") or "").strip()
        scene = re.sub(r"\s+", " ", scene).strip(' "\'')
        if not scene:
            raise ValueError("empty scene")
        if "mascot" not in scene.lower():
            scene = f"the mascot character with {scene}"
        # A scene that smuggles text into the image defeats the title overlay.
        scene = re.sub(r'\b(text|caption|words?|letters?|title)\b', "", scene,
                       flags=re.I)
        # ...and a scene that dictates its own background overrides STYLE_SUFFIX,
        # which is what turned a "bold vivid" brief into flat white.
        scene = _BG_RE.sub("", scene)
        scene = re.sub(r"\s+", " ", scene).strip(" ,")
        if len(scene.split()) > 40:
            scene = " ".join(scene.split()[:40])
        log.info(f"Mascot scene: {scene}")
        return scene
    except Exception as e:
        log.warning(f"Mascot scene LLM failed ({e}); using '{fb}'")
        return fb


# ==============================================================================
# RENDER
# ==============================================================================

def render_scene(scene: str, out_png: Path, aspect: str = "9x16",
                 seed: Optional[int] = None,
                 reference_images: Optional[list] = None) -> Optional[Path]:
    """Render `scene` with the mascot as the identity reference.

    `reference_images` (up to 3, e.g. front + three-quarter + side) is honoured
    by Qwen-Edit. Pass SEPARATE images, never a collage: a grid reference
    destroys the subject — verified, the character vanished entirely.

    Returns None (never raises) when the mascot or the backend is unavailable.
    """
    ref = mascot_path()
    if ref is None:
        log.info("no mascot image; skipping mascot thumbnail")
        return None
    try:
        ok, msg = backend_healthy()
        if not ok:
            log.warning(f"mascot thumbnail skipped: {msg}")
            return None

        if seed is None:
            seed = random.randint(1, 2**31 - 1)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        prompt = f"{scene}, {STYLE_SUFFIX}"
        target = NATIVE_ASPECTS.get(aspect, "9:16")
        bid = active_backend_id()

        if bid == BACKEND_ID:
            from modules.image_backends import comfyui_qwen_edit as qwen
            result = qwen.generate(
                prompt=prompt, output_path=out_png, aspect_ratio=target,
                seed=seed, negative_prompt=NEGATIVE,
                reference_image=ref, reference_images=reference_images,
            )
        else:
            # The USO Backend CLASS reads lora_strength from models.json and
            # ignores the argument, so call its module-level generate().
            from modules.image_backends import comfyui_uso as uso
            result = uso.generate(
                prompt=prompt, output_path=out_png, aspect_ratio=target,
                seed=seed, lora_strength=POSE_LORA_STRENGTH,
                reference_image=ref, reference_images=reference_images,
            )

        if not getattr(result, "success", False):
            log.warning(f"mascot render failed ({getattr(result, 'error', '?')}); "
                        f"falling back to a still")
            return None
        log.info(f"Mascot base rendered: {out_png.name} ({aspect}, {bid})")
        return Path(result.image_path)
    except Exception as e:
        log.warning(f"mascot render failed ({e}); falling back to a still")
        return None


def render_for_video(title: str, context: str, out_dir: Path, stem: str,
                     topic: str = "", aspects: tuple = ("9x16", "16x9"),
                     seed: Optional[int] = None,
                     release_after: bool = True) -> dict:
    """One mascot scene, rendered at each aspect. {aspect: Path} — may be empty.

    The SAME seed and scene are used for every aspect so the two thumbnails are
    recognisably the same artwork, not two unrelated images.

    Memory: the scene is written by Ollama FIRST, then Ollama and whatever
    ComfyUI held are evicted, then every aspect renders while the model stays
    warm. `release_after=False` keeps it resident — use that when looping over
    many videos (a bulk backfill), and call release() once at the end.
    """
    available, why = is_available()
    if not available:
        log.info(f"mascot thumbnails off: {why}")
        return {}

    # 1. LLM first, while the GPU still belongs to whoever had it.
    scene = scene_prompt(title, context, topic)
    if seed is None:
        seed = random.randint(1, 2**31 - 1)

    # 2. Evict Ollama + ComfyUI's current model before loading ~13.5 GB.
    prepare_gpu()

    # 3. Render every aspect warm.
    out: dict = {}
    try:
        for aspect in aspects:
            png = out_dir / f"{stem}_mascot_{aspect}.png"
            got = render_scene(scene, png, aspect=aspect, seed=seed)
            if got:
                out[aspect] = got
    finally:
        # 4. Give the card back so the next pipeline stage starts clean.
        if release_after:
            release()

    if out:
        out["_scene"] = scene
        out["_seed"] = seed
        out["_backend"] = active_backend_id()
    return out
