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

BACKEND_ID = "comfyui_uso"

# Aspects USO renders natively (see comfyui_uso._RESOLUTION_TABLE).
NATIVE_ASPECTS = {"16x9": "16:9", "9x16": "9:16", "1x1": "1:1"}

# USO transfers POSE together with identity, and the reference is a T-pose. At
# the default strength (1.0) every thumbnail came out as the same arms-out
# stance with a prop floating at the mouth — no amount of negative prompting
# broke it. Measured on the cake scene, same seed:
#   1.00  identity perfect, T-pose locked, cake floats (no hand on it)
#   0.80  identity perfect, still T-pose
#   0.65  identity holds, real action pose — arm bent, cake gripped at the mouth
#   0.50  identity BREAKS: blue eyes, spots gone, wrong species
# 0.65 is the knee of the curve: the weakest reference that still is the mascot.
POSE_LORA_STRENGTH = 0.65

# USO ignores negative prompts (see the note below), so every constraint has to
# be stated positively — including "not a T-pose", phrased as what we DO want.
STYLE_SUFFIX = (
    "dynamic expressive action pose, exaggerated cartoon body language, "
    "leaning into the action, elbows bent, both hands busy with the object, "
    "big readable facial expression, three-quarter view, "
    "subject large in frame, cropped at the knees, "
    "bold vivid solid color background, crisp studio lighting, high contrast, "
    "generous empty space at the bottom for a title, "
    "professional youtube thumbnail art, no text, no letters, no words"
)

# NOTE: USO takes NO negative prompt. Its workflow wires ConditioningZeroOut
# into the sampler's `negative` input (comfyui_uso.py node "48"), and the module
# level generate() has no negative parameter — anything passed is discarded.
# Flux-dev at cfg 1.0 works that way. So everything that shapes the image has to
# live in the POSITIVE prompt above, and the pose is fixed by POSE_LORA_STRENGTH.
# (This was learned the hard way: an anti-T-pose negative prompt did nothing.)

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


def backend_healthy() -> tuple[bool, str]:
    try:
        from modules import image_backend as ib
        return ib.get_named_backend(BACKEND_ID).health_check()
    except Exception as e:
        return False, f"{BACKEND_ID} unavailable: {e}"


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
                 seed: Optional[int] = None) -> Optional[Path]:
    """Render `scene` with the mascot as USO's identity reference.

    Returns None (never raises) when the mascot or the backend is unavailable.
    """
    ref = mascot_path()
    if ref is None:
        log.info("no mascot image; skipping mascot thumbnail")
        return None
    try:
        from modules import gpu_utils
        # The Backend CLASS reads lora_strength from models.json and ignores the
        # argument, so call the adapter's module-level generate() — POSE_LORA_
        # STRENGTH is the whole reason the mascot isn't stuck in a T-pose.
        from modules.image_backends import comfyui_uso as uso

        ok, msg = backend_healthy()
        if not ok:
            log.warning(f"mascot thumbnail skipped: {msg}")
            return None

        gpu_utils.ensure_vram_free(min_gb=6.0)
        if seed is None:
            seed = random.randint(1, 2**31 - 1)

        out_png.parent.mkdir(parents=True, exist_ok=True)
        result = uso.generate(
            prompt=f"{scene}, {STYLE_SUFFIX}",
            output_path=out_png,
            aspect_ratio=NATIVE_ASPECTS.get(aspect, "9:16"),
            seed=seed,
            lora_strength=POSE_LORA_STRENGTH,
            reference_image=ref,
        )
        if not getattr(result, "success", False):
            log.warning(f"mascot render failed ({getattr(result, 'error', '?')}); "
                        f"falling back to a still")
            return None
        log.info(f"Mascot base rendered: {out_png.name} ({aspect}, "
                 f"lora={POSE_LORA_STRENGTH})")
        return Path(result.image_path)
    except Exception as e:
        log.warning(f"mascot render failed ({e}); falling back to a still")
        return None


def render_for_video(title: str, context: str, out_dir: Path, stem: str,
                     topic: str = "", aspects: tuple = ("9x16", "16x9"),
                     seed: Optional[int] = None) -> dict:
    """One mascot scene, rendered at each aspect. {aspect: Path} — may be empty.

    The SAME seed and scene are used for every aspect so the two thumbnails are
    recognisably the same artwork, not two unrelated images.
    """
    available, why = is_available()
    if not available:
        log.info(f"mascot thumbnails off: {why}")
        return {}
    scene = scene_prompt(title, context, topic)
    if seed is None:
        seed = random.randint(1, 2**31 - 1)
    out: dict = {}
    for aspect in aspects:
        png = out_dir / f"{stem}_mascot_{aspect}.png"
        got = render_scene(scene, png, aspect=aspect, seed=seed)
        if got:
            out[aspect] = got
    if out:
        out["_scene"] = scene
        out["_seed"] = seed
    return out
