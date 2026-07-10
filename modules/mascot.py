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


class MascotGpuFault(RuntimeError):
    """ComfyUI's CUDA context died. Not a per-image failure: every subsequent
    render fails too, so a batch must stop instead of silently emitting
    still-frame thumbnails that look like successes."""

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
ASSETS_DIR = PROJECT_ROOT / "02_Agent" / "assets"

# Checked in priority order — the first that exists wins.
MASCOT_NAMES = ("mascot.png", "mascot.jpg", "mascot.jpeg", "mascot.webp")

# Optional extra angles. Qwen-Edit takes at most 3 references, so the back view
# is deliberately last: it carries no face and no chest logo, which are the two
# things identity transfer keys on. Front + three-quarter + side covers every
# camera angle a thumbnail needs.
#
# These must be SEPARATE FILES. Do not stitch them into a contact sheet: a
# 4-view grid reference annihilated the character in testing (the subject
# vanished, leaving a disembodied hand holding the prop).
MASCOT_ANGLE_NAMES = (
    "mascot_front.png",
    "mascot_threequarter.png",
    "mascot_side.png",
    "mascot_back.png",
)

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

# Qwen is a text-rendering model, so it can draw the TITLE into the artwork
# instead of us stamping it on afterwards. Measured on the bee reel, one seed:
#   "BEE FACTS"                      -> perfect
#   "Bees Have 5 Eyes"               -> perfect, numeral and all
#   "Worker Bees Flap Wings 230x/sec" -> perfect, wrapped to two lines by itself
# It sizes and places the type as part of the composition, which a PIL overlay
# can never do. Long titles are likelier to garble, so past BAKE_MAX_CHARS we
# fall back to the overlay.
BAKE_MAX_CHARS = 44

# The banned-text clauses are dropped when baking — they would fight the headline.
# What replaces them bans the *overlay look*: flat type pasted on the picture is
# exactly what we are trying to get away from.
NEGATIVE_BAKED = (
    "youtube logo, play button, ui overlay, watermark, signature, "
    "misspelled text, gibberish text, distorted letters, duplicate text, "
    "flat text overlay, sticker text, subtitle caption, caption bar, title bar, "
    "plain flat sans-serif caption, text pasted on top of the image, "
    "character covering the letters, letters hidden behind the character, "
    "character centred in front of the text, cropped letters, "
    "blurry, low quality, deformed hands, extra limbs, extra characters, "
    "frame, border, t-pose, arms spread wide, stiff symmetrical standing pose, "
    "cluttered background, busy background"
)

# Style without the "no text" clauses. The background stays simple and the
# character stays off-centre, because the letters need somewhere to live.
STYLE_BAKED = (
    "dynamic off-balance action pose, exaggerated cartoon body language, "
    "leaning into the action, one arm raised, asymmetrical stance, "
    "big readable facial expression, full body visible, "
    "bold vivid solid color background, crisp studio lighting, high contrast, "
    "3d character key art, cinematic title card"
)


NEGATIVE = (
    # The caption-bar family. Qwen will happily render a black title bar with
    # garbled words and a counterfeit YouTube play button unless told not to.
    "caption bar, title bar, subtitle bar, banner, lower third, letterbox bars, "
    "youtube logo, play button, ui overlay, watermark, signature, "
    "text, letters, words, numbers, "
    # Everything else.
    "blurry, low quality, deformed hands, extra limbs, extra characters, "
    "frame, border, t-pose, arms spread wide, stiff symmetrical standing pose, "
    "cluttered background, busy background"
)

# Stated positively as well as in NEGATIVE, because the fallback backend (USO)
# discards negatives entirely — see the note further down.
# Do NOT say "youtube thumbnail" or "space for a title" here. Qwen is a
# text-rendering model: asked for thumbnail art it obligingly PAINTS a caption
# bar and a fake YouTube logo across the bottom — garbled words, real trademark.
# Describe the picture, never the format it will end up in.
STYLE_SUFFIX = (
    "dynamic expressive action pose, exaggerated cartoon body language, "
    "leaning into the action, elbows bent, both hands busy with the object, "
    "big readable facial expression, "
    "full body visible, subject fills the frame, "
    "bold vivid solid color background, empty uncluttered lower area, "
    "crisp studio lighting, high contrast, 3d character key art"
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
    "- This is a FRIENDLY BRAND MASCOT. Never show violence, weapons, blood, "
    "gore, body organs, death, hate, drugs, alcohol or adult content — not even "
    "as a joke, and not even when the video is about those subjects.\n"
    "- If the topic is abstract (love, hatred, time, money), do NOT stage the "
    "abstraction. Use a harmless everyday object, or just a big reaction.\n"
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
    """The primary mascot reference, or None when none has been added yet.

    Falls back to the front angle: an angle set alone is a perfectly good
    install, and without this the whole feature silently no-ops when someone
    drops in mascot_front.png but no mascot.png.
    """
    for name in MASCOT_NAMES:
        p = ASSETS_DIR / name
        if p.exists() and p.stat().st_size > 0:
            return p
    for name in MASCOT_ANGLE_NAMES:
        p = ASSETS_DIR / name
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def mascot_refs(max_refs: int = 1) -> list:
    """The reference images to condition on, best first.

    ONE front reference by default, and that is a measured choice, not laziness.
    Qwen-Edit accepts three (image1..image3), and feeding it front +
    three-quarter + side makes it *copy a reference's stance* instead of acting
    out the scene. Same six scenes, same seeds:

        scene            1 ref                     3 refs
        -----            -----                     ------
        run + look back  panicked, mid-stride,     calm profile jog, small in
                         fills the frame           frame, no look-back
        lift overhead    straining, teeth gritted  fine, but subject shrinks
        point at a       three-quarter, jaw        correct profile, flat
        black hole       dropped                   expression
        cake, caught     guilty, cheeks bulging    good

    A front view already gives Qwen every profile and back-turn we asked for.
    The extra angles only add pose priors that compete with the prompt, and the
    comic expression is what sells a thumbnail. The plumbing stays: pass
    `reference_images=` explicitly, or raise max_refs, when a scene really needs
    the far side of the character.
    """
    angles = [ASSETS_DIR / n for n in MASCOT_ANGLE_NAMES]
    angles = [p for p in angles if p.exists() and p.stat().st_size > 0]
    if angles:
        return angles[:max_refs]
    single = mascot_path()
    return [single] if single else []


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

# Residency is owned by modules/gpu_memory.py — one place that knows WHICH model
# holds the card. Before that existed, prepare_gpu evicted the very model it was
# about to use: ensure_vram_free(14 GB) looked at a card holding a resident
# 13.5 GB Qwen, saw ~1 GB free, and freed ComfyUI. A 14-video "warm" batch paid
# the 4-minute cold load 14 times.


def _model_label() -> str:
    from modules import gpu_memory
    return (gpu_memory.QWEN_EDIT if active_backend_id() == BACKEND_ID
            else gpu_memory.FLUX_USO)


def _set_resident(value: bool) -> None:
    """Kept for the fatal-fault path: a dead CUDA context is not a resident
    model, and the next acquire() must not skip its eviction."""
    from modules import gpu_memory
    if value:
        gpu_memory.acquire(_model_label())
    else:
        gpu_memory.forget()


def prepare_gpu() -> None:
    """Evict everything else before loading a ~13.5 GB model on a 16 GB card.

    Ollama first: the scene prompt has already been written by then, so its
    12.6 GB is dead weight. Then ComfyUI's current model (Wan / Flux / Z-Image).
    A no-op when our own model is already resident.
    """
    from modules import gpu_memory
    try:
        gpu_memory.acquire(_model_label())
    except Exception as e:
        log.warning(f"VRAM pre-flight failed ({e}); rendering anyway")


def release() -> None:
    """Hand the card back. Call once a batch of thumbnails is done, never
    between aspects — a cold reload costs ~4 min, a warm render 15 s."""
    from modules import gpu_memory
    try:
        gpu_memory.release(_model_label())
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


# The scene prompt tells the model to be physical and funny. On a concrete topic
# (goldfish, cake) that lands. On an ABSTRACT one it takes the instruction
# literally and stages the abstraction: asked for "Hatred's Hidden Costs" it
# wrote "the mascot caught trying to type hate speech on a laptop"; asked for
# "Love: The Hidden Truths" it wrote "stuffing a heart-shaped liver into its
# mouth". Neither is anything you would put a branded character in.
#
# Freeform image prompts already go through safety_filter (see manual_mode).
# Mascot scenes did not — this is that gate.
_SCENE_BANNED = (
    # hate / harassment
    "hate speech", "hate", "hateful", "racist", "slur", "nazi", "swastika",
    # violence / weapons / gore
    "gun", "rifle", "pistol", "weapon", "knife", "stab", "shoot", "shooting",
    "blood", "bloody", "gore", "corpse", "wound", "kill", "killing", "murder",
    "bomb", "explosive", "war", "torture",
    # anatomy that reads as gore on a cartoon
    "liver", "organ", "organs", "intestine", "guts", "brain matter", "eyeball",
    "flesh", "carcass",
    # self-harm
    "suicide", "self-harm", "hang", "hanging", "noose",
    # adult / substances
    "nude", "naked", "sexual", "sexy", "erotic", "drug", "drugs", "cocaine",
    "heroin", "cigarette", "smoking", "alcohol", "beer", "vodka", "whiskey",
)

_NEUTRAL_SCENE = ("the mascot character shrugging with both paws up, "
                  "one eyebrow raised, puzzled curious expression")


def scene_violation(scene: str) -> Optional[str]:
    """The banned term a scene contains, or None. Whole-word matching, so
    'hangar' does not trip 'hang' and 'organic' does not trip 'organ'."""
    text = (scene or "").lower()
    for term in _SCENE_BANNED:
        if re.search(rf"\b{re.escape(term)}\b", text):
            return term
    return None


def can_bake(title: str) -> bool:
    """Is this headline short enough for the model to spell reliably?"""
    from modules.publish_kit import strip_emoji
    t = strip_emoji(title or "")
    return 0 < len(t) <= BAKE_MAX_CHARS


# Qwen composes to the centre by default. "Off to one side" in prose did not move
# it — the mascot stood across the words twice. Naming the REGION each element
# owns does move it, and the region has to differ by aspect: a portrait frame
# stacks (text above, character below), a landscape frame splits (text right,
# character left).
_BAKE_LAYOUT = {
    # Portrait took four tries. A tall frame cannot hold a four-word phrase and a
    # full-body character at the same scale, and every way of saying "fit" failed
    # differently:
    #   "letters fill the upper half"      -> line 3 spilled behind the mascot
    #   "the phrase sits ENTIRELY above"   -> Qwen DELETED the last two words
    #   "on two lines across the top half" -> it drew the whole phrase TWICE
    # What works is giving the type room (two thirds, three lines) and making the
    # character small enough to live under it.
    "9x16": ("the letters fill the upper half of the frame on one or two lines, "
             "no word is left out and the phrase appears exactly once, and the "
             "character stands in the lower half of the frame, below all of the "
             "letters"),
    "16x9": ("every word of the phrase sits entirely within the right half of "
             "the frame and the character stands on the far left of the frame, "
             "clear of the words"),
    "1x1":  ("all of the words are written on two lines across the upper half of "
             "the frame, no word is left out, and the character stands below "
             "the letters"),
}


# A 9:16 frame holding a full-body character has room for about two words of
# giant type. Given four, Qwen silently drops the last two — the art then stops
# saying what the kit records. Rather than let that happen invisibly, shorten the
# phrase ourselves and record what each aspect really says. 16:9 has the width
# for the whole hook.
PORTRAIT_MAX_WORDS = 2
PORTRAIT_MAX_CHARS = 14


def fit_headline(headline: str, aspect: str, short: Optional[str] = None) -> str:
    """The words this aspect can actually render whole.

    Portrait gets the LLM's short form ("5 Eyes"), never a chopped headline
    ("Bees Have"). With no short form supplied, the full headline is used: a
    smaller point size beats a mutilated phrase.
    """
    if aspect != "9x16":
        return headline
    if short and len(short) <= PORTRAIT_MAX_CHARS:
        return short
    if len(headline) <= PORTRAIT_MAX_CHARS:
        return headline
    return short or headline


def bake_clause(title: str, aspect: str = "9x16") -> str:
    """The instruction that builds the headline INTO the artwork.

    Asking for "white sans-serif letters across the bottom" only makes the model
    paint the overlay we were trying to escape — flat type sitting on the picture.
    So the letters are described as objects in the scene: they have volume, they
    take the scene's light, they cast shadows, and the character stands among
    them. That is the difference between a caption and a title treatment.
    """
    from modules.publish_kit import strip_emoji
    t = strip_emoji(title).strip()
    return (f'the words "{t}" are built into the scene itself as giant chunky '
            f"three-dimensional block letters standing upright on the ground "
            f"behind the character, the letters have real thickness and volume, "
            f"lit by the same light as the character and casting soft shadows onto "
            f"the ground, sculpted in the same art style and colour palette as the "
            f"character, "
            # Learned the hard way: "the character overlaps the letters" put the
            # mascot dead centre with its body across the words — "230 FLAPS"
            # read as "2?0 F??S". The character has to clear the type, and only
            # an explicit region moves it.
            f"{_BAKE_LAYOUT.get(aspect, _BAKE_LAYOUT['9x16'])}, "
            f"the character does not cover any letter, every letter is completely "
            f"visible and unobstructed, the words are perfectly spelled, large and "
            f"instantly readable")


def fallback_scene(title: str, context: str = "", topic: str = "") -> str:
    """Deterministic scene when the LLM can't be reached or wrote something
    unusable. Deliberately plain — it never invents anything beyond a noun
    already in the video, and never stages an abstraction."""
    noun = _topic_noun(title, context, topic)
    if scene_violation(noun):
        # e.g. topic "hatred" -> do not put the mascot next to hatred.
        return _NEUTRAL_SCENE
    return (f"the mascot character standing next to {noun}, "
            f"friendly pose, plain bold background")


def _clean_scene(raw_scene: str) -> str:
    scene = re.sub(r"\s+", " ", (raw_scene or "")).strip(' "\'')
    if not scene:
        return ""
    if "mascot" not in scene.lower():
        scene = f"the mascot character with {scene}"
    # A scene that smuggles text into the image defeats the title overlay.
    scene = re.sub(r'\b(text|caption|words?|letters?|title)\b', "", scene, flags=re.I)
    # ...and a scene that dictates its own background overrides STYLE_SUFFIX,
    # which is what turned a "bold vivid" brief into flat white.
    scene = _BG_RE.sub("", scene)
    scene = re.sub(r"\s+", " ", scene).strip(" ,")
    if len(scene.split()) > 40:
        scene = " ".join(scene.split()[:40])
    return scene


def scene_prompt(title: str, context: str = "", topic: str = "") -> str:
    """Ask the LLM for one concrete mascot scene. Falls back, never raises.

    The scene is SAFETY-GATED. "Be physical and funny" applied to an abstract
    topic produces things you would never brand: hatred became "typing hate
    speech on a laptop", love became "stuffing a heart-shaped liver into its
    mouth". A violating scene is re-rolled once, then replaced with a plain one.
    """
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
        for attempt in (1, 2):
            raw = _call_llm(prompt, _SCENE_SYS, role="creative")
            scene = _clean_scene(_extract_json(raw).get("scene") or "")
            if not scene:
                continue
            bad = scene_violation(scene)
            if not bad:
                log.info(f"Mascot scene: {scene}")
                return scene
            log.warning(f"Mascot scene rejected (contains {bad!r}): {scene}")
            prompt += (
                "\n\nYour previous scene was rejected: it depicted something "
                "unsuitable for a friendly brand mascot. Never show violence, "
                "weapons, gore, body organs, hate, drugs or adult content. "
                "If the topic is abstract, use a harmless everyday object "
                "instead, or simply have the mascot react with a big expression."
            )
        log.warning(f"Mascot scene unusable after 2 tries; using '{fb}'")
        return fb
    except Exception as e:
        log.warning(f"Mascot scene LLM failed ({e}); using '{fb}'")
        return fb


# ==============================================================================
# RENDER
# ==============================================================================

def render_scene(scene: str, out_png: Path, aspect: str = "9x16",
                 seed: Optional[int] = None,
                 reference_images: Optional[list] = None,
                 headline: str = "") -> Optional[Path]:
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
        target = NATIVE_ASPECTS.get(aspect, "9:16")
        bid = active_backend_id()

        # Bake the headline into the art when the backend can spell it. USO
        # cannot (Flux garbles small text), so it always gets the overlay.
        baked = bool(headline) and bid == BACKEND_ID and can_bake(headline)
        if baked:
            prompt = f"{scene}, {STYLE_BAKED}, {bake_clause(headline, aspect)}"
            negative = NEGATIVE_BAKED
        else:
            prompt = f"{scene}, {STYLE_SUFFIX}"
            negative = NEGATIVE

        # One front reference by default — see mascot_refs() for the measurement.
        refs = reference_images or mascot_refs()

        if bid == BACKEND_ID:
            from modules.image_backends import comfyui_qwen_edit as qwen
            result = qwen.generate(
                prompt=prompt, output_path=out_png, aspect_ratio=target,
                seed=seed, negative_prompt=negative,
                reference_images=[str(p) for p in refs],
            )
        else:
            # The USO Backend CLASS reads lora_strength from models.json and
            # ignores the argument, so call its module-level generate().
            from modules.image_backends import comfyui_uso as uso
            result = uso.generate(
                prompt=prompt, output_path=out_png, aspect_ratio=target,
                seed=seed, lora_strength=POSE_LORA_STRENGTH,
                reference_image=refs[0] if refs else ref,
            )

        if not getattr(result, "success", False):
            if getattr(result, "fatal", False):
                # The CUDA context is dead: every later render in this batch
                # would silently fall back to a still and look like a success.
                from modules import gpu_memory
                gpu_memory.forget()   # a dead CUDA context is not a resident model
                raise MascotGpuFault(
                    "ComfyUI hit a fatal GPU error (CUDA context lost). "
                    "Restart ComfyUI before rendering more thumbnails."
                )
            log.warning(f"mascot render failed ({getattr(result, 'error', '?')}); "
                        f"falling back to a still")
            return None
        # gpu_memory already marked us resident in prepare_gpu().
        log.info(f"Mascot base rendered: {out_png.name} ({aspect}, {bid}, "
                 f"refs={len(refs)}, headline={'baked' if baked else 'overlay'})")
        return Path(result.image_path)
    except MascotGpuFault:
        raise                      # never downgrade a dead GPU to "use a still"
    except Exception as e:
        log.warning(f"mascot render failed ({e}); falling back to a still")
        return None


def render_for_video(title: str, context: str, out_dir: Path, stem: str,
                     topic: str = "", aspects: tuple = ("9x16", "16x9"),
                     seed: Optional[int] = None,
                     release_after: bool = True,
                     scene: Optional[str] = None,
                     headline: Optional[str] = None,
                     headline_short: Optional[str] = None) -> dict:
    """One mascot scene, rendered at each aspect. {aspect: Path} — may be empty.

    The SAME seed and scene are used for every aspect so the two thumbnails are
    recognisably the same artwork, not two unrelated images.

    Memory: the scene is written by Ollama FIRST, then Ollama and whatever
    ComfyUI held are evicted, then every aspect renders while the model stays
    warm. `release_after=False` keeps it resident — use that when looping over
    many videos (a bulk backfill), and call release() once at the end.

    `scene` lets a bulk caller pre-write EVERY scene with the LLM, unload Ollama
    once, and then render the whole batch without ever reloading it. Ollama
    (12.6 GB) and Qwen (13.5 GB) cannot both sit on a 16 GB card, so alternating
    between them per video would thrash.
    """
    available, why = is_available()
    if not available:
        log.info(f"mascot thumbnails off: {why}")
        return {}

    # 1. LLM first, while the GPU still belongs to whoever had it.
    if not scene:
        scene = scene_prompt(title, context, topic)
    if seed is None:
        seed = random.randint(1, 2**31 - 1)

    # 2. Evict Ollama + ComfyUI's current model before loading ~13.5 GB.
    prepare_gpu()

    # 3. Render every aspect warm.
    out: dict = {}
    # The headline is the short hook, not the upload title: a 47-character title
    # baked into the art renders as tiny type. Callers that pass nothing get the
    # title, and can_bake() then decides.
    headline = headline if headline is not None else title
    baked = bool(headline) and active_backend_id() == BACKEND_ID and can_bake(headline)
    try:
        for aspect in aspects:
            png = out_dir / f"{stem}_mascot_{aspect}.png"
            shown = fit_headline(headline, aspect, headline_short) if baked else ""
            got = render_scene(scene, png, aspect=aspect, seed=seed,
                               headline=shown)
            if got:
                out[aspect] = got
                if shown:
                    # What this aspect ACTUALLY says — portrait may carry fewer
                    # words than landscape.
                    out.setdefault("_headline_shown", {})[aspect] = shown
    finally:
        # 4. Give the card back so the next pipeline stage starts clean.
        if release_after:
            release()

    if out:
        out["_scene"] = scene
        out["_seed"] = seed
        out["_backend"] = active_backend_id()
        out["_baked_headline"] = baked   # publish_kit must not paint over it
    return out
