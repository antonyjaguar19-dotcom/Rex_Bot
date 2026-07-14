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

import json
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

# --- Presenter framing (stills that will be animated by Wan S2V) -------------
# The presenter still: the mascot, full body, mid-action, holding the fact's prop.
#
# It used to be nailed to the floor ("feet planted") because Wan-S2V mangled any
# leg that left the ground. The video stage no longer uses S2V, so that rule is
# gone and the poses are free to be fun again. What stays is the PROPORTION lock
# (Qwen quietly slims the cub into a lanky humanoid when a costume changes) and
# the anti-intersection negative (a paw went straight through a hat brim).
# THE STYLE MUST NOT SAY WHAT THE MASCOT IS.
#
# This block is appended to EVERY presenter still, and it used to describe the old
# mascot: "a short chunky CUB", "both PAWS clearly interacting with the prop". That was
# written when the mascot was a jaguar cub, and it was true then. It is now a lie told
# to the model on every single frame — and the model believed it: a lesson still came
# back with a human child wearing MOUSE EARS, because the prompt kept insisting she had
# paws and a cub's build.
#
# The reference image already says what the character is. The style must only say what
# it is DOING. Anything here that names a species, a build or an anatomy is a claim
# about a mascot we may not have.
STYLE_PRESENTER = (
    # PROPORTIONS BY REFERENCE, NEVER BY DESCRIPTION.
    #
    # This line has now broken the character TWICE, both times because someone (me)
    # described the body instead of pointing at it:
    #   "a short chunky cub ... both paws"     -> a human girl grew MOUSE EARS
    #   "chibi proportions, big head, small body" -> a BOBBLEHEAD, head bigger than torso
    # The second was my own repair of the first, and the comment sitting directly above
    # it already said "by REFERENCE, not by description". Read your own comments.
    #
    # The reference image HAS the proportions. Any adjective here competes with it, and
    # the adjective wins — Qwen weights the text prompt over the reference. So: no
    # adjective. Point at the picture and say "that one, in different clothes".
    "exactly the same character as the reference image: the same face, the same head "
    "size, the same body proportions, the same build, the same species — only the "
    "clothes and the props change, "  # facts: a costume per fact IS the joke

    "fully clothed, the shirt covers the whole torso, "
    "full body visible, dynamic playful action pose, exaggerated cartoon body "
    "language, mid-motion, big readable facial expression, mouth open "
    "mid-sentence, "
    # GRIP, not proximity. "Held cleanly in front of the body, nothing intersecting"
    # was read as "keep the prop AWAY from the hand", and the plate of food came back
    # HOVERING in mid-air beside her. A held object is touching the hand that holds it.
    "her fingers wrapped firmly around the prop, gripping it, the hand visibly "
    "closed on the object, the object resting in her palm, "
    "bold vivid solid color background, crisp studio lighting, high contrast, "
    "3d character key art"
)

# --- LESSON-ONLY tuning -------------------------------------------------------
# Everything above is shared with FACTS mode, and facts wants the opposite of some of
# what a lesson wants: a facts reel is spectacle. It is ALLOWED a giant honey dipper and
# a grinning cartoon sun — that is the joke, and the joke is the point when the picture's
# only job is to hold attention on a line you already said out loud.
#
# A lesson's picture is what a child who cannot read is learning FROM. So the lesson gets
# its own overlay, and facts is left exactly as it was.
STYLE_TEACHING = STYLE_PRESENTER.replace(
    # A LESSON is one girl, on one day, in one film. The shared line invites a costume
    # change on every shot — right for a facts reel (each fact gets its own gag costume)
    # and wrong here: shot 1 came back in a pinafore and shot 3 in her white top and
    # denim skirt, and thirteen of those watch like thirteen different days.
    "only the clothes and the props change",
    "she is wearing EXACTLY THE SAME CLOTHES as in the reference image — the same top, "
    "the same skirt, the same shoes, unchanged in every shot; only the PROPS change",
) + (
    # SCALE. Nothing constrained how BIG a prop was, and "holding up a grey rock" came
    # back as a BOULDER bigger than her torso, hugged with both arms — which also ate the
    # SECOND prop the scene asked for, because both her arms were full of the first.
    # A lesson's prop is a child's toy: it fits in her hand. A facts reel's need not.
    ", the props are toys a child can hold — roughly the size of her head, "
    "clearly held in her hands, never bigger than her body and never doll-house tiny"
    # The flat colour card made every shot look like a void. A real place is better
    # television AND better teaching — but the child and the prop are the lesson, so the
    # setting stays soft and behind her, and the captions have to stay readable over it.
    ", the character is sharp and fully separated from the background, the setting is "
    "soft and gently out of focus behind her, uncluttered, nothing growing out of her "
    "head, bright cheerful daylight"
)

NEGATIVE_PRESENTER = (
    # Species drift: told the character had "paws" and a "cub" build, Qwen gave a human
    # child MOUSE EARS. The style prompt no longer claims a species — and the negative
    # now bans the parts it used to invite.
    "animal ears on a human, mouse ears, cat ears, whiskers, snout, muzzle, tail, "
    "fur on a human face, paws instead of hands, changed species, different creature, "
    # THE TWIN. Qwen applies the ONE identity reference to every human-shaped thing in
    # the frame: "hugged by her mother" came back as two identical mascots.
    "twins, clone, duplicate character, two identical characters, the same character "
    "twice, mirrored character, second identical child, "
    # And the ANIMALS duplicate too, entirely on their own. A scene naming ONE puppy
    # came back with a puppy in each hand. Qwen fills a spare hand by copying whatever it
    # already drew, so a scene that gives her two things to hold and names only one
    # animal gets the animal twice.
    "two puppies, two dogs, duplicate animal, two identical animals, the same animal "
    "twice, cloned pet, extra animals, more than one puppy, "
    # A TAIL growing out of the GIRL'S HIP, and a doll and a puppy FUSED into one
    # creature (a puppy's head on a cloth body with button eyes). Two similar objects
    # held close together merge; an animal's parts wander onto the nearest body.
    "tail growing from a person, tail on a girl, tail on a child, animal tail on a human, "
    "merged creature, doll fused with an animal, chimera, two objects melted together, "
    "toy merged with a pet, "
    # ASYMMETRIC EYES — one pupil bigger, one eye lower, a lazy eye. The face is the
    # identity and it must be clean.
    "asymmetric eyes, lazy eye, misaligned pupils, uneven eyes, wonky eye, "
    "one eye larger than the other, cross-eyed, "
    # THE LIVING DOLL. A doll is human-SHAPED, so it is the likeliest thing in any scene
    # to capture the reference — "holding up a doll" returned a LIVING CHILD with the
    # mascot's own face, dangling by the wrist, in a lesson about how toys are NOT alive.
    "doll with a human face, realistic child instead of a doll, living doll, "
    "a real child held by the arm, child dangling, child in distress, "
    # ...and the OPPOSITE, which is what "limp and lifeless, stitched button eyes" got:
    # a HEADLESS cloth sack with buttons sewn onto its torso, and then a blank white
    # faceless figure. Both read as a voodoo doll. A rag doll has a head and a smile.
    "headless doll, doll with no head, faceless doll, blank white doll, featureless doll, "
    "eyes on the torso, voodoo doll, creepy doll, sinister toy, "
    # ...and the over-correction of THAT: naming only "a round stitched head, yarn hair
    # and a stitched smile" got a DISEMBODIED HEAD — a knitted ball with a face, no body
    # at all, in two shots out of three. The model draws exactly what you name.
    "disembodied head, doll head with no body, severed head, a ball with a face, "
    "floating head, "
    # BOBBLEHEAD. Telling the model "big head, small body" (my own botched repair of the
    # cub language) gave her a head bigger than her torso.
    "bobblehead, oversized head, giant head, head bigger than the body, "
    "shrunken body, tiny torso, stunted limbs, distorted proportions, "
    # FLOATING PROPS. "Held cleanly in front of the body, nothing intersecting" was read
    # as "keep it away from the hand" — the plate of food hovered in mid-air beside her.
    "floating object, hovering prop, object suspended in mid-air, "
    "prop not touching the hand, disembodied object, object detached from the hand, "
    "empty open palm under a floating object, "
    # The face IS the identity. Asked for "heart eyes" the model deleted her eyes and
    # pasted two red emoji over the sockets — that is not an expression, it is a
    # different character. Expressions come from eyebrows and mouths.
    "heart-shaped eyes, emoji eyes, star eyes, spiral eyes, symbols instead of eyes, "
    "eyes covered, eyes replaced, "
    # Modesty. This is a lesson for six-year-olds and the mascot is a small child.
    # A "chef's outfit" came back as a crop top with a bare midriff.
    "crop top, bare midriff, exposed belly, bare stomach, bare chest, undressed, "
    # Identity, lost to a COSTUME rather than to a species swap. A "toy repairman's
    # outfit" (dungarees + a cap over the hair) came back as a completely different
    # child — the face and the hair are all the artist has to recognise her by.
    "different character, different child, different face, changed hairstyle, "
    "hair hidden under a hat, head covered, hood up, helmet, "
    # Proportion drift: the character kept coming back with long thin legs and a slim
    # body. The build is fixed; only the costume changes.
    "thin legs, skinny legs, long legs, slender legs, lanky, elongated limbs, "
    "tall slim body, adult body, human proportions, changed body shape, "
    "different character, stretched torso, "
    # Geometry intersection: a paw went straight THROUGH the hat brim.
    "hand passing through object, hand clipping through hat, limbs intersecting "
    "props, merged geometry, object embedded in body, overlapping shapes, "
    "hand inside prop, "
    "deformed legs, broken limbs, twisted body, extra legs, bent backwards, "
    "mangled feet, distorted anatomy, "
    "caption bar, title bar, banner, youtube logo, play button, ui overlay, "
    "watermark, text, letters, words, numbers, "
    "blurry, low quality, deformed hands, extra limbs, extra characters, "
    "frame, border, t-pose, cluttered background"
)

# --- LESSON-ONLY bans ---------------------------------------------------------
# These are wrong for a FACTS reel and right for a lesson, which is why they are not in
# the shared negative. A facts reel is spectacle: a grinning cartoon sun is a joke, a
# giant honey dipper is a joke, and a mascot pulling a shocked face at a horrifying fact
# is the whole point. In a lesson every one of those is a defect.
NEGATIVE_TEACHING = NEGATIVE_PRESENTER + (
    # A FACE ON AN INANIMATE OBJECT. Half these lessons teach what is alive and what is
    # not, and we drew a smiling cartoon face on the Earth for the line about being
    # hugged by your mother. A grinning rock teaches a child the wrong answer.
    ", googly eyes on an object, smiling face on a ball, face drawn on a toy, "
    "anthropomorphic object, cartoon eyes on an inanimate object, "
    # ANGER. A "playful frown" came back as a small girl SCOWLING in real anger. The
    # teacher in a lesson for six-year-olds is never cross with them.
    "angry, scowling, furious, glaring, mean expression, upset, crying, worried, "
    # SCALE. "A grey rock" came back as a BOULDER hugged with both arms — which also cost
    # her the second prop, since both arms were full of the first.
    "giant prop, oversized object, boulder, huge rock, prop bigger than the character, "
    "prop hugged with both arms, "
    # ...and the opposite. "Hand-sized, never larger than her head" over-corrected into
    # DOLL-HOUSE scale: a puppy the size of a finger, perched on her fingertips.
    "miniature prop, doll-house sized toy, finger-sized animal, tiny objects, "
    "props perched on fingertips, "
    # BUBBLES. Qwen fills a thought bubble with a garbled picture-within-a-picture; it
    # reads as a rendering fault, not a device.
    "thought bubble, speech bubble, comic balloon, inset picture, picture-in-picture"
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

def _library():
    """The mascot shelf, when there is one. Late import: mascot_library reads
    ASSETS_DIR back off this module."""
    from modules import mascot_library as ml
    return ml if ml.library_exists() else None


def active_mascot_name() -> str:
    """Display name of the mascot in use — for logs and the UI."""
    ml = _library()
    got = ml.active() if ml else None
    return got["name"] if got else "default"


def mascot_path() -> Optional[Path]:
    """The primary mascot reference, or None when none has been added yet.

    With a library on disk (assets/mascots/), this is the ACTIVE mascot's art —
    switching mascot switches every thumbnail and presenter still with it. The
    flat `assets/mascot.png` layout below is the pre-library fallback, and is only
    consulted while no library exists (so deleting every mascot leaves you with
    none, instead of resurrecting the old one).

    Falls back to the front angle: an angle set alone is a perfectly good
    install, and without this the whole feature silently no-ops when someone
    drops in mascot_front.png but no mascot.png.
    """
    ml = _library()
    if ml:
        return ml.primary_image()
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
    ml = _library()
    if ml:
        return ml.refs(max_refs=max_refs)
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
        ml = _library()
        if ml:
            # An ACTIVE mascot with no art is the confusing case: the shelf is not
            # empty and the dropdown names a character, so "no mascot" reads like a
            # lie. Name the one that is missing its image.
            got = ml.active()
            where = (f"“{got['name']}” has no image yet — add one in the Mascots tab"
                     if got else "no mascots on the shelf — add one in the Mascots tab")
            return False, where
        where = str(ASSETS_DIR / "mascot.png")
        return False, f"no mascot image — add one at {where}"
    ok, msg = backend_healthy()
    if not ok:
        return False, msg
    return True, f"mascot {active_mascot_name()} ({p.name}) + {BACKEND_ID}"


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

_NEUTRAL_SCENE = ("the mascot character shrugging with both hands up, "
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


def _salvage_scene(raw: str) -> str:
    """Pull a scene out of an answer that ignored the JSON contract.

    Measured on a live reel: the model replied with prose — "Note: To meet your
    requirement strictly with no description of backgrounds and fo..." — the JSON
    parse returned nothing, and that shot silently fell back to a generic
    "standing next to honeybees" pose. The scene was usually still in there
    somewhere; look for the sentence that actually names the mascot.
    """
    text = (raw or "").strip()
    if not text:
        return ""
    for line in re.split(r"[\n.]+", text):
        line = line.strip(' "\'`,')
        # Drop the model's own commentary about the task.
        if re.match(r"^(note|okay|sure|here|i |output|json)\b", line, flags=re.I):
            continue
        if "mascot" in line.lower() and len(line.split()) >= 5:
            return line
    return ""


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
            # scene_prompt is the THUMBNAIL path — a facts reel's cover art, which is
            # spectacle by design. Only the universal guards apply (the mascot must
            # survive its own picture); the lesson's pedagogy guards do not.
            scene = clean_scene_for_the_mascot(
                _clean_scene(_extract_json(raw).get("scene") or ""))
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


_EXPLAINER_SYS = (
    "You dress a friendly brand mascot to ILLUSTRATE and EXPLAIN one fact in a "
    "short educational video. The mascot is the on-screen presenter for this "
    "single fact.\n"
    'Output ONLY valid JSON: {"scene": "..."}\n'
    "Rules:\n"
    "- The scene ALWAYS stars 'the mascot character'. Name it exactly that.\n"
    "- Put the mascot in a COSTUME or ROLE that fits THIS fact, and make it "
    "ACTIVELY DEMONSTRATE the fact with one concrete prop — it must be DOING "
    "something, mid-motion, never just standing.\n"
    "- THE MASCOT NEVER BECOMES SOMETHING ELSE. It is always the same character. It "
    "may WEAR things and HOLD things — a chef's hat, a lab coat, a cape, a ladle — but "
    "it is never 'dressed as' an animal, a creature or another being, and its body, "
    "species and face never change. To show an animal, put the animal BESIDE it: "
    "'the mascot character kneeling beside a puppy, holding out a bone' — never 'the "
    "mascot character dressed as a puppy'.\n"
    "  Fact 'bees make honey' -> 'the mascot character in a beekeeper suit "
    "scooping dripping honey from a jar and holding up the dripping dipper'.\n"
    "  Fact 'a bee flaps 230 times a second' -> 'the mascot character dressed as "
    "a racing pilot flapping tiny wings in a blur, zooming forward'.\n"
    "- Make it FUN and VISUALLY INTERESTING. Big, playful, physical comedy: the "
    "mascot can leap, hover, zoom, spin, balance, dangle, ride, surf, tumble, "
    "peek out of something, be chased by something. Surprise the viewer.\n"
    "- Use STRONG ACTION VERBS and put the whole body into it: showing, waving, "
    "scooping, pointing, lifting, hauling, launching, dodging, juggling.\n"
    "- Hold the prop CLEARLY, out in front of the body. It must never overlap or "
    "pass through the mascot — a hand once went straight through a hat brim.\n"
    "- ONE MOMENT. A still picture cannot show a sequence. Never write 'then', "
    "'finally' or 'after that' — 'eats, then grows taller, then hops' comes back as a "
    "smear of limbs. Pick the single action the line is about.\n"
    "- NEVER touch the face. Do not write 'heart eyes', 'star eyes' or any symbol in "
    "place of an eye: the model DELETES her eyes and pastes emoji over the sockets, and "
    "the face is the whole identity. Expressions are made of eyebrows and mouths.\n"
    "- The mascot stays DRESSED. This is a lesson for six-year-olds. Never write about "
    "her belly, tummy or stomach — 'belly expanding comically' got a small child drawn "
    "in a crop top.\n"
    "- NEVER say what the mascot is made of or what body parts it has. Do not write "
    "'paw', 'snout', 'tail', 'fur' or any other animal part: the mascot may be a child, "
    "and telling the artist it has paws is how a child ends up with mouse ears. Say "
    "'hand', 'holding', 'the character'.\n"
    "- FULL BODY in frame, FACING THE CAMERA, mouth visible, caught mid-sentence, "
    "talking to the viewer WHILE doing the action.\n"
    "- Only the COSTUME and the PROP change between shots. NEVER describe the "
    "mascot's body, height, build or legs — its proportions are fixed and any "
    "word about them makes the renderer redraw the character.\n"
    "- Friendly, cheerful, energetic. The motion and the costume selling the "
    "fact matter most.\n"
    "- FRIENDLY BRAND MASCOT: never violence, weapons, blood, gore, organs, "
    "death, hate, drugs, alcohol or adult content — not even as a joke.\n"
    "- If the fact is abstract, use a harmless everyday prop or costume.\n"
    "- Under 30 words. Describe only what is visible: costume, the ACTION in "
    "motion, prop, the facing-camera framing, expression.\n"
    "- Do NOT describe background, lighting or camera settings — those are set.\n"
    "- No text, captions, letters or numbers in the image.\n"
    "Examples:\n"
    '{"scene": "the mascot character in a beekeeper suit facing the camera, '
    'waist-up, lifting a dripping honeycomb and waving it, honey splashing, '
    'proud excited grin"}\n'
    '{"scene": "the mascot character dressed as a scuba diver facing the camera, '
    'spinning to point both hands at a glowing jellyfish, eyes wide with wonder"}'
)


# The mascot may not be TURNED INTO something else — and the prompt saying so is not
# enough. Measured on a real lesson: asked to illustrate "living things eat, grow and
# move", the model wrote "the mascot character dressed as a playful puppy, mid-leap
# with a chew toy in mouth", and Qwen-Edit did exactly that. It kept the child's
# clothes and replaced the child with a dog. The character was simply gone.
#
# A costume is a hat, a coat, a cape. A creature is not a costume.
_CREATURES = (
    "puppy", "dog", "cat", "kitten", "bird", "duck", "chicken", "hen", "cow", "horse",
    "pig", "sheep", "goat", "rabbit", "bunny", "mouse", "rat", "squirrel", "monkey",
    "elephant", "lion", "tiger", "bear", "wolf", "fox", "deer", "frog", "toad",
    "snake", "lizard", "turtle", "tortoise", "fish", "shark", "whale", "dolphin",
    "octopus", "crab", "bee", "wasp", "ant", "spider", "butterfly", "caterpillar",
    "worm", "snail", "penguin", "owl", "eagle", "parrot", "peacock", "dinosaur",
    "dragon", "unicorn", "jaguar", "leopard", "cheetah", "giraffe", "zebra",
    "kangaroo", "panda", "koala", "hippo", "rhino", "crocodile", "camel", "donkey",
)

# Up to two adjectives may sit between the article and the animal — the scene that
# actually broke a render said "dressed as a PLAYFUL puppy", and a regex anchored
# straight onto the noun sailed past it.
_AS_A_CREATURE = re.compile(
    r"\b(?:dressed|disguised|costumed)?\s*as\s+(?:a|an|the)\s+(?:\w+\s+){0,2}?(" +
    "|".join(_CREATURES) + r")s?\b", re.I)


def species_swap(scene: str) -> Optional[str]:
    """The creature this scene would turn the mascot INTO, or None."""
    m = _AS_A_CREATURE.search(scene or "")
    return m.group(1).lower() if m else None


# Animal ANATOMY, handed to the artist as fact. "Mid-balance on one paw" produced a
# human child with mouse ears — the model reasons backwards from the body part to the
# creature it belongs to. The mascot's species lives in the reference image, and nothing
# in a prompt is allowed to contradict it.
_ANIMAL_PARTS = {
    "paw": "hand", "paws": "hands", "snout": "nose", "muzzle": "face",
    "tail": "back", "fur": "clothes", "whiskers": "face", "hooves": "feet",
    "hoof": "foot", "claws": "fingers", "beak": "mouth", "wings": "arms",
}
_PART_RE = re.compile(r"\b(" + "|".join(_ANIMAL_PARTS) + r")\b", re.I)


_ANIMAL_NOUN = re.compile(r"\b(" + "|".join(_CREATURES) + r")s?\b", re.I)


def _clauses(scene: str) -> list:
    """Split on commas. A scene is a comma-separated list of beats, and each beat has
    exactly one subject — which is what lets us tell whose paw is whose."""
    out, i = [], 0
    for part in (scene or "").split(","):
        out.append((i, part))
        i += len(part) + 1
    return out


def animal_parts(scene: str) -> list:
    """Animal body parts the scene claims THE MASCOT has.

    A lesson about living things legitimately puts a real puppy in the frame, and that
    puppy is allowed its tail. Only a part in a clause that names no animal can belong
    to the mascot — "mid-balance on one paw" is hers; "puppy wagging its tail" is his.
    """
    found = set()
    for _, clause in _clauses(scene):
        if _ANIMAL_NOUN.search(clause):
            continue                      # this beat is about a real animal; its body is its own
        found |= {m.group(1).lower() for m in _PART_RE.finditer(clause)}
    return sorted(found)


def keep_the_body(scene: str) -> str:
    """Strip animal anatomy the scene handed the MASCOT. The reference image is the only
    thing allowed to say what her body is — told she has paws, Qwen reasoned backwards to
    the creature paws belong to and drew a human child with mouse ears."""
    if not animal_parts(scene):
        return scene
    out, found = [], []
    for _, clause in _clauses(scene):
        if _ANIMAL_NOUN.search(clause):
            out.append(clause)            # leave the puppy's tail on the puppy
            continue
        found += [m.group(1).lower() for m in _PART_RE.finditer(clause)]
        out.append(_PART_RE.sub(lambda m: _ANIMAL_PARTS[m.group(1).lower()], clause))
    log.warning(f"scene gave the mascot {', '.join(sorted(set(found)))} — the reference "
                f"decides what body she has, not the scene")
    return ",".join(out)


# --- The face is the identity ------------------------------------------------
# The scene writer asked for "waving happily with heart eyes" and Qwen did exactly
# that: it deleted her eyes and pasted two red emoji hearts over the sockets. The
# mascot's face is the one thing identity transfer keys on — a cartoon symbol in
# place of an eye is not an expression, it is a different character.
_SYMBOL_EYES = re.compile(
    r",?\s*(?:with\s+)?(?:big\s+)?(?:heart|hearts|star|stars|spiral|spirals|x|dollar)"
    r"[- ]?(?:shaped\s+)?eyes\b", re.I)

# A still is ONE moment. Handed "holding a plate to its mouth, then growing taller,
# waving happily, finally hopping on one foot", the model tries to draw all four at
# once and the result is a smear of limbs. Cut at the first sequence word: the scene
# keeps its first action, which is the one the line is actually about.
_SEQUENCE = re.compile(
    r"[,;]?\s*\b(?:then|finally|afterwards|after that|next|"
    # "pointing at each IN TURN" is a sequence wearing a disguise, and it is also how
    # you get a T-pose: two props, one on each side, and she reaches for both. The
    # reference photo is a T-pose, and arms-spread is the model's road home to it.
    r"in turn|one by one|one after another|back and forth)\b", re.I)

# Skin. A "belly expanding comically with each bite" got a six-year-old drawn in a
# crop top with a bare midriff. This is a children's lesson: the mascot is dressed.
_SKIN = re.compile(
    r",?\s*\b(?:belly|tummy|stomach|midriff)\b[^,]*", re.I)


def keep_the_face(scene: str) -> str:
    """Symbols are not expressions. Her eyes stay her eyes."""
    if not _SYMBOL_EYES.search(scene or ""):
        return scene
    log.warning("scene replaced the mascot's eyes with a symbol — the face is the identity")
    return _tidy(_SYMBOL_EYES.sub("", scene))


def one_moment(scene: str) -> str:
    """A still is one moment, not a storyboard. Keep the first action."""
    m = _SEQUENCE.search(scene or "")
    if not m:
        return scene
    kept = _tidy(scene[:m.start()].rstrip(" ,;"))
    log.warning(f"scene described a SEQUENCE; a still shows one moment — kept: {kept[:60]}")
    return kept


def keep_it_dressed(scene: str) -> str:
    """No bare skin on a child in a lesson for children.

    Conservative on purpose. "Scratching its belly" belongs to the puppy named a clause
    EARLIER, so clause-scoping alone (which works for anatomy) strips the wrong belly
    here — the pronoun reaches back across the comma. So the cut only happens when the
    scene mentions no animal at all, and therefore the only body in it is hers. When an
    animal IS present the negative prompt is what keeps her shirt on, which is the
    weaker guarantee but never mangles a legitimate scene.
    """
    if _ANIMAL_NOUN.search(scene or "") or not _SKIN.search(scene or ""):
        return scene
    log.warning("scene bared the mascot's midriff — she stays dressed")
    return _tidy(_SKIN.sub("", scene))


# Anger, again — and this time the NEGATIVE did not stop it. Line 9 of the first lesson
# is a negative statement ("they don't eat, they don't grow, they're not alive"), the
# scene named no expression at all, and Qwen supplied one from the sentiment of the
# words: a six-year-old scowling and snarling at the camera.
#
# Banning "angry" is not enough, because a face is not optional — the model WILL choose
# one. The scene has to SAY which. So: strip the cross words, and if the scene names no
# expression, give it a warm one. A teacher's face is never left to the renderer.
_CROSS = re.compile(
    r",?\s*[^,]*\b(?:angry|angrily|furious|cross|scowl\w*|glar\w*|snarl\w*|"
    r"frown\w*|grumpy|annoyed|upset|sad|scared|worried|stern)\b[^,]*", re.I)

_WARM_WORDS = re.compile(
    r"\b(?:smil\w*|grin\w*|laugh\w*|happy|cheerful|delight\w*|joyful|curious|"
    r"wonder\w*|excited|eyebrows raised|questioning|gentle|warm|kind|proud|"
    r"eyes wide|beaming)\b", re.I)

_WARM_DEFAULT = "warm friendly smile, eyebrows raised in curiosity"


def warm_face(scene: str) -> str:
    """The teacher's face is never left to the renderer.

    Strips a cross expression, and NAMES a warm one when the scene named none — a
    picture has a face whether or not the prompt asked for one, and on a line about
    what something CANNOT do the model picks the face from the sentiment of the words.
    """
    out = _tidy(_CROSS.sub("", scene or ""))
    if not out:
        return scene
    if not _WARM_WORDS.search(out):
        out = f"{out}, {_WARM_DEFAULT}"
        log.info("scene named no expression — the teacher is warm by default")
    return out


# The T-pose has three roads home, and all of them end at the reference photo:
#   1. an EMPTY scene — nothing to hold, so she stands like the reference (still_12);
#   2. TOO MUCH to hold — the artist drops the lot and stands her like the reference;
#   3. one thing on each SIDE — she reaches for both, and arms-spread IS the T-pose
#      (still_08: "standing between a toy car and a plant, pointing at each in turn").
#
# Roads 1 and 2 are the writer's business, and the prompt now covers them. Road 3 is a
# specific sentence shape, so it can be caught: strip the words that spread her arms.
# Two shapes, two repairs. "Standing BETWEEN a car and a plant" still wants both props
# in frame — she just must not reach for both, so she stands BESIDE them. A bare
# "arms spread wide" wants no props at all and is simply the T-pose spelled out; it
# becomes a real, asymmetric gesture.
_STANDING_BETWEEN = re.compile(r"\bstanding between\b", re.I)
_ARMS_WIDE = re.compile(
    r",?\s*\b(?:standing |with )?(?:her |both )?arms (?:spread|stretched|outstretched|"
    r"held|out|open|wide)(?: wide| out| open)?(?: to (?:the |both )?sides?)?\b"
    r"|,?\s*\b(?:both |her )?outstretched arms\b"
    r"|,?\s*\bboth arms (?:out|wide|raised|lifted|open)\b"
    r"|,?\s*\b(?:wide )?open arms\b", re.I)

# Whatever a guard cuts, the SUBJECT survives. An early version's clause-scoped delete
# took "the mascot character" out with the offending phrase and handed the backend a
# scene with nobody in it.
_SUBJECT = "the mascot character"


def one_focus(scene: str) -> str:
    """Kill the arms-spread construction. A reached-for prop on each side is a T-pose."""
    s = scene or ""
    if not (_ARMS_WIDE.search(s) or _STANDING_BETWEEN.search(s)):
        return scene
    s = _STANDING_BETWEEN.sub("standing beside", s)
    s = _ARMS_WIDE.sub(", one hand raised", s)
    log.warning("scene spread the mascot's arms — that is the reference photo's T-pose")
    return _tidy(s)


def _tidy(scene: str) -> str:
    """Clean up after a cut, and never lose who the picture is of."""
    out = ", ".join(c.strip() for c in (scene or "").split(",") if c.strip())
    out = re.sub(r"\s{2,}", " ", out).strip(" ,")
    # A cut can leave a preposition hanging onto nothing ("the mascot character with,
    # one hand raised"). Harmless to a diffusion model, but it reads as a bug in the log
    # and the log is how anyone ever notices these.
    out = re.sub(r"\b(?:with|and|holding|beside)\s*,", ",", out)
    out = re.sub(r",\s*,", ",", out).strip(" ,")
    if out and _SUBJECT not in out.lower():
        out = f"{_SUBJECT} {out}"
    return out


# Toys and animals look the same to a diffusion model unless you insist otherwise. Asked
# for "a doll in one hand and a puppy in the other", Qwen drew TWO PLUSH TOY DOGS — and
# the line was "why doesn't your doll need to eat like Jimmy does?", whose entire point
# is that one of them is alive and the other is not. A lesson that cannot show the
# difference cannot teach it.
_TOY_WORDS = re.compile(r"\b(?:doll|toy|teddy|stuffed|plush|puppet|figurine)\b", re.I)
_LIVE_ANIMAL = re.compile(
    r"\b(?:real |live |living )?(" + "|".join(_CREATURES) + r")s?\b", re.I)


def alive_looks_alive(scene: str) -> str:
    """If a scene puts a real animal next to a toy, insist the animal looks alive."""
    s = scene or ""
    if not (_TOY_WORDS.search(s) and _LIVE_ANIMAL.search(s)):
        return scene
    if re.search(r"\breal (?:live )?\w+|\blive \w+|wagging|blinking|breathing", s, re.I):
        return scene                       # the scene already says so
    def _mark(m):
        word = m.group(0)
        if re.match(r"(?:real|live|living)\b", word, re.I):
            return word                    # already said
        # IN PLACE, and ONLY in place. The first version also appended ", the puppy alive
        # and moving" on the end — which named the puppy a SECOND time, and Qwen duly
        # drew TWO PUPPIES, one in each hand. Every extra mention of a thing is another
        # copy of that thing. Say it once.
        return f"real live {m.group(1)}, wagging and blinking"
    out = _LIVE_ANIMAL.sub(_mark, s, count=1)
    log.info("scene put a real animal beside a toy — saying so, or both come back plush")
    return _tidy(out)


# THE actual invariant, arrived at the long way round. Every T-pose we have shipped —
# and there were four, by four different routes — has one thing in common: HER HANDS
# WERE DOING NOTHING. Nothing to hold, or too much to hold and so nothing held, or a
# thing on each side to reach for. Idle hands go home to the reference photo, and the
# reference photo is a T-pose.
#
# So the rule is not about props or sides or counts. It is: her hands are always busy.
# "held" and "cradling" are busy hands, and this pattern did not know it — "hold\w*"
# does not match the irregular past tense, and "cradle" was simply missing. So a scene
# whose hands were ALREADY FULL ("a plant held in her right hand and a rock held in her
# left") was judged idle, and never_empty_handed() bolted "waving one arm high, the other
# hand resting at her side" onto the end of it.
#
# That is a flat contradiction — both hands full AND one waving AND one at her side — and
# the model resolved it the only way it could: it DROPPED A PROP. That is where the rock
# went in shot 1. A guard that misfires is worse than no guard, because it actively
# fights the scene.
_HANDS_BUSY = re.compile(
    r"\b(?:hold\w*|held|holds|lift\w*|carry\w*|carrie\w*|cradl\w*|clutch\w*|grip\w*|"
    r"grasp\w*|hug\w*|cuddl\w*|stroking|stroke\w*|petting|pat\w*|scratch\w*|wav\w*|"
    r"wave|point\w*|show\w*|offering|feeding|feed\w*|reach\w*|touch\w*|push\w*|pull\w*|"
    r"scoop\w*|balanc\w*|cover\w*|clap\w*|counting|"
    # ...and the whole-body actions, which are not IDLE either. This pattern has now
    # misfired twice, each time on a scene whose hands were plainly occupied:
    #   "a plant HELD in her right hand"  (hold\w* does not match the past tense)
    #   "running with both arms SWINGING" (a whole-body action, no object at all)
    # and each time it bolted "waving one arm high, the other hand resting at her side"
    # onto a scene that contradicted it. The model then dropped a prop to resolve the
    # contradiction. Err towards LEAVING A SCENE ALONE: the cost of a missed idle scene
    # is a stiff pose, and the cost of a false positive is a lost prop.
    r"swing\w*|running|runs|climb\w*|dancing|jumping|leap\w*|throw\w*|catch\w*|"
    r"digging|planting|watering|eating|biting|drinking|shrug\w*|"
    r"in (?:her|both|one) (?:hand|hands|arm|arms)|hands? full)\b", re.I)

# _BUSY_DEFAULT must itself satisfy _HANDS_BUSY, or the guard is not idempotent: run it
# twice and it appends twice. It said "a cheerful wave" (the noun) while the pattern only
# knew "waving" (the verb), so a scene came out of the guard still looking idle.
_BUSY_DEFAULT = ("waving one arm high in a cheerful hello, the other hand resting at "
                 "her side")


def never_empty_handed(scene: str) -> str:
    """Idle hands go home to the reference photo, and the reference photo is a T-pose."""
    if _HANDS_BUSY.search(scene or ""):
        return scene
    log.info("scene left the mascot's hands empty — giving her something to do with them")
    return _tidy(f"{(scene or '').rstrip(' ,')}, {_BUSY_DEFAULT}")


# SCALE. Nothing in the prompt said how BIG a prop was, so "holding up a grey rock" came
# back as a BOULDER — bigger than her torso, hugged with both arms. That also ate the
# SECOND prop the scene asked for (a plant), because both her arms were full of the
# first. A prop is a child's toy. It goes in a hand.
_PROP_NOUNS = (
    "rock", "stone", "pebble", "plant", "seedling", "flower", "leaf", "doll", "toy",
    "car", "ball", "book", "plate", "bowl", "cup", "apple", "fruit", "carrot",
    "vegetable", "seed", "bone", "brush", "spoon", "globe", "teddy", "wrench",
)
# The article is captured SEPARATELY so "small" lands after it. The first version
# anchored on "a|the|up" and rewrote "holding up a rock" into "holding up small a rock".
_PROP_RE = re.compile(r"\b(a|an|the|one)\s+((?:\w+\s+){0,2}(?:" +
                      "|".join(_PROP_NOUNS) + r"))\b", re.I)
_ALREADY_SMALL = re.compile(r"\b(?:small|little|tiny|toy|hand-sized|miniature)\b", re.I)


def props_fit_in_a_hand(scene: str) -> str:
    """A prop is a child's toy, not a boulder."""
    s = scene or ""
    if not _PROP_RE.search(s):
        return scene

    def _shrink(m):
        article, phrase = m.group(1), m.group(2)
        if _ALREADY_SMALL.search(phrase):
            return m.group(0)
        return f"{article} small {phrase}"

    out = _PROP_RE.sub(_shrink, s)
    if out != s:
        log.info("scene named a prop with no size — a prop is hand-sized or it becomes "
                 "a boulder")
    return _tidy(out)


# THE REFERENCE BLEEDS. Qwen-Edit is handed ONE identity reference — the mascot — and it
# applies that identity to every human-shaped thing in the frame. Three ruined stills,
# one cause:
#
#   "being hugged by her smiling mother"  -> TWO IDENTICAL NAKSHUS. Same clothes, same
#                                            hair, same butterflies, same bindi.
#   "holding up a doll named Ammu"        -> a LIVING CHILD with Nakshu's own face,
#                                            held dangling by the wrist. The line is
#                                            "Ammu is a toy and isn't alive". The
#                                            picture taught the opposite AND read as a
#                                            child being hurt. This is the worst thing
#                                            this pipeline has produced.
#
# A doll is human-SHAPED, which is exactly what the reference is, so it is the likeliest
# thing in any scene to be captured. The scene must say, out loud, what a doll is made
# of and that another person is another person.
# The name travels WITH the doll ("a doll named Ammu"), or the replacement orphans it
# into "...not a person named Ammu", which reads as the mascot naming a corpse.
_DOLL = re.compile(r"\b(?:a|an|the|one|her|his)\s+(?:\w+\s+){0,2}"
                   r"(?:doll|dolly|teddy|puppet|figurine|action figure)"
                   r"(\s+(?:named|called)\s+\w+)?", re.I)
# The name goes straight after "rag doll", not on the end — "obviously a lifeless toy
# and not a person named Ammu" reads as the mascot naming a corpse.
_DOLL_HEAD = "a cheerful rag doll"
# Short, and it names the WHOLE toy. Both halves matter and each was learned the hard
# way: 25 words DILUTED and the doll was dropped for a second puppy; naming only the head
# got a disembodied head, a knitted ball with a face.
_DOLL_TAIL = " with a stitched smile, yarn hair, a dress, and cloth arms and legs"

_OTHER_PERSON = re.compile(
    r"\b(mother|mum|mummy|mom|mommy|father|dad|daddy|papa|grandmother|grandma|"
    r"grandfather|grandpa|teacher|friend|brother|sister|parent)\b", re.I)
_ADULTS = {"mother", "mum", "mummy", "mom", "mommy", "father", "dad", "daddy", "papa",
           "grandmother", "grandma", "grandfather", "grandpa", "teacher", "parent"}


def toys_look_like_toys(scene: str) -> str:
    """A doll is human-shaped, and the identity reference is a human. Say what it is
    made of, or the doll comes back as a living child wearing the mascot's face."""
    m = _DOLL.search(scene or "")
    if not m:
        return scene
    if re.search(r"\b(?:rag|cloth|stitched|button eyes|stuffed|plastic|lifeless)\b",
                 scene, re.I):
        return scene                       # the scene already says what it is
    out = _DOLL.sub(
        lambda x: f"{_DOLL_HEAD}{x.group(1) or ''}{_DOLL_TAIL}", scene, count=1)
    log.info("scene named a doll without saying it is a TOY — a doll is human-shaped, "
             "and unsaid it comes back as a living child with the mascot's face")
    return _tidy(out)


def other_people_are_other_people(scene: str) -> str:
    """THERE IS ONE PERSON IN A MASCOT PICTURE, AND IT IS THE MASCOT.

    Qwen-Edit is handed exactly ONE identity reference, so it can draw exactly ONE
    person. It has nothing to draw a second person FROM. Three attempts, each failing
    worse than the last:

        no note        -> two identical Nakshus hugging. A twin, not a mother.
        parenthesised  -> "her smiling mother (a grown adult, much taller, a completely
                          different face...)" read as a PROP DESCRIPTION. The mother
                          vanished and the child was left holding a small doll.
        appositive     -> "her mother, a tall grown-up woman with a completely different
                          face and long hair" — and BOTH girls came back with the long
                          brown hair and the different face. Nakshu was gone from her own
                          lesson: no bindi, no topknot, no butterflies, on either of them.

    There is no wording that fixes this. The model cannot invent a consistent second
    character from a reference of the first, and every instruction meant for the second
    person lands on both. So the second person does not go in the picture.

    A six-year-old does not need to SEE the mother to understand the line "you feel happy
    when mummy hugs you" — she needs to see the FEELING, and the mascot can carry that on
    her own face and in her own arms.
    """
    m = _OTHER_PERSON.search(scene or "")
    if not m:
        return scene

    # If THIS MASCOT HAS A PICTURE OF THEM, they stay. Qwen-Edit takes image1..image3, and
    # handed an actual drawing of the mother it draws an actual mother — measured, first
    # try. See modules/cast.py: this whole guard was built on a conclusion ("one reference
    # draws one person") that was a limit of how we CALLED the model, not of the model.
    # Jeffy is the one who spotted it.
    #
    # A picture, not merely a NAME: a relation with an empty slot still has nothing for
    # Qwen to draw from, and would still come back as a twin of the mascot.
    try:
        from modules import cast
        from modules import mascot_library as _ml
        mid = _ml.get_active_id()
        if mid and cast.ref_for(scene, mid)[1]:
            return scene
    except Exception as e:          # pragma: no cover - the family is optional
        log.debug(f"family unavailable ({e}); keeping the mascot alone")

    who = m.group(1).lower()
    log.warning(f"scene put a {who} in the frame and we have no reference for them — "
                f"one reference draws one person, so a second would come back as a TWIN. "
                f"Keeping the mascot alone.")
    out = re.sub(r",?\s*[^,]*\b" + re.escape(who) + r"\b[^,]*", "", scene, count=1,
                 flags=re.I)
    # A plural pronoun is left pointing at nobody once the second person is gone:
    # "both of them laughing" with only one person in the frame.
    out = _tidy(re.sub(r",?\s*\b(?:both of them|the two of them|each other|together)\b",
                       "", out, flags=re.I))
    if not _HANDS_BUSY.search(out):
        out = _tidy(out + ", both arms wrapped around herself in a happy hug, eyes "
                          "closed, a big blissful smile")
    return out


# A LESSON is one girl, on one day, in one film. Her clothes do not change.
#
# The shared style says "only the clothes and the props change" — correct for a FACTS
# reel, where a costume per fact is the joke, and wrong here: shot 1 came back in a
# pinafore and shot 3 in her white top and denim skirt. Thirteen of those watch like
# thirteen different days.
#
# The costume clause is also how the mascot lost her identity outright once: a "toy
# repairman's costume" (dungarees + a cap over the hair) returned a different child.
_GARMENT = (r"(?:coat|hat|cap|apron|suit|costume|outfit|uniform|jacket|sweater|dress|"
            r"clothes|helmet|gown|robe|cape|goggles|scarf|boots|gloves)")
# The trailing "(and <garment>)*" matters: "dressed in a chef hat and apron" stopped at
# "hat" and left a dangling "and apron" behind.
_COSTUME = re.compile(
    r",?\s*\b(?:wearing|dressed (?:in|as)|clad in)\s+"
    r"(?:(?!facing|holding|lifting|kneeling|hugging|pointing)[^,])*?" + _GARMENT +
    r"(?:\s+and\s+(?:an?\s+)?(?:\w+\s+){0,2}" + _GARMENT + r")*\b", re.I)


def her_clothes_do_not_change(scene: str) -> str:
    """Strip the costume. In a lesson she wears her own clothes, shot after shot."""
    if not _COSTUME.search(scene or ""):
        return scene
    log.info("scene put the mascot in a costume — in a lesson her clothes never change")
    return _tidy(_COSTUME.sub("", scene))


def clean_scene_for_the_mascot(scene: str, teaching: bool = False) -> str:
    """Every guard, in one call. Order matters: drop the montage first, so the guards
    below never spend their effort on a clause that is about to be cut anyway.

    The UNIVERSAL guards run for every mode — they are all "the mascot must survive its
    own picture": no species swap, no borrowed anatomy, no emoji where an eye should be,
    no bare skin on a child, no T-pose, one moment.

    Three are LESSON pedagogy and would damage a facts reel, which is spectacle by
    design and allowed its jokes:
      warm_face          — a facts mascot may pull a shocked face at a horrifying fact
      alive_looks_alive  — only a lesson needs a real puppy to read as ALIVE beside a toy
      props_fit_in_a_hand— a facts reel's giant honey dipper IS the joke
    """
    out = other_people_are_other_people(toys_look_like_toys(
        never_empty_handed(one_focus(keep_it_dressed(keep_the_face(keep_the_body(
            keep_the_mascot(one_moment(scene)))))))))
    if teaching:
        out = props_fit_in_a_hand(alive_looks_alive(warm_face(
            her_clothes_do_not_change(out))))
    return out


def keep_the_mascot(scene: str) -> str:
    """Rewrite a scene that would replace the mascot with an animal.

    The animal is not dropped — it is moved OUT of the mascot and put beside it, which
    is what the lesson wanted anyway: the mascot SHOWING you a puppy teaches the same
    thing as the mascot BEING one, and it is still the mascot.
    """
    creature = species_swap(scene)
    if not creature:
        return scene
    fixed = _AS_A_CREATURE.sub(f"standing beside a {creature}", scene, count=1)
    log.warning(f"scene would have turned the mascot into a {creature}; "
                f"put the {creature} beside it instead")
    return fixed


# --- Teaching, which is not the same job as entertaining ----------------------
# _EXPLAINER_SYS is a FACTS prompt. It asks for spectacle: "make it FUN", "surprise the
# viewer", "physical comedy", "the mascot can leap, hover, zoom, spin". For a fact reel
# that is right — the picture's job is to hold attention on a thing you already said.
#
# In a LESSON the picture's job is to TEACH, and spectacle actively fights it. Told to
# surprise, the model gave "living things eat, grow, move and have babies" a chef's
# costume, and gave "you feel happy when mommy or daddy hugs you" a girl hugging a
# smiling cartoon Earth. That second one is not merely off-topic: this lesson exists to
# teach that non-living things do not feel, and we drew a FACE on a ball. The picture
# taught the opposite of the words.
#
# So a teaching scene shows what the LINE SAYS. If the line names her mother, a puppy or
# a doll, that is what is in the frame.
_TEACHING_SYS = (
    "You describe ONE picture for ONE line of a lesson read aloud to six-year-olds. "
    "Answer with JSON only: {\"scene\": \"...\"}\n"
    "\n"
    "THE PICTURE MUST SHOW WHAT THE LINE SAYS. This is a lesson, not a joke. A child "
    "who cannot read is looking at the picture to understand the words. If the picture "
    "shows something else, the line is wasted.\n"
    "- If the line names a THING (a doll, a puppy, a rock, a plant, a toy car, a plate "
    "of food), that thing is IN THE PICTURE and IN HER HANDS. Say it plainly — 'holding "
    "a doll in one hand', 'lifting the rock up' — and name ONE prop, two at the most. A "
    "scene that asked for a doll AND a toy car AND both arms raised came back with "
    "EMPTY HANDS, standing in the reference photo's own arms-out pose: given too much "
    "to hold, the artist drops the lot and falls back on the reference.\n"
    "- NEVER put one thing on her left and another on her right and have her point at "
    "or compare BOTH. That gives her ARMS SPREAD WIDE — which is the pose of the "
    "reference photo, and the picture becomes a T-pose with props on the floor. To "
    "compare two things, she HOLDS one up and the other waits: 'holding up a toy car "
    "in one hand, a potted plant on the table beside her'.\n"
    "- If the line names a PERSON (mummy, daddy, a friend), that person is IN THE "
    "PICTURE with the mascot — and you MUST say they are a DIFFERENT person: 'her "
    "mother, a grown adult, much taller, a completely different face and different "
    "hair'. The artist is given ONE reference photo (the mascot) and copies it onto "
    "every human it draws: 'being hugged by her smiling mother' came back as TWO "
    "IDENTICAL MASCOTS hugging.\n"
    "- A DOLL IS A TOY, AND YOU MUST SAY WHAT IT IS MADE OF. A doll is human-shaped, so "
    "it is the FIRST thing the artist copies the reference photo onto: 'holding up a "
    "doll named Ammu' came back as a LIVING CHILD with the mascot's own face, held "
    "dangling by the wrist — in a lesson about how toys are NOT alive. Always write 'a "
    "limp cloth rag doll with stitched button eyes, floppy and lifeless'.\n"
    "- If the line asks a QUESTION, show the mascot ASKING it — holding up the thing "
    "she is asking about, head tilted, eyebrows raised.\n"
    "- NEVER put a face, eyes or a smile on an object that is not alive. No smiling "
    "sun, no happy cloud, no cartoon eyes on a ball or a rock or a toy. Half of these "
    "lessons are about what is alive and what is not, and a grinning rock teaches a "
    "child the wrong answer.\n"
    "- When the line CONTRASTS a living thing with a lifeless one, the two must be "
    "UNMISTAKABLY different in the picture. Say 'a REAL LIVE puppy, wagging and "
    "blinking' and 'a cloth rag doll, limp and still'. A scene that asked for 'a doll "
    "in one hand and a puppy in the other' came back holding TWO PLUSH TOY DOGS — both "
    "read as toys, and the one contrast the whole line is built on was gone. The living "
    "thing is ALIVE and MOVING; the toy is obviously a toy.\n"
    "- NEVER MENTION HER CLOTHES AT ALL. She wears her own everyday clothes in every "
    "shot of the lesson, and they do not change — this is one girl, on one day, in one "
    "film. A lesson whose character is in a pinafore in shot 1 and a skirt in shot 3 "
    "watches like thirteen different days. Do not write 'wearing', 'dressed in', or "
    "name any costume: a chef's hat does not explain that living things grow, and a toy "
    "repairman's outfit does not explain that toys are not alive. Describe what she is "
    "DOING and what she is HOLDING. Nothing else.\n"
    "- NEVER a head-to-toe costume, and NEVER a cap, helmet, hood or anything else over "
    "her hair. Her FACE and her HAIR are the only two things the artist has to "
    "recognise her by. A 'toy repairman's costume' — dungarees and a cap — came back as "
    "a COMPLETELY DIFFERENT CHILD, and the lesson's last shot starred a stranger.\n"
    "- ONE MOMENT, one action. Never write 'then', 'finally' or 'after that' — a still "
    "picture cannot show a sequence.\n"
    "- The mascot NEVER becomes something else. She may WEAR things and HOLD things, "
    "but she is never 'dressed as' an animal or another being. To show an animal, put "
    "the animal BESIDE her: 'the mascot character kneeling beside a puppy'.\n"
    "- NEVER touch her face. No 'heart eyes', 'star eyes' or any symbol where an eye "
    "should be — the artist deletes her eyes and pastes emoji over the sockets. "
    "Expressions are eyebrows and mouths.\n"
    "- She stays DRESSED. Never write about her belly, tummy or stomach.\n"
    "- NEVER name a body part she may not have. No 'paw', 'snout', 'tail', 'fur' — she "
    "may be a human child, and telling the artist she has paws is how a child ends up "
    "drawn with mouse ears. Say 'hand', 'holding', 'the character'.\n"
    "- HER HANDS ARE ALWAYS BUSY. This is the most important rule in this list. Every "
    "ruined picture so far — four of them, by four different routes — had her hands "
    "doing NOTHING: nothing to hold, or so much to hold that she held none of it, or "
    "one thing on each side to reach for. Idle hands go home to the reference photo, "
    "and the reference photo is a T-POSE: arms straight out, staring ahead. She is "
    "always holding, lifting, stroking, hugging, offering or showing something.\n"
    "- FULL BODY, FACING THE CAMERA, mouth open mid-sentence — she is talking to the "
    "child while she does it.\n"
    "- ALWAYS END WITH HER FACE, and it is ALWAYS warm — smiling, laughing, delighted, "
    "curious, wide-eyed with wonder. The picture has a face whether you ask for one or "
    "not, and if you do not say, the artist picks it from the MOOD OF THE WORDS: on the "
    "line 'they don't eat, they don't grow, they're not alive' it drew a six-year-old "
    "SCOWLING and snarling at the camera. When a line says what something CANNOT do, "
    "she is CURIOUS, not cross — head tilted, eyebrows raised, a questioning smile.\n"
    "- Never scary, never violent, never sad, never angry.\n"
    "- No thought bubbles, no speech bubbles, no picture-inside-the-picture. The artist "
    "fills them with garbled nonsense.\n"
    "- Under 30 words. Only what is VISIBLE: who is there, what she is doing, what she "
    "is holding, her expression.\n"
    "- No background, lighting or camera notes. No text, letters or numbers.\n"
    "\n"
    "Line: 'Your puppy Jimmy feels happy when you pet him!'\n"
    '{"scene": "the mascot character kneeling beside a happy puppy, stroking its back '
    'with one hand, laughing, facing the camera"}\n'
    "Line: 'Can you tell me why your doll does not need to eat?'\n"
    '{"scene": "the mascot character facing the camera holding up a doll in one hand '
    'and an empty plate in the other, head tilted, eyebrows raised in a question"}'
)


def explainer_scene(fact: str, topic: str = "", context: str = "",
                    teaching: bool = False) -> str:
    """A costumed, camera-facing mascot scene that ILLUSTRATES one fact.

    Unlike scene_prompt (a single funny thumbnail), this is the per-shot presenter
    frame for facts-mascot mode: the mascot dressed for THIS fact, facing camera,
    mouth visible so S2V can animate it speaking the narration. Safety-gated and
    fallback-safe like scene_prompt — never raises.

    `teaching=True` swaps the facts prompt (which asks for spectacle) for the lesson
    prompt (which asks the picture to show what the line SAYS). See _TEACHING_SYS.
    """
    fb = fallback_scene(fact or topic, context, topic)
    if not (fact or "").strip():
        return fb
    sys_prompt = _TEACHING_SYS if teaching else _EXPLAINER_SYS
    try:
        from modules.script_generator import _call_llm, _extract_json
        prompt = (f"Topic: {topic or '(general)'}\n"
                  f"The fact to illustrate:\n{fact.strip()[:400]}\n\n"
                  f"Describe the mascot presenter scene for this fact.")
        if teaching:
            prompt = (f"Lesson: {topic or '(general)'}\n"
                      f"The line the mascot speaks, out loud, to the child:\n"
                      f"{fact.strip()[:400]}\n\n"
                      f"Describe the ONE picture the child sees while hearing this line.")
        for _ in (1, 2):
            raw = _call_llm(prompt, sys_prompt, role="creative")
            got = (_extract_json(raw) or {}).get("scene") or ""
            if not got:
                # The model answered in prose instead of JSON. The scene is
                # usually still in there — a live reel lost a costumed shot to a
                # generic pose because nobody looked.
                got = _salvage_scene(raw)
                if got:
                    log.info(f"Explainer scene salvaged from prose: {got[:70]}")
            scene = _clean_scene(got)
            if not scene:
                continue
            # The mascot must survive its own scene. "Dressed as a puppy" is not a
            # costume — Qwen-Edit renders a puppy, keeps the clothes, and the
            # character is gone. Telling the model not to is not enough; this is the
            # check.
            scene = clean_scene_for_the_mascot(scene, teaching=teaching)
            bad = scene_violation(scene)
            if not bad:
                log.info(f"Mascot explainer scene: {scene}")
                return scene
            log.warning(f"Explainer scene rejected (contains {bad!r}): {scene}")
            prompt += ("\n\nThat was rejected as unsuitable for a friendly mascot. "
                       "Use a harmless costume/prop and a big friendly expression.")
        return fb
    except Exception as e:
        log.warning(f"Explainer scene LLM failed ({e}); using '{fb}'")
        return fb


# ==============================================================================
# RENDER
# ==============================================================================

def build_presenter_prompt(scene: str, background: str = "",
                           teaching: bool = False) -> tuple:
    """The EXACT strings a presenter still is rendered from: (positive, negative).

    This exists so the dashboard can SHOW you what the model actually gets. It must be the
    same code the renderer runs — not a re-creation of it. A preview that can drift from
    reality is the "a comment is not a test" bug wearing a new hat, and this project has
    already paid for that one: the watermark's docstring said "bottom-right" for months
    while the code said `(H-h)/2`, and the logo floated at mid-height in every frame.

    So render_scene() calls THIS. There is no second copy.

    A LESSON gets its own style and its own bans. A facts reel is spectacle — it is ALLOWED
    a giant honey dipper and a grinning cartoon sun, because the picture's only job is to
    hold attention on a line you already said out loud. A lesson's picture is what a child
    who cannot read is learning FROM, and every one of those is a defect there.
    """
    style = STYLE_TEACHING if teaching else STYLE_PRESENTER
    if background:
        # One lesson, one look. The style only asks for "a bold vivid solid color
        # background", so Qwen picked a NEW colour every shot: the first lesson ran blue,
        # purple, beige, grey, dark grey — thirteen pictures that watch like clips from
        # four different videos. A facts reel is one shot and does not care; a lesson is a
        # film.
        style = style.replace("bold vivid solid color background", background)
    negative = NEGATIVE_TEACHING if teaching else NEGATIVE_PRESENTER
    return f"{scene}, {style}", negative


def render_scene(scene: str, out_png: Path, aspect: str = "9x16",
                 seed: Optional[int] = None,
                 reference_images: Optional[list] = None,
                 headline: str = "",
                 presenter: bool = False,
                 full_quality: bool = False,
                 background: str = "",
                 teaching: bool = False) -> Optional[Path]:
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
        if presenter:
            prompt, negative = build_presenter_prompt(scene, background, teaching)
            baked = False
        elif baked:
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
                # full_quality = 20 steps at cfg 2.5, instead of the 4-step Lightning
                # LoRA at cfg 1.0. Two things change, and only one of them is what you
                # would expect.
                #
                # The negative prompt starts working. Lightning is DISTILLED to run at
                # cfg 1.0, and at cfg 1.0 there is no classifier-free guidance at all —
                # the model never looks at the negative conditioning. So
                # NEGATIVE_PRESENTER's ban on "t-pose, arms spread wide, stiff
                # symmetrical standing pose" was inert. The ban was written; the sampler
                # was not listening.
                #
                # But measured A/B (same scene, same seed, same reference), that does
                # NOT reliably fix the pose: on one scene the fast path was the more
                # dynamic of the two. The T-pose was coming from somewhere else — a
                # scene that turned the mascot into an animal (see keep_the_mascot) and
                # a reference image that is itself a T-pose.
                #
                # What full_quality DOES buy is the PROPS: a plate of sliced vegetables
                # instead of candy-coloured blobs. That matters in a lesson, where the
                # prop is the thing being taught, and not much on a facts reel, where
                # the mascot is the joke. So lessons ask for it (~100s a still) and
                # facts stay fast (~27s).
                use_lightning=not full_quality,
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
        # Park the flags beside the art. publish_kit REUSES these PNGs on a rerun
        # (a GPU render costs minutes, re-compositing a title costs nothing), and
        # a cache hit that forgets `baked` makes the next run paint the title on
        # top of type the model already drew — two headlines on one thumbnail.
        try:
            (out_dir / f"{stem}_mascot.json").write_text(json.dumps({
                "baked": baked, "scene": scene, "seed": seed,
                "backend": out["_backend"],
                "headline_shown": out.get("_headline_shown", {}),
            }), encoding="utf-8")
        except Exception as e:
            log.warning(f"could not park mascot art flags: {e}")
    return out
