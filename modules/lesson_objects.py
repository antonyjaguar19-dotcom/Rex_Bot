"""Claw Bot — the lesson's recurring OBJECTS, kept the same shot to shot.

A lesson keeps coming back to the same thing. "Living vs Non-living" is built on ONE doll
(the not-living example) and ONE puppy (the living one): "your doll never eats or grows, but
your puppy does". If the doll is a plush rabbit in shot 2, a peg doll in shot 8 and a pink
bear in shot 11, the through-line the whole lesson rests on is gone — and that is exactly
what the real render did (nakshu, 2026-07-15).

This is the `cast.py` idea, for props instead of people. Two layers, so an object stays
itself even when it cannot have a reference slot:

1. DESCRIPTION-LOCK (always on, free). Each recurring object has ONE canonical vivid
   description, pinned when the lesson is written. Every scene that names the object gets
   that exact text — so "a doll" can never be re-invented into a new toy.

2. REFERENCE IMAGE (the upgrade, when a slot is free). The object is drawn ONCE in isolation
   and handed to Qwen-Edit as an extra reference on shots that name it — the strongest
   consistency. cast.py's hard-won rule holds here too: a ref'd object is pointed at, NOT
   described (the adjective fights the picture and the adjective wins), so only the
   desc-locked objects carry their words.

Qwen-Edit takes THREE references (image1..image3). The mascot is always slot 1; a named
person (cast.py) takes slot 2 — person wins — so an object gets a real reference only when a
slot is left, and otherwise falls back to the description-lock. The budget is enforced by the
caller (`lesson_pipeline._scene_and_refs`); this module supplies the parts.
"""
import logging
import re
import sys
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modules.facts_memory import _singular      # the proven plural fold (keeps octopus)

log = logging.getLogger("claw_bot.lesson_objects")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
LESSONS_DIR = PROJECT_ROOT / "04_Outputs" / "lessons"

_ORDINALS = {1: "first", 2: "second", 3: "third", 4: "fourth"}

# The object is drawn ALONE — no character, no hands — so the reference carries only the
# object's own look, the way a cast portrait carries only the person. A prop in the frame
# with the mascot would drag the mascot into the reference and back into every shot.
_PIN_STYLE = ("centred on a plain soft pastel studio background, the whole object clearly "
              "visible, soft even lighting, 3d pixar style, no person, no child, no "
              "character, no hands, nobody holding it, a single object only")
_PIN_NEGATIVE = ("a person, a child, a character, a mascot, hands, fingers, holding, "
                 "arms, two of them, multiple objects, duplicate, text, letters, numbers, "
                 "watermark, busy background, cluttered")


def objects_for(lesson: dict) -> list:
    """The recurring objects the writer pinned for this lesson (may be empty)."""
    got = (lesson or {}).get("objects")
    return got if isinstance(got, list) else []


def _tokens(text: str) -> set:
    """The plural-folded content words of a piece of text, for whole-word matching."""
    return {_singular(w) for w in re.findall(r"[a-z]+", (text or "").lower())}


def _names(obj: dict) -> list:
    """Every word this object answers to — its noun and any aliases, plural-folded."""
    words = [obj.get("noun", "")] + list(obj.get("aliases", []))
    return sorted({_singular(w.lower()) for w in words if w})


def detect(text: str, objects: list) -> list:
    """The KEYS of the objects this text names, in the objects' own order (which the writer
    sorts by recurrence, so the more central object comes first — and therefore wins the
    ref slot). Whole-word, plural-folded: 'dolls' finds the 'doll' object, 'dollhouse' does
    not."""
    toks = _tokens(text)
    out = []
    for obj in objects or []:
        key = obj.get("key")
        if key and toks & set(_names(obj)):
            out.append(key)
    return out


# A hand action near a mention means THAT is the mention to keep — the object is meant to be
# in her hands, not lying on a table as well.
_HELD_NEAR = re.compile(
    r"\b(?:holding|carrying|showing|offering|lifting|handing|presenting|cradling|"
    r"gripping|clutching|in\s+(?:her|his)\s+hand)\b", re.I)


def one_mention_per_object(scene: str, objects: list) -> str:
    """One object, ONE mention. Asked for a ball twice, a model draws two balls.

    Measured: the writer produced "...a table where a small faceless blue rubber ball and a
    real live golden Labrador puppy sit beside each other, holding up a small bright blue
    rubber play ball..." — the ball placed AND held. Both Qwen-Edit 2509 and 2511 rendered
    two balls, correctly: the sentence ordered two.

    The mention kept is the HELD one (that is what a lesson shot wants — the prop in her
    hand); with no held mention the first survives. Every other mention has its noun phrase
    (article + up to five adjectives + the noun) cut out, and a stranded "and" is swept up.
    Whole-word and plural-folded, using the object's own noun + aliases — so the 'doll' whose
    desc is now a ball and whose aliases are ball/block/toy is matched however it was written.
    """
    s = scene or ""
    if not s.strip() or not objects:
        return scene
    for obj in objects:
        names = [re.escape(n) for n in _names(obj) if n]
        if not names:
            continue
        pat = re.compile(r"\b(?:" + "|".join(names) + r")s?\b", re.I)
        hits = list(pat.finditer(s))
        if len(hits) < 2:
            continue
        # which mention survives: the held one, else the first
        keep = hits[0]
        for h in hits:
            window = s[max(0, h.start() - 60):h.start()]
            if _HELD_NEAR.search(window):
                keep = h
                break
        # cut the OTHER mentions, noun phrase and all, back-to-front so offsets hold
        for h in reversed(hits):
            if h.start() == keep.start():
                continue
            phrase = re.compile(
                r"(?:\b(?:a|an|the)\s+)?(?:(?!and\b)[\w-]+\s+){0,5}$")
            m = phrase.search(s[:h.start()])
            cut_from = m.start() if m else h.start()
            s = s[:cut_from] + s[h.end():]
        log.info("scene named %r more than once — kept one mention", obj.get("noun"))
    # a cut can strand the "and" that joined the two objects, either at the head of a clause
    # ("..., and a puppy sit beside each other") or straight after the preposition that
    # introduced them ("in front of a table where and a puppy sit beside each other").
    s = re.sub(r",\s*and\s+", ", ", s)
    s = re.sub(r"\b(where|with|of|in|on|at|beside|near|holding|showing)\s+and\s+",
               r"\1 ", s, flags=re.I)
    s = re.sub(r"\band\s+and\b", "and", s, flags=re.I)
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r",\s*,", ",", s).strip(" ,")
    return s


def _by_key(objects: list, key: str) -> Optional[dict]:
    return next((o for o in (objects or []) if o.get("key") == key), None)


# ==============================================================================
# PINNING — draw the object once, cache it
# ==============================================================================

def objects_dir(lesson_id: str) -> Path:
    return LESSONS_DIR / lesson_id / "objects"


def image(lesson_id: str, key: str) -> Optional[Path]:
    """The pinned picture of this object, or None."""
    p = objects_dir(lesson_id) / f"{key}.png"
    return p if p.is_file() and p.stat().st_size else None


def pin(lesson_id: str, obj: dict, seed: int = 555) -> Optional[Path]:
    """Draw the object ALONE, once, and cache it. The reference every shot naming it will
    share. Never raises: a pin that fails leaves the object on the description-lock, which is
    a weaker guarantee but not a broken one."""
    key, desc = obj.get("key"), (obj.get("desc") or obj.get("noun") or "").strip()
    if not key or not desc:
        return None
    out = image(lesson_id, key)
    if out:
        return out                          # drawn already — never pay twice
    out = objects_dir(lesson_id) / f"{key}.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    from modules.image_backends import comfyui_qwen_edit as qe
    log.info(f"pinning lesson object {key!r} (once; cached at {out})")
    try:
        r = qe.generate(
            prompt=f"{desc}, {_PIN_STYLE}",
            output_path=out,
            aspect_ratio="1:1",
            seed=seed,
            use_lightning=False,            # drawn once — pay for the quality
            negative_prompt=_PIN_NEGATIVE,
        )
    except Exception as e:
        log.warning(f"could not pin {key!r}: {e}")
        return None
    if not (r and getattr(r, "success", False) and out.is_file()):
        log.warning(f"{key!r} did not render; it will use the description-lock instead")
        return None
    return out


# ==============================================================================
# BUILDING THE SCENE TEXT
# ==============================================================================

def name_object_refs(scene: str, ref_specs: list) -> str:
    """Point at the reference for each object that GOT a slot — 'by reference, not by
    description' (cast.name_the_refs). `ref_specs` = [(label, slot)] where slot is the
    object's 1-based position in the final reference list. Says WHICH image is which; says
    nothing about how it looks (the picture is the description)."""
    if not ref_specs:
        return scene
    parts = []
    for label, slot in ref_specs:
        ordinal = _ORDINALS.get(slot, f"{slot}th")
        parts.append(f"the {label} is exactly the one in the {ordinal} reference image, "
                     f"copied exactly")
    return f"{scene}. " + "; ".join(parts) + "."


def lock_descriptions(scene: str, keys: list, objects: list) -> str:
    """Inject the canonical description for each object that did NOT get a reference slot —
    so 'a doll' is drawn as THE doll even without its picture. This is the floor the whole
    feature stands on: it costs nothing and needs no slot."""
    clauses = []
    for key in keys or []:
        obj = _by_key(objects, key)
        if not obj:
            continue
        noun, desc = obj.get("noun", "it"), (obj.get("desc") or "").strip()
        if desc:
            clauses.append(f"the {noun} is exactly {desc}")
    if not clauses:
        return scene
    return f"{scene}. " + "; ".join(clauses) + "."
