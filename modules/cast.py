"""
Claw Bot — the mascot's family.

A lesson keeps naming a second person. "You feel happy when mummy hugs you", "you run and
play with your friends", "daddy reads to you" — the line names someone, and a picture that
does not show them teaches half the sentence.

For a long time we could not draw them. Qwen-Edit was handed ONE identity reference (the
mascot) and it painted that identity onto every human-shaped thing in the frame:

    no note        -> two identical mascots hugging. A twin, not a mother.
    parenthesised  -> "her mother (a grown adult, much taller...)" read as a PROP
                      description; the mother vanished and the child was left holding a
                      doll.
    appositive     -> "her mother, a tall grown-up woman with a completely different face
                      and long hair" — and BOTH girls came back with the long brown hair
                      and the different face. The mascot was gone from her own lesson.

I concluded the model could only ever draw one person, and worked around it by keeping the
mascot alone. That conclusion was WRONG, and Jeffy is the one who said so: "what if we
give another char as the ref for mom?"

It is not a limit of the model. It is a limit of how we were calling it.
`TextEncodeQwenImageEditPlus` takes image1..image3 — the backend has accepted three
references all along — and we had only ever passed one. Hand it an actual drawing of the
mother and it draws an actual mother. Measured, first try: the mascot with her own topknot,
butterflies and bindi, hugged by an adult woman in a teal kurta with a braid. Two people,
two identities, no twin.

The single-reference rule we DID measure was about three views of the SAME character, where
the extra references made Qwen copy a reference's stance instead of acting out the scene.
A different character is a different experiment, and I folded them together without ever
running it.

THE FAMILY BELONGS TO THE MASCOT. `assets/mascots/<id>/family/` — because a second mascot
inheriting the first one's mother is just the twin problem with extra steps.
"""

import json
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modules import mascot_library as ml          # noqa: E402
from modules.file_utils import atomic_write_json  # noqa: E402

log = logging.getLogger("claw_bot.cast")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
META = "meta.json"

# The relations a lesson asks for. `prompt` only feeds the optional Generate button and
# the "who is in the second reference image" sentence — an uploaded picture needs none of
# it. The adults are explicitly A GROWN ADULT: a mother who reads as a big child is the
# twin problem wearing a different hat.
PRESETS = {
    "mother": {
        "label": "Mom",
        "prompt": ("a warm friendly mother in her thirties, a grown adult woman, long "
                   "black hair in a braid, a simple teal kurta and dark trousers, "
                   "gentle smile"),
    },
    "father": {
        "label": "Dad",
        "prompt": ("a warm friendly father in his thirties, a grown adult man, short "
                   "black hair, a neat moustache, a simple blue shirt and dark trousers, "
                   "gentle smile"),
    },
    "grandmother": {
        "label": "Grandmother",
        "prompt": ("a kind grandmother in her sixties, a grown adult woman, grey hair in "
                   "a bun, round glasses, a soft cream saree, warm smile"),
    },
    "grandfather": {
        "label": "Grandfather",
        "prompt": ("a kind grandfather in his sixties, a grown adult man, grey hair, "
                   "round glasses, a plain white kurta, warm smile"),
    },
    "uncle": {
        "label": "Uncle",
        "prompt": ("a cheerful uncle in his forties, a grown adult man, short black hair, "
                   "a checked shirt and dark trousers, big friendly grin"),
    },
    "aunt": {
        "label": "Aunty",
        "prompt": ("a cheerful aunty in her forties, a grown adult woman, hair in a bun, "
                   "a bright orange salwar kameez, big friendly grin"),
    },
    "teacher": {
        "label": "Teacher",
        "prompt": ("a friendly schoolteacher in her thirties, a grown adult woman, hair "
                   "in a neat bun, a green saree, kind smile"),
    },
    "friend": {
        "label": "Friend",
        "prompt": ("a cheerful child of about six, short black hair, a yellow t-shirt and "
                   "blue shorts, big grin"),
    },
}

# What a scene calls them.
ALIASES = {
    "mum": "mother", "mummy": "mother", "mom": "mother", "mommy": "mother", "ma": "mother",
    "dad": "father", "daddy": "father", "papa": "father", "pa": "father",
    "grandma": "grandmother", "granny": "grandmother", "nani": "grandmother",
    "grandpa": "grandfather", "thatha": "grandfather",
    "aunty": "aunt", "auntie": "aunt",
    "brother": "friend", "sister": "friend", "classmate": "friend",
}

# A cast reference is PLAIN: front-on, full body, empty hands, nothing behind them.
# Identity transfer keys on the face and the clothes — and a prop in a reference gets
# copied into every scene that character ever appears in.
STYLE = ("full body, standing facing the camera, arms relaxed at their sides, holding "
         "nothing, 3d pixar style character, soft even studio lighting, plain white "
         "background, full body visible from head to toe")
NEGATIVE = ("chibi, big head, toddler, cropped, close-up, headshot, props, held objects, "
            "text, letters, watermark, busy background")


# ==============================================================================
# WHERE THE FAMILY LIVES
# ==============================================================================

def family_dir(mid: str) -> Optional[Path]:
    d = ml.mascot_dir(mid)
    return (d / "family") if d else None


def _meta(mid: str) -> dict:
    d = family_dir(mid)
    if not d:
        return {"members": {}}
    f = d / META
    if not f.is_file():
        return {"members": {}}
    try:
        got = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"members": {}}
    if not isinstance(got.get("members"), dict):
        return {"members": {}}
    return got


def _save_meta(mid: str, meta: dict) -> None:
    d = family_dir(mid)
    if not d:
        return
    d.mkdir(parents=True, exist_ok=True)
    atomic_write_json(d / META, meta)


def _image(mid: str, relation: str) -> Optional[Path]:
    d = family_dir(mid)
    if not d or not d.is_dir():
        return None
    for ext in IMAGE_EXTS:
        p = d / f"{relation}{ext}"
        if p.is_file() and p.stat().st_size:
            return p
    return None


def spec(mid: str, relation: str) -> Optional[dict]:
    """What we know about one family member: their label and what they look like."""
    if relation in PRESETS:
        base = dict(PRESETS[relation])
    else:
        custom = _meta(mid)["members"].get(relation)
        if not custom:
            return None
        base = {"label": custom.get("label") or relation.title(),
                "prompt": custom.get("desc") or ""}
    base["relation"] = relation
    base["path"] = _image(mid, relation)
    return base


def members(mid: str) -> list:
    """Every slot this mascot has — the presets first, then any relation you added.

    Slots with no picture are included, empty: the UI needs somewhere to drop the upload.
    """
    out = [spec(mid, r) for r in PRESETS]
    for r in _meta(mid)["members"]:
        if r not in PRESETS:
            got = spec(mid, r)
            if got:
                out.append(got)
    return [m for m in out if m]


def have(mid: str, relation: str) -> bool:
    return _image(mid, relation) is not None


# ==============================================================================
# CHANGING IT
# ==============================================================================

def add_relation(mid: str, label: str, desc: str = "") -> str:
    """A relation of your own — Cousin Ravi, the neighbour, whoever the book needs."""
    if not ml.describe(mid):
        raise ValueError(f"no mascot named {mid!r}")
    label = (label or "").strip()
    if not label:
        raise ValueError("a relation needs a name")
    relation = ml.slugify(label)
    if relation in PRESETS:
        return relation                     # already a slot; nothing to add
    meta = _meta(mid)
    meta["members"][relation] = {"label": label, "desc": (desc or "").strip()}
    _save_meta(mid, meta)
    log.info(f"{mid}: added the relation {label!r} ({relation})")
    return relation


def put_image(mid: str, relation: str, src: Optional[Path] = None,
              data: Optional[bytes] = None, filename: str = "") -> Path:
    """Install a family member's picture.

    STAGES the incoming file before deleting the old one. mascot_library.put_file learned
    that the hard way: the "one voice per mascot" cleanup unlinked voice.mp3 and then tried
    to read it as the source, and it destroyed Jeffy's only recording. The source can BE
    the file you are replacing.
    """
    if not ml.describe(mid):
        raise ValueError(f"no mascot named {mid!r}")
    if not spec(mid, relation):
        raise ValueError(f"{mid} has no relation {relation!r} — add it first")

    ext = Path(filename or (src.name if src else "")).suffix.lower()
    if ext not in IMAGE_EXTS:
        raise ValueError(f"{ext or 'that'} is not an image ({', '.join(sorted(IMAGE_EXTS))})")

    d = family_dir(mid)
    d.mkdir(parents=True, exist_ok=True)
    staged = d / f"_incoming_{relation}{ext}"
    if data is not None:
        staged.write_bytes(data)
    elif src and Path(src).is_file():
        shutil.copyfile(src, staged)
    else:
        raise ValueError("nothing to install")

    for old_ext in IMAGE_EXTS:                # one picture per relation
        old = d / f"{relation}{old_ext}"
        if old.is_file():
            old.unlink()
    dest = d / f"{relation}{ext}"
    staged.replace(dest)
    log.info(f"{mid}: {relation} portrait installed ({dest.name})")
    return dest


def remove_member(mid: str, relation: str) -> None:
    """Forget one family member: the picture, and the relation if it was a custom one."""
    d = family_dir(mid)
    if not d or not d.is_dir():
        return
    for ext in IMAGE_EXTS:
        p = d / f"{relation}{ext}"
        if p.is_file():
            p.unlink()
    meta = _meta(mid)
    if relation in meta["members"]:
        meta["members"].pop(relation)
        _save_meta(mid, meta)
    log.info(f"{mid}: removed {relation}")


def remove_family(mid: str) -> None:
    """The whole family, and nobody else's."""
    d = family_dir(mid)
    if d and d.is_dir():
        shutil.rmtree(d)
        log.info(f"{mid}: family removed")


def draw(mid: str, relation: str, seed: int = 777) -> Optional[Path]:
    """Draw a family member (the optional path — uploading your own is the main one).

    Cached: a lesson never pays for this twice.
    """
    got = spec(mid, relation)
    if not got:
        log.warning(f"{mid} has no relation {relation!r}")
        return None
    if got["path"]:
        return got["path"]
    if not got["prompt"]:
        log.warning(f"{mid}/{relation} has no description to draw from — upload a picture")
        return None

    d = family_dir(mid)
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"{relation}.png"

    from modules.image_backends import comfyui_qwen_edit as qe
    log.info(f"drawing {mid}/{relation} (once; cached at {out})")
    try:
        r = qe.generate(
            prompt=f"{got['prompt']}, {STYLE}",
            output_path=out,
            aspect_ratio="9:16",
            seed=seed,
            use_lightning=False,      # drawn once — pay for it
            negative_prompt=NEGATIVE,
        )
    except Exception as e:
        log.warning(f"could not draw {mid}/{relation}: {e}")
        return None
    if not (r and getattr(r, "success", False) and out.is_file()):
        log.warning(f"{mid}/{relation} did not render")
        return None
    return out


# ==============================================================================
# READING A SCENE
# ==============================================================================

def _pattern(mid: str) -> re.Pattern:
    """Every word this mascot's family answers to. Longest first, so "grandmother" is not
    eaten by "mother"."""
    words = set(PRESETS) | set(ALIASES)
    for r, m in _meta(mid)["members"].items():
        words.add(r.replace("-", " "))
        if m.get("label"):
            words.add(m["label"].lower())
    words = sorted((w for w in words if w), key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")\b", re.I)


def role_in(scene: str, mid: str) -> Optional[str]:
    """The family member this scene asks for, if any."""
    if not scene or not mid:
        return None
    m = _pattern(mid).search(scene)
    if not m:
        return None
    who = m.group(1).lower()
    if who in ALIASES:
        return ALIASES[who]
    if who in PRESETS:
        return who
    # a custom relation, matched on its slug or its label
    for r, meta in _meta(mid)["members"].items():
        if who in (r.replace("-", " "), (meta.get("label") or "").lower()):
            return r
    return None


def ref_for(scene: str, mid: str) -> tuple:
    """(relation, image) for the family member this scene names — or (None, None).

    None for a solo scene, which is most of them, and they must NOT pay for a second
    reference: a spare reference is another thing for Qwen to copy into a picture that did
    not ask for it.
    """
    relation = role_in(scene, mid)
    if not relation:
        return None, None
    img = _image(mid, relation)
    if not img:
        log.info(f"scene names {relation} but {mid} has no picture of them — "
                 f"add one on the Mascots tab, or they cannot be drawn")
        return None, None
    return relation, img


def name_the_refs(scene: str, mid: str, relation: str) -> str:
    """Tell the model which reference is which — and NOTHING ELSE about them.

    Two references and no explanation is an invitation to blend them. The render that
    proved this works said so explicitly — "the little girl in the FIRST reference image
    ... the grown woman in the SECOND" — and came back with two correct, separate people
    on the first try.

    WE DO NOT DESCRIBE THEM. This sentence used to carry the relation's appearance text
    ("long black hair in a braid, a simple teal kurta and dark trousers") — the PRESET's
    words — and it carried them even when Jeffy had uploaded a picture of his own mother.
    So the prompt described one woman while the reference image showed another, and the
    prompt won: she came back in a teal kurta with a braid instead of the tan patterned
    kurta and dupatta he had given her.

    It is the same rule the mascot's own body taught, twice, at the cost of a defect each
    time: BY REFERENCE, NEVER BY DESCRIPTION. Any adjective competes with the picture, and
    the adjective wins, because Qwen weights the text over the reference. Point at the
    picture and say "that one".

    The appearance text still has a job — it is what `draw()` renders a member FROM, and
    it is how a custom relation gets a face at all. It just has no business in a prompt
    that already has their photograph attached.
    """
    if not relation or "reference image" in (scene or ""):
        return scene
    got = spec(mid, relation) or {}
    who = ml.describe(mid) or {}
    mascot_name = who.get("name") or "the mascot"
    label = got.get("label") or relation
    kind = "child" if relation == "friend" else "grown-up"
    return (f"{mascot_name}, the child in the FIRST reference image, together with her "
            f"{label} — the {kind} in the SECOND reference image, copied exactly from that "
            f"image. {scene}")


# ==============================================================================
# MIGRATION
# ==============================================================================

def migrate() -> Optional[str]:
    """Move the old GLOBAL cast into the active mascot's family, once.

    assets/cast/mother.png was drawn before the family belonged to anybody. It is the
    active mascot's mother — there was only one mascot when it was made.
    """
    old = Path(__file__).parent.parent / "assets" / "cast"
    if not old.is_dir():
        return None
    mid = ml.get_active_id()
    if not mid:
        return None
    d = family_dir(mid)
    if not d:
        return None
    moved = []
    for p in sorted(old.iterdir()):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        relation = p.stem.lower()
        if relation not in PRESETS or have(mid, relation):
            continue
        d.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, d / f"{relation}{p.suffix.lower()}")
        moved.append(relation)
    if moved:
        log.info(f"moved the old shared cast into {mid}'s family: {', '.join(moved)}")
    return mid if moved else None
