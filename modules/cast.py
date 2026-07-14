"""
Claw Bot — the supporting cast.

A lesson keeps needing a second person. "You feel happy when mummy hugs you", "you run
and play with your friends", "your teacher shows you a picture" — the line names someone,
and a picture that does not show them teaches half the sentence.

For a long time we could not draw them. Qwen-Edit was handed ONE identity reference (the
mascot) and it painted that identity onto every human-shaped thing in the frame:

    no note        -> two identical Nakshus hugging. A twin, not a mother.
    parenthesised  -> "her mother (a grown adult, much taller...)" read as a PROP
                      description; the mother vanished and the child was left holding a
                      doll.
    appositive     -> "her mother, a tall grown-up woman with a completely different face
                      and long hair" — and BOTH girls came back with the long brown hair
                      and the different face. The mascot was gone from her own lesson.

I concluded from that the model could only ever draw one person, and worked around it by
keeping the mascot alone. That conclusion was WRONG, and Jeffy is the one who said so:
"what if we give another char as the ref for mom?"

It is not a limit of the model. It is a limit of how we were calling it.
`TextEncodeQwenImageEditPlus` takes image1..image3 and we had only ever passed one. Give
it a SECOND reference — an actual drawing of the mother — and it draws an actual mother.
Measured, first try: Nakshu with her topknot, butterflies and bindi, hugged by an adult
woman in a teal kurta with a braid. Two people, two identities, no twin.

The single-reference rule we DID measure was about three views of the SAME character,
where the extra references made Qwen copy a reference's stance instead of acting out the
scene. A different character is a different experiment, and I folded them together
without running it.

Cast members are drawn once, on demand, and cached in `assets/cast/`. They are deliberately
plain: front-on, full body, plain background, no props — that is what identity transfer
keys on.
"""

import logging
import re
import sys
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

log = logging.getLogger("claw_bot.cast")

CAST_DIR = Path(__file__).parent.parent / "assets" / "cast"

# Who a lesson can ask for, and what they look like. Adults are unmistakably ADULT —
# a mother who reads as a big child is the twin problem wearing a different hat.
ROLES = {
    "mother": ("a warm friendly indian mother in her thirties, a grown adult woman, "
               "long black hair in a braid, a simple teal kurta and dark trousers, "
               "gentle smile"),
    "father": ("a warm friendly indian father in his thirties, a grown adult man, "
               "short black hair, a neat moustache, a simple blue shirt and dark "
               "trousers, gentle smile"),
    "grandmother": ("a kind indian grandmother in her sixties, a grown adult woman, "
                    "grey hair in a bun, round glasses, a soft cream saree, warm smile"),
    "grandfather": ("a kind indian grandfather in his sixties, a grown adult man, grey "
                    "hair, round glasses, a plain white kurta, warm smile"),
    "teacher": ("a friendly indian schoolteacher in her thirties, a grown adult woman, "
                "hair in a neat bun, a green saree, holding nothing, kind smile"),
    "friend": ("a cheerful indian boy of about six, short black hair, a yellow t-shirt "
               "and blue shorts, big grin"),
}

# The words a scene uses for them.
_ALIASES = {
    "mum": "mother", "mummy": "mother", "mom": "mother", "mommy": "mother",
    "dad": "father", "daddy": "father", "papa": "father",
    "grandma": "grandmother", "granny": "grandmother",
    "grandpa": "grandfather",
    "brother": "friend", "sister": "friend", "classmate": "friend",
}

_ROLE_RE = re.compile(
    r"\b(" + "|".join(sorted(set(ROLES) | set(_ALIASES), key=len, reverse=True)) + r")\b",
    re.I)

# Plain, front-on, full body, nothing in their hands. Identity transfer keys on the face
# and the clothes, and a prop in the reference gets copied into every scene they appear in.
_STYLE = ("full body, standing facing the camera, arms relaxed at their sides, holding "
          "nothing, 3d pixar style character, soft even studio lighting, plain white "
          "background, full body visible from head to toe")
_NEGATIVE = ("chibi, big head, toddler, small child, cropped, close-up, headshot, "
             "props, held objects, text, letters, watermark, busy background")


def role_in(scene: str) -> Optional[str]:
    """The supporting character this scene asks for, if any."""
    m = _ROLE_RE.search(scene or "")
    if not m:
        return None
    who = m.group(1).lower()
    return _ALIASES.get(who, who)


def ref_path(role: str) -> Path:
    return CAST_DIR / f"{role}.png"


def have(role: str) -> bool:
    p = ref_path(role)
    return p.is_file() and p.stat().st_size > 0


def draw(role: str, seed: int = 777) -> Optional[Path]:
    """Draw a cast member once. Cached — a lesson never pays for this twice."""
    if role not in ROLES:
        log.warning(f"no such cast role: {role!r}")
        return None
    out = ref_path(role)
    if have(role):
        return out

    CAST_DIR.mkdir(parents=True, exist_ok=True)
    from modules.image_backends import comfyui_qwen_edit as qe

    log.info(f"drawing the {role} reference (once; cached at {out})")
    try:
        r = qe.generate(
            prompt=f"{ROLES[role]}, {_STYLE}",
            output_path=out,
            aspect_ratio="9:16",
            seed=seed,
            use_lightning=False,          # the cast reference is drawn ONCE — pay for it
            negative_prompt=_NEGATIVE,
        )
    except Exception as e:
        log.warning(f"could not draw the {role} reference: {e}")
        return None
    if not (r and r.success and out.is_file()):
        log.warning(f"the {role} reference did not render")
        return None
    return out


def name_the_refs(scene: str, role: str) -> str:
    """Tell the model which reference is which.

    Two references and no explanation is an invitation to blend them. The test render
    that proved this works said so explicitly — "the little girl from the first reference
    image ... the grown woman from the second reference image" — and came back with two
    correct, separate people on the first try.
    """
    if not role or "reference image" in (scene or ""):
        return scene
    who = "grown-up" if role != "friend" else "child"
    return (f"the mascot character, the little girl in the FIRST reference image, "
            f"together with her {role}, the {who} in the SECOND reference image. "
            f"{scene}")


def ref_for(scene: str) -> Optional[Path]:
    """The cast reference this scene needs, drawing it if we have never needed it before.

    None when the scene has nobody but the mascot in it — which is most of them, and
    those must NOT pay for a second reference: a spare reference is another thing for
    Qwen to copy into the picture.
    """
    role = role_in(scene)
    if not role:
        return None
    return draw(role)
