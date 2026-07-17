"""Claw Bot — the lesson SHOT CONTRACT: what this picture is supposed to be.

Every defect this pipeline ever shipped was SILENT. Nothing errored, nothing warned, and the
video looked right at every intermediate step and was wrong in the file. The reason is not
that the bot cannot tell — it is that **nothing ever wrote down what the shot was supposed to
look like**, so there was nothing to compare the picture against. Not for the QC model, and
not for Jeffy at a 104px thumbnail.

THE CONTRACT ALREADY EXISTS. It is dissolved into prompt prose. Every claim below is computed
today and then melted into one paragraph handed to a sampler:

    cast.role_in            knows who is in the shot
    never_empty_handed      already CHOSE the exact pose (`_BUSY_POSES`)
    warm_face               already NAMED the expression (`_WARM_DEFAULT`)
    lesson_objects          already forced the canonical prop desc (a doll IS a faceless block)
    setting_for             already fixed the one backdrop for the lesson
    build_presenter_prompt  already decided the apple ban, conditionally

This module does not compute any of that again. It **names those claims as separate, checkable
things** — reading them back off the finished scene with the same patterns that put them
there. No LLM. Nothing here decides anything; it only writes down what was decided.

THE READER IS THE GATE — a person. Printed beside the picture, the contract is what turns
"does this look OK?", which nobody can answer at a glance, into "is she holding the blue block
and nothing else?", which anyone can.

There is no vision model here, deliberately (jeffy, 2026-07-17). One was built, wired to this
contract, and measured on the ledger's own known-bad stills:

    qwen2.5vl:7b    0% recall on EVERY check — 0/3 on a face-on-an-object (it said, of a ball
                    with a face on it, "No faces on objects"), 0/3 on the phantom apple, 0/1
                    on the T-pose. Not uncertain: confidently blind.
    qwen2.5vl:32b   caught the T-pose 1/1, the doll-with-a-face 1/1, 2/3 faces-on-objects —
                    but at 60-140s an image (≈20 min a lesson) plus a 21 GB unload of
                    ComfyUI, for a hint that could not be trusted to redraw anything.

Both models had ALSO been misjudged before that, by a parser bug (they answer under
"[key]" and "1" respectively; the lookup wanted the bare key, so every verdict was
discarded and the old fail-open default called it "pass"). That bug is why the record said
the 32b could not see a face on a ball. It could.

The whole apparatus was deleted anyway: the checking is `.claude/skills/image-supervisor`,
which is a person looking at the pixels. This module's job is to tell that person what to
look FOR.

And `diff(contract, delivered)` catches the losses the bot ALREADY DETECTS and throws away
into a progress string that scrolls off the feed long before the gate:

    object_desc_locked   a named person + two pinned props = prop #2 silently loses its ref
                         slot to the person and falls back to words. It will drift, and there
                         is NO other symptom. This is the consistency hole.
    person_dropped       the scene names mummy, the shelf has no photo of her, she is cut
                         (correctly — the alternative is a TWIN) and nobody is told.
    framing_not_delivered  the crop failed, so the closeup is a full-body shot. A log.warning
                         nobody reads.
    negative_not_sent    the default lesson backend (USO) discards the negative prompt
                         entirely — see `mascot.render_scene`. Every negative-side ban in the
                         ledger is inert on it.

NOTHING IN THIS MODULE MAY RAISE, BLOCK, OR REDRAW. A wrong code costs one extra line in a
list at the gate. That is the price ceiling, on purpose: a check that can sink a lesson is
worse than the defect it was written for.
"""
import logging
import re
import sys
from pathlib import Path

_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modules import mascot

log = logging.getLogger("claw_bot.lesson_contract")

# The fruit words `build_presenter_prompt` itself tests before it appends the apple ban. Named
# here from the same list so the contract cannot promise a ban the prompt did not make.
_FRUIT = re.compile(r"apple|fruit|banana|orange|berry|berries|grape", re.I)

# Toy nouns whose canonical description is a FACELESS one. A doll is drawn as a building
# block on purpose (IMAGE_FEEDBACK #11: a rag doll with a face is a horror-film doll, and a
# faceless HUMAN-SHAPED doll reads as a voodoo doll — the shape was the problem), so the
# contract has to say the picture's truth, not the narration's word.
_TOY = re.compile(r"\b(?:doll|toy|block|brick|ball|teddy|figurine)\b", re.I)


# ==============================================================================
# THE CONTRACT — what this shot promised
# ==============================================================================

def contract_for(lesson: dict, i: int, mid: str = "") -> dict:
    """What shot `i` is supposed to be. Derived, deterministic, no LLM.

    Read off the FINISHED scene — the one the guards have already rewritten — because the
    guards are where the promises are made. `never_empty_handed` does not merely forbid a
    T-pose, it picks the pose; that pose IS the contract, and reading it back is the only way
    to check the picture kept it.
    """
    from modules import cast
    from modules import lesson_objects as lo
    from modules import lesson_pipeline as lp

    beats = (lesson or {}).get("beats") or []
    if not (0 <= i < len(beats)):
        return {}
    b = beats[i]
    scene = b.get("mascot_scene", "") or ""
    framing = b.get("framing", "medium")

    # WHO. A second person only counts when the shelf has a PICTURE of them — without one
    # they are cut from the shot (cast.ref_for), and a contract that still promised them
    # would flag every correct render.
    relation = cast.ref_for(scene, mid)[0] if mid else None
    named = cast.role_in(scene, mid) if mid else None

    # WHAT SHE HOLDS. The canonical desc, not the narration's word for it.
    objects = lo.objects_for(lesson)
    props = []
    for key in lo.detect(scene, objects):
        obj = next((o for o in objects if o.get("key") == key), None)
        if obj:
            props.append({"key": key,
                          "noun": obj.get("noun", key),
                          "desc": (obj.get("desc") or "").strip()})

    return {
        "line": i + 1,
        "framing": framing,
        "framing_says": mascot.FRAMING_CLAUSE.get(framing, ""),
        "people": {"count": 2 if relation else 1, "relation": relation, "named": named},
        "objects": props,
        "hands": _hands_of(scene),
        "face": "warm",                 # warm_face guarantees it or names it
        "background": lp.setting_for(lesson.get("topic", "") or lesson.get("title", "")),
        "banned": _banned_for(scene, props),
    }


def _hands_of(scene: str) -> dict:
    """What her hands were promised to be doing.

    `never_empty_handed` appends one of `_BUSY_POSES` to an idle scene, so when a pose is
    present verbatim it IS the promise and can be quoted at the QC model. Otherwise the scene
    named its own action and the busy-pattern found it.
    """
    s = scene or ""
    for pose in mascot._BUSY_POSES:
        if pose in s:
            return {"state": "busy", "says": pose, "injected": True}
    m = mascot._HANDS_BUSY.search(s) or mascot._HANDS_BUSY_EXTRA.search(s)
    if m:
        return {"state": "busy", "says": m.group(0), "injected": False}
    # Reaching here means the guard chain did not run or did not fire. Not an error — the
    # contract simply cannot promise a pose nobody chose.
    return {"state": "unstated", "says": "", "injected": False}


def _banned_for(scene: str, props: list) -> list:
    """What must NOT be in this picture. Shot-specific, and only what is actually true.

    A blanket ban list would be a lie on the shot that legitimately holds an apple, and the
    QC would flag it — so this mirrors `build_presenter_prompt`'s own conditional logic
    rather than restating a constant.
    """
    out = ["a face, eyes, a mouth or a smile on any non-living object",
           "a second copy of the child"]
    if not mascot._HELD_PROP.search(scene or ""):
        # No held prop named = anything in her hands is a hallucination. Measured: every
        # hallucinated-prop shot was a scene with only a hand GESTURE, and Qwen filled the
        # free hand.
        out.append("anything at all held in her hands")
    if not _FRUIT.search(scene or ""):
        # The same condition build_presenter_prompt uses before it appends the apple ban.
        out.append("an apple or any fruit")
    if any(_TOY.search(p.get("noun", "")) for p in props):
        out.append("a doll, figure or plush toy with a face, eyes or hair")
    return out


# ==============================================================================
# WHAT WAS DELIVERED — filled in by the renderer, not guessed here
# ==============================================================================

def delivered_blank() -> dict:
    """The shape `_scene_and_refs` fills in. Kept here so the two halves of the diff are
    defined in one place."""
    return {"refs": [], "people_drawn": 1, "people_dropped": None,
            "objects_by_ref": [], "objects_desc_locked": [],
            "framing": "", "crop_applied": None, "negative_sent": None}


# ==============================================================================
# THE DIFF — the losses nobody is currently told about
# ==============================================================================

_CODE_TEXT = {
    "person_dropped":
        "line {line}: the scene needs a {who}, but this mascot has no picture of them — she "
        "was left out of the shot (she would come back as a twin of the child). Add a photo "
        "on the Mascots tab and redraw.",
    "object_desc_locked":
        "line {line}: the {who} could not have its own reference picture on this shot (the "
        "person took the slot), so it is held to its description only — it may drift from "
        "the way it looks in the other shots. Check it.",
    "framing_not_delivered":
        "line {line}: the {who} crop did not apply, so this shot is framed wider than the "
        "shot list says. Redraw it.",
    "negative_not_sent":
        "line {line}: this backend takes no negative prompt, so the bans (no face on "
        "objects, no phantom apple) are not enforced by the sampler on this shot — only by "
        "the scene wording and the visual check.",
}


def diff(contract: dict, delivered: dict) -> list:
    """[(code, who)] — what the shot promised that the render did not deliver.

    Pure. Never raises: a contract or a delivered record that is missing or malformed yields
    no codes, because a broken detector must not manufacture warnings about good pictures.
    """
    if not contract or not delivered:
        return []
    out = []

    # A person the scene asked for, that we have no photo of. Correctly cut — but say so.
    dropped = delivered.get("people_dropped")
    if dropped:
        out.append(("person_dropped", dropped))

    # THE CONSISTENCY HOLE. Nothing else in the pipeline shows this.
    for key in delivered.get("objects_desc_locked") or []:
        noun = next((o["noun"] for o in contract.get("objects", [])
                     if o.get("key") == key), key)
        out.append(("object_desc_locked", noun))

    # The crop is the ONLY thing that delivers framing — Qwen-Edit ignores the prompt clause
    # and anchors to the full-body reference. `False` means it was asked for and failed;
    # `None` means it was never attempted (no crop needed), which is not a loss.
    framing = contract.get("framing", "")
    if delivered.get("crop_applied") is False and framing:
        out.append(("framing_not_delivered", framing))

    if delivered.get("negative_sent") is False:
        out.append(("negative_not_sent", ""))

    return out


def warnings_for(contract: dict, delivered: dict) -> list:
    """The diff, as {code, line, text} ready for `lesson_writer.add_warning`."""
    line = contract.get("line", 0)
    out = []
    for code, who in diff(contract, delivered):
        text = _CODE_TEXT.get(code, "line {line}: {who}")
        out.append({"code": code, "line": line,
                    "text": text.format(line=line, who=who or "shot")})
    return out
