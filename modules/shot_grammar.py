"""Claw Bot — the shared SHOT GRAMMAR: one framing vocabulary for every pipeline.

Two pipelines grew their own copies of the same idea. Lesson mode carries a per-beat
`framing` (mascot.FRAMING_CLAUSE + shot_framing crop ladder); the kids/story pipeline has
`shot_type` (prompt_assembler.SHOT_TYPE_FRAMING). Same concept — how close the camera is —
written twice, and neither knew about `establishing` or `montage`.

This module is the SINGLE SOURCE OF TRUTH for the STRUCTURE of a shot grammar:

- `FRAMINGS`      — the canonical names, superset of both pipelines + establishing + montage.
- `FRAME_WINDOW`  — the crop ladder (top-fraction, height-fraction) that DELIVERS a framing
                    for backends that ignore the prompt clause (Qwen-Edit). Lifted from
                    shot_framing; establishing/montage/insert added.
- `framing_for()` — assign a framing per beat from its `kind`, then de-duplicate neighbours.
                    Generalized from lesson_writer._framing_for (which now delegates here).
- `sequences()`   — group an ordered list of beats into named Sequences for a storyboard
                    readout, deterministically (no LLM — same doctrine as the coverage and
                    outro checks).

Deliberately NOT unified here: the PROSE clause each pipeline injects into its prompt. The
mascot presenter clause ("the mascot's face and shoulders fill the frame") and the kids
clause ("the character's face fills most of the frame") legitimately differ in subject and
wording, and existing lessons/tests are pinned to the exact strings. This module owns the
vocabulary and the geometry; each pipeline keeps its own words, validated against `FRAMINGS`.
"""
from typing import Callable, Optional

# The canonical framing names, closest-to-farthest is not the order — this is the union of
# the lesson ladder (establishing/wide/medium/closeup), the kids grammar (wide/medium/
# closeup/insert), plus montage. `medium` is every pipeline's implicit default.
FRAMINGS = ("establishing", "wide", "medium", "closeup", "insert", "montage")
DEFAULT_FRAMING = "medium"


def is_framing(name: str) -> bool:
    return (name or "").strip().lower() in FRAMINGS


# The crop ladder: (top_fraction, height_fraction) of the source, aspect kept, applied by
# shot_framing.window() for backends that anchor to the full-body reference and ignore the
# framing clause. `establishing`/`montage` = the whole frame (a montage is composed as a
# grid, not cropped in). `insert` drops to the hands/object band. Measured on the real
# Class-1 lesson (nakshu, 2026-07-15); a full window (0.0, 1.0) == the picture unchanged.
FRAME_WINDOW = {
    "establishing": (0.00, 1.00),
    "wide":         (0.00, 0.88),
    "medium":       (0.00, 0.74),
    "closeup":      (0.02, 0.52),
    "insert":       (0.30, 0.55),   # the hands and the object they touch, face out of frame
    "montage":      (0.00, 1.00),   # composed as a grid — never cropped
}
NO_CROP = (0.00, 1.00)


def window_factors(framing: str) -> tuple:
    """The (top_fraction, height_fraction) for a framing; NO_CROP for anything unknown."""
    return FRAME_WINDOW.get((framing or "").strip().lower(), NO_CROP)


def framing_for(kinds: list, kind_map: dict, default: str = DEFAULT_FRAMING,
                nudge: Optional[dict] = None) -> list:
    """A camera framing per beat: assigned by kind, then no two NEIGHBOURS the same.

    Mechanical and deterministic. `kind_map` gives each kind its wanted framing; a kind not
    in the map falls back to `default`. `nudge` names the workhorse framings that may be
    swapped when two neighbours collide (e.g. {"medium": "wide", "wide": "medium"}); framings
    NOT in `nudge` (closeup, establishing — rare and load-bearing) are left as written even
    when they repeat. Generalized from lesson_writer._framing_for; the lesson passes its own
    maps so its behaviour is byte-for-byte unchanged.
    """
    nudge = nudge or {}
    out = [kind_map.get(k, default) for k in kinds]
    for i in range(1, len(out)):
        if out[i] == out[i - 1] and out[i] in nudge:
            out[i] = nudge[out[i]]
    return out


def sequences(items: list, title_for: Callable[[object], str]) -> list:
    """Group an ordered list into named Sequences for a storyboard readout.

    Returns [{"title": str, "shots": [index, ...]}], order preserved; consecutive items with
    the same title merge into one group. `title_for(item)` names the sequence an item belongs
    to. No LLM — deterministic from the items themselves, so a reorder regroups on the next
    read with nothing stored. Generalized from lesson_writer.sequences.
    """
    out: list = []
    for i, item in enumerate(items or []):
        title = title_for(item)
        if out and out[-1]["title"] == title:
            out[-1]["shots"].append(i)
        else:
            out.append({"title": title, "shots": [i]})
    return out
