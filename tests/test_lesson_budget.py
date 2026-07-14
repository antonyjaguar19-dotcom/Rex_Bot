"""A lesson must never be squeezed into a facts reel's 40-second ceiling.

This test exists BEFORE the code that could break it, because the trap is so easy to
fall into: the obvious way to voice a lesson is `facts_pipeline._voice_beats_mascot()`
— it clones the mascot, it has the short-take guard, it is right there. And it ends in
`_fit_to_budget()`, which reads `rs.get_facts_max_seconds()` (40s) and speeds the
narration up by as much as 1.45x to fit.

A five-minute lesson through that path comes out as a teacher gabbling. Nothing errors.
Lessons call `_voice_beats_clone()` instead — one level down, same clone, same guard,
no budget — and their length is set in WORDS by the writer, where you can see it.
"""
import inspect
from pathlib import Path

import pytest

from modules import facts_pipeline as fp
from modules import lesson_writer as lw

MODULES = Path(__file__).parent.parent / "modules"


def _called_names() -> dict:
    """{module: every function name it CALLS}.

    Parsed, not grepped. The lesson modules carry long comments explaining this exact
    trap — and a substring scan cannot tell a warning about a landmine from stepping
    on one. It failed on its own documentation.
    """
    import ast

    out = {}
    for p in MODULES.glob("lesson_*.py"):
        names = set()
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call):
                f = node.func
                if isinstance(f, ast.Attribute):
                    names.add(f.attr)
                elif isinstance(f, ast.Name):
                    names.add(f.id)
        out[p.name] = names
    return out


def test_the_facts_40s_ceiling_is_never_read_by_lesson_code():
    for name, called in _called_names().items():
        assert "get_facts_max_seconds" not in called, (
            f"{name} reads the FACTS reel ceiling (40s). A lesson is minutes long — "
            f"this would trim the teacher's pace to fit a Short.")
        assert "_fit_to_budget" not in called, (
            f"{name} calls _fit_to_budget, which speeds the narration up to hit the "
            f"facts ceiling. A lesson's length is set in words by the writer.")


def test_the_budgeted_voice_path_is_never_used_by_lesson_code():
    """_voice_beats_mascot is the trap; _voice_beats_clone is the door next to it."""
    for name, called in _called_names().items():
        assert "_voice_beats_mascot" not in called, (
            f"{name} uses the facts voicing wrapper, which applies the 40s budget. "
            f"Use facts_pipeline._voice_beats_clone (no budget, same short-take guard).")


def test_the_lesson_pipeline_voices_through_the_unbudgeted_path():
    """Not just 'does not call the bad one' — proves it calls the RIGHT one."""
    called = _called_names()["lesson_pipeline.py"]
    assert "_voice_beats_clone" in called


def test_the_trap_is_real_and_still_there():
    """If facts ever stops trimming, the guards above become cargo cult. Prove the
    thing we are avoiding still exists and still reads the ceiling."""
    assert "_fit_to_budget" in inspect.getsource(fp._voice_beats_mascot)
    assert "get_facts_max_seconds" in inspect.getsource(fp._fit_to_budget)


def test_a_lessons_length_comes_from_its_words():
    """The voice reads at a fixed pace, so the script IS the runtime. 90 seconds is
    about 160 spoken words — not 'whatever the model wrote, sped up'."""
    beats, budget = lw.plan_shape(90)
    assert abs(budget / lw.WORDS_PER_SEC + lw.PAD_PER_BEAT * beats - 90) < 6

    # And the writer's own estimate is the same arithmetic, not a guess.
    assert lw.word_budget(90, 10) == int((90 - 0.7 * 10) * lw.WORDS_PER_SEC)


def test_a_lesson_is_never_forced_into_a_shorts_length():
    """45s floor, 180s ceiling. A Class-1 topic has a few hundred words behind it;
    beyond three minutes the lesson stops being the book's."""
    assert lw.plan_shape(5)[1] == lw.plan_shape(lw.MIN_SECONDS)[1]      # clamped up
    assert lw.plan_shape(600)[1] == lw.plan_shape(lw.MAX_SECONDS)[1]    # clamped down
