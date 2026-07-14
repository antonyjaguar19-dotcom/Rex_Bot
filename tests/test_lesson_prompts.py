"""Reveal the prompts — and never let the reveal lie.

What reaches Qwen is `f"{scene}, {style}"`: the scene PLUS ~120 words of STYLE_TEACHING,
against a ~90-word negative. None of it appeared in the dashboard, in lesson.json, or in
any log line — the backend deliberately logs only seed/steps/cfg. Every rule hammered in
this week (proportions by reference, the props fit in a hand, nothing inanimate gets a
face, one prop per hand) lived in a string nobody could read.

The video prompt was worse: invented at render time inside a throwaway dict and never
stored, so it could not be seen, let alone changed.

THE REVEAL MUST BE THE RENDERER'S OWN CODE. A preview assembled by a second copy would
drift from the truth, and a preview that can lie is worse than none. This project has
already paid for exactly that bug: the watermark's docstring said "bottom-right" for
months while the code said `(H-h)/2`, and the logo floated at mid-height in every frame
Jeffy looked at. A comment is not a test.
"""
import inspect

import pytest

from modules import lesson_pipeline as lp
from modules import lesson_writer as lw
from modules import mascot as mas


@pytest.fixture
def lesson(tmp_path, monkeypatch):
    monkeypatch.setattr(lw, "LESSONS_DIR", tmp_path)
    monkeypatch.setattr(lp, "LESSONS_DIR", tmp_path)
    lid = "20260714_150000"
    (tmp_path / lid).mkdir()
    beats = [{
        "kind": "teach",
        "narration": "Living things eat and grow.",
        "on_screen": "Alive",
        "image_prompt": "",
        "mascot_scene": "the mascot character holding up a small green plant, smiling",
        "animate": False, "approved": True, "index": 1,
    }]
    lw._save({"lesson_id": lid, "title": "Living Things",
              "topic": "Living and Non-living Things", "beats": beats,
              "stage": "stills", "word_count": 5, "estimated_seconds": 4.0})
    return lid


# --- the reveal is the truth ----------------------------------------------------

def test_there_is_only_one_prompt_builder():
    # render_scene must CALL the builder, not carry its own copy. Two copies is how a
    # preview starts telling you something the renderer is not doing.
    src = inspect.getsource(mas.render_scene)
    assert "build_presenter_prompt(scene, background, teaching)" in src
    assert "STYLE_TEACHING if teaching" not in src, "render_scene has its own copy again"


def test_the_revealed_prompt_is_byte_for_byte_what_qwen_gets(lesson):
    p = lp.prompts_for(lesson, 0)

    # rebuild it the way the RENDERER does — through the same two functions
    got = lw.load_lesson(lesson)
    from modules import mascot_library as ml
    scene_sent, _, _ = lp._scene_and_refs(got, 0, ml.get_active_id())
    backdrop = lp.setting_for(got["topic"])
    positive, negative = mas.build_presenter_prompt(scene_sent, backdrop, teaching=True)

    assert p["image_positive"] == positive
    assert p["image_negative"] == negative
    assert p["scene_sent"] == scene_sent


def test_the_style_and_the_negative_really_are_in_there(lesson):
    # The point of the reveal: these rules were invisible. Each cost a defect to learn.
    p = lp.prompts_for(lesson, 0)
    pos, neg = p["image_positive"], p["image_negative"]

    assert p["scene"] in pos, "the scene leads the prompt"
    assert "the same body proportions" in pos          # the bobblehead
    assert "the props are toys a child can hold" in pos  # the boulder
    assert "a sunny home garden" in pos, "the backdrop was substituted in"
    assert "bold vivid solid color background" not in pos

    assert "mouse ears" in neg                          # the cub's paws
    assert "bobblehead" in neg
    assert "floating object" in neg
    assert "twins" in neg                               # the mother
    assert "headless doll" in neg


def test_the_numbers_come_from_the_backend_not_from_a_retype(lesson):
    from modules.image_backends import comfyui_qwen_edit as qe
    p = lp.prompts_for(lesson, 0)
    assert p["steps"] == qe.DEFAULT_STEPS_FULL
    assert p["cfg"] == qe.DEFAULT_CFG_FULL
    # the same seed expression the renderer uses — a redraw moves it, or you would get the
    # identical picture back (which is exactly what "Redo the pictures" did for 20 minutes)
    assert p["seed"] == lp._seed_for(lw.load_lesson(lesson), 0)


# --- the video prompt exists now -------------------------------------------------

def test_the_motion_prompt_is_stored_not_invented_and_thrown_away(lesson):
    p = lp.prompts_for(lesson, 0)
    assert p["motion_is_default"] is True
    assert p["motion"] == lp.default_motion(lw.load_lesson(lesson)["beats"][0])

    lw.set_beat_field(lesson, 0, "motion_prompt", "she leans in, the camera pushes past")
    p = lp.prompts_for(lesson, 0)
    assert p["motion_is_default"] is False
    assert p["motion"] == "she leans in, the camera pushes past"

    # and the RENDER prefers what you wrote
    src = inspect.getsource(lp.render_lesson)
    assert 'beats[i].get("motion_prompt") or default_motion(beats[i])' in src

    # prepare writes the default down, so it exists to be looked at
    assert 'b["motion_prompt"] = default_motion(b)' in inspect.getsource(lp.prepare_lesson)


# --- an edited scene is guarded --------------------------------------------------

def test_a_hand_edited_scene_gets_the_same_guards(lesson):
    # The guards used to run at WRITE time only (inside explainer_scene); render_scene
    # never called them. So a hand-edited scene reached Qwen byte-for-byte and could walk
    # straight back into a defect we had already paid for.
    got = lw.set_scene(lesson, 0,
                       "the mascot character dressed as a playful puppy, mid-leap")
    assert got["saved"] and got["guarded"]
    assert "dressed as" not in got["after"]
    assert "standing beside a puppy" in got["after"], "the mascot survives her own scene"

    on_disk = lw.load_lesson(lesson)["beats"][0]["mascot_scene"]
    assert on_disk == got["after"], "the CLEANED text is what gets saved"


def test_the_guards_report_what_they_changed(lesson):
    # Silently rewriting a person's words is its own kind of lie.
    got = lw.set_scene(lesson, 0, "the mascot character holding a rock, angry scowl")
    assert got["guarded"] is True
    assert got["before"] != got["after"]
    assert "angry" not in got["after"]

    # a scene the guards are happy with is left alone, and reported as such
    clean = "the mascot character holding up a small toy car in one hand, smiling"
    got = lw.set_scene(lesson, 0, clean)
    assert got["guarded"] is False
    assert got["after"] == clean


def test_editing_the_scene_clears_the_tick(lesson):
    # A different scene is a different picture, and nobody has seen it. It must not be
    # able to reach Wan on the strength of a tick given to the picture it replaced.
    assert lw.load_lesson(lesson)["beats"][0]["approved"] is True
    lw.set_scene(lesson, 0, "the mascot character waving, smiling")
    assert lw.load_lesson(lesson)["beats"][0]["approved"] is False
    assert lw.unapproved(lw.load_lesson(lesson)) == [1]
