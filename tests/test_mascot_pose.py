"""The mascot must survive its own picture.

Two failures, found together on a real lesson still: the child had been replaced by a
DOG (wearing her clothes), standing in the reference image's T-pose.

Neither was the model being stupid. We told it to do both.
"""
import inspect

import pytest

from modules import mascot as mas


# --- the mascot is never turned into something else ---------------------------

def test_dressed_as_an_animal_is_caught():
    """The scene that actually shipped: "the mascot character dressed as a playful
    puppy, mid-leap with a chew toy in mouth". Qwen-Edit did exactly that — it drew a
    puppy, kept the child's clothes, and the character was gone."""
    scene = ("the mascot character dressed as a playful puppy facing the camera, "
             "mid-leap with a chew toy in mouth, wagging tail")
    assert mas.species_swap(scene) == "puppy"


def test_an_adjective_does_not_smuggle_the_animal_past():
    """A regex anchored straight onto the noun missed "a PLAYFUL puppy" — which is
    the exact string that broke the render."""
    for scene, want in [
        ("the mascot character dressed as a tiny busy bee", "bee"),
        ("the mascot character disguised as an octopus", "octopus"),
        ("the mascot character dressed as a big friendly elephant", "elephant"),
    ]:
        assert mas.species_swap(scene) == want


def test_a_real_costume_is_left_alone():
    """A hat is a costume. A creature is not. The mascot may be a chef, a beekeeper,
    a scuba diver — those keep the character and only change its clothes."""
    for scene in [
        "the mascot character in a beekeeper suit lifting a dripping honeycomb",
        "the mascot character dressed as a chef dishing out a plate of vegetables",
        "the mascot character dressed as a scuba diver pointing at a glowing jellyfish",
        "the mascot character kneeling beside a puppy, holding out a bone",
    ]:
        assert mas.species_swap(scene) is None


def test_the_animal_is_moved_beside_the_mascot_not_deleted():
    """The lesson still wants its puppy — it just cannot BE the puppy. Showing one
    teaches the same thing as being one, and the mascot survives."""
    got = mas.keep_the_mascot(
        "the mascot character dressed as a playful puppy, mid-leap with a chew toy")
    assert "standing beside a puppy" in got
    assert "dressed as" not in got
    assert mas.species_swap(got) is None


def test_the_scene_writer_is_told_and_then_checked():
    """The prompt is not the feature — the check is. Both must be there."""
    assert "NEVER BECOMES SOMETHING ELSE" in mas._EXPLAINER_SYS
    src = inspect.getsource(mas.explainer_scene)
    assert "keep_the_mascot" in src, "the guard must run on what the model returns"


# --- the pose ------------------------------------------------------------------

def test_full_quality_turns_lightning_off():
    """At cfg 1.0 (the 4-step Lightning LoRA) there is no classifier-free guidance, so
    NEGATIVE_PRESENTER's ban on "t-pose, arms spread wide" is never even read.

    Measured A/B, though: that is NOT what was giving us T-poses — on one scene the
    fast path was the more dynamic of the two. What the full path really buys is the
    PROPS (sliced vegetables instead of candy-coloured blobs), which is why a LESSON
    asks for it and a facts reel does not.
    """
    assert "t-pose" in mas.NEGATIVE_PRESENTER.lower()

    src = inspect.getsource(mas.render_scene)
    assert "use_lightning=not full_quality" in src


def test_a_lesson_pays_for_its_props_and_a_facts_reel_does_not():
    """A lesson's prop is the teaching. A facts reel's mascot is the joke."""
    from modules import facts_pipeline as fp
    from modules import lesson_pipeline as lp

    assert "full_quality=True" in inspect.getsource(lp.prepare_lesson)
    assert "full_quality" not in inspect.getsource(fp._render_facts_mascot)


def test_the_lightning_path_really_is_cfg_1():
    """If the backend ever stops distilling at cfg 1.0, the guard above becomes cargo
    cult. Prove the thing we are avoiding still exists."""
    from modules.image_backends import comfyui_qwen_edit as qe
    assert qe.DEFAULT_CFG_LIGHTNING == 1.0
    assert qe.DEFAULT_CFG_FULL > 1.0
