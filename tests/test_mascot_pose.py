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
    # clean_scene_for_the_mascot() is keep_the_mascot + every sibling guard that came
    # after it (body, face, montage, modesty) in one call.
    assert "clean_scene_for_the_mascot" in src, "the guards must run on what the model returns"


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


# --- The mascot's body belongs to the reference image, not the prompt ---------
# still_01 of the first real lesson came back with a HUMAN CHILD WEARING MOUSE EARS.
# The scene said "mid-balance on one paw" and the style suffix claimed the character was
# "a short chunky cub" with "both paws" on the prop. Told the subject has paws, the model
# reasons backwards to the creature paws belong to and grows the ears to match. Nakshu is
# a little girl. Nothing in a prompt may state what the mascot's body IS.

def test_the_presenter_style_names_no_species_and_no_animal_parts():
    style = mas.STYLE_PRESENTER.lower()
    for word in ("cub", "paw", "paws", "fur", "snout", "tail", "muzzle", "whiskers"):
        assert word not in style.split(), f"presenter style calls the mascot's body a {word!r}"
    # and it must actively pin the body to the reference
    assert "same species" in style


def test_the_negative_bans_the_parts_the_old_prompt_invited():
    neg = mas.NEGATIVE_PRESENTER.lower()
    assert "mouse ears" in neg
    assert "changed species" in neg


def test_animal_anatomy_is_stripped_out_of_a_scene():
    scene = ("the mascot character dressed in a chef coat, mid-balance on one paw "
             "holding a carrot, tail wagging")
    assert mas.animal_parts(scene) == ["paw", "tail"]
    fixed = mas.keep_the_body(scene)
    assert "paw" not in fixed and "tail" not in fixed
    assert "one hand holding a carrot" in fixed
    assert "chef coat" in fixed          # the costume is untouched


def test_a_real_animal_in_the_scene_keeps_its_own_tail():
    # A lesson about living things puts a real puppy on screen, and the puppy is ALLOWED
    # a tail. Only anatomy in a clause that names no animal can be the mascot's. Blunt
    # substitution turned "puppy wagging its tail" into "puppy wagging its back".
    scene = ("the mascot character kneeling beside a playful puppy, "
             "scratching its belly with both hands, puppy wagging its tail")
    assert mas.animal_parts(scene) == []
    assert mas.keep_the_body(scene) == scene


def test_the_mascots_paw_is_taken_even_when_an_animal_shares_the_frame():
    scene = ("the mascot character balancing on one paw, "
             "pointing at a puppy wagging its tail")
    assert mas.animal_parts(scene) == ["paw"]
    fixed = mas.keep_the_body(scene)
    assert "on one hand" in fixed          # hers is rewritten
    assert "wagging its tail" in fixed     # his is not


def test_a_human_scene_passes_through_unchanged():
    scene = "the mascot character holding up a plant with one hand, smiling"
    assert mas.animal_parts(scene) == []
    assert mas.keep_the_body(scene) == scene


def test_the_scene_writer_is_told_not_to_hand_out_body_parts():
    sys_prompt = mas._EXPLAINER_SYS.lower()
    assert "'paw'" in sys_prompt or "paw" in sys_prompt
    assert "mouse ears" in sys_prompt     # the rule states WHY, or it gets deleted later


# --- Three more ways a scene can wreck the mascot, all seen in one real lesson ----

def test_heart_eyes_are_stripped_because_the_face_is_the_identity():
    # Shipped: the writer asked for "waving happily with heart eyes" and Qwen DELETED
    # her eyes, pasting two red emoji hearts over the sockets. Identity transfer keys
    # on the face; a symbol in an eye socket is a different character.
    scene = "the mascot character waving happily with heart eyes, holding a plate"
    out = mas.keep_the_face(scene)
    assert "heart" not in out
    assert "waving happily" in out and "holding a plate" in out
    for variant in ("with star eyes", "with big heart-shaped eyes", "with spiral eyes"):
        assert "eyes" not in mas.keep_the_face(f"the mascot {variant}, smiling")
    assert "mouse ears" in mas.NEGATIVE_PRESENTER  # the sibling guard is still there
    assert "heart-shaped eyes" in mas.NEGATIVE_PRESENTER


def test_a_still_is_one_moment_not_a_storyboard():
    # Shipped: "holding a plate of food to its mouth, then growing taller, waving
    # happily, finally hopping on one foot, showing movement" — four actions in one
    # frame came back as a smear of limbs.
    scene = ("the mascot character holding a plate of food to her mouth, then growing "
             "taller, finally hopping on one foot")
    out = mas.one_moment(scene)
    assert out == "the mascot character holding a plate of food to her mouth"
    # a scene with no sequence word is untouched
    plain = "the mascot character holding up a rock, playful frown"
    assert mas.one_moment(plain) == plain


def test_the_mascot_stays_dressed():
    # Shipped: "belly expanding comically with each bite" drew a six-year-old in a crop
    # top with a bare midriff. This is a lesson for six-year-olds.
    scene = ("the mascot character in a chef's apron, holding up a fruit slice mid-chew, "
             "belly expanding comically with each bite, cheerful smile")
    out = mas.keep_it_dressed(scene)
    assert "belly" not in out
    assert "fruit slice mid-chew" in out and "cheerful smile" in out
    for banned in ("crop top", "bare midriff", "exposed belly"):
        assert banned in mas.NEGATIVE_PRESENTER, banned   # modesty is UNIVERSAL
    assert "fully clothed" in mas.STYLE_PRESENTER
    # the PUPPY may still have a belly to scratch
    dog = "the mascot character kneeling beside a puppy, scratching its belly"
    assert mas.keep_it_dressed(dog) == dog


def test_every_guard_runs_on_one_scene():
    scene = ("the mascot character dressed as a playful puppy, balancing on one paw, "
             "belly showing, with heart eyes, then hopping away")
    out = mas.clean_scene_for_the_mascot(scene)
    for banned in ("dressed as a playful puppy", "paw", "heart eyes", "then"):
        assert banned not in out, f"{banned!r} survived every guard: {out!r}"
    assert "standing beside a puppy" in out      # she is herself, next to the animal
    # "belly" SURVIVES here, and that is the documented trade. keep_it_dressed only cuts
    # when no animal is in the scene, because a pronoun reaches back across a comma
    # ("puppy, scratching its belly") and the blunt version stripped the DOG's belly.
    # With an animal present the NEGATIVE prompt is what keeps her shirt on.
    assert "crop top" in mas.NEGATIVE_PRESENTER


def test_the_belly_is_cut_when_the_scene_is_hers_alone():
    scene = "the mascot character holding a fruit slice, belly expanding comically"
    assert "belly" not in mas.keep_it_dressed(scene)


def test_the_teachers_face_is_never_left_to_the_renderer():
    # still_08, SECOND time: line 9 is a negative statement ("they don't eat, they don't
    # grow, they're not alive"), the scene named no expression at all, and Qwen picked
    # one from the SENTIMENT of the words — a six-year-old scowling and snarling at the
    # camera. Banning "angry" in the negative did not stop it, because a face is not
    # optional: the model will choose one. The scene must SAY which.
    plain = "the mascot character standing between a toy car and a plant, pointing"
    out = mas.warm_face(plain)
    assert "warm friendly smile" in out
    assert "pointing" in out

    cross = "the mascot character holding a rock, playful frown, shaking her head"
    out = mas.warm_face(cross)
    assert "frown" not in out
    assert "shaking her head" in out          # the ACTION survives
    assert "warm friendly smile" in out

    # a scene that already names a warm face is left alone
    warm = "the mascot character kneeling beside a puppy, laughing"
    assert mas.warm_face(warm) == warm


def test_warm_face_runs_in_the_full_guard():
    out = mas.clean_scene_for_the_mascot(
        "the mascot character holding a rock, angry scowl", teaching=True)
    assert "angry" not in out and "scowl" not in out
    assert "warm friendly smile" in out


def test_the_teaching_prompt_puts_the_prop_in_her_hands():
    # still_12: the scene named a doll AND a toy car AND both arms lifted, and came back
    # with EMPTY HANDS in the reference's own arms-out pose. Given too much to hold, the
    # artist drops the lot and falls back on the reference — which is the T-pose.
    sysp = mas._TEACHING_SYS.lower()
    assert "in her hands" in sysp
    assert "empty hands" in sysp, "the rule must keep its reason"
    assert "one prop, two at the most" in sysp


# --- The T-pose has three roads home, and they all end at the reference photo ------
# The reference image is a T-pose, so arms-spread is the model's road back to it. We
# have now watched it take that road three separate ways:
#   1. an EMPTY scene       — nothing to hold, so she stands like the reference
#   2. TOO MUCH to hold     — the artist drops the lot and stands her like the reference
#   3. one thing on each SIDE — she reaches for both, and arms-spread IS the T-pose
# (1) and (2) are the writer's business and the prompt covers them. (3) is a sentence
# shape, so it can be caught.

def test_a_prop_on_each_side_is_a_t_pose():
    # Shipped: "standing between a toy car and a plant, pointing at each in turn" — she
    # came back with both arms straight out, a toy on each side of her feet.
    scene = ("the mascot character standing between a toy car and a plant, "
             "pointing at each in turn, facing the camera")
    out = mas.clean_scene_for_the_mascot(scene)
    assert "standing between" not in out
    assert "standing beside" in out
    assert "in turn" not in out            # a sequence wearing a disguise
    assert "toy car" in out and "plant" in out   # both props stay IN the frame


def test_arms_spread_wide_becomes_a_real_gesture():
    for scene in (
        "the mascot character with both arms out wide, smiling",
        "the mascot character standing with arms open wide, smiling at the camera",
        "the mascot character arms spread wide, holding nothing",
        "the mascot character with wide open arms, laughing",
    ):
        out = mas.clean_scene_for_the_mascot(scene)
        assert "one hand raised" in out, out
        for banned in ("arms out wide", "arms open wide", "arms spread wide",
                       "open arms"):
            assert banned not in out, f"{banned!r} survived: {out!r}"


def test_a_guard_never_cuts_away_the_subject():
    # An early version's clause-scoped delete took "the mascot character" out along with
    # the offending phrase, and handed the backend a picture with nobody in it.
    for scene in (
        "the mascot character with both arms out wide",
        "the mascot character, arms spread wide, angry scowl",
        "the mascot character standing between a car and a plant",
    ):
        out = mas.clean_scene_for_the_mascot(scene)
        assert out.lower().startswith("the mascot character"), out


def test_a_good_scene_survives_every_guard_intact():
    # The guards must be a net, not a mangler. This is what a correct scene looks like:
    # hands busy, one prop held, sizes stated, a warm face, no sequence, no bare skin.
    scene = ("the mascot character facing the camera holding up a small cloth rag doll "
             "with stitched eyes in one hand, a small toy car on the floor beside her, "
             "shaking her head with a warm knowing smile")
    assert mas.clean_scene_for_the_mascot(scene, teaching=True) == scene


def test_a_real_animal_beside_a_toy_is_told_to_look_alive():
    # Line 10 of the first lesson: "why doesn't your doll Ammu need to eat like Jimmy
    # does?" — its ENTIRE point is that one of them is alive and the other is not. The
    # scene asked for "a doll in one hand and a puppy in the other" and Qwen drew TWO
    # PLUSH TOY DOGS. Both read as toys, and the contrast the line is built on was gone.
    # A lesson that cannot show the difference cannot teach it.
    scene = ("the mascot character facing the camera holding a doll named Ammu in one "
             "hand and a puppy in the other, curious")
    out = mas.alive_looks_alive(scene)
    assert "real live puppy" in out
    assert "wagging and blinking" in out   # marked ALIVE, in place, ONCE
    assert "doll named Ammu" in out        # the toy is untouched


def test_an_animal_with_no_toy_beside_it_is_left_alone():
    # Nothing to contrast with — no need to labour the point, and every word added to a
    # scene is a word competing with the ones that matter.
    scene = "the mascot character kneeling beside a happy puppy, stroking its back, laughing"
    assert mas.alive_looks_alive(scene) == scene
    assert mas.alive_looks_alive("the mascot character holding a toy car, smiling") == \
        "the mascot character holding a toy car, smiling"


def test_a_scene_that_already_says_it_is_not_said_twice():
    scene = "the mascot character holding a doll and a real live puppy, wagging, smiling"
    assert mas.alive_looks_alive(scene) == scene


def test_alive_looks_alive_runs_in_the_full_guard():
    out = mas.clean_scene_for_the_mascot(
        "the mascot character holding a doll in one hand and a puppy in the other",
        teaching=True)
    assert "real live puppy" in out


# --- The actual invariant, arrived at the long way round -------------------------
# Four T-poses shipped, by four different routes, and they all have ONE thing in common:
# HER HANDS WERE DOING NOTHING.
#   still_12  nothing to hold           (empty scene)
#   still_12  a doll AND a car AND both arms raised -> she held neither
#   still_08  a car on one side, a plant on the other -> she reached for both
#   still_11  a plant, a puppy and a friend flanking her, hands empty
# The rule is not about props, or sides, or counts. Idle hands go home to the reference
# photo, and the reference photo is a T-pose.

def test_the_mascot_is_never_empty_handed():
    idle = ("the mascot character standing beside a plant, a puppy, and her friend, "
            "facing the camera, smiling")
    out = mas.never_empty_handed(idle)
    assert "waving one arm high" in out
    assert "the other hand resting at her side" in out   # ASYMMETRIC, so not a T-pose
    # and the repair must satisfy the guard's own test, or it appends on every pass
    assert mas.never_empty_handed(out) == out

    for busy in (
        "the mascot character kneeling beside a puppy, stroking its back, laughing",
        "the mascot character holding up a toy car in one hand, smiling",
        "the mascot character hugging her mother, beaming",
        "the mascot character pointing at a rock, curious",
        "the mascot character lifting a plate of food, delighted",
    ):
        assert mas.never_empty_handed(busy) == busy, f"busy hands were meddled with: {busy}"


def test_never_empty_handed_runs_in_the_full_guard():
    out = mas.clean_scene_for_the_mascot(
        "the mascot character standing beside a plant and a puppy, facing the camera")
    assert "waving one arm high" in out


def test_the_writer_is_told_the_rule_that_matters_most():
    sysp = mas._TEACHING_SYS
    assert "HER HANDS ARE ALWAYS BUSY" in sysp
    assert "T-POSE" in sysp
    assert "Idle hands go home to the reference photo" in sysp


def test_every_guard_is_idempotent():
    # A scene is cleaned when it is written AND again when it is redrawn. If a guard's
    # own repair does not satisfy the guard's own test, it appends every single time.
    # _BUSY_DEFAULT said "a cheerful wave" (noun) while _HANDS_BUSY only knew "waving"
    # (verb) — so the guard ran, wrote its fix, and STILL considered the scene idle.
    for scene in (
        "the mascot character facing the camera, smiling widely",
        "the mascot character standing beside a plant and a puppy",
        "the mascot character with arms outstretched to the sides, grinning",
        "the mascot character holding up a doll in one hand, laughing",
    ):
        once = mas.clean_scene_for_the_mascot(scene)
        twice = mas.clean_scene_for_the_mascot(once)
        assert once == twice, f"guard is not idempotent:\n  1x: {once}\n  2x: {twice}"


def test_outstretched_arms_are_a_t_pose_too():
    # The real scene the model wrote: "arms outstretched to the sides". _ARMS_WIDE knew
    # "arms stretched" and "arms out" but not the single word "outstretched", so the
    # T-pose sailed past and never_empty_handed then bolted a raised arm onto it —
    # leaving a scene that asked for BOTH arms wide AND one arm up.
    for scene in ("the mascot character with arms outstretched to the sides, smiling",
                  "the mascot character, outstretched arms, grinning at the camera"):
        out = mas.clean_scene_for_the_mascot(scene)
        assert "outstretched" not in out, out
        assert "waving one arm high" in out


# --- Proportions by reference, never by description ------------------------------
# This line has now broken the character TWICE, both times because the body was
# DESCRIBED instead of pointed at:
#   "a short chunky cub ... both paws"        -> a human girl grew MOUSE EARS
#   "chibi proportions, big head, small body" -> a BOBBLEHEAD, head bigger than her torso
# The second was my own repair of the first. The reference image HAS the proportions;
# any adjective in the prompt competes with it, and the adjective wins.

def test_the_build_is_pinned_to_the_reference_never_described():
    style = mas.STYLE_PRESENTER.lower()
    assert "the same head size" in style
    assert "the same body proportions" in style
    for described in ("chibi", "big head", "small body", "chunky", "cub", "bobblehead"):
        assert described not in style, f"the build is described as {described!r}"
    for banned in ("bobblehead", "oversized head", "head bigger than the body",
                   "distorted proportions"):
        assert banned in mas.NEGATIVE_PRESENTER, banned


def test_a_held_prop_touches_the_hand_that_holds_it():
    # "props held cleanly in front of the body, nothing intersecting" was read as KEEP
    # THE PROP AWAY FROM THE HAND — a plate of food came back HOVERING in mid-air beside
    # her, with her arm truncated behind it. A held object is touching the hand.
    style = mas.STYLE_PRESENTER.lower()
    assert "fingers wrapped firmly around the prop" in style
    assert "the hand visibly closed on the object" in style
    assert "nothing intersecting" not in style, "this is what pushed the prop off the hand"
    for banned in ("floating object", "hovering prop", "prop not touching the hand",
                   "object suspended in mid-air"):
        assert banned in mas.NEGATIVE_PRESENTER, banned


def test_a_prop_fits_in_a_hand():
    # "holding up a grey rock" came back as a BOULDER — bigger than her torso, hugged
    # with both arms. That also cost her the SECOND prop the scene asked for (a plant),
    # because both her arms were full of the first. Nothing in the prompt had ever said
    # how big a prop was. A prop is a child's toy: it goes in a hand.
    scene = ("the mascot character holding up a green leafy plant in one hand and a "
             "grey rock in the other, elbows bent")
    out = mas.props_fit_in_a_hand(scene)
    assert "a small green leafy plant" in out
    assert "a small grey rock" in out

    # a prop already described as small is not re-shrunk (and the guard is idempotent)
    already = "the mascot character holding a small toy car, smiling"
    assert mas.props_fit_in_a_hand(already) == already
    once = mas.clean_scene_for_the_mascot(scene, teaching=True)
    assert mas.clean_scene_for_the_mascot(once, teaching=True) == once

    # prop SCALE is a lesson rule — a facts reel's giant honey dipper is the joke
    for banned in ("giant prop", "boulder", "prop bigger than the character",
                   "prop hugged with both arms"):
        assert banned in mas.NEGATIVE_TEACHING, banned
    assert "every prop is SMALL" in mas.STYLE_TEACHING


def test_a_living_creature_is_not_shrunk_into_a_toy():
    # The puppy is a real animal, not a prop. Calling it "a small puppy" is harmless,
    # but the guard must not touch a scene that has no prop nouns in it at all.
    scene = "the mascot character kneeling beside a puppy, stroking its back, laughing"
    assert mas.props_fit_in_a_hand(scene) == scene


# --- Each mode tunes its own LLM; facts must not pay for the lesson's fixes -------
# A FACTS reel is spectacle by design. It is ALLOWED a giant honey dipper, a grinning
# cartoon sun, and a mascot pulling a shocked face at a horrifying fact — that is the
# joke, and the joke is the point when the picture's only job is to hold attention on a
# line you already said out loud. In a LESSON every one of those is a defect. So the
# lesson gets an overlay and facts keeps exactly what it had.

def test_the_lesson_tunes_its_own_prompt_and_facts_is_untouched():
    assert "every prop is SMALL" in mas.STYLE_TEACHING
    assert "every prop is SMALL" not in mas.STYLE_PRESENTER, \
        "a facts reel's giant honey dipper IS the joke"

    for lesson_only in ("angry", "scowling", "anthropomorphic object", "giant prop",
                        "thought bubble"):
        assert lesson_only in mas.NEGATIVE_TEACHING, lesson_only
        assert lesson_only not in mas.NEGATIVE_PRESENTER, \
            f"{lesson_only!r} is banned for FACTS too — that is the lesson's rule, not a shared one"

    # but everything that is "the mascot must survive its own picture" IS shared
    for universal in ("mouse ears", "bobblehead", "floating object", "changed species",
                      "heart-shaped eyes", "crop top", "t-pose"):
        assert universal in mas.NEGATIVE_PRESENTER, universal
        assert universal in mas.NEGATIVE_TEACHING, universal


def test_the_pedagogy_guards_only_run_for_a_lesson():
    cross = "the mascot character holding up a rock, angry scowl"
    # facts: left alone — a shocked face at a horrifying fact is the point
    assert mas.clean_scene_for_the_mascot(cross) == cross
    # lesson: the teacher is never cross with a six-year-old
    taught = mas.clean_scene_for_the_mascot(cross, teaching=True)
    assert "angry" not in taught and "scowl" not in taught
    assert "warm friendly smile" in taught
    assert "a small rock" in taught            # and the prop fits in a hand

    # the UNIVERSAL guards run for both
    swap = "the mascot character dressed as a playful puppy, mid-leap"
    for teaching in (False, True):
        out = mas.clean_scene_for_the_mascot(swap, teaching=teaching)
        assert "dressed as" not in out, "the species guard is universal"


def test_render_scene_picks_the_right_style_for_the_mode():
    import inspect
    src = inspect.getsource(mas.render_scene)
    assert "STYLE_TEACHING if teaching else STYLE_PRESENTER" in src
    assert "NEGATIVE_TEACHING if teaching else NEGATIVE_PRESENTER" in src
    assert "teaching" in inspect.signature(mas.render_scene).parameters

    from modules import lesson_pipeline as lp
    assert "teaching=True" in inspect.getsource(lp.prepare_lesson)
    # facts must NOT ask for the lesson's tuning
    from modules import facts_pipeline as fp
    assert "teaching=True" not in inspect.getsource(fp._render_facts_mascot)


# --- THE REFERENCE BLEEDS ---------------------------------------------------------
# Qwen-Edit is handed ONE identity reference — the mascot — and it applies that identity
# to every human-SHAPED thing in the frame. Two ruined stills from one cause, and the
# second is the worst thing this pipeline has produced:
#
#   "being hugged by her smiling mother" -> TWO IDENTICAL NAKSHUS, same clothes, same
#                                           hair, same butterflies, same bindi.
#   "holding up a doll named Ammu"       -> a LIVING CHILD with Nakshu's own face,
#                                           dangling by the wrist. The line it
#                                           illustrates is "Ammu is a toy and isn't
#                                           alive": the picture taught the OPPOSITE of
#                                           the words, and read as a child being hurt.
#
# A doll is human-shaped, which is exactly what the reference is, so it is the likeliest
# thing in any scene to capture it.

def test_a_doll_must_look_like_a_toy_not_a_child():
    scene = "the mascot character holding up a doll named Ammu in one hand"
    out = mas.toys_look_like_toys(scene)
    assert "cloth rag doll" in out
    assert "stitched button eyes" in out
    assert "lifeless toy and not a person" in out
    assert "named Ammu" in out, "the doll's name must survive — the child knows it by name"
    # and the name must not land on the word 'person'
    assert "not a person named Ammu" not in out

    for banned in ("doll with a human face", "living doll", "a real child held by the arm",
                   "child in distress", "realistic child instead of a doll"):
        assert banned in mas.NEGATIVE_PRESENTER, banned


def test_a_doll_already_described_as_cloth_is_left_alone():
    scene = "the mascot character holding a cloth rag doll with stitched eyes, smiling"
    assert mas.toys_look_like_toys(scene) == scene


def test_the_second_person_is_not_a_second_mascot():
    scene = "the mascot character being hugged by her smiling mother, laughing"
    out = mas.other_people_are_other_people(scene)
    assert "grown adult" in out
    assert "a completely different face" in out
    assert "much taller" in out
    # the note goes IN PLACE, beside the word "mother" — appending it named her twice,
    # and every extra mention of a thing is another copy of that thing
    assert "mother (a grown adult" in out
    for banned in ("twins", "clone", "duplicate character", "the same character twice"):
        assert banned in mas.NEGATIVE_PRESENTER, banned

    # a FRIEND is another child, not an adult
    friend = mas.other_people_are_other_people(
        "the mascot character playing with her friend")
    assert "a different child" in friend
    assert "grown adult" not in friend


def test_the_bleed_guards_are_universal_not_lesson_only():
    # A twinned mascot is broken in a FACTS reel too — this is "the mascot must survive
    # its own picture", not pedagogy.
    for teaching in (False, True):
        out = mas.clean_scene_for_the_mascot(
            "the mascot character being hugged by her mother", teaching=teaching)
        assert "grown adult" in out
        out = mas.clean_scene_for_the_mascot(
            "the mascot character holding a doll", teaching=teaching)
        assert "cloth rag doll" in out


def test_a_scene_with_no_doll_and_no_people_is_untouched():
    scene = "the mascot character kneeling beside a puppy, stroking its back, laughing"
    assert mas.toys_look_like_toys(scene) == scene
    assert mas.other_people_are_other_people(scene) == scene


# --- One girl, one day, one film -------------------------------------------------
# The shared style says "only the clothes and the props change". Correct for a FACTS
# reel — a costume per fact IS the joke. Wrong for a LESSON: shot 1 came back in a
# pinafore and shot 3 in her white top and denim skirt, and thirteen of those watch like
# thirteen different days.
#
# The costume clause is also how she lost her identity outright: a "toy repairman's
# costume" (dungarees + a cap over her hair) returned a completely different child.

def test_a_lesson_pins_her_clothes_and_facts_keeps_its_costumes():
    assert "only the clothes and the props change" in mas.STYLE_PRESENTER
    assert "EXACTLY THE SAME CLOTHES" in mas.STYLE_TEACHING
    assert "only the clothes and the props change" not in mas.STYLE_TEACHING


def test_the_costume_is_stripped_from_a_lesson_scene_only():
    for scene, gone in (
        ("the mascot character wearing a scientist lab coat facing the camera, "
         "holding up a small plant", "lab coat"),
        ("the mascot character dressed in a chef hat and apron, holding a small plate "
         "of food, smiling", "apron"),
        ("the mascot character dressed in a toy repairman costume facing the camera, "
         "holding up a small broken toy car", "repairman"),
    ):
        lesson = mas.clean_scene_for_the_mascot(scene, teaching=True)
        assert gone not in lesson, f"{gone!r} survived the lesson guard: {lesson}"
        assert lesson.lower().startswith("the mascot character")
        # a FACTS reel keeps its costume — that is the joke
        assert gone in mas.clean_scene_for_the_mascot(scene)


def test_the_costume_guard_keeps_the_action_and_the_framing():
    scene = ("the mascot character wearing a tiny jacket facing the camera, kneeling "
             "beside a puppy, stroking its back")
    out = mas.clean_scene_for_the_mascot(scene, teaching=True)
    assert "jacket" not in out
    assert "facing the camera" in out       # the framing survives
    assert "kneeling beside a puppy" in out  # the action survives
    assert "stroking its back" in out


def test_chained_garments_do_not_leave_a_dangling_and():
    # "dressed in a chef hat and apron" stopped at "hat" and left ", and apron" behind.
    out = mas.her_clothes_do_not_change(
        "the mascot character dressed in a chef hat and apron, holding a plate")
    assert "apron" not in out and "and apron" not in out
    assert "holding a plate" in out


def test_a_scene_naming_no_clothes_is_untouched():
    scene = "the mascot character holding up a small toy car in one hand, smiling"
    assert mas.her_clothes_do_not_change(scene) == scene
    assert mas.clean_scene_for_the_mascot(scene, teaching=True) == scene


# --- Every extra mention of a thing is another copy of that thing -----------------
# alive_looks_alive() marked the animal in place AND appended ", the puppy alive and
# moving" on the end. That named the puppy TWICE, and Qwen drew TWO PUPPIES, one in each
# hand, in a scene that asked for one. other_people_are_other_people() had the same
# shape and would have produced two mothers.
#
# A guard may rewrite a noun where it stands. It may never mention it again.

def _times_named(scene, noun):
    import re
    return len(re.findall(rf"\b{noun}\b", scene, re.I))


def test_a_guard_never_names_a_thing_twice():
    puppy = ("the mascot character holding up a puppy in one hand and a potted plant "
             "beside her, smiling widely")
    out = mas.clean_scene_for_the_mascot(puppy, teaching=True)
    assert _times_named(out, "puppy") == 1, f"the puppy is named twice: {out}"

    both = "the mascot character holding a doll in one hand and a puppy in the other"
    out = mas.clean_scene_for_the_mascot(both, teaching=True)
    assert _times_named(out, "puppy") == 1, f"the puppy is named twice: {out}"
    assert _times_named(out, "doll") == 1, f"the doll is named twice: {out}"
    assert "real live puppy" in out          # still marked ALIVE, just once

    mum = "the mascot character being hugged by her smiling mother, laughing"
    out = mas.clean_scene_for_the_mascot(mum, teaching=True)
    assert _times_named(out, "mother") == 1, f"the mother is named twice: {out}"
    assert "grown adult" in out              # still marked a DIFFERENT person, just once


def test_the_marks_are_made_in_place_not_appended():
    import inspect
    for fn in (mas.alive_looks_alive, mas.other_people_are_other_people):
        src = inspect.getsource(fn)
        assert ".sub(" in src, f"{fn.__name__} must rewrite in place"
    # the appended-note shape is what caused two puppies
    assert 'f"{out.rstrip' not in inspect.getsource(mas.alive_looks_alive)
