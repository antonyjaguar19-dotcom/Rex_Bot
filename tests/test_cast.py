"""A second person needs a second reference image.

For a long time a lesson could not show anyone but the mascot. Handed ONE identity
reference, Qwen-Edit painted that identity onto every human in the frame:

    no note        -> two identical Nakshus hugging. A twin, not a mother.
    parenthesised  -> the mother read as a PROP; she vanished and the child was left
                      holding a doll.
    appositive     -> BOTH girls came back with the mother's hair and face. The mascot
                      was gone from her own lesson.

I concluded the model could only ever draw one person, and worked around it by keeping
the mascot alone. That was WRONG, and Jeffy is the one who said so: "what if we give
another char as the ref for mom?"

TextEncodeQwenImageEditPlus takes image1..image3 and we had only ever passed one. Give it
a drawing of the mother and it draws a mother — measured, first try: Nakshu with her
topknot, butterflies and bindi, hugged by an adult woman in a teal kurta.

The single-reference rule we DID measure was about three views of the SAME character,
where the extras made Qwen copy a reference's stance. A different character is a different
experiment, and I folded them together without running it.
"""
import inspect

from modules import cast


def test_the_role_a_scene_asks_for():
    assert cast.role_in("the mascot character being hugged by her mother") == "mother"
    assert cast.role_in("the mascot character holding mummy's hand") == "mother"
    assert cast.role_in("the mascot character waving at daddy") == "father"
    assert cast.role_in("the mascot character with her grandma") == "grandmother"
    assert cast.role_in("the mascot character playing with her friend") == "friend"
    # most scenes have nobody but her, and those must NOT pay for a second reference —
    # a spare reference is another thing for Qwen to copy into the picture
    assert cast.role_in("the mascot character holding up a toy car") is None


def test_the_adults_are_unmistakably_adult():
    # A mother who reads as a big child is the twin problem wearing a different hat.
    for role in ("mother", "father", "grandmother", "grandfather", "teacher"):
        assert "grown adult" in cast.ROLES[role]
    assert "chibi" in cast._NEGATIVE and "toddler" in cast._NEGATIVE


def test_a_cast_reference_is_plain_and_empty_handed():
    # Identity transfer keys on the face and the clothes. A prop in the reference gets
    # copied into every scene that character ever appears in.
    assert "holding nothing" in cast._STYLE
    assert "plain white background" in cast._STYLE
    assert "props" in cast._NEGATIVE


def test_the_scene_says_which_reference_is_which():
    # Two references and no explanation is an invitation to blend them.
    out = cast.name_the_refs(
        "the mascot character being hugged by her mother, laughing", "mother")
    assert "FIRST reference image" in out
    assert "SECOND reference image" in out
    assert "mother" in out
    # and it is not applied twice
    assert cast.name_the_refs(out, "mother") == out


def test_a_second_person_survives_the_guard_when_we_can_draw_them():
    from modules import mascot as mas
    scene = "the mascot character being hugged by her mother, laughing"
    out = mas.clean_scene_for_the_mascot(scene, teaching=True)
    assert "mother" in out.lower(), \
        "we have a reference for her now — she stays in the picture"


def test_a_person_we_cannot_draw_is_still_removed():
    # The guard is not gone, it is conditional. Someone with no reference would still
    # come back as a twin of the mascot, so they still do not go in the picture.
    from modules import mascot as mas
    scene = "the mascot character standing beside her postman, waving"
    out = mas.clean_scene_for_the_mascot(scene, teaching=True)
    assert out.lower().startswith("the mascot character")


def test_the_lesson_actually_passes_the_second_reference():
    from modules import lesson_pipeline as lp
    src = inspect.getsource(lp.prepare_lesson)
    assert "cast.ref_for(" in src
    assert "reference_images=refs" in src
    assert "cast.name_the_refs(" in src
