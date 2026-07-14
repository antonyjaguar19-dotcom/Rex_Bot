"""The mascot's family.

A lesson keeps naming a second person — "when mummy hugs you", "you play with your
friends". For a long time a picture could not show them: handed ONE identity reference,
Qwen-Edit painted the mascot's identity onto every human in the frame and the mother came
back as the child's TWIN. Three wordings, three failures:

    no note        -> two identical mascots hugging.
    parenthesised  -> the mother read as a PROP; she vanished and the child was left
                      holding a doll.
    appositive     -> BOTH girls came back with the mother's face and hair. The mascot was
                      gone from her own lesson.

I concluded that was a hard limit of Qwen-Edit and kept the mascot alone. Jeffy said "what
if we give another char as the ref for mom?" — and he was right. The backend takes
image1..image3 and we were only ever passing one. Hand it a picture of the mother and it
draws a mother; proven first try, and then in the real lesson.

The family belongs to THE MASCOT. A second mascot inheriting the first one's mother is the
twin problem with extra steps.
"""
import inspect

import pytest
from PIL import Image

from modules import cast
from modules import mascot_library as ml


@pytest.fixture
def mid(tmp_path, monkeypatch):
    """A mascot on a shelf of its own. conftest already points mascot.ASSETS_DIR at a tmp
    dir; this puts a mascot on it to hang a family from."""
    from modules import mascot as mas
    monkeypatch.setattr(mas, "ASSETS_DIR", tmp_path)
    art = tmp_path / "art.png"
    Image.new("RGB", (64, 64)).save(art)
    return ml.create("Nakshu", image=art)


def _png(p):
    Image.new("RGB", (32, 32)).save(p)
    return p


# --- the shelf -----------------------------------------------------------------

def test_the_family_belongs_to_the_mascot(mid, tmp_path):
    # Not a global assets/cast/ — a second mascot would inherit the first one's mother.
    d = cast.family_dir(mid)
    assert d.parent == ml.mascot_dir(mid)
    assert d.name == "family"

    other = ml.create("Someone Else")
    assert cast.family_dir(other) != d

    cast.put_image(mid, "mother", src=_png(tmp_path / "mum.png"), filename="mum.png")
    assert cast.have(mid, "mother")
    assert not cast.have(other, "mother"), "families do not leak between mascots"


def test_every_relation_jeffy_asked_for_has_a_slot(mid):
    slots = {m["relation"] for m in cast.members(mid)}
    for wanted in ("mother", "father", "grandmother", "grandfather", "uncle", "aunt"):
        assert wanted in slots, wanted
    # empty slots are still listed — the UI needs somewhere to drop the upload
    assert all(m["path"] is None for m in cast.members(mid))


def test_the_adults_are_unmistakably_adult():
    # A mother who reads as a big child is the twin problem wearing a different hat.
    for role in ("mother", "father", "grandmother", "grandfather", "uncle", "aunt",
                 "teacher"):
        assert "grown adult" in cast.PRESETS[role]["prompt"], role
    assert "chibi" in cast.NEGATIVE and "toddler" in cast.NEGATIVE


def test_a_reference_is_plain_and_empty_handed():
    # Identity transfer keys on the face and the clothes. A prop in the reference gets
    # copied into every scene that character ever appears in.
    assert "holding nothing" in cast.STYLE
    assert "plain white background" in cast.STYLE
    assert "props" in cast.NEGATIVE


# --- changing it ---------------------------------------------------------------

def test_a_relation_of_your_own(mid):
    rel = cast.add_relation(mid, "Cousin Ravi",
                            "a cheerful boy of about ten, short hair, red t-shirt")
    assert rel == "cousin-ravi"
    got = cast.spec(mid, rel)
    assert got["label"] == "Cousin Ravi"
    assert "red t-shirt" in got["prompt"], "the description IS what the artist is told"
    assert rel in {m["relation"] for m in cast.members(mid)}

    # and a scene finds him, by label or by slug
    assert cast.role_in("the mascot playing with Cousin Ravi", mid) == rel
    assert cast.role_in("the mascot beside her cousin ravi", mid) == rel


def test_put_image_stages_before_it_deletes(mid, tmp_path):
    # mascot_library.put_file learned this the hard way: the "one voice per mascot"
    # cleanup unlinked voice.mp3 and THEN tried to read it as the source. It destroyed
    # Jeffy's only recording. The source can BE the file you are replacing.
    first = cast.put_image(mid, "mother", src=_png(tmp_path / "a.png"), filename="a.png")
    assert first.is_file()

    again = cast.put_image(mid, "mother", src=first, filename="a.png")
    assert again.is_file() and again.stat().st_size, "replacing with itself destroyed it"

    src = inspect.getsource(cast.put_image)
    assert "_incoming_" in src, "stage first, delete after"


def test_one_picture_per_relation(mid, tmp_path):
    cast.put_image(mid, "mother", src=_png(tmp_path / "a.png"), filename="a.png")
    cast.put_image(mid, "mother", src=_png(tmp_path / "b.jpg"), filename="b.jpg")
    d = cast.family_dir(mid)
    assert not (d / "mother.png").exists(), "the old picture is gone"
    assert (d / "mother.jpg").is_file()


def test_a_file_that_is_not_an_image_is_refused(mid, tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("hello")
    with pytest.raises(ValueError):
        cast.put_image(mid, "mother", src=bad, filename="notes.txt")


def test_removing_one_and_removing_the_lot(mid, tmp_path):
    cast.put_image(mid, "mother", src=_png(tmp_path / "m.png"), filename="m.png")
    cast.put_image(mid, "father", src=_png(tmp_path / "f.png"), filename="f.png")
    rel = cast.add_relation(mid, "Cousin Ravi", "a boy of ten")

    cast.remove_member(mid, "mother")
    assert not cast.have(mid, "mother")
    assert cast.have(mid, "father"), "only the one asked for"

    cast.remove_member(mid, rel)
    assert rel not in {m["relation"] for m in cast.members(mid)}, \
        "a custom relation is forgotten, not merely emptied"

    other = ml.create("Someone Else")
    cast.put_image(other, "mother", src=_png(tmp_path / "o.png"), filename="o.png")
    cast.remove_family(mid)
    assert not cast.family_dir(mid).exists()
    assert cast.have(other, "mother"), "and nobody else's family"


# --- reading a scene -----------------------------------------------------------

def test_the_words_a_scene_uses_for_them(mid):
    assert cast.role_in("the mascot being hugged by her mother", mid) == "mother"
    assert cast.role_in("the mascot holding mummy's hand", mid) == "mother"
    assert cast.role_in("the mascot waving at daddy", mid) == "father"
    assert cast.role_in("the mascot with her grandma", mid) == "grandmother"
    assert cast.role_in("the mascot beside her aunty", mid) == "aunt"
    assert cast.role_in("the mascot playing with her friend", mid) == "friend"
    # "grandmother" must not be eaten by "mother" — longest match first
    assert cast.role_in("the mascot hugging her grandmother", mid) == "grandmother"


def test_a_solo_scene_pays_for_no_second_reference(mid):
    # Most scenes. A spare reference is another thing for Qwen to copy into a picture that
    # did not ask for it.
    assert cast.ref_for("the mascot holding up a toy car", mid) == (None, None)


def test_a_relation_named_but_never_pictured_is_not_drawn(mid):
    # The slot exists; there is no picture in it. There is still nothing for Qwen to draw
    # them FROM, so they would still come back as a twin of the mascot.
    assert cast.role_in("the mascot hugged by her mother", mid) == "mother"
    assert cast.ref_for("the mascot hugged by her mother", mid) == (None, None)


def test_a_relation_with_a_picture_is_drawn(mid, tmp_path):
    cast.put_image(mid, "mother", src=_png(tmp_path / "m.png"), filename="m.png")
    relation, img = cast.ref_for("the mascot hugged by her mother", mid)
    assert relation == "mother"
    assert img and img.is_file()


def test_the_scene_says_which_reference_is_which(mid, tmp_path):
    # Two references and no explanation is an invitation to blend them. The proving render
    # said so explicitly and came back with two correct, separate people first try.
    cast.put_image(mid, "mother", src=_png(tmp_path / "m.png"), filename="m.png")
    out = cast.name_the_refs("the mascot hugged by her mother, laughing", mid, "mother")
    assert "FIRST reference image" in out
    assert "SECOND reference image" in out
    assert "Nakshu" in out, "the mascot is named, not called 'the little girl'"
    assert "Mom" in out
    # and it is not applied twice
    assert cast.name_the_refs(out, mid, "mother") == out


# --- the mascot guard, and the pipeline ----------------------------------------

def test_a_person_we_can_draw_survives_the_scene_guard(mid, tmp_path, monkeypatch):
    from modules import mascot as mas
    monkeypatch.setattr(ml, "get_active_id", lambda: mid)
    cast.put_image(mid, "mother", src=_png(tmp_path / "m.png"), filename="m.png")
    out = mas.clean_scene_for_the_mascot(
        "the mascot character being hugged by her mother, laughing", teaching=True)
    assert "mother" in out.lower(), "we have a picture of her — she stays in the shot"


def test_a_person_we_cannot_draw_is_still_removed(mid, monkeypatch):
    # The guard is not gone, it is conditional. With no picture of her, the mother would
    # still come back as a twin of the mascot, so she still does not go in the picture.
    from modules import mascot as mas
    monkeypatch.setattr(ml, "get_active_id", lambda: mid)
    out = mas.clean_scene_for_the_mascot(
        "the mascot character being hugged by her mother, laughing", teaching=True)
    assert "mother" not in out.lower()
    assert out.lower().startswith("the mascot character")


def test_the_lesson_passes_the_second_reference():
    # The draw lives in _draw_one now — shared by the whole batch and by a single-picture
    # redraw, so the code you use to FIX a bad picture cannot drift from the code that
    # made it.
    from modules import lesson_pipeline as lp
    # _scene_and_refs is the single source: the RENDERER uses it, and so does the reveal
    # panel. A preview built from a second copy would drift from what is actually sent.
    src = inspect.getsource(lp._scene_and_refs)
    assert "cast.ref_for(scene_text, mid)" in src
    assert "cast.name_the_refs(scene_text, mid, relation)" in src

    assert "reference_images=refs" in inspect.getsource(lp._draw_one)
    for caller in (lp._draw_one, lp.prompts_for):
        assert "_scene_and_refs(" in inspect.getsource(caller)
    for caller in (lp.prepare_lesson, lp.redraw_still):
        assert "_draw_one(" in inspect.getsource(caller)


# --- Name a family member and the bot finds them ---------------------------------
# Jeffy: "if i mention the family in the prompt will the bot automatically get the family
# characters tagged under the mascot?" — yes, and asking exposed two real bugs.

def test_the_guards_vocabulary_is_the_familys_vocabulary():
    # It was a SECOND hand-maintained list, and it drifted the moment the family gained
    # Uncle and Aunty: the words were in cast.PRESETS but not in mascot._OTHER_PERSON, so
    # an "aunty" with no picture of her sailed straight past the guard and would have been
    # drawn as a TWIN OF THE MASCOT — precisely the bug that guard exists to stop,
    # reintroduced by two lists that had to agree and did not.
    from modules import mascot as mas
    words = mas._other_person_words()
    for who in ("mother", "father", "grandmother", "grandfather", "uncle", "aunt",
                "teacher", "friend"):
        assert who in words, who
    for spoken in ("aunty", "mummy", "daddy", "grandma"):
        assert mas._OTHER_PERSON.search(f"the mascot with her {spoken}"), spoken


def test_naming_someone_you_have_a_picture_of_tags_them(mid, tmp_path, monkeypatch):
    from modules import lesson_pipeline as lp
    monkeypatch.setattr(ml, "get_active_id", lambda: mid)
    cast.put_image(mid, "mother", src=_png(tmp_path / "m.png"), filename="m.png")

    lesson = {"beats": [{"mascot_scene":
                         "the mascot character being hugged by her mother, laughing"}]}
    scene_sent, refs, relation = lp._scene_and_refs(lesson, 0, mid)

    assert relation == "mother"
    assert refs and len(refs) == 2, "the mascot AND the mother"
    assert "SECOND reference image" in scene_sent, "the model is told which is which"
    assert "mother" in scene_sent


def test_naming_someone_you_have_no_picture_of_leaves_them_out_of_the_SHOT(mid, monkeypatch):
    # Qwen has nothing to draw her FROM, so it would paint the mascot's own identity onto
    # her — a twin. She comes out of the picture.
    from modules import lesson_pipeline as lp
    monkeypatch.setattr(ml, "get_active_id", lambda: mid)

    lesson = {"beats": [{"mascot_scene":
                         "the mascot character playing with her aunty in the garden"}]}
    scene_sent, refs, relation = lp._scene_and_refs(lesson, 0, mid)

    assert refs is None and relation is None
    assert "aunty" not in scene_sent.lower(), "she would have come back as a twin"


def test_but_your_WORDS_are_never_deleted(mid, tmp_path, monkeypatch):
    # ...only out of the PICTURE. The strip used to happen at SAVE time, destructively: it
    # deleted the mother from your sentence for good, and WHICH mascot happened to be
    # active at that moment decided whether she was deleted at all. Add her picture
    # afterwards and she never came back — your words had been rewritten on disk.
    #
    # Now: give this mascot an aunty and the very next redraw has her in it, with no
    # editing.
    from modules import lesson_pipeline as lp
    from modules import lesson_writer as lw
    monkeypatch.setattr(ml, "get_active_id", lambda: mid)

    scene = "the mascot character playing with her aunty in the garden"
    kept = mas_clean(scene)
    assert "aunty" in kept, "your sentence is yours"

    # ...and once she has a picture, the very next draw tags her — no edit needed
    cast.put_image(mid, "aunt", src=_png(tmp_path / "a.png"), filename="a.png")
    lesson = {"beats": [{"mascot_scene": kept}]}
    _, refs, relation = lp._scene_and_refs(lesson, 0, mid)
    assert relation == "aunt"
    assert refs and len(refs) == 2


def mas_clean(scene):
    from modules import mascot as mas
    return mas.clean_scene_for_the_mascot(scene, teaching=True, keep_people=True)
