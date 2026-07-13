"""The mascot shelf: more than one character, one of them active.

The pipeline used to have exactly one mascot — `assets/mascot.png`. These tests
pin the two things that make several of them safe: the ACTIVE one is what every
render reaches for, and the old flat layout still works (and is migrated, not
broken) for anyone who never adds a second.
"""
from pathlib import Path

import pytest

import modules.mascot as mas
import modules.mascot_library as ml
import modules.runtime_settings as rs

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64
WAV = b"RIFF" + b"0" * 64


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """A tmp assets dir + a tmp settings file. mascot_library reads ASSETS_DIR off
    mascot.py at call time, so patching that one attribute moves the whole shelf."""
    assets = tmp_path / "assets"
    assets.mkdir()
    monkeypatch.setattr(mas, "ASSETS_DIR", assets)
    monkeypatch.setattr(rs, "SETTINGS_PATH", tmp_path / "runtime_settings.json")
    # The shared voice clip is a module constant, not an ASSETS_DIR lookup — left
    # alone, these tests would read the owner's real recording off disk.
    monkeypatch.setattr(rs, "DEFAULT_MASCOT_VOICE_REF", assets / "mascot_voice.wav")
    return assets


def _flat(assets: Path, *names: str) -> None:
    for n in names:
        (assets / n).write_bytes(PNG)


# --- the shelf --------------------------------------------------------------

def test_empty_shelf_is_not_a_library():
    assert ml.list_mascots() == []
    assert ml.get_active_id() is None
    assert ml.primary_image() is None
    assert not ml.library_exists()


def test_create_lists_and_activates():
    mid = ml.create("Robot Owl", image_bytes=PNG)
    assert mid == "robot-owl"
    ml.set_active_id(mid)

    shelf = ml.list_mascots()
    assert [m["id"] for m in shelf] == ["robot-owl"]
    assert shelf[0]["name"] == "Robot Owl"
    assert shelf[0]["ready"] is True
    assert ml.get_active_id() == "robot-owl"
    assert ml.primary_image().name == "mascot.png"


def test_ids_never_collide():
    a = ml.create("Robot Owl", image_bytes=PNG)
    b = ml.create("Robot Owl", image_bytes=PNG)
    assert a == "robot-owl" and b == "robot-owl-2"


def test_active_mascot_is_the_one_that_renders():
    """The whole point of the shelf: switching the dropdown must switch the art
    every downstream stage conditions on — not just a label."""
    owl = ml.create("Robot Owl", image_bytes=PNG)
    cub = ml.create("Jaguar Cub", image_bytes=PNG)

    ml.set_active_id(owl)
    assert mas.mascot_path().parent.name == owl
    assert mas.active_mascot_name() == "Robot Owl"

    ml.set_active_id(cub)
    assert mas.mascot_path().parent.name == cub
    assert [p.parent.name for p in mas.mascot_refs()] == [cub]


def test_a_stale_active_id_falls_back_instead_of_disabling_the_mascot():
    """A deleted folder left behind an id nobody could see. Silently rendering
    NO mascot is the failure mode this project keeps paying for."""
    mid = ml.create("Robot Owl", image_bytes=PNG)
    rs.set_active_mascot("a-mascot-that-was-deleted")
    assert ml.get_active_id() == mid
    assert mas.mascot_path() is not None


def test_remove_hands_the_crown_to_whats_left():
    owl = ml.create("Robot Owl", image_bytes=PNG)
    cub = ml.create("Jaguar Cub", image_bytes=PNG)
    ml.set_active_id(owl)

    ml.remove(owl)
    assert [m["id"] for m in ml.list_mascots()] == [cub]
    assert ml.get_active_id() == cub

    ml.remove(cub)
    assert ml.list_mascots() == []
    assert mas.mascot_path() is None      # empty shelf = no mascot, full stop
    ok, why = mas.is_available()
    assert not ok and "no mascot" in why


def test_remove_unknown_raises():
    with pytest.raises(ValueError):
        ml.remove("nobody")


# --- files ------------------------------------------------------------------

def test_angles_become_the_references():
    mid = ml.create("Robot Owl", image_bytes=PNG)
    ml.set_active_id(mid)
    for role in ("front", "threequarter", "side"):
        ml.put_file(mid, role, data=PNG, filename=f"{role}.png")

    # Same rule as the flat layout: the angle set wins, front first, capped at 3.
    assert [p.name for p in mas.mascot_refs()] == ["mascot_front.png"]
    assert len(mas.mascot_refs(3)) == 3


def test_a_new_main_replaces_the_old_one_rather_than_stacking():
    """Two primaries in one folder (mascot.png + mascot.jpg) means the priority
    order silently decides which character renders."""
    mid = ml.create("Robot Owl", image_bytes=PNG, filename="a.png")
    ml.put_file(mid, "main", data=PNG, filename="b.jpg")
    d = ml.mascots_dir() / mid
    assert (d / "mascot.jpg").exists()
    assert not (d / "mascot.png").exists()


def test_bad_file_types_are_refused():
    mid = ml.create("Robot Owl", image_bytes=PNG)
    with pytest.raises(ValueError):
        ml.put_file(mid, "main", data=PNG, filename="mascot.txt")
    with pytest.raises(ValueError):
        ml.put_file(mid, "voice", data=WAV, filename="voice.png")
    with pytest.raises(ValueError):
        ml.put_file(mid, "hat", data=PNG, filename="hat.png")


# --- voice ------------------------------------------------------------------

def test_each_mascot_speaks_in_its_own_voice():
    """A second character must not inherit the first one's cloned voice."""
    owl = ml.create("Robot Owl", image_bytes=PNG)
    cub = ml.create("Jaguar Cub", image_bytes=PNG)
    ml.put_file(cub, "voice", data=WAV, filename="cub.wav")

    ml.set_active_id(cub)
    assert ml.voice_ref().parent.name == cub

    ml.set_active_id(owl)
    assert ml.voice_ref() is None       # no clip of its own, and no global one


def test_a_mascot_without_a_voice_falls_back_to_the_shared_clip(isolated, tmp_path):
    shared = tmp_path / "shared.wav"
    shared.write_bytes(WAV)
    rs.set_mascot_voice_ref(shared)

    mid = ml.create("Robot Owl", image_bytes=PNG)
    ml.set_active_id(mid)
    assert ml.voice_ref() == shared.resolve()


# --- migration --------------------------------------------------------------

def test_the_old_flat_mascot_still_works_untouched(isolated):
    _flat(isolated, "mascot.png", "mascot_front.png")
    assert not ml.library_exists()
    assert mas.mascot_path() == isolated / "mascot.png"   # no shelf, no change


def test_migrate_copies_the_flat_mascot_onto_the_shelf(isolated):
    _flat(isolated, "mascot.png", "mascot_front.png", "mascot_side.png")
    (isolated / "mascot_voice.wav").write_bytes(WAV)

    assert ml.migrate() == "default"
    assert ml.get_active_id() == "default"
    assert mas.mascot_path().parent.name == "default"
    assert ml.voice_ref().name == "voice.wav"
    assert (isolated / "mascot.png").exists()            # the original is kept
    assert ml.migrate() is None                          # and it only runs once


def test_migrate_does_nothing_when_there_is_no_flat_mascot():
    assert ml.migrate() is None
    assert not ml.library_exists()


def test_deleting_every_mascot_does_not_resurrect_the_flat_one(isolated):
    """Once the shelf exists it is the only truth. Falling back to the flat file
    after a delete would bring back the character you just removed."""
    _flat(isolated, "mascot.png")
    ml.migrate()
    ml.remove("default")
    assert mas.mascot_path() is None
