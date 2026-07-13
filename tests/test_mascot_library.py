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
    assert not ok and "no mascots on the shelf" in why


def test_an_active_mascot_with_no_art_says_so_by_name():
    """The confusing case: the shelf is not empty and the dropdown names a
    character, so a bare "no mascot image" reads like a lie."""
    mid = ml.create("Nakshu")             # named, never given a picture
    ml.set_active_id(mid)
    ok, why = mas.is_available()
    assert not ok
    assert "Nakshu" in why and "no image" in why


def test_rename_changes_the_label_and_moves_nothing():
    """The id is the mascot's ADDRESS — active_mascot points at it and the art
    lives under it. Renaming the folder would orphan the active mascot and move
    files a render may be reading. Only the label changes."""
    mid = ml.create("Robot Owl", image_bytes=PNG)
    ml.set_active_id(mid)
    art = ml.primary_image()

    assert ml.rename(mid, "  Captain Owl  ") == "Captain Owl"   # trimmed

    got = ml.describe(mid)
    assert got["name"] == "Captain Owl"
    assert got["id"] == mid == "robot-owl"          # the id does NOT follow the name
    assert ml.get_active_id() == mid                # still active
    assert ml.primary_image() == art                # art did not move
    assert mas.active_mascot_name() == "Captain Owl"


def test_rename_refuses_an_empty_name_and_an_unknown_mascot():
    mid = ml.create("Robot Owl", image_bytes=PNG)
    with pytest.raises(ValueError):
        ml.rename(mid, "   ")
    with pytest.raises(ValueError):
        ml.rename("nobody", "Ghost")
    assert ml.describe(mid)["name"] == "Robot Owl"


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


def test_there_is_no_back_view():
    """It carries no face and no chest logo — the two things identity transfer
    keys on — so it was never actually used as a reference. Asking for a fourth
    image nobody looks at is just work."""
    assert "back" not in ml.INTAKE_VIEWS
    assert "back" not in ml.ROLE_FILES
    assert not any("back" in n for n in ml.ANGLE_NAMES)

    mid = ml.create("Robot Owl", image_bytes=PNG)
    with pytest.raises(ValueError):
        ml.put_file(mid, "back", data=PNG, filename="back.png")


def test_file_for_reports_what_is_actually_in_each_slot():
    """The card shows the slot the dropdown names. It used to show the primary
    image whatever you picked, so choosing 'voice clip' told you nothing about
    whether a clip had ever landed."""
    mid = ml.create("Robot Owl", image_bytes=PNG)
    ml.put_file(mid, "side", data=PNG, filename="s.png")
    ml.put_file(mid, "voice", data=WAV, filename="v.wav")

    assert ml.file_for(mid, "main").name == "mascot.png"
    assert ml.file_for(mid, "side").name == "mascot_side.png"
    assert ml.file_for(mid, "voice").name == "voice.wav"
    assert ml.file_for(mid, "threequarter") is None      # empty slot, not the main image
    assert ml.file_for("nobody", "main") is None


def test_replacing_a_slot_from_its_own_file_does_not_destroy_it(monkeypatch):
    """The cleanup that keeps ONE voice per mascot deleted the mascot's current
    clip — and the current clip can be the SOURCE (re-importing voice.mp3 to
    convert it). It was deleted before it was read, and the copy that was meant to
    replace it then failed on a missing file. The clip was gone.
    """
    # No ffmpeg in the test: the fallback keeps the bytes, which is what we check.
    monkeypatch.setattr(ml, "FFMPEG_EXE", Path("no_such_ffmpeg.exe"))

    mid = ml.create("Robot Owl", image_bytes=PNG)
    mp3 = ml.mascots_dir() / mid / "voice.mp3"
    mp3.write_bytes(WAV)

    got = ml.put_file(mid, "voice", src=mp3, filename="voice.mp3")

    assert got.name == "voice.wav"
    assert got.read_bytes() == WAV          # the audio survived the round trip
    assert not mp3.exists()                 # and only ONE voice is left
    assert ml.file_for(mid, "voice") == got


def test_a_voice_clip_is_normalised_to_the_format_the_clone_wants(tmp_path, monkeypatch):
    """Mono 16k wav. It is the one thing the voice is conditioned on, and the only
    clips that ever cloned cleanly here were mono 16k wavs — a 48 kHz mp3 got in
    through the dashboard (Discord had always converted) and the reel came back
    with a 0.12s hook."""
    calls = []
    monkeypatch.setattr(ml, "_to_wav_mono16k",
                        lambda s, dst: calls.append((Path(s).suffix, dst.name))
                        or dst.write_bytes(WAV) or dst)

    mid = ml.create("Robot Owl", image_bytes=PNG)
    src = tmp_path / "recording.mp3"
    src.write_bytes(WAV)
    got = ml.put_file(mid, "voice", src=src, filename="recording.mp3")

    assert calls == [(".mp3", "voice.wav")]
    assert got.name == "voice.wav"


def test_bad_file_types_are_refused():
    mid = ml.create("Robot Owl", image_bytes=PNG)
    with pytest.raises(ValueError):
        ml.put_file(mid, "main", data=PNG, filename="mascot.txt")
    with pytest.raises(ValueError):
        ml.put_file(mid, "voice", data=WAV, filename="voice.png")
    with pytest.raises(ValueError):
        ml.put_file(mid, "hat", data=PNG, filename="hat.png")


# --- intake (the creation form) ---------------------------------------------

def test_intake_installs_every_view_and_the_voice(tmp_path):
    views = {}
    for v in ml.INTAKE_VIEWS:
        p = tmp_path / f"{v}.png"
        p.write_bytes(PNG)
        views[v] = p
    clip = tmp_path / "clip.wav"
    clip.write_bytes(WAV)

    mid = ml.create_from_intake("Robot Owl", views, clip)
    ml.set_active_id(mid)
    got = ml.describe(mid)

    assert len(got["angles"]) == 3      # front, three-quarter, side — no BACK view
    assert got["voice"].name == "voice.wav"
    # The FRONT view is also the primary: a mascot whose primary is a side view
    # renders a character seen from the side in every thumbnail.
    assert got["image"].name == "mascot.png"
    assert got["image"].read_bytes() == views["front"].read_bytes()


def test_intake_without_the_front_view_is_refused():
    with pytest.raises(ValueError, match="front"):
        ml.create_from_intake("Robot Owl", {}, None)
    assert ml.list_mascots() == []


def test_a_failed_intake_leaves_nothing_behind(tmp_path):
    """Half a mascot is worse than none: it lists, it activates, and it renders
    an empty reference."""
    front = tmp_path / "front.png"
    front.write_bytes(PNG)
    bad = tmp_path / "voice.txt"          # not an audio file
    bad.write_bytes(WAV)

    with pytest.raises(ValueError):
        ml.create_from_intake("Robot Owl", {"front": front}, bad)
    assert ml.list_mascots() == []


def test_intake_takes_the_front_view_alone(tmp_path):
    front = tmp_path / "front.png"
    front.write_bytes(PNG)
    mid = ml.create_from_intake("Robot Owl", {"front": front}, None)
    got = ml.describe(mid)
    assert got["ready"] and got["voice"] is None


def test_voice_warning_is_advice_not_a_veto(monkeypatch, tmp_path):
    """A short clip still clones — thinly. Warn, never refuse: the owner is the
    one listening to the result."""
    clip = tmp_path / "clip.wav"
    clip.write_bytes(WAV)

    monkeypatch.setattr(ml, "audio_duration", lambda p: 2.0)
    assert "under" in ml.voice_warning(clip)

    monkeypatch.setattr(ml, "audio_duration", lambda p: 90.0)
    assert "past" in ml.voice_warning(clip)

    monkeypatch.setattr(ml, "audio_duration", lambda p: 10.0)
    assert ml.voice_warning(clip) is None

    # ffprobe missing / unreadable file: unknown is not "bad".
    monkeypatch.setattr(ml, "audio_duration", lambda p: None)
    assert ml.voice_warning(clip) is None

    # And a warned-about clip is still installed.
    monkeypatch.setattr(ml, "audio_duration", lambda p: 2.0)
    front = tmp_path / "front.png"
    front.write_bytes(PNG)
    mid = ml.create_from_intake("Robot Owl", {"front": front}, clip)
    assert ml.describe(mid)["voice"] is not None


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
