"""manual_mode — project store, board ops, duration math. No GPU/ffmpeg/LLM.

Generation (image/video/TTS/music) and assembly encodes are NOT exercised here
(GPU + ffmpeg); those paths are covered by the shared-shape checks below
(manifest fields, target-duration rule, safety gate).
"""

import json

import pytest

from modules import manual_mode as mm


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Redirect the manual store into tmp."""
    manual_dir = tmp_path / "manual"
    monkeypatch.setattr(mm, "MANUAL_DIR", manual_dir)
    monkeypatch.setattr(mm, "CURRENT_MARKER", manual_dir / "_current.txt")
    return manual_dir


def _png(tmp_path, name="src.png"):
    p = tmp_path / name
    p.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    return p


# ---------------------------------------------------------------- project CRUD

def test_create_load_roundtrip(world):
    proj = mm.create_project("My film", aspect_ratio="9:16")
    assert proj["_id"]
    loaded = mm.load_project(proj["_id"])
    assert loaded["name"] == "My film"
    assert loaded["aspect_ratio"] == "9:16"
    assert loaded["shots"] == []
    # create sets the current marker
    assert mm.current_project_id() == proj["_id"]


def test_list_projects_newest_first(world):
    import time
    a = mm.create_project("A")
    time.sleep(1.1)  # ids are second-resolution timestamps
    b = mm.create_project("B")
    ids = [pid for pid, _ in mm.list_projects()]
    assert ids[0] == b["_id"]
    assert a["_id"] in ids


def test_current_falls_back_to_newest(world):
    proj = mm.create_project("X")
    mm.CURRENT_MARKER.unlink()  # marker gone → newest on disk
    assert mm.current_project_id() == proj["_id"]


def test_load_missing_returns_none(world):
    assert mm.load_project("nope") is None


# ---------------------------------------------------------------- board ops

def test_add_shot_copies_image_and_numbers(world, tmp_path):
    proj = mm.create_project("P")
    src = _png(tmp_path)
    shot = mm.add_shot(proj, src, prompt="a red barn", seed=42, duration=4.0)
    assert shot["n"] == 1
    assert shot["seed"] == 42
    img = mm.abs_path(proj, shot["image"])
    assert img.exists() and img.parent.name == "images"
    # manifest persisted
    again = mm.load_project(proj["_id"])
    assert len(again["shots"]) == 1
    assert again["shots"][0]["prompt"] == "a red barn"


def test_add_shot_insert_position(world, tmp_path):
    proj = mm.create_project("P")
    for i in range(3):
        mm.add_shot(proj, _png(tmp_path, f"s{i}.png"), prompt=f"p{i}")
    mm.add_shot(proj, _png(tmp_path, "mid.png"), prompt="MID", position=2)
    prompts = [s["prompt"] for s in proj["shots"]]
    assert prompts == ["p0", "MID", "p1", "p2"]
    assert [s["n"] for s in proj["shots"]] == [1, 2, 3, 4]


def test_remove_and_move(world, tmp_path):
    proj = mm.create_project("P")
    for i in range(3):
        mm.add_shot(proj, _png(tmp_path, f"s{i}.png"), prompt=f"p{i}")
    mm.remove_shot(proj, 2)
    assert [s["prompt"] for s in proj["shots"]] == ["p0", "p2"]
    assert [s["n"] for s in proj["shots"]] == [1, 2]
    mm.move_shot(proj, 2, "up")
    assert [s["prompt"] for s in proj["shots"]] == ["p2", "p0"]
    # moving top up is a no-op, not an error
    mm.move_shot(proj, 1, "up")
    assert proj["shots"][0]["prompt"] == "p2"


def test_get_shot_missing_raises(world, tmp_path):
    proj = mm.create_project("P")
    with pytest.raises(IndexError):
        mm.get_shot(proj, 1)


def test_set_shot_fields_validates(world, tmp_path):
    proj = mm.create_project("P")
    mm.add_shot(proj, _png(tmp_path), prompt="x")
    mm.set_shot_fields(proj, 1, motion_prompt="slow pan", duration="6.5",
                       narration="Hello there.")
    s = mm.load_project(proj["_id"])["shots"][0]
    assert s["motion_prompt"] == "slow pan"
    assert s["duration"] == 6.5
    assert s["narration"] == "Hello there."
    with pytest.raises(KeyError):
        mm.set_shot_fields(proj, 1, clip="clips/hack.mp4")


def test_replace_image_clears_stale_clip(world, tmp_path):
    proj = mm.create_project("P")
    mm.add_shot(proj, _png(tmp_path), prompt="v1")
    proj["shots"][0]["clip"] = "clips/old.mp4"
    mm.save_project(proj)
    mm.replace_shot_image(proj, 1, _png(tmp_path, "v2.png"), prompt="v2")
    s = mm.load_project(proj["_id"])["shots"][0]
    assert s["clip"] is None
    assert s["prompt"] == "v2"


# ---------------------------------------------------------------- durations

def test_shot_target_duration_narration_wins():
    shot = {"duration": 3.0}
    # no narration → base duration
    assert mm.shot_target_duration(shot, None) == 3.0
    # narration shorter than base → base holds
    assert mm.shot_target_duration(shot, 1.0) == 3.0
    # narration longer → narration + pad wins (Rule 5: never crop audio)
    assert mm.shot_target_duration(shot, 5.0) == 5.0 + mm.NARR_PAD_SEC


def test_total_duration_sums_shots(world, tmp_path):
    proj = mm.create_project("P")
    mm.add_shot(proj, _png(tmp_path, "a.png"), duration=2.0)
    mm.add_shot(proj, _png(tmp_path, "b.png"), duration=3.5)
    assert mm.total_duration(proj) == pytest.approx(5.5)


# ---------------------------------------------------------------- safety gate

def test_prompt_safety_adult_profile():
    ok, blocked = mm.check_prompt_safety("a foggy graveyard, skeletal trees")
    assert ok  # horror vocabulary allowed (adult profile)
    ok, blocked = mm.check_prompt_safety("nude figure on a beach")
    assert not ok and blocked


# ---------------------------------------------------------------- assemble guards

def test_assemble_empty_board_raises(world):
    proj = mm.create_project("P")
    with pytest.raises(ValueError):
        mm.assemble(proj)


def test_board_summary_mentions_shots(world, tmp_path):
    proj = mm.create_project("Film")
    mm.add_shot(proj, _png(tmp_path), prompt="castle at dawn", duration=4.0)
    text = mm.board_summary(proj)
    assert "Film" in text and "castle at dawn" in text and "1" in text
