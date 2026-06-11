"""shot_editor.insert_shot — renumbering + artifact-shift logic, no LLM/GPU.

The LLM call and the prompt_assembler calls are monkeypatched so these tests
exercise only the bookkeeping: script renumber, approved-prompts key shift,
storyboard PNG renames + manifest, clip renames (incl. _v revisions), seeds.
"""

import json

import pytest

from modules import shot_editor as se


SID = "20990101_000000"


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Fake 04_Outputs tree with a 3-shot script + all artifact kinds."""
    scripts = tmp_path / "scripts"
    storyboards = tmp_path / "storyboards"
    clips = tmp_path / "clips"
    approved = tmp_path / "approved_prompts"
    monkeypatch.setattr(se, "SCRIPTS_DIR", scripts)
    monkeypatch.setattr(se, "STORYBOARDS_DIR", storyboards)
    monkeypatch.setattr(se, "CLIPS_DIR", clips)
    monkeypatch.setattr(se, "APPROVED_PROMPTS_DIR", approved)

    script = {
        "_id": SID,
        "title": "Test",
        "setting": "a meadow",
        "moral": "be kind",
        "characters": [{"name": "Rex", "type": "animal",
                        "locked_visual_token": "a young brown rabbit"}],
        "shots": [
            {"shot_number": 1, "beat": "hook", "narration": "one",
             "visual_description": "v1"},
            {"shot_number": 2, "beat": "struggle", "narration": "two",
             "visual_description": "v2"},
            {"shot_number": 3, "beat": "consequence", "narration": "three",
             "visual_description": "v3"},
        ],
    }
    _write_json(scripts / f"script_{SID}.json", script)

    _write_json(approved / f"{SID}.json", {
        "script_id": SID,
        "prompts": {
            "1": {"image_prompt": "p1", "approved": True},
            "2": {"image_prompt": "p2", "approved": True},
            "3": {"image_prompt": "p3", "approved": True},
        },
    })

    story_dir = storyboards / SID
    story_dir.mkdir(parents=True)
    for n in (1, 2, 3):
        (story_dir / f"shot{n}_first.png").write_bytes(f"png{n}".encode())
    _write_json(story_dir / "storyboard.json", {
        "script_id": SID,
        "total_frames": 3,
        "frames": [
            {"shot_number": n, "frame_type": "first",
             "image_path": f"04_Outputs/storyboards/{SID}/shot{n}_first.png"}
            for n in (1, 2, 3)
        ],
    })

    clips.mkdir(parents=True)
    (clips / f"clip_{SID}_shot1.mp4").write_bytes(b"c1")
    (clips / f"clip_{SID}_shot2.mp4").write_bytes(b"c2")
    (clips / f"clip_{SID}_shot2_v2.mp4").write_bytes(b"c2v2")
    (clips / f"clip_{SID}_shot3.mp4").write_bytes(b"c3")
    _write_json(clips / f"clip_{SID}_seeds.json", {"1": 11, "2": 22, "3": 33})

    # No network: LLM shot gen + prompt assembly both fail → fallback paths
    monkeypatch.setattr(se, "_generate_shot_llm",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no llm")))
    monkeypatch.setattr(se.pa, "assemble_image_prompt",
                        lambda **k: (_ for _ in ()).throw(RuntimeError("no llm")))
    monkeypatch.setattr(se.pa, "assemble_motion_prompt",
                        lambda **k: (_ for _ in ()).throw(RuntimeError("no llm")))

    return {"scripts": scripts, "storyboards": storyboards,
            "clips": clips, "approved": approved}


def test_insert_middle_renumbers_everything(world):
    result = se.insert_shot(SID, 2, "a closeup of paws on the rope")

    script = json.loads(
        (world["scripts"] / f"script_{SID}.json").read_text(encoding="utf-8"))
    nums = [s["shot_number"] for s in script["shots"]]
    assert nums == [1, 2, 3, 4]
    narrs = {s["shot_number"]: s["narration"] for s in script["shots"]}
    assert narrs[1] == "one"
    assert narrs[3] == "two"      # old shot 2 moved
    assert narrs[4] == "three"    # old shot 3 moved
    assert "paws" in narrs[2]     # fallback shot built from the brief
    assert result["new_shot"]["shot_type"] in ("wide", "medium", "closeup", "insert")
    assert result["shifted_shots"] == 2

    # approved prompts keys shifted, new unapproved entry at "2"
    ap = json.loads(
        (world["approved"] / f"{SID}.json").read_text(encoding="utf-8"))
    assert ap["prompts"]["1"]["image_prompt"] == "p1"
    assert ap["prompts"]["3"]["image_prompt"] == "p2"
    assert ap["prompts"]["4"]["image_prompt"] == "p3"
    assert ap["prompts"]["2"]["approved"] is False
    assert result["prompts_entry_added"] is True

    # storyboard PNGs renamed, slot for new shot empty
    story_dir = world["storyboards"] / SID
    assert (story_dir / "shot1_first.png").read_bytes() == b"png1"
    assert not (story_dir / "shot2_first.png").exists()
    assert (story_dir / "shot3_first.png").read_bytes() == b"png2"
    assert (story_dir / "shot4_first.png").read_bytes() == b"png3"

    # manifest follows
    md = json.loads((story_dir / "storyboard.json").read_text(encoding="utf-8"))
    by_num = {f["shot_number"]: f for f in md["frames"]}
    assert set(by_num) == {1, 3, 4}
    assert by_num[3]["image_path"].endswith("shot3_first.png")

    # clips renamed including _v revisions; seeds keys shifted
    clips = world["clips"]
    assert (clips / f"clip_{SID}_shot1.mp4").exists()
    assert not (clips / f"clip_{SID}_shot2.mp4").exists()
    assert (clips / f"clip_{SID}_shot3.mp4").read_bytes() == b"c2"
    assert (clips / f"clip_{SID}_shot3_v2.mp4").read_bytes() == b"c2v2"
    assert (clips / f"clip_{SID}_shot4.mp4").read_bytes() == b"c3"
    seeds = json.loads(
        (clips / f"clip_{SID}_seeds.json").read_text(encoding="utf-8"))
    assert seeds == {"1": 11, "3": 22, "4": 33}


def test_insert_at_start_and_end(world):
    se.insert_shot(SID, 1, "establishing wide of the meadow")
    script = json.loads(
        (world["scripts"] / f"script_{SID}.json").read_text(encoding="utf-8"))
    assert [s["shot_number"] for s in script["shots"]] == [1, 2, 3, 4]
    assert script["shots"][1]["narration"] == "one"

    # position beyond the end clamps to append (now 4 shots → new shot 5)
    se.insert_shot(SID, 99, "the sun sets")
    script = json.loads(
        (world["scripts"] / f"script_{SID}.json").read_text(encoding="utf-8"))
    assert [s["shot_number"] for s in script["shots"]] == [1, 2, 3, 4, 5]
    assert "sun sets" in script["shots"][4]["narration"]


def test_no_artifacts_is_fine(world):
    """Script-only insert (nothing rendered yet) must not crash."""
    for p in (world["approved"] / f"{SID}.json",
              world["clips"] / f"clip_{SID}_seeds.json"):
        p.unlink()
    for p in world["clips"].glob("*.mp4"):
        p.unlink()
    import shutil
    shutil.rmtree(world["storyboards"] / SID)

    result = se.insert_shot(SID, 2, "")
    assert result["shifted_shots"] == 2
    assert result["renamed_frames"] == 0
    assert result["renamed_clips"] == 0
    assert result["prompts_entry_added"] is False
