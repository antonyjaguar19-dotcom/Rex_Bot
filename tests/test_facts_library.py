"""The reel library: every finished reel, and a delete that takes ALL of it.

A reel is scattered over four places (story, stills, finals + publish kit,
poster-less master). Deleting one by hand meant remembering all four, which is why
04_Outputs/final had 212 facts files in it.

Delete is irreversible and glob-driven, so the two things pinned hardest here are:
it removes everything the reel owns, and it CANNOT be talked into removing anything
else.
"""

import json

import pytest

from modules import facts_library as fl


@pytest.fixture(autouse=True)
def outputs(tmp_path, monkeypatch):
    """A whole 04_Outputs tree with two finished reels and one unfinished."""
    facts = tmp_path / "facts"
    stills = tmp_path / "storyboards"
    final = tmp_path / "final"
    posterless = final / "_posterless"
    for d in (facts, stills, final, posterless):
        d.mkdir(parents=True)

    monkeypatch.setattr(fl, "FACTS_DIR", facts)
    monkeypatch.setattr(fl, "STILLS_DIR", stills)
    monkeypatch.setattr(fl, "FINAL_DIR", final)
    monkeypatch.setattr(fl, "POSTERLESS_DIR", posterless)

    def reel(fid, topic, title, finished=True):
        (facts / f"facts_{fid}.json").write_text(json.dumps({
            "facts_id": fid, "title": title, "topic": topic,
            "beats": [{"kind": "hook", "narration": "Watch this."},
                      {"kind": "fact", "narration": f"A fact about {topic}."},
                      {"kind": "fact", "narration": f"Another {topic} fact."},
                      {"kind": "outro", "narration": "Follow."}],
        }), encoding="utf-8")
        d = stills / f"facts_{fid}"
        d.mkdir()
        (d / "still_00.png").write_bytes(b"png")
        if not finished:
            return
        for aspect in ("9x16", "16x9"):
            (final / f"facts_{fid}_{aspect}.mp4").write_bytes(b"video" * 100)
        stem = f"facts_{fid}_9x16"
        (final / f"{stem}_title.txt").write_text(title, encoding="utf-8")
        (final / f"{stem}_description.txt").write_text("desc #facts", encoding="utf-8")
        (final / f"{stem}_thumb_9x16.jpg").write_bytes(b"jpg")
        (final / f"{stem}_discord.mp4").write_bytes(b"preview")
        (posterless / f"{stem}.mp4").write_bytes(b"master")

    reel("20260713_152428", "octopuses", "Octopus Facts")
    reel("20260712_101500", "bees", "Bee Facts")
    reel("20260711_090000", "volcanoes", "Never Rendered", finished=False)
    return tmp_path


# --- listing ----------------------------------------------------------------

def test_only_finished_reels_are_listed():
    """The story JSON exists from the moment the LLM writes it. A list built from
    those would be half full of runs that crashed at the render."""
    reels = fl.list_reels()
    assert [r["id"] for r in reels] == ["20260713_152428", "20260712_101500"]


def test_a_reel_carries_what_the_outliner_shows():
    r = fl.describe("20260713_152428")
    assert r["title"] == "Octopus Facts"
    assert r["topic"] == "octopuses"
    assert r["n_facts"] == 2                        # the hook and outro are not facts
    assert r["facts"] == ["A fact about octopuses.", "Another octopuses fact."]
    assert r["description"] == "desc #facts"
    assert r["thumb"] is not None
    assert r["when"].year == 2026 and r["when"].day == 13
    assert r["video"].name.endswith("_9x16.mp4")    # portrait is the reel


def test_the_discord_preview_is_not_a_deliverable():
    """`_discord.mp4` is a re-encode for the phone, not an aspect to download."""
    r = fl.describe("20260713_152428")
    assert not any("_discord" in v.name for v in r["videos"])
    assert len(r["videos"]) == 2                    # 9x16 + 16x9


def test_describe_is_none_for_an_unfinished_or_unknown_reel():
    assert fl.describe("20260711_090000") is None   # written, never rendered
    assert fl.describe("nonsense") is None


# --- deleting ---------------------------------------------------------------

def test_delete_takes_every_last_file_of_the_reel(outputs):
    n_files, mb = fl.footprint("20260713_152428")
    assert n_files == 9        # story + still + 2 videos + title + desc + thumb
                               # + discord preview + posterless master
    got = fl.delete_reel("20260713_152428")
    assert len(got["removed"]) and not got["failed"]

    assert not list(outputs.rglob("*20260713_152428*")), "something was left behind"
    assert fl.describe("20260713_152428") is None


def test_deleting_one_reel_leaves_the_others_alone(outputs):
    fl.delete_reel("20260713_152428")
    assert [r["id"] for r in fl.list_reels()] == ["20260712_101500"]
    assert (outputs / "facts" / "facts_20260712_101500.json").exists()
    assert (outputs / "storyboards" / "facts_20260712_101500").is_dir()


def test_a_bad_id_cannot_glob_the_whole_library(outputs):
    """The id is interpolated into a glob that DELETES. '*' would take the lot."""
    for bad in ("*", "", "..", "facts_*", "20260713", "../../final"):
        with pytest.raises(ValueError):
            fl.delete_reel(bad)
    assert len(fl.list_reels()) == 2, "nothing may have been deleted"


def test_the_facts_are_remembered_after_a_delete_unless_you_say_otherwise(monkeypatch):
    """You delete a bad reel because you want the next one to be DIFFERENT — so
    what it said stays on the do-not-repeat list."""
    from modules import facts_memory as fm
    forgot = []
    monkeypatch.setattr(fm, "forget_reel", lambda fid: forgot.append(fid) or 3)

    fl.delete_reel("20260712_101500")
    assert forgot == []

    fl.delete_reel("20260713_152428", forget_facts=True)
    assert forgot == ["20260713_152428"]
