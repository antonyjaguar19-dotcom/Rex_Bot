"""The lessons you have made, and throwing one away.

Also the guard for a bug that cost a lesson its whole upload kit and said nothing:
`publish_kit.attach(..., aspect="16x9")` when the parameter is `aspects` (a tuple). It
raised TypeError, a try/except swallowed it, and the lesson finished with no title, no
description and no thumbnail — a warning in a log nobody reads.
"""
import inspect
import json

import pytest

from modules import lesson_library as ll
from modules import lesson_pipeline as lp
from modules import lesson_writer as lw
from modules import publish_kit


@pytest.fixture(autouse=True)
def lessons(tmp_path, monkeypatch):
    monkeypatch.setattr(lw, "LESSONS_DIR", tmp_path / "lessons")
    monkeypatch.setattr(ll, "LESSONS_DIR", tmp_path / "lessons")
    return tmp_path


def _lesson(lid="20260714_120000", stage="rendered") -> dict:
    d = ll.LESSONS_DIR / lid
    (d / "stills").mkdir(parents=True, exist_ok=True)
    (d / "audio").mkdir(parents=True, exist_ok=True)
    (d / "final").mkdir(parents=True, exist_ok=True)
    for i in range(3):
        (d / "stills" / f"still_{i:02d}.png").write_bytes(b"png" * 100)
        (d / "audio" / f"beat_{i:02d}.wav").write_bytes(b"RIFF" * 100)
    video = d / "final" / f"lesson_{lid}_16x9.mp4"
    video.write_bytes(b"video" * 500)

    lesson = {
        "lesson_id": lid, "_id": lid, "title": "Living Things",
        "topic": "Living and Non-living Things", "book_id": "b1",
        "book_title": "Class 1 EVS", "pages": [4, 10], "stage": stage,
        "estimated_seconds": 74.0, "word_count": 130,
        "beats": [{"kind": "teach", "narration": f"line {i}", "on_screen": "x",
                   "image_prompt": "y", "animate": i == 1, "index": i + 1}
                  for i in range(3)],
        "video": str(video),
    }
    (d / "lesson.json").write_text(json.dumps(lesson), encoding="utf-8")
    return lesson


# --- the list ------------------------------------------------------------------

def test_a_lesson_shows_what_you_need_to_choose_between_them():
    _lesson()
    got = ll.describe("20260714_120000")

    assert got["title"] == "Living Things"
    assert got["topic"] == "Living and Non-living Things"
    assert got["stage"] == "rendered"
    assert got["beats"] == 3 and got["animated"] == 1
    assert got["video"].name.endswith("_16x9.mp4")
    assert len(got["stills"]) == 3
    assert got["when"].day == 14


def test_lessons_are_listed_newest_first_and_filtered_by_book():
    _lesson("20260714_100000")
    _lesson("20260714_120000")
    assert [x["id"] for x in ll.list_lessons()] == ["20260714_120000",
                                                    "20260714_100000"]
    assert len(ll.list_lessons(book_id="b1")) == 2
    assert ll.list_lessons(book_id="other") == []


def test_a_half_written_lesson_still_lists():
    """A lesson that was never rendered is still a lesson you may want to delete."""
    _lesson(stage="written")
    got = ll.describe("20260714_120000")
    assert got["stage"] == "written"


# --- delete --------------------------------------------------------------------

def test_delete_takes_the_whole_lesson(lessons):
    _lesson()
    n, mb = ll.footprint("20260714_120000")
    assert n == 8            # 3 stills + 3 wavs + video + lesson.json

    ll.delete_lesson("20260714_120000")
    assert not list(lessons.rglob("*20260714_120000*")), "something was left behind"
    assert ll.list_lessons() == []


def test_a_bad_id_never_reaches_a_path():
    """This id is joined to a path that gets rmtree'd."""
    for bad in ("..", "*", "", "../../04_Outputs", "20260714"):
        with pytest.raises(ValueError):
            ll.delete_lesson(bad)


def test_deleting_a_lesson_leaves_the_book_alone():
    """You will want to teach that topic again, and re-reading a 64-page scan is ten
    minutes of vision model."""
    _lesson()
    src = inspect.getsource(ll.delete_lesson)
    assert "BOOKS_DIR" not in src and "books" not in src.lower().split("book it was")[0]


# --- the kit that silently never attached --------------------------------------

def test_the_lesson_asks_publish_kit_for_something_it_actually_takes():
    """`aspect=` vs `aspects=`. One character, and the lesson ships with no title, no
    description and no thumbnail — and finishes 'successfully'."""
    params = inspect.signature(publish_kit.attach).parameters
    assert "aspects" in params and "aspect" not in params

    src = inspect.getsource(lp.render_lesson)
    assert "aspects=(ASPECT,)" in src
    assert "aspect=ASPECT)" not in src


def test_a_failed_kit_is_said_out_loud():
    """It used to be a log warning under a bare except. The user watches the progress
    feed, not the log."""
    src = inspect.getsource(lp.render_lesson)
    assert "upload kit failed" in src


# --- A lesson does not live in one tree -----------------------------------------
# The Wan stage is `horror_video`, shared with horror mode, and it writes its raw clips
# to its OWN scratch dirs, named after the lesson but outside the lesson's folder:
#   04_Outputs/videos/hwan_<id>_NNNN.mp4
#   04_Outputs/clips/_mv_temp/hseg_<id>_NNNN.mp4
# rmtree on the lesson folder leaves both. That is exactly how facts mode accumulated
# 212 orphaned files in final/ — a delete that misses a directory hoards gigabytes and
# nobody finds out until the disk does.

def test_delete_takes_the_wan_files_that_live_outside_the_lesson(tmp_path, monkeypatch):
    from modules import lesson_library as llib

    monkeypatch.setattr(llib, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(llib, "LESSONS_DIR", tmp_path / "04_Outputs" / "lessons")

    lid = "20260714_113840"
    d = llib.LESSONS_DIR / lid
    (d / "stills").mkdir(parents=True)
    (d / "stills" / "still_00.png").write_bytes(b"x" * 10)
    (d / "lesson.json").write_text("{}")

    vids = tmp_path / "04_Outputs" / "videos"
    segs = tmp_path / "04_Outputs" / "clips" / "_mv_temp"
    vids.mkdir(parents=True)
    segs.mkdir(parents=True)
    stray_a = vids / f"hwan_{lid}_0000.mp4"
    stray_b = segs / f"hseg_{lid}_0001.mp4"
    stray_a.write_bytes(b"y" * 100)
    stray_b.write_bytes(b"z" * 100)
    # another lesson's files must survive
    other = vids / "hwan_20260101_000000_0000.mp4"
    other.write_bytes(b"keep")

    assert llib._strays(lid) == sorted([stray_a, stray_b])
    n, mb = llib.footprint(lid)
    assert n == 4                       # 2 in the tree + 2 strays

    res = llib.delete_lesson(lid)
    assert res["strays"] == 2
    assert not d.exists()
    assert not stray_a.exists() and not stray_b.exists()
    assert other.exists(), "another lesson's Wan clip was deleted"


def test_a_bad_id_never_reaches_a_glob(tmp_path, monkeypatch):
    # "*" in a glob would take every lesson's Wan clips with it.
    from modules import lesson_library as llib
    for bad in ("*", "../..", "", "20260714", "; rm -rf /"):
        with pytest.raises(ValueError):
            llib.delete_lesson(bad)
