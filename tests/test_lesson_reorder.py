"""Reordering a shot must take its picture, its voice and its clip with it.

EVERY artifact is named by the beat's POSITION — still_04.png, beat_04.wav, clip_04.mp4 —
and the narration track is rebuilt by GLOBBING those filenames, not by walking the beats:

    wavs = sorted((d / "audio").glob("beat_*.wav"))      # lesson_pipeline

So moving a beat in the list and stopping there would leave the VOICE in the old order
while the pictures, the captions and the durations followed the new one. The video would
render cleanly, log nothing, and be wrong in the file — which is the shape of every bug
this pipeline has ever shipped.
"""
import json

import pytest

from modules import lesson_writer as lw


@pytest.fixture
def lesson(tmp_path, monkeypatch):
    monkeypatch.setattr(lw, "LESSONS_DIR", tmp_path)
    lid = "20260714_120000"
    d = tmp_path / lid
    for sub in ("stills", "audio", "clips"):
        (d / sub).mkdir(parents=True)

    beats = []
    for i in range(4):
        beats.append({
            "kind": "teach", "narration": f"line {i}", "on_screen": f"L{i}",
            "image_prompt": "", "animate": i == 1, "approved": True, "index": i + 1,
            "still": str(d / "stills" / f"still_{i:02d}.png"),
            "duration": 3.0 + i,
        })
        # the artifacts carry their own beat's number INSIDE them, so a rename that put
        # the wrong file in a slot is visible
        (d / "stills" / f"still_{i:02d}.png").write_text(f"picture of line {i}")
        (d / "audio" / f"beat_{i:02d}.wav").write_text(f"voice of line {i}")
        (d / "clips" / f"clip_{i:02d}.mp4").write_text(f"clip of line {i}")

    lw._save({"lesson_id": lid, "title": "T", "topic": "T", "beats": beats,
              "stage": "stills", "word_count": 8, "estimated_seconds": 12.0})
    (d / "audio" / "spoken.json").write_text(
        json.dumps([b["narration"] for b in beats]))
    return lid, d


def _on_disk(d, sub, template, n):
    """What each numbered slot actually contains, in order."""
    return [(d / sub / template.format(i)).read_text() for i in range(n)]


def test_the_picture_the_voice_and_the_clip_all_follow_the_beat(lesson):
    lid, d = lesson
    assert lw.move_beat(lid, 3, 0) is True          # last shot to the front

    got = lw.load_lesson(lid)
    order = [b["narration"] for b in got["beats"]]
    assert order == ["line 3", "line 0", "line 1", "line 2"]

    # and every artifact moved WITH it — slot i holds line order[i]'s file
    assert _on_disk(d, "stills", "still_{:02d}.png", 4) == [
        "picture of line 3", "picture of line 0", "picture of line 1", "picture of line 2"]
    assert _on_disk(d, "audio", "beat_{:02d}.wav", 4) == [
        "voice of line 3", "voice of line 0", "voice of line 1", "voice of line 2"]
    assert _on_disk(d, "clips", "clip_{:02d}.mp4", 4) == [
        "clip of line 3", "clip of line 0", "clip of line 1", "clip of line 2"]


def test_the_narration_track_is_globbed_so_the_wavs_MUST_be_renamed(lesson):
    # lesson_pipeline builds the spoken track with sorted(glob("beat_*.wav")). If the wavs
    # do not move, the voice keeps the OLD order while the pictures follow the new one —
    # and nothing anywhere would say so.
    lid, d = lesson
    lw.move_beat(lid, 0, 3)

    spoken_order = [p.read_text() for p in sorted((d / "audio").glob("beat_*.wav"))]
    beat_order = [f"voice of {b['narration']}" for b in lw.load_lesson(lid)["beats"]]
    assert spoken_order == beat_order, "the voice and the pictures have desynced"


def test_a_permutation_not_just_a_swap(lesson):
    # shot_editor renames highest-first, which is collision-free for an INSERT (+1 shift).
    # A reorder is a permutation, and for a permutation "highest first" collides: moving
    # 2->0 needs 0->1 and 1->2 at the same time. Hence the two-phase rename.
    lid, d = lesson
    lw.move_beat(lid, 2, 0)
    lw.move_beat(lid, 3, 1)

    got = [b["narration"] for b in lw.load_lesson(lid)["beats"]]
    pics = _on_disk(d, "stills", "still_{:02d}.png", 4)
    assert pics == [f"picture of {n}" for n in got], "a picture is on the wrong line"


def test_the_beats_stored_still_path_is_rewritten(lesson):
    # The beat carries the ABSOLUTE path of its picture, and the file it names has just
    # been renamed out from under it.
    lid, d = lesson
    lw.move_beat(lid, 3, 0)
    for i, b in enumerate(lw.load_lesson(lid)["beats"]):
        assert b["still"].endswith(f"still_{i:02d}.png")
        assert (d / "stills" / f"still_{i:02d}.png").is_file()


def test_the_voice_takes_are_still_reused(lesson):
    # spoken.json remembers WHICH WORDS the takes on disk are of, in order. Leave it stale
    # and prepare_lesson finds a mismatch and re-records the whole lesson — the clone is
    # stochastic, and a re-read can collapse a line and drop the lesson to a preset voice
    # as the price of moving a shot.
    from modules import lesson_pipeline as lp
    lid, d = lesson
    lw.move_beat(lid, 3, 0)

    got = lw.load_lesson(lid)
    said = json.loads((d / "audio" / "spoken.json").read_text())
    assert said == [b["narration"] for b in got["beats"]]

    narrations = [b["narration"] for b in got["beats"]]
    assert len(lp._reusable_takes(narrations, d / "audio")) == 4, \
        "the takes were thrown away and the lesson would be re-voiced"


def test_the_animate_tick_and_the_approval_travel_with_the_beat(lesson):
    lid, _ = lesson
    lw.move_beat(lid, 1, 3)                 # the ticked beat goes last
    got = lw.load_lesson(lid)["beats"]
    assert got[3]["narration"] == "line 1" and got[3]["animate"] is True
    assert [b["animate"] for b in got] == [False, False, False, True]


def test_a_move_that_changes_nothing_is_refused(lesson):
    lid, _ = lesson
    assert lw.move_beat(lid, 1, 1) is False
    assert lw.move_beat(lid, -1, 0) is False
    assert lw.move_beat(lid, 0, 9) is False


# --- RESET -----------------------------------------------------------------------

def test_reset_puts_every_shot_and_every_file_back(lesson):
    lid, d = lesson
    lw.move_beat(lid, 3, 0)
    lw.move_beat(lid, 2, 1)
    assert [b["narration"] for b in lw.load_lesson(lid)["beats"]] != \
        ["line 0", "line 1", "line 2", "line 3"]

    assert lw.reset_order(lid) is True
    got = lw.load_lesson(lid)["beats"]
    assert [b["narration"] for b in got] == ["line 0", "line 1", "line 2", "line 3"]

    # and the files came back with them — the whole point
    assert _on_disk(d, "stills", "still_{:02d}.png", 4) == [
        f"picture of line {i}" for i in range(4)]
    assert _on_disk(d, "audio", "beat_{:02d}.wav", 4) == [
        f"voice of line {i}" for i in range(4)]
    assert _on_disk(d, "clips", "clip_{:02d}.mp4", 4) == [
        f"clip of line {i}" for i in range(4)]


def test_the_original_order_is_stamped_before_the_first_move_not_after(lesson):
    # Stamping it lazily is the only honest moment: a lesson written before this existed
    # has no record of its original order, and inventing one from the CURRENT order — after
    # it has already been shuffled — would make Reset a no-op that LOOKED like it worked.
    lid, _ = lesson
    assert all("orig" not in b for b in lw.load_lesson(lid)["beats"])

    lw.move_beat(lid, 3, 0)
    got = lw.load_lesson(lid)["beats"]
    assert [b["orig"] for b in got] == [3, 0, 1, 2], "it remembers where each shot STARTED"


def test_reset_is_not_offered_when_nothing_moved(lesson):
    # A button that does nothing is worse than no button — this codebase has already
    # shipped one ("Redo the pictures" was a twenty-minute no-op).
    lid, _ = lesson
    assert lw.is_reordered(lw.load_lesson(lid)) is False
    assert lw.reset_order(lid) is False

    lw.move_beat(lid, 0, 2)
    assert lw.is_reordered(lw.load_lesson(lid)) is True

    lw.reset_order(lid)
    assert lw.is_reordered(lw.load_lesson(lid)) is False, "back home — nothing left to reset"


def test_reset_keeps_the_voice_takes(lesson):
    from modules import lesson_pipeline as lp
    lid, d = lesson
    lw.move_beat(lid, 3, 0)
    lw.reset_order(lid)

    narrations = [b["narration"] for b in lw.load_lesson(lid)["beats"]]
    assert len(lp._reusable_takes(narrations, d / "audio")) == 4, \
        "a reset must not cost you a re-record"


def test_move_and_reset_share_one_permutation(lesson):
    # Two copies of the rename would be two chances to desync the voice from the pictures.
    import inspect
    assert "_apply_order(" in inspect.getsource(lw.move_beat)
    assert "_apply_order(" in inspect.getsource(lw.reset_order)
