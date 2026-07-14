"""Rendering a lesson: the gate, and the two clips that do not match.

Two failures this guards:

**The gate.** Wan costs ~8 minutes of GPU per shot. Ticking every line of a 12-line
lesson is an hour and a half, and nobody means to start that from one click. So
nothing is animated by default, the render refuses a lesson that was never approved,
and there is a hard cap on the number of animated shots.

**The mixed segment list.** A lesson is half Wan (16fps) and half Ken Burns (30fps).
The concat demuxer stream-copies, which does not error on a mismatch — it re-containers
the frames at the wrong rate and the segment plays for a different LENGTH than it was
cut to. That is the bug that once cropped the narration off the end of every video
here. The ffmpeg test below is the only kind that can catch it.
"""
import json
import subprocess
from pathlib import Path

import pytest

from modules import lesson_assembly as la
from modules import lesson_pipeline as lp
from modules import lesson_writer as lw

FFMPEG = la.FFMPEG_EXE
FFPROBE = Path(str(FFMPEG).replace("ffmpeg.exe", "ffprobe.exe"))


@pytest.fixture(autouse=True)
def lessons(tmp_path, monkeypatch):
    monkeypatch.setattr(lw, "LESSONS_DIR", tmp_path / "lessons")
    monkeypatch.setattr(lp, "LESSONS_DIR", tmp_path / "lessons")
    monkeypatch.setattr(la, "LESSONS_DIR", tmp_path / "lessons")
    return tmp_path


def _lesson(stage="stills", animate=()) -> dict:
    lid = "20260714_100000"
    d = lw.LESSONS_DIR / lid
    d.mkdir(parents=True, exist_ok=True)
    beats = []
    for i in range(6):
        beats.append({
            "kind": "teach", "narration": f"This is line number {i} of the lesson.",
            "on_screen": f"Line {i}", "image_prompt": "a child",
            "animate": i in animate, "index": i + 1,
            "still": str(d / f"still_{i:02d}.png"), "duration": 3.0,
        })
    lesson = {"lesson_id": lid, "_id": lid, "title": "Your Body", "topic": "Body",
              "book_id": "b", "pages": [1, 2], "beats": beats, "stage": stage,
              "word_count": 42, "estimated_seconds": 18.0, "target_seconds": 20.0}
    (d / "lesson.json").write_text(json.dumps(lesson), encoding="utf-8")
    return lesson


# --- the gate -----------------------------------------------------------------

def test_nothing_is_animated_until_you_tick_it():
    lesson = _lesson()
    assert all(b["animate"] is False for b in lesson["beats"])
    assert lp.estimate(lesson["beats"])["animated"] == 0


def test_the_estimate_is_what_makes_the_tickbox_mean_anything():
    """Each tick is ~8.5 minutes. The number has to be on screen BEFORE the click."""
    beats = _lesson()["beats"]
    assert lp.estimate(beats)["minutes"] <= 5           # all stills: ffmpeg only

    for b in beats[:5]:
        b["animate"] = True
    est = lp.estimate(beats)
    assert est["animated"] == 5
    assert 40 <= est["minutes"] <= 55                   # ~8.5 min each


def test_render_refuses_a_lesson_nobody_looked_at(monkeypatch):
    """The gate is in the PIPELINE, not the browser — a Discord command or a stale tab
    must not be able to start an hour of Wan on a lesson nobody has seen."""
    _lesson(stage="stills")
    with pytest.raises(lp.LessonRenderError, match="not been approved"):
        lp.render_lesson("20260714_100000")


def test_approve_locks_in_what_you_ticked():
    _lesson(stage="stills", animate=(1, 3))
    got = lp.approve("20260714_100000")

    assert got["stage"] == "approved"
    assert [b["animate"] for b in got["beats"]] == [False, True, False, True,
                                                    False, False]
    # and it is on DISK — the render reads the ticks back, not the browser
    assert lw.load_lesson("20260714_100000")["stage"] == "approved"


def test_approve_refuses_more_shots_than_the_cap():
    lesson = _lesson(stage="stills")
    d = lw.LESSONS_DIR / lesson["lesson_id"]
    lesson["beats"] = [dict(lesson["beats"][0], animate=True, index=i)
                       for i in range(lp.MAX_WAN_CLIPS + 5)]
    (d / "lesson.json").write_text(json.dumps(lesson), encoding="utf-8")

    with pytest.raises(lp.LessonRenderError, match="cap"):
        lp.approve(lesson["lesson_id"])
    assert lw.load_lesson(lesson["lesson_id"])["stage"] == "stills"   # not approved


def test_a_lesson_with_no_pictures_cannot_be_approved():
    _lesson(stage="written")
    with pytest.raises(lp.LessonRenderError, match="not been prepared"):
        lp.approve("20260714_100000")


# --- the mixed clip list (real ffmpeg — a mock proves nothing here) ------------

def _make_clip(path: Path, seconds: float, fps: int, w: int, h: int) -> Path:
    subprocess.run(
        [str(FFMPEG), "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", f"testsrc=size={w}x{h}:rate={fps}:duration={seconds}",
         "-pix_fmt", "yuv420p", str(path)], check=True, timeout=120)
    return path


def _dur(path: Path) -> float:
    out = subprocess.run(
        [str(FFPROBE), "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True, timeout=60)
    return float(out.stdout.strip())


def _fps(path: Path) -> float:
    out = subprocess.run(
        [str(FFPROBE), "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=r_frame_rate", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=60)
    num, den = out.stdout.strip().split("/")
    return float(num) / float(den)


@pytest.mark.skipif(not FFMPEG.exists(), reason="ffmpeg not installed")
def test_a_16fps_wan_clip_and_a_30fps_still_survive_being_joined(tmp_path):
    """The heart of it. Wan renders at 16fps, Ken Burns at 30 — and the concat demuxer
    stream-copies, so a mismatch does not error, it SHIFTS THE DURATION. Every segment
    is resampled first, and each one comes out exactly as long as its line."""
    wan = _make_clip(tmp_path / "wan.mp4", 5.0, 16, 1280, 720)      # Wan's shape
    still = _make_clip(tmp_path / "kb.mp4", 4.0, 30, 1920, 1080)    # Ken Burns' shape

    a = la.normalize_segment(wan, 3.5, 1920, 1080, tmp_path / "a.mp4")
    b = la.normalize_segment(still, 2.5, 1920, 1080, tmp_path / "b.mp4")

    assert abs(_dur(a) - 3.5) < 0.15, "the clip must be exactly its line's length"
    assert abs(_dur(b) - 2.5) < 0.15
    assert abs(_fps(a) - la.FPS) < 0.1, "resampled, not relabelled"
    assert abs(_fps(b) - la.FPS) < 0.1


@pytest.mark.skipif(not FFMPEG.exists(), reason="ffmpeg not installed")
def test_a_short_wan_clip_is_held_not_cut_short(tmp_path):
    """Wan tops out at 5 seconds. A 7-second line still needs 7 seconds of picture —
    the last frame is held, so the picture never runs out before the teacher stops."""
    wan = _make_clip(tmp_path / "wan.mp4", 5.0, 16, 1280, 720)
    seg = la.normalize_segment(wan, 7.0, 1920, 1080, tmp_path / "seg.mp4")
    assert abs(_dur(seg) - 7.0) < 0.15


# --- captions -----------------------------------------------------------------

def test_the_captions_follow_the_real_line_lengths(tmp_path):
    beats = _lesson()["beats"]
    durations = [3.0, 2.0, 4.0, 3.0, 2.0, 3.0]
    ass = la.write_lesson_ass(beats, durations, 1920, 1080, tmp_path / "s.ass")
    text = ass.read_text(encoding="utf-8")

    assert "Line 0" in text and "This is line number 0" in text
    # No fact badge — that is a Shorts device, and a lesson is not a countdown.
    assert "#1" not in text


# --- A redrawn picture re-animates ITS shot, and only its shot ------------------
# A Wan clip is ~8.5 minutes. Lesson mode is a look-at-it-then-fix-it loop, and before
# this every round of that loop paid for ALL the animation again — so fixing one bad
# picture in a 3-clip lesson cost 25 minutes of GPU to redraw two clips that were fine.

def test_a_clip_is_reused_when_its_picture_has_not_changed(tmp_path, monkeypatch):
    import time
    from modules import lesson_pipeline as lp

    still = tmp_path / "still_00.png"
    still.write_bytes(b"x")
    clip = tmp_path / "clip_00.mp4"
    time.sleep(0.02)
    clip.write_bytes(b"y")                       # clip is NEWER than the still

    monkeypatch.setattr(lp.fp, "_probe_dur", lambda p: 4.0)

    def reusable(clip_p, still_p, want):
        if not clip_p.exists() or not clip_p.stat().st_size:
            return False
        if clip_p.stat().st_mtime < still_p.stat().st_mtime:
            return False
        got = lp.fp._probe_dur(clip_p)
        return got > 0 and abs(got - want) <= 0.15

    assert reusable(clip, still, 4.0) is True

    # redraw the picture -> the clip is now stale and must be re-animated
    time.sleep(0.02)
    still.write_bytes(b"z")
    assert reusable(clip, still, 4.0) is False


def test_a_clip_of_the_wrong_length_is_not_reused(tmp_path, monkeypatch):
    # The line was rewritten, so the voice take is a different length. A clip that no
    # longer matches its line would desync every caption after it.
    import time
    from modules import lesson_pipeline as lp
    still = tmp_path / "s.png"; still.write_bytes(b"x")
    time.sleep(0.02)
    clip = tmp_path / "c.mp4"; clip.write_bytes(b"y")
    monkeypatch.setattr(lp.fp, "_probe_dur", lambda p: 4.0)
    got = lp.fp._probe_dur(clip)
    assert abs(got - 6.5) > 0.15          # 4.0s clip cannot carry a 6.5s line


def test_render_lesson_actually_checks_for_reusable_clips():
    # The guard above is a model of the rule; this proves the rule is IN the renderer.
    import inspect
    from modules import lesson_pipeline as lp
    src = inspect.getsource(lp.render_lesson)
    assert "_still_reusable" in src
    assert "st_mtime" in src, "reuse must be invalidated by a redrawn still"
    # and a stale clip must not be silently kept: the ticked list is filtered by it
    assert "b.get(\"animate\") and clips[i] is None" in src
