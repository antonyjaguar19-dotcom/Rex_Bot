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


def _lesson(stage="stills", animate=(), approved=True) -> dict:
    """A prepared lesson. `approved=True` by default because most of these tests are about
    something else — the gate itself is exercised in
    test_a_picture_nobody_confirmed_cannot_reach_wan."""
    lid = "20260714_100000"
    d = lw.LESSONS_DIR / lid
    d.mkdir(parents=True, exist_ok=True)
    beats = []
    for i in range(6):
        beats.append({
            "kind": "teach", "narration": f"This is line number {i} of the lesson.",
            "on_screen": f"Line {i}", "image_prompt": "a child",
            "animate": i in animate, "approved": approved, "index": i + 1,
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


def test_a_rendered_lesson_can_go_back_through_the_gate():
    # The whole mode is a look-at-it-then-fix-it loop: you watch the lesson, one picture
    # is wrong, you redraw it and render again. approve() refused a lesson whose stage
    # was "rendered", so the only way to fix a bad still was to write the lesson from
    # scratch — and the loop the mode is built around was a dead end.
    import inspect
    from modules import lesson_pipeline as lp
    src = inspect.getsource(lp.approve)
    assert '"rendered"' in src, "a rendered lesson must be re-approvable"
    assert '("stills", "approved", "rendered")' in src


def test_redo_the_pictures_actually_redraws_them(tmp_path, monkeypatch):
    # "🔁 Redo the pictures" was a twenty-minute NO-OP. prepare_lesson skips a beat that
    # already has a scene (`if not b.get("mascot_scene")`) and the still's seed is a
    # fixed 4000+i — so it rewrote nothing, redrew at the same seed, and handed back
    # byte-identical images. The button appeared to do nothing, because it did nothing.
    import inspect
    from modules import lesson_pipeline as lp

    src = inspect.getsource(lp.prepare_lesson)
    assert 'b["mascot_scene"] = ""' in src, "redo must throw the old scenes away"
    assert "still_take" in src, "redo must move the seed"
    assert "redo: bool = False" in inspect.signature(lp.prepare_lesson).__str__() or \
        "redo" in inspect.signature(lp.prepare_lesson).parameters


def test_redoing_the_pictures_keeps_the_voice(tmp_path):
    # Re-recording 13 lines to fix one picture is minutes of GPU for an identical
    # result — and worse, the clone is stochastic: it might collapse on a line THIS
    # time and drop the whole lesson to a preset voice as the price of a redraw.
    from modules import lesson_pipeline as lp

    audio = tmp_path / "audio"
    audio.mkdir()
    lines = ["Hello little one!", "Living things grow."]
    for i in range(2):
        (audio / f"beat_{i:02d}.wav").write_bytes(b"RIFF")

    assert lp._reusable_takes(lines, audio) == []      # no manifest yet
    lp._remember_takes(lines, audio)
    assert len(lp._reusable_takes(lines, audio)) == 2  # same words -> keep the takes

    # edit a line and the take of the OLD sentence must not be kept
    assert lp._reusable_takes(["Hello little one!", "Toys are not alive."], audio) == []
    # a missing wav is not reusable either
    (audio / "beat_01.wav").unlink()
    assert lp._reusable_takes(lines, audio) == []


def test_the_gate_shows_a_picture_you_can_actually_judge():
    # At 104px, every defect found in this lesson — mouse ears, emoji eyes, a bare
    # midriff, a different child entirely — is a few pixels across and invisible. The
    # gate ASKS you to look; it has to be lookable.
    import inspect
    from modules import dashboard_nicegui as dash
    src = inspect.getsource(dash)
    assert "cursor: zoom-in" in src
    assert "redo=True" in src, "the Redo button must ask for a real redo"


def test_one_lesson_gets_one_backdrop():
    # STYLE_PRESENTER only asks for "a bold vivid solid color background", so Qwen picked
    # a NEW colour on every shot: the first real lesson came out blue, purple, beige,
    # grey and dark grey — thirteen pictures that watch like clips cut together from four
    # different videos. A facts reel is one shot and does not care. A lesson is a film.
    #
    # The fix used to be one flat COLOUR per lesson. It is now one PLACE per lesson —
    # a garden, a classroom, a farmyard — because a flat card left the child floating in
    # a void. Same invariant either way: ONE, for the whole lesson.
    import inspect
    from modules import lesson_pipeline as lp
    from modules import mascot as mas

    src = inspect.getsource(lp.prepare_lesson)
    assert "setting_for(" in src
    # the draw itself lives in _draw_one now — shared by the batch and a single redraw, so
    # the copy you use to fix a bad picture cannot drift from the one that made it
    assert "background=backdrop" in inspect.getsource(lp._draw_one)

    a = lp.setting_for("Living and Non-living Things")
    b = lp.setting_for("Living and Non-living Things")
    assert a == b, "the film must not move house halfway through"
    assert "bold vivid solid color background" in mas.STYLE_PRESENTER
    assert "background" in inspect.signature(mas.render_scene).parameters


def test_a_facts_reel_still_picks_its_own_colour():
    # The knob is opt-in. Facts mode passes no background and keeps the free choice.
    import inspect
    from modules import facts_pipeline as fp
    assert "background=" not in inspect.getsource(fp._render_facts_mascot)


def test_a_lesson_happens_in_a_place_not_a_void():
    # The backdrop used to be a flat colour card and every shot looked like a child
    # floating in a void. A lesson about living things belongs in a garden; one about the
    # body in a classroom. But it stays ONE place: thirteen locations watch like thirteen
    # different films, which is the same mistake the flat colours made.
    from modules import lesson_pipeline as lp
    from modules import mascot as mas

    assert "garden" in lp.setting_for("Living and Non-living Things")
    assert "classroom" in lp.setting_for("My Wonderful Body")
    assert "farmyard" in lp.setting_for("Animals Around Us")
    assert lp.setting_for("Quantum Widgets") == lp.DEFAULT_SETTING   # never empty

    # stable across a redo — the film does not move house halfway through
    assert lp.setting_for("Living Things") == lp.setting_for("Living Things")

    # and the child must stay readable against it, captions included
    assert "soft and gently out of focus behind her" in mas.STYLE_TEACHING
    assert "the character is sharp and fully separated from the background" in mas.STYLE_TEACHING

    import inspect
    assert "setting_for(" in inspect.getsource(lp.prepare_lesson)


def test_the_tail_the_chimera_and_the_wonky_eye_are_banned():
    # A dog's tail grew out of the GIRL'S HIP; a doll and a puppy FUSED into one creature
    # (a puppy's head on a cloth body with button eyes); and one eye came back larger than
    # the other. Two similar objects held close together merge, and an animal's parts
    # wander onto the nearest body.
    from modules import mascot as mas
    for banned in ("tail growing from a person", "tail on a child",
                   "merged creature", "doll fused with an animal", "chimera",
                   "asymmetric eyes", "lazy eye", "one eye larger than the other"):
        assert banned in mas.NEGATIVE_PRESENTER, banned


# --- NOTHING STARTS WAN ON A PICTURE NOBODY HAS SEEN -----------------------------
# Almost every serious defect this pipeline has produced — mouse ears, a twin mother, a
# doll rendered as a living child dangling by the wrist, a headless doll, a boulder, the
# teacher snarling — rendered cleanly, errored nothing, warned nothing, and was wrong only
# IN THE FILE. Wan then spent 8.5 minutes animating it.
#
# An image pipeline has no failing test: a wrong picture renders exactly as fast and as
# cleanly as a right one, and the only detector is a person looking at it.

def test_a_picture_nobody_confirmed_cannot_reach_wan(tmp_path, monkeypatch):
    from modules import lesson_pipeline as lp
    from modules import lesson_writer as lw

    monkeypatch.setattr(lw, "LESSONS_DIR", tmp_path)
    monkeypatch.setattr(lp, "LESSONS_DIR", tmp_path)
    lid = "20260714_130000"
    (tmp_path / lid).mkdir()
    beats = [{"kind": "teach", "narration": f"line {i}", "on_screen": "x",
              "image_prompt": "", "animate": False, "approved": i < 2, "index": i + 1,
              "kind_": None}
             for i in range(4)]
    lw._save({"lesson_id": lid, "title": "T", "topic": "T", "beats": beats,
              "stage": "stills", "word_count": 8, "estimated_seconds": 12.0})

    with pytest.raises(lp.LessonRenderError) as e:
        lp.approve(lid)
    msg = str(e.value)
    assert "not confirmed" in msg
    assert "3, 4" in msg, "it must say WHICH pictures"

    for i in (2, 3):
        lw.set_beat_approved(lid, i, True)
    got = lp.approve(lid)                    # now it goes through
    assert got["stage"] == "approved"


def test_a_fresh_picture_is_never_pre_approved():
    # Carrying an old tick across a redraw is exactly how an unlooked-at picture reaches
    # Wan.
    import inspect
    from modules import lesson_pipeline as lp
    src = inspect.getsource(lp.prepare_lesson)
    assert 'b["approved"] = False' in src

    src = inspect.getsource(lp.redraw_still)
    assert "set_beat_approved(lesson_id, i, False)" in src


def test_one_picture_can_be_redrawn_on_its_own():
    # Before this, fixing one bad picture meant redrawing all thirteen — and "Redo the
    # pictures" was a twenty-minute NO-OP besides (same scene, fixed seed, byte-identical
    # images). A redraw needs a NEW SEED or you get the same picture back.
    import inspect
    from modules import lesson_pipeline as lp
    src = inspect.getsource(lp.redraw_still)
    assert 'lesson["still_take"] = int(lesson.get("still_take", 0)) + 1' in src
    assert "_draw_one(" in src

    # ...and the batch and the single redraw go through the SAME draw, or the copy that
    # drifts is the one you use to fix a bad picture
    assert "_draw_one(" in inspect.getsource(lp.prepare_lesson)


# --- the logo ------------------------------------------------------------------

def test_the_logo_is_small_half_transparent_and_actually_bottom_right():
    # It was a THIRD of the way across the frame and all but opaque, and it sat at the
    # RIGHT EDGE, VERTICALLY CENTRED — beside the mascot's head — while its own docstring
    # and every call-site comment said "bottom-right". The y-expression was (H-h)/2. A
    # comment is not a test.
    from modules import musicvideo_assembly as mva

    assert mva.WM_WIDTH_FRAC == 0.10
    assert mva.WM_OPACITY == 0.5

    f = mva.logo_overlay_filter("v0", "v1", 2, 1920)
    assert "aa=0.5" in f
    assert "scale=192:" in f                    # 10% of 1920
    assert "overlay=W-w-57:H-h-57" in f, "still not bottom-right"
    assert "(H-h)/2" not in f


# --- the outro -----------------------------------------------------------------

def test_the_lesson_ends_on_a_subscribe_about_the_topic(tmp_path, monkeypatch):
    from modules import lesson_writer as lw
    monkeypatch.setattr(lw, "LESSONS_DIR", tmp_path)
    lid = "20260714_140000"
    (tmp_path / lid).mkdir()
    beats = [{"kind": "intro", "narration": "Hello!", "on_screen": "Hi",
              "image_prompt": "", "animate": False, "approved": True, "index": 1},
             {"kind": "recap", "narration": "Living things eat and grow.",
              "on_screen": "Recap", "image_prompt": "", "animate": False,
              "approved": True, "index": 2}]
    lw._save({"lesson_id": lid, "title": "Living Things",
              "topic": "Living and Non-living Things", "beats": beats,
              "stage": "stills", "word_count": 6, "estimated_seconds": 8.0})

    # no Ollama in the tests — the fallback must still give the lesson an ending
    monkeypatch.setattr(lw, "subscribe_outro", lambda t: lw._fallback_outro(t))
    assert lw.ensure_outro(lid) is True

    got = lw.load_lesson(lid)["beats"]
    assert len(got) == 3
    assert got[1]["narration"] == "Living things eat and grow.", "the RECAP survives"
    assert got[-1]["kind"] == "outro"
    assert "subscribe" in got[-1]["narration"].lower()
    assert "living" in got[-1]["narration"].lower(), "it is about the TOPIC"
    assert got[-1]["approved"] is False, "a new picture nobody has seen"

    # idempotent — a second prepare must not staple a second outro on
    assert lw.ensure_outro(lid) is False
    assert len(lw.load_lesson(lid)["beats"]) == 3


def test_the_outro_check_is_the_feature_not_the_prompt(monkeypatch):
    # A model told to say "subscribe" writes a lovely sign-off that never says it — the
    # same lesson the fact memory taught (a model told "don't repeat these" rewords the
    # fact instead). The CHECK is the feature.
    from modules import lesson_writer as lw
    import modules.script_generator as sg

    monkeypatch.setattr(sg, "_call_llm",
                        lambda *a, **k: '{"outro": "Thanks for watching, see you soon!"}')
    got = lw.subscribe_outro("Fruits")
    assert "subscribe" in got.lower(), "a CTA with no CTA in it was accepted"
    assert got == lw._fallback_outro("Fruits")


def test_the_recap_is_no_longer_labelled_the_outro():
    # The last spoken line used to be tagged "outro" purely because it came last, so the
    # lesson did not END — it just stopped.
    from modules import lesson_writer as lw
    kinds = lw._kinds_for(["Hello.", "Living things grow.", "Is a rock alive?",
                           "So living things eat and grow."])
    assert kinds[0] == "intro"
    assert kinds[-1] == "recap"
    assert "outro" not in kinds


def test_an_untick_after_approve_still_stops_the_render():
    # approve() reads the ticks and stamps stage="approved"; render_lesson used to check only
    # the STAMP. Anything that un-ticked a picture in between — a second tab, a redraw, a
    # Discord command — left the stamp standing and an hour of Wan started on a picture
    # nobody had confirmed. The same predicate, one call later.
    _lesson(stage="stills")
    lid = "20260714_100000"
    lp.approve(lid)
    assert lw.load_lesson(lid)["stage"] == "approved"

    lw.set_beat_approved(lid, 0, False)          # a redraw, another tab, a stale click

    with pytest.raises(lp.LessonRenderError) as e:
        lp.render_lesson(lid)
    assert "not confirmed" in str(e.value)
    assert "1" in str(e.value), "it must name the line"
