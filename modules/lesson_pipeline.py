"""
Claw Bot — lesson mode: rendering the lesson

Two jobs, with a human in between.

    prepare_lesson()   scenes → voice → stills          (~10-15 min of GPU)
        ↓
    THE GATE           you look at every still and TICK the ones worth animating
        ↓
    render_lesson()    Wan on the ticked, a slow pan on the rest, then assemble

They are two jobs on purpose. Wan costs about **8 minutes of GPU per shot** (measured:
previous reels peaked at 482s) and Ken Burns costs nothing, so on a 12-beat lesson the
difference between ticking none and ticking all is 5 minutes against an hour and a
half. That choice is yours, and it has to be made while LOOKING at the pictures — so
the GPU lock is released before the gate and the tickboxes are read back off disk, not
out of the browser.

Nothing is animated by default. `animate: false` on every beat, set by the writer.

The landmine this file must not step on: `facts_pipeline._voice_beats_mascot()` ends in
`_fit_to_budget()`, which reads the FACTS reel's 40-second ceiling and speeds the
narration up by as much as 1.45x to hit it. A ninety-second lesson through that comes
out as a teacher gabbling, and nothing errors. We call `_voice_beats_clone()` — one
level down: same cloned mascot, same short-take guard, no budget. Pinned by
tests/test_lesson_budget.py.
"""

import logging
import shutil
import time as _t
from pathlib import Path
from typing import Callable, Optional

from modules import facts_assembly as fasm
from modules import facts_pipeline as fp
from modules import gpu_memory, gpu_utils
from modules import lesson_assembly as la
from modules import lesson_writer as lw
from modules import mascot
from modules import publish_kit
from modules.assembly import ASPECTS

log = logging.getLogger("claw_bot.lesson_pipeline")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
LESSONS_DIR = PROJECT_ROOT / "04_Outputs" / "lessons"

ASPECT = "16x9"

# Measured on this machine: Wan at 720p runs 8-9 minutes a clip. Ken Burns is ffmpeg
# on the CPU — seconds. These are what the gate's estimate is built from, and they are
# why the tickbox exists at all.
WAN_MIN_PER_CLIP = 8.5
STILL_SEC_PER_CLIP = 3.0

# A hard stop. Past this many animated shots the render is measured in hours, and
# nobody means to start that from one click.
MAX_WAN_CLIPS = 20


class LessonRenderError(RuntimeError):
    """The lesson cannot be rendered. Raised rather than quietly making a worse one."""


def preflight(_p=lambda m: None) -> None:
    """Prove the renderer answers BEFORE anything is written or voiced.

    A dead ComfyUI used to cost a 40-minute render of the wrong film. The mascot IS
    the lesson: there is no 'abstract backdrop' fallback here, because a lesson of
    gradient cards is not a lesson.
    """
    ok, why = mascot.is_available()
    if not ok:
        raise LessonRenderError(
            f"the mascot cannot render: {why}. Nothing was started.")
    _p(f"✅ {why}")


def estimate(beats: list) -> dict:
    """What this render will cost, given what is ticked. Shown BEFORE you commit."""
    animated = sum(1 for b in beats if b.get("animate"))
    still = len(beats) - animated
    minutes = animated * WAN_MIN_PER_CLIP + still * STILL_SEC_PER_CLIP / 60 + 3
    return {"animated": animated, "still": still, "minutes": round(minutes),
            "over_cap": animated > MAX_WAN_CLIPS}


def _voice_with_preset(narrations: list, out_dir: Path, _p) -> list:
    """Kokoro, one wav per line. The fallback when the clone cannot read the lesson.

    Deliberately NOT `facts_pipeline._voice_beats_mascot()`, which would give us the
    same cascade AND `_fit_to_budget()` — the 40-second facts ceiling, applied to a
    ninety-second lesson, at up to 1.45x. A preset voice is a downgrade you can hear
    and live with; a gabbling one is a broken video.
    """
    from modules import runtime_settings as rs
    from modules.tts_engine import TTSEngine

    tts = TTSEngine()
    voice = rs.get_facts_voice()
    _p(f"🎙️ preset voice: kokoro / {voice}")
    wavs = []
    for i, text in enumerate(narrations):
        wp = out_dir / f"beat_{i:02d}.wav"
        tts.synthesize(text, output_path=wp, voice=voice)
        wavs.append(fp._pad_wav(wp))
    return wavs


# ==============================================================================
# PHASE 1 — scenes, voice, stills. Then STOP.
# ==============================================================================

def prepare_lesson(lesson: dict,
                   progress_cb: Optional[Callable[[str], None]] = None) -> dict:
    """Voice the lesson and draw a still for every beat. Renders no video.

    Ends by handing the lesson back with `stage="stills"` — the gate is next, and the
    GPU is free while you look at the pictures.
    """
    def _p(msg: str):
        log.info(msg)
        if progress_cb:
            try:
                progress_cb(msg)
            except Exception:
                pass

    t0 = _t.time()
    lesson_id = lesson["lesson_id"]
    beats = lesson.get("beats", [])
    if not beats:
        raise LessonRenderError("this lesson has no beats")

    preflight(_p)

    d = LESSONS_DIR / lesson_id
    stills_dir, audio_dir = d / "stills", d / "audio"
    stills_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    # 1. The scenes — the mascot, in costume, doing what this line is about. Ollama is
    #    on the card for this and nothing else, and is unloaded before the pictures.
    _p(f"🎭 writing {len(beats)} scenes for {mascot.active_mascot_name()}…")
    with gpu_memory.llm():
        for i, b in enumerate(beats):
            if not b.get("mascot_scene"):
                b["mascot_scene"] = mascot.explainer_scene(
                    b["narration"], lesson.get("topic", ""), teaching=True)
            lw.set_beat_field(lesson_id, i, "mascot_scene", b["mascot_scene"])
    gpu_utils.free_ollama_vram()

    # 2. The voice. _voice_beats_clone, NOT _voice_beats_mascot — see the module note:
    #    the wrapper would trim this lesson's pace to fit a 40-second reel.
    _p("🎙️ voicing the lesson in the mascot's own voice…")
    narrations = [b["narration"] for b in beats]
    wavs = fp._voice_beats_clone(narrations, audio_dir, _p)
    if not wavs:
        # The clone collapses on a line now and then and returns None for the whole
        # lesson rather than shipping a blip (facts_pipeline._short_takes). Fall back
        # to a preset voice for ALL of it — one voice, consistent — but NOT through
        # facts' wrapper, which would trim the pace to a 40-second ceiling.
        _p("⚠️ the mascot's cloned voice could not read the lesson; using a preset voice")
        wavs = _voice_with_preset(narrations, audio_dir, _p)
    if not wavs:
        raise LessonRenderError(
            "the lesson could not be voiced. Nothing was rendered.")

    durations = [fp._probe_dur(Path(w)) for w in wavs]
    total = sum(durations)
    _p(f"🗣️ the lesson runs {total:.0f}s ({total/60:.1f} min)")

    # 3. The stills — one per beat, the mascot presenting, 16x9. Qwen-Edit stays warm
    #    for the whole batch (a cold load is ~4 min, a warm render ~20s).
    # ~100s each at full quality — say so, rather than let it look hung.
    _p(f"🖼️ drawing {len(beats)} pictures (~{len(beats) * 100 / 60:.0f} min; a lesson "
       f"draws at full quality so the props are real)…")
    gpu_memory.acquire(gpu_memory.QWEN_EDIT)
    stills = []
    try:
        for i, b in enumerate(beats):
            sp = stills_dir / f"still_{i:02d}.png"
            # full_quality: 20 steps at cfg 2.5 (~100s) instead of the 4-step
            # Lightning path (~27s). A lesson's prop IS the teaching — a plate of
            # vegetables has to look like vegetables, not candy — so a lesson pays
            # the extra minute a shot. A facts reel does not, and stays fast.
            got = mascot.render_scene(b["mascot_scene"], sp, aspect=ASPECT,
                                      seed=4000 + i, presenter=True,
                                      full_quality=True)
            if not got:
                # No black frames, no gradients. A lesson with a missing picture is a
                # lesson with a hole in it, and it must not be discovered in the file.
                raise LessonRenderError(
                    f"the picture for line {i+1} could not be drawn. Nothing was "
                    f"assembled — fix ComfyUI and prepare the lesson again.")
            stills.append(got)
            if (i + 1) % 3 == 0 or i == len(beats) - 1:
                _p(f"🖼️ {i+1}/{len(beats)} pictures drawn")
    finally:
        gpu_memory.release(gpu_memory.QWEN_EDIT)

    lesson = lw.load_lesson(lesson_id) or lesson
    for i, b in enumerate(lesson["beats"]):
        b["still"] = str(stills[i])
        b["duration"] = round(durations[i], 3)
    lesson["narration_dir"] = str(audio_dir)
    lesson["stage"] = "stills"
    lesson["estimated_seconds"] = round(total, 1)
    lw._save(lesson)

    est = estimate(lesson["beats"])
    _p(f"✅ ready in {(_t.time()-t0)/60:.0f} min. Now TICK the lines worth animating — "
       f"nothing is animated yet, and each tick adds about {WAN_MIN_PER_CLIP:.0f} min.")
    _p(f"   as it stands: {est['still']} still shots ≈ {est['minutes']} min to render")
    return lesson


# ==============================================================================
# THE GATE
# ==============================================================================

def approve(lesson_id: str) -> dict:
    """You have looked at the pictures and ticked what to animate. Lock it in."""
    lesson = lw.load_lesson(lesson_id)
    if not lesson:
        raise LessonRenderError(f"no lesson {lesson_id!r}")
    if lesson.get("stage") not in ("stills", "approved"):
        raise LessonRenderError(
            f"this lesson has not been prepared yet (stage={lesson.get('stage')!r}) — "
            f"there are no pictures to look at.")

    est = estimate(lesson["beats"])
    if est["over_cap"]:
        raise LessonRenderError(
            f"{est['animated']} animated shots is about {est['minutes']} minutes of "
            f"rendering — over the {MAX_WAN_CLIPS}-shot cap. Untick some: a still with "
            f"a slow pan costs nothing.")
    lesson["stage"] = "approved"
    lw._save(lesson)
    return lesson


# ==============================================================================
# PHASE 2 — animate what was ticked, pan the rest, assemble
# ==============================================================================

def render_lesson(lesson_id: str,
                  progress_cb: Optional[Callable[[str], None]] = None) -> dict:
    """Render the approved lesson. Raises unless it went through the gate."""
    def _p(msg: str):
        log.info(msg)
        if progress_cb:
            try:
                progress_cb(msg)
            except Exception:
                pass

    lesson = lw.load_lesson(lesson_id)
    if not lesson:
        raise LessonRenderError(f"no lesson {lesson_id!r}")

    # The gate lives HERE, not in the browser: a Discord command or a stale tab must
    # not be able to start an hour of Wan on a lesson nobody looked at.
    if lesson.get("stage") != "approved":
        raise LessonRenderError(
            "this lesson has not been approved. Look at the pictures, tick the lines "
            "worth animating, then press Render.")

    beats = lesson["beats"]
    stills = [Path(b["still"]) for b in beats]
    durations = [float(b["duration"]) for b in beats]
    missing = [i + 1 for i, s in enumerate(stills) if not s.exists()]
    if missing:
        raise LessonRenderError(f"the picture for line(s) {missing} is gone — prepare "
                                f"the lesson again.")

    est = estimate(beats)
    _p(f"🎬 {est['animated']} animated + {est['still']} still shots — "
       f"about {est['minutes']} min")

    d = LESSONS_DIR / lesson_id
    clips_dir = d / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    w, h = ASPECTS[ASPECT]
    clips: list = [None] * len(beats)

    # 0. Reuse the Wan clips that are still good.
    #
    # A Wan clip costs ~8.5 minutes. Lesson mode is a look-at-it-and-fix-it loop — you
    # redraw ONE bad picture and render again — and without this every one of those
    # rounds paid for all the animation a second time. A clip survives if it is newer
    # than the still it was drawn from and still the length of the line it carries; a
    # redrawn still is newer than its clip, so fixing a picture re-animates exactly that
    # shot and nothing else.
    def _still_reusable(i: int) -> bool:
        clip = clips_dir / f"clip_{i:02d}.mp4"
        if not clip.exists() or not clip.stat().st_size:
            return False
        if clip.stat().st_mtime < stills[i].stat().st_mtime:
            return False                      # the picture was redrawn after the clip
        got = fp._probe_dur(clip)
        return got > 0 and abs(got - durations[i]) <= 0.15

    for i in range(len(beats)):
        if beats[i].get("animate") and _still_reusable(i):
            clips[i] = clips_dir / f"clip_{i:02d}.mp4"
    reused = sum(1 for c in clips if c is not None)
    if reused:
        _p(f"♻️ reusing {reused} animated clip(s) whose picture has not changed "
           f"(~{reused * WAN_MIN_PER_CLIP:.0f} min saved)")

    # 1. The ticked beats: real motion. One Wan clip each, ~8 min apiece.
    ticked = [i for i, b in enumerate(beats) if b.get("animate") and clips[i] is None]
    if ticked:
        from modules import horror_video
        _p(f"🎥 animating {len(ticked)} shot(s) with Wan — this is the slow part")
        gpu_memory.acquire(gpu_memory.WAN_VIDEO)
        try:
            sub = {"_id": lesson_id,
                   "beats": [dict(beats[i], motion_prompt=(
                       beats[i].get("motion_prompt")
                       or f"{beats[i].get('mascot_scene', '')}, gentle lively gestures, "
                          f"subtle camera push-in")) for i in ticked]}
            made = horror_video.render_shot_clips(
                sub, [stills[i] for i in ticked], [durations[i] for i in ticked],
                aspect_ratio="16:9", progress_cb=_p, fill_mode="retime")
            for n, i in enumerate(ticked):
                dst = clips_dir / f"clip_{i:02d}.mp4"
                shutil.copyfile(made[n], dst)
                clips[i] = dst
        finally:
            gpu_memory.release(gpu_memory.WAN_VIDEO)

    # 2. Everything else: the still, with a slow pan. No GPU, seconds each.
    rest = [i for i in range(len(beats)) if clips[i] is None]
    if rest:
        _p(f"🖼️ panning {len(rest)} still shot(s)…")
        for i in rest:
            clips[i] = la.still_segment(stills[i], durations[i], w, h,
                                        clips_dir / f"clip_{i:02d}.mp4",
                                        zoom_in=(i % 2 == 0))

    # 3. One narration track for the whole lesson, in the order the lines are spoken.
    wavs = sorted((d / "audio").glob("beat_*.wav"))
    if len(wavs) != len(beats):
        raise LessonRenderError(
            f"{len(wavs)} voice file(s) for {len(beats)} lines — prepare the lesson "
            f"again.")
    narration = d / "audio" / "narration.wav"
    fp._concat_wavs(wavs, narration)

    out = la.assemble_lesson(lesson, narration, clips, durations, progress_cb=_p)

    # 4. The upload kit: a title, a description and a thumbnail beside the video.
    #
    # This was `aspect=ASPECT`, and the parameter is `aspects` — a tuple. It raised
    # TypeError, the except below swallowed it, and the lesson shipped with no title,
    # no description and no thumbnail: a warning in a log nobody reads. The kit must
    # not be able to fail quietly, so a failure is now SAID, in the progress feed the
    # user is actually watching.
    try:
        context = " ".join(b["narration"] for b in beats)
        out["kit"] = publish_kit.attach(
            out[ASPECT], fallback_title=lesson["title"], context=context,
            description=f"A lesson on {lesson['topic']}, from "
                        f"{lesson.get('book_title', 'the textbook')}.",
            mode="school lesson (16x9, kid-friendly explainer)",
            source_image=stills[0], aspects=(ASPECT,))
        _p("🏷️ upload kit written (title, description, thumbnail)")
    except Exception as e:
        log.exception("publish kit failed")
        _p(f"⚠️ the lesson rendered, but its upload kit failed ({e}) — no title, "
           f"description or thumbnail was written.")

    lesson = lw.load_lesson(lesson_id)
    lesson["stage"] = "rendered"
    lesson["video"] = str(out[ASPECT])
    lw._save(lesson)

    _p(f"✅ '{lesson['title']}' — {out['duration']:.0f}s, {out[ASPECT].name}")
    return out
