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

import json
import logging
import shutil
import time as _t
from pathlib import Path
from typing import Callable, Optional

from modules import cast
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

# ONE LESSON, ONE PLACE.
#
# This used to be a flat colour card, and every shot looked like a child floating in a
# void. A real place is better television and better teaching: a lesson about living
# things belongs in a garden, one about the body in a bright classroom.
#
# It stays ONE place for the whole lesson, though. Thirteen different locations watch
# like thirteen different films — the same mistake the flat colours made when Qwen was
# left to pick one per shot (blue, purple, beige, grey).
#
# The setting is deliberately soft and behind her: the child and the prop are the
# teaching, and the burned-in caption has to stay readable across the bottom.
SETTINGS = {
    "living": "a sunny home garden with a low wooden fence, potted plants and grass",
    "body": "a bright cheerful classroom with a soft rug and low shelves",
    "plant": "a sunny home garden with flower beds and a small watering can",
    "flower": "a sunny home garden with flower beds and a small watering can",
    "vegetable": "a bright kitchen with a low table and a basket of vegetables",
    "fruit": "a bright kitchen with a low table and a bowl of fruit",
    "animal": "a green farmyard with a wooden fence and a red barn far behind",
    "food": "a bright kitchen with a low table and a fruit bowl",
    "water": "a sunny garden beside a small pond with reeds",
    "air": "a breezy hilltop park with kites in the far distance",
    "sky": "a breezy hilltop park under a wide blue sky",
    "home": "a warm tidy living room with a sofa and a toy box",
    "school": "a bright cheerful classroom with a soft rug and low shelves",
    "play": "a sunny playground with a swing set and slide behind her",
}
DEFAULT_SETTING = "a sunny home garden with a low wooden fence, grass and potted plants"


def setting_for(topic: str) -> str:
    """The one place this lesson happens. Keyword-matched, not LLM-guessed — a wrong
    setting is cheap to live with and an LLM round-trip per lesson is not."""
    t = (topic or "").lower()
    for key, place in SETTINGS.items():
        if key in t:
            return place
    return DEFAULT_SETTING


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


def _reusable_takes(narrations: list, audio_dir: Path) -> list:
    """The voice takes on disk, IF they are still the takes these words need.

    A take is stale the moment its line is edited, and only then. The words are written
    beside the audio when it is recorded, so this is an exact check, not a guess — a
    length or mtime heuristic would happily keep a take of the wrong sentence.
    """
    manifest = audio_dir / "spoken.json"
    if not manifest.is_file():
        return []
    try:
        said = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if said != narrations:
        return []
    wavs = [audio_dir / f"beat_{i:02d}.wav" for i in range(len(narrations))]
    if not all(w.is_file() and w.stat().st_size for w in wavs):
        return []
    return wavs


def _remember_takes(narrations: list, audio_dir: Path) -> None:
    """Write down which words these takes are of."""
    from modules.file_utils import atomic_write_json
    atomic_write_json(audio_dir / "spoken.json", narrations)


def _scene_and_refs(lesson: dict, i: int, mid: str, _p=lambda m: None) -> tuple:
    """(scene_text, refs, relation) — the scene EXACTLY as the model will receive it.

    Shared by the renderer and by the reveal panel. A preview built from a second copy of
    this would drift from what is actually sent, and a preview that can lie is worse than
    no preview.

    A SECOND PERSON GETS A SECOND REFERENCE. "You feel happy when mummy hugs you" needs a
    mummy in the picture, and for a long time we could not draw one: handed a single
    reference (the mascot), Qwen painted that identity onto every human in the frame and
    the mother came back as the child's TWIN. TextEncodeQwenImageEditPlus takes
    image1..image3 and we were only ever passing one. See modules/cast.py.

    Only when the scene actually names someone: a spare reference is another thing for Qwen
    to copy into a picture that did not ask for it.
    """
    from modules import lesson_objects as lo
    MAX_REFS = 3        # Qwen-Edit image1..image3 (comfyui_qwen_edit.MAX_REFS)

    orig_scene = lesson["beats"][i].get("mascot_scene", "")
    scene_text = orig_scene
    lesson_id = lesson.get("lesson_id")
    objects = lo.objects_for(lesson)

    # SLOT 1 — the mascot, always.
    refs = [str(r) for r in mascot.mascot_refs(max_refs=1)]

    # scene_text is still the original here (nothing has mutated it yet), so this reads the
    # person off the same words the objects are detected from below.
    relation, second = cast.ref_for(scene_text, mid)
    if second:
        # SLOT 2 — a named person. PERSON WINS the budget: a person keeps its slot ahead of
        # any prop, because a mislaid person becomes a TWIN and a mislaid prop only drifts.
        refs.append(str(second))
        # Two references and no explanation is an invitation to blend them. Say which is
        # which — that is what the proving render did, and it came back with two correct,
        # separate people first try.
        scene_text = cast.name_the_refs(scene_text, mid, relation)
        _p(f"👥 line {i+1} has a second person — {relation} joins the shot")
    else:
        # NAMED, BUT WE HAVE NO PICTURE OF THEM. Qwen would paint the mascot's own identity
        # onto her — a TWIN. She comes out of the PICTURE; the words on disk are untouched.
        named = cast.role_in(orig_scene, mid)
        if named:
            scene_text = mascot.other_people_are_other_people(scene_text)
            _p(f"⚠️ line {i+1} names a {named}, but this mascot has no picture of them — "
               f"leaving them out of the shot (they would come back as a twin). Add one on "
               f"the Mascots tab and redraw.")

    # THE RECURRING OBJECTS this shot names — a reference each while the slots last (so the
    # doll is the SAME doll every shot), the rest on the description-lock. Detected on the
    # ORIGINAL scene, before the family sentence was prepended.
    if objects:
        ref_specs, desc_keys = [], []
        for key in lo.detect(orig_scene, objects):
            img = lo.image(lesson_id, key)
            label = next((o["noun"] for o in objects if o["key"] == key), key)
            if img and len(refs) < MAX_REFS:
                refs.append(str(img))
                ref_specs.append((label, len(refs)))     # its 1-based slot in the ref list
            else:
                desc_keys.append(key)                    # no slot (or no pin) -> desc-lock
        if ref_specs:
            scene_text = lo.name_object_refs(scene_text, ref_specs)
            _p(f"🧸 line {i+1}: {', '.join(l for l, _ in ref_specs)} kept identical by "
               f"reference")
        if desc_keys:
            scene_text = lo.lock_descriptions(scene_text, desc_keys, objects)

    return scene_text, refs, relation


def _seed_for(lesson: dict, i: int) -> int:
    """A NEW seed after a redraw, or you get the same picture back — which is exactly what
    "Redo the pictures" did for twenty minutes."""
    return 4000 + i + 1000 * int(lesson.get("still_take", 0))


def default_motion(beat: dict) -> str:
    """What Wan is told when nobody has written a motion prompt by hand."""
    return (f"{beat.get('mascot_scene', '')}, gentle lively gestures, "
            f"subtle camera push-in")


def _draw_one(lesson: dict, i: int, backdrop: str, mid: str, out: Path,
              _p=lambda m: None):
    """One picture. The whole batch and a single redraw go through HERE — two copies of
    this would drift, and the copy that drifted would be the one you use to fix a bad
    picture.

    full_quality: 20 steps at cfg 2.5 (~100s) instead of the 4-step Lightning path (~27s).
    A lesson's prop IS the teaching — a plate of vegetables has to look like vegetables,
    not candy — so a lesson pays the extra minute a shot. A facts reel does not.
    """
    scene_text, refs, relation = _scene_and_refs(lesson, i, mid, _p)
    framing = lesson["beats"][i].get("framing", "medium")
    from modules import runtime_settings as rs
    got = mascot.render_scene(
        scene_text, out, aspect=ASPECT,
        seed=_seed_for(lesson, i),
        presenter=True, full_quality=True,
        reference_images=refs, background=backdrop, teaching=True,
        framing=framing,
        # a SECOND PERSON, not merely a second reference: an object ref must not trip the
        # two-people identity clause.
        two_people=bool(relation),
        # lessons render on USO by default (multi-subject: mascot + person + props); flip
        # with rs.set_lesson_image_backend("qwen").
        prefer_backend=rs.get_lesson_image_backend())
    if got:
        # DELIVER the framing. Qwen-Edit anchors to the full-body reference and ignores the
        # prompt's framing clause (measured: a closeup came back full-body), so the camera
        # distance is a deterministic crop of the render — see shot_framing. Runs once, here.
        from modules import shot_framing as sf
        sf.crop_to_framing(got, framing)
    return got


def prompts_for(lesson_id: str, i: int) -> dict:
    """Everything this shot actually sends to the models — the image prompt, its negative,
    the video prompt, and the numbers.

    None of it was visible anywhere: what reaches Qwen is `f"{scene}, {style}"`, and the
    ~120 words of STYLE_TEACHING and ~90 of NEGATIVE_TEACHING appear in no log, no JSON and
    no screen. Every rule hammered in this week — proportions by reference, the props fit
    in a hand, nothing inanimate gets a face — lives in a string nobody could read.

    Built from the SAME functions the renderer calls, so it cannot drift from the truth.
    """
    lesson = lw.load_lesson(lesson_id)
    if not lesson or not (0 <= i < len(lesson.get("beats", []))):
        return {}
    b = lesson["beats"][i]

    from modules import mascot_library as _ml
    from modules.image_backends import comfyui_qwen_edit as qe

    mid = _ml.get_active_id()
    backdrop = setting_for(lesson.get("topic", "") or lesson.get("title", ""))
    scene_text, refs, relation = _scene_and_refs(lesson, i, mid)
    framing = b.get("framing", "medium")
    # A SECOND PERSON keeps the identity clause apart — an object reference does not count.
    # The framing swaps the full-body clause for this shot's camera distance.
    positive, negative = mascot.build_presenter_prompt(
        scene_text, backdrop, teaching=True,
        two_people=bool(relation),
        framing=framing)

    from modules import lesson_objects as _lo
    return {
        "scene": b.get("mascot_scene", ""),      # what you edit
        "framing": framing,                      # this shot's camera distance
        "scene_sent": scene_text,                # what the model is told (+ the family line)
        "image_positive": positive,              # ...plus the style. THIS is what Qwen gets.
        "image_negative": negative,
        "motion": b.get("motion_prompt") or default_motion(b),
        "motion_is_default": not b.get("motion_prompt"),
        "seed": _seed_for(lesson, i),
        "steps": qe.DEFAULT_STEPS_FULL,          # read from the backend, never retyped
        "cfg": qe.DEFAULT_CFG_FULL,
        "backdrop": backdrop,
        "relation": relation,
        "refs": [Path(r).name for r in (refs or mascot.mascot_refs(max_refs=1))],
        # the recurring objects this shot names (pinned to stay identical shot to shot)
        "objects": _lo.detect(b.get("mascot_scene", ""), _lo.objects_for(lesson)),
    }


def redraw_still(lesson_id: str, i: int,
                 progress_cb: Optional[Callable[[str], None]] = None) -> bool:
    """Redraw ONE picture. A new seed, and its confirmation is cleared.

    Before this, fixing one bad picture meant redrawing all thirteen — and "🔁 Redo the
    pictures" was for twenty minutes a no-op besides (it kept the scene and the seed, so
    it handed back byte-identical images). Clip reuse keys on the still's mtime, so
    redrawing one shot re-animates that shot and no other.
    """
    def _p(msg: str):
        log.info(msg)
        if progress_cb:
            try:
                progress_cb(msg)
            except Exception:
                pass

    lesson = lw.load_lesson(lesson_id)
    if not lesson or not (0 <= i < len(lesson.get("beats", []))):
        return False

    preflight(_p)
    from modules import mascot_library as _ml
    cast.migrate()
    mid = _ml.get_active_id()
    backdrop = setting_for(lesson.get("topic", "") or lesson.get("title", ""))

    # A NEW SEED, or you get the same picture back. That was the whole bug in "Redo the
    # pictures": same scene, fixed seed, byte-identical images, twenty minutes gone.
    lesson["still_take"] = int(lesson.get("still_take", 0)) + 1
    sp = LESSONS_DIR / lesson_id / "stills" / f"still_{i:02d}.png"

    _p(f"🖼️ redrawing picture {i+1}…")
    from modules import runtime_settings as _rs
    _label = (gpu_memory.FLUX_USO if _rs.get_lesson_image_backend() == "uso"
              else gpu_memory.QWEN_EDIT)
    gpu_memory.acquire(_label)
    try:
        got = _draw_one(lesson, i, backdrop, mid, sp, _p)
    finally:
        gpu_memory.release(_label)
    if not got:
        _p(f"❌ picture {i+1} could not be drawn")
        return False

    lw._save(lesson)                        # keep the bumped seed
    lw.set_beat_approved(lesson_id, i, False)   # a NEW picture has not been looked at
    _p(f"✅ picture {i+1} redrawn — confirm it when you have looked")
    return True


def run_visual_qc(lesson_id: str, max_rounds: int = 2,
                  progress_cb: Optional[Callable[[str], None]] = None) -> dict:
    """LOOK at every rendered still with the QC vision model and re-draw the ones it flags.

    A separate PASS, not per-render, because VRAM: the 21 GB QC model (qwen2.5vl:32b) cannot
    share a 16 GB card with ComfyUI's image model. So ComfyUI is freed before the QC model
    loads, and the QC model is freed before any re-draw. Capped at `max_rounds` re-rolls; a
    shot still flagged after that is SURFACED for a human, never looped forever. Best-effort:
    if the QC model is not available the pass is skipped, and QC never blocks a render.

    Returns {skipped?|rounds, flagged:[i,...], why?}.
    """
    def _p(msg: str):
        log.info(msg)
        if progress_cb:
            try:
                progress_cb(msg)
            except Exception:
                pass

    from modules import image_qc
    ok, why = image_qc.available()
    if not ok:
        _p(f"👁️ visual QC skipped — {why}")
        return {"skipped": True, "why": why, "flagged": []}

    stills_dir = LESSONS_DIR / lesson_id / "stills"
    n = len(lw.load_lesson(lesson_id).get("beats", []))
    flagged: list = []
    for rnd in range(max_rounds + 1):
        gpu_utils.free_comfyui_vram()               # give the 21 GB QC model the card
        flagged = []
        for i in range(n):
            sp = stills_dir / f"still_{i:02d}.png"
            if not sp.exists():
                continue
            v = image_qc.qc_still(sp, "teaching")
            if v.get("ok", True):
                _p(f"✅ shot {i+1}/{n}: passed visual QC")
            else:
                keys = ", ".join(f["key"] for f in v.get("fails", []))
                _p(f"⚠️ shot {i+1}/{n}: visual QC flagged {keys}")
                flagged.append(i)
        if not flagged:
            _p(f"👁️ all {n} shots passed visual QC")
            return {"rounds": rnd, "flagged": []}
        if rnd == max_rounds:
            _p(f"👁️ {len(flagged)} shot(s) still flagged after {max_rounds} re-roll(s): "
               f"{[i + 1 for i in flagged]} — surfacing for you to judge")
            return {"rounds": rnd, "flagged": flagged}
        gpu_utils.free_ollama_vram()                # free the QC model before re-drawing
        _p(f"👁️ re-rolling {len(flagged)} flagged shot(s): {[i + 1 for i in flagged]}")
        for i in flagged:
            redraw_still(lesson_id, i, progress_cb)
    return {"rounds": max_rounds, "flagged": flagged}


# ==============================================================================
# PHASE 1 — scenes, voice, stills. Then STOP.
# ==============================================================================

def prepare_lesson(lesson: dict,
                   progress_cb: Optional[Callable[[str], None]] = None,
                   redo: bool = False) -> dict:
    """Voice the lesson and draw a still for every beat. Renders no video.

    Ends by handing the lesson back with `stage="stills"` — the gate is next, and the
    GPU is free while you look at the pictures.

    `redo=True` means "these pictures are wrong, give me different ones". It throws the
    scenes away so they are WRITTEN AGAIN, and moves the seed. Without both, "Redo the
    pictures" was a twenty-minute no-op: the scene was kept (the code skips a beat that
    already has one) and the seed is a fixed 4000+i, so Qwen redrew the identical image
    and the button appeared to do nothing at all.
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

    # The lesson ENDS on something. Its last spoken line used to be the last sentence of
    # the recap, labelled "outro" purely because it came last — so the lesson did not end,
    # it just stopped. Idempotent, so a lesson written before this gains an ending the next
    # time it is prepared, without being rewritten.
    if lw.ensure_outro(lesson_id):
        lesson = lw.load_lesson(lesson_id)
        _p(f"👋 the lesson now ends on: {lesson['beats'][-1]['narration']}")

    # Every beat needs a camera framing (establishing/wide/medium/closeup). A lesson written
    # before framing existed has none; back-fill it from each beat's kind, idempotently, and
    # never over a framing set by hand.
    if lw.ensure_framing(lesson_id):
        lesson = lw.load_lesson(lesson_id)
        _p("🎥 gave each shot a camera framing")

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
    if redo:
        for b in beats:
            b["mascot_scene"] = ""
        lesson["still_take"] = int(lesson.get("still_take", 0)) + 1

    from modules import lesson_objects as _lo
    _obj_nouns = [o.get("noun") for o in _lo.objects_for(lesson)]
    _p(f"🎭 writing {len(beats)} scenes for {mascot.active_mascot_name()}…")
    with gpu_memory.llm():
        for i, b in enumerate(beats):
            if not b.get("mascot_scene"):
                b["mascot_scene"] = mascot.explainer_scene(
                    b["narration"], lesson.get("topic", ""), teaching=True,
                    framing=b.get("framing", "medium"), object_nouns=_obj_nouns)
            lw.set_beat_field(lesson_id, i, "mascot_scene", b["mascot_scene"])
    gpu_utils.free_ollama_vram()

    # 2. The voice. _voice_beats_clone, NOT _voice_beats_mascot — see the module note:
    #    the wrapper would trim this lesson's pace to fit a 40-second reel.
    narrations = [b["narration"] for b in beats]

    # Redoing the PICTURES must not re-record the VOICE. The takes are only stale if the
    # words changed, and a clone re-read of 13 lines is minutes of GPU for an identical
    # result — worse, it is minutes of a stochastic model that might collapse on a line
    # this time (facts_pipeline._short_takes) and drop the whole lesson to a preset voice
    # as the price of fixing one picture.
    wavs = _reusable_takes(narrations, audio_dir)
    if wavs:
        _p(f"♻️ keeping {len(wavs)} voice takes — the words have not changed")
    else:
        _p("🎙️ voicing the lesson in the mascot's own voice…")
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
    _remember_takes(narrations, audio_dir)

    durations = [fp._probe_dur(Path(w)) for w in wavs]
    total = sum(durations)
    _p(f"🗣️ the lesson runs {total:.0f}s ({total/60:.1f} min)")

    # 3. The stills — one per beat, the mascot presenting, 16x9. Qwen-Edit stays warm
    #    for the whole batch (a cold load is ~4 min, a warm render ~20s).
    # ~100s each at full quality — say so, rather than let it look hung.
    _p(f"🖼️ drawing {len(beats)} pictures (~{len(beats) * 100 / 60:.0f} min; a lesson "
       f"draws at full quality so the props are real)…")
    # ONE LESSON, ONE LOOK. The presenter style only asks for "a bold vivid solid color
    # background", so Qwen picked a new colour on every shot: the first lesson came out
    # blue, purple, beige, grey and dark grey — thirteen pictures that watch like clips
    # cut together from four different videos. A facts reel is one shot and does not
    # care. A lesson is a film, and a child should not be told, five times, that the
    # scene has changed when it has not.
    backdrop = setting_for(lesson.get("topic", "") or lesson.get("title", ""))

    # Whose family. A second mascot inheriting the first one's mother is the twin problem
    # with extra steps, so the family belongs to the mascot (assets/mascots/<id>/family/).
    from modules import mascot_library as _ml
    cast.migrate()                      # the old shared assets/cast/, moved in once
    mid = _ml.get_active_id()
    _p(f"🎬 this lesson is set in {backdrop}")

    from modules import runtime_settings as _rs
    _draw_label = (gpu_memory.FLUX_USO if _rs.get_lesson_image_backend() == "uso"
                   else gpu_memory.QWEN_EDIT)
    gpu_memory.acquire(_draw_label)
    stills = []
    try:
        # PIN the recurring objects first, while the image model is warm. The doll and puppy
        # are each drawn ONCE, in isolation, and every shot that names them references the
        # same picture — so the not-living example and the living one stay themselves across
        # the lesson (they used to be a different toy every shot). A pin that fails leaves
        # that object on the description-lock; never fatal.
        from modules import lesson_objects as lo
        objs = lo.objects_for(lesson)
        if objs:
            _p(f"🧸 pinning {len(objs)} recurring object(s) so they stay the same every "
               f"shot…")
            for o in objs:
                pinned = lo.pin(lesson_id, o)
                _p(f"   • {o['noun']}: "
                   + ("pinned" if pinned else "could not pin — describing it instead"))

        for i, b in enumerate(beats):
            sp = stills_dir / f"still_{i:02d}.png"
            got = _draw_one(lesson, i, backdrop, mid, sp, _p)
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
        gpu_memory.release(_draw_label)

    lesson = lw.load_lesson(lesson_id) or lesson
    for i, b in enumerate(lesson["beats"]):
        b["still"] = str(stills[i])
        b["duration"] = round(durations[i], 3)
        # A FRESH PICTURE IS NEVER PRE-APPROVED. These are new images nobody has seen;
        # carrying an old tick across a redraw is exactly how an unlooked-at picture would
        # reach Wan.
        b["approved"] = False
        # The video prompt used to be invented at render time and thrown away, so it could
        # not be seen or edited. Write it down. A hand-edited one is never overwritten.
        if not b.get("motion_prompt"):
            b["motion_prompt"] = default_motion(b)
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
    # "rendered" is allowed, and that is the whole point of the mode. You watch the
    # lesson, one picture is wrong, you redraw it and render again — and the gate has to
    # let you back through or the loop is a dead end. (It was: approve() refused a
    # rendered lesson, so the only way to fix a bad still was to write the lesson from
    # scratch.) Clip reuse in render_lesson means the second pass only re-animates the
    # shots whose picture actually changed.
    if lesson.get("stage") not in ("stills", "approved", "rendered"):
        raise LessonRenderError(
            f"this lesson has not been prepared yet (stage={lesson.get('stage')!r}) — "
            f"there are no pictures to look at.")

    est = estimate(lesson["beats"])
    if est["over_cap"]:
        raise LessonRenderError(
            f"{est['animated']} animated shots is about {est['minutes']} minutes of "
            f"rendering — over the {MAX_WAN_CLIPS}-shot cap. Untick some: a still with "
            f"a slow pan costs nothing.")

    # EVERY PICTURE MUST HAVE BEEN LOOKED AT. This is the last thing checked and the one
    # that matters most.
    #
    # Almost every serious defect this pipeline has produced — mouse ears, a twin mother, a
    # doll rendered as a living child dangling by the wrist, a headless doll, a boulder,
    # the teacher snarling at the camera — rendered cleanly, errored nothing, warned
    # nothing, and was wrong only IN THE FILE. Wan then spent 8.5 minutes animating it.
    #
    # An image pipeline has no failing test: a wrong picture renders exactly as fast and as
    # cleanly as a right one, and the only detector is a person looking at it. So nothing
    # starts Wan on a picture nobody has confirmed.
    missing = lw.unapproved(lesson)
    if missing:
        raise LessonRenderError(
            f"{len(missing)} picture(s) not confirmed yet — line(s) "
            f"{', '.join(str(n) for n in missing)}. Open each one, look at it properly, "
            f"and tick ✓. Nothing starts an hour of Wan on a picture nobody has seen.")
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
            # The motion prompt is STORED on the beat now (prepare_lesson writes the
            # default), so what the reveal panel shows is what Wan is actually handed. It
            # used to be invented here and thrown away — a throwaway dict copy — which is
            # why it could not be seen or changed.
            sub = {"_id": lesson_id,
                   "beats": [dict(beats[i], motion_prompt=(
                       beats[i].get("motion_prompt") or default_motion(beats[i])))
                       for i in ticked]}
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
