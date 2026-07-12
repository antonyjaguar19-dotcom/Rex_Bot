"""
Claw Bot — Facts Shorts Pipeline (orchestrator)

Renders a facts reel:
  1. Narration — kokoro voices each beat separately; the real per-beat spans give
     exact scene timing + perfectly-synced on-screen text (reuses the horror
     pipeline's span machinery).
  2. Loose mood backdrop per beat (active image backend; NO character refs — there
     is no recurring subject). Falls back to a generated gradient if the backend
     is unavailable, so the reel renders even with ComfyUI down.
  3. facts_assembly — Ken Burns + BIG centered fact text + optional music → 9x16.

Front-end agnostic (returns paths).
"""

import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modules import image_backend
from modules import gpu_utils
from modules import runtime_settings as rs
from modules import facts_assembly as fasm
from modules.assembly import ASPECTS
# Reuse the horror pipeline's kokoro per-beat narration + span helpers. Facts
# voices with its OWN bright/fast voice (not horror's deep narrator).
from modules.horrorstory_pipeline import (
    _kokoro_narrate, _load_spans, _spans_path, _scene_durations,
)


def _voice_facts(narrations: list, out_path: Path):
    """Kokoro narration in the FACTS voice (bright + slightly fast = energetic).
    Returns (wav, spans)."""
    return _kokoro_narrate(narrations, out_path,
                           voice=rs.get_facts_voice(),
                           speed=rs.get_facts_voice_speed())

log = logging.getLogger("claw_bot.facts_pipeline")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
STILLS_DIR = PROJECT_ROOT / "04_Outputs" / "storyboards"
NARRATION_DIR = PROJECT_ROOT / "04_Outputs" / "audio"

# palette cycled for gradient fallbacks (dark, punchy, readable under white text)
_GRADIENTS = [
    ("0x1a1a2e", "0x16213e"), ("0x2b1055", "0x7597de"), ("0x0f2027", "0x2c5364"),
    ("0x232526", "0x414345"), ("0x141e30", "0x243b55"), ("0x3a1c71", "0xd76d77"),
]


# S2V renders at its native window size; assembly scales each clip onto the
# final 9x16 / 16x9 / 1x1 canvas. Rendering S2V at full 1080x1920 would blow
# VRAM and time for no gain.
_S2V_DIMS = {"9x16": (480, 832), "16x9": (832, 480), "1x1": (624, 624)}


def _probe_dur(path: Path) -> float:
    try:
        out = subprocess.run(
            [str(fasm.FFMPEG_EXE).replace("ffmpeg.exe", "ffprobe.exe"),
             "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
             str(path)], capture_output=True, text=True, timeout=30).stdout.strip()
        return float(out)
    except Exception:
        return 0.0


def _trim_video(src: Path, dur: float, dst: Path) -> Path:
    """Trim a clip to exactly `dur` seconds (drops S2V's frame-rounding tail so
    the concatenated reel stays in lock-step with the narration)."""
    if dur <= 0:
        return src
    r = subprocess.run(
        [str(fasm.FFMPEG_EXE), "-y", "-loglevel", "error", "-i", str(src),
         "-t", f"{dur:.3f}", "-c:v", "libx264", "-c:a", "aac", "-pix_fmt",
         "yuv420p", str(dst)], capture_output=True, text=True, timeout=300)
    return dst if dst.exists() and r.returncode == 0 else src


def _concat_wavs(wavs: list, dst: Path) -> Path:
    lst = dst.with_suffix(".txt")
    lst.write_text("".join(f"file '{w.as_posix()}'\n" for w in wavs), encoding="utf-8")
    subprocess.run(
        [str(fasm.FFMPEG_EXE), "-y", "-loglevel", "error", "-f", "concat",
         "-safe", "0", "-i", str(lst), "-c", "copy", str(dst)],
        capture_output=True, text=True, timeout=120)
    if not dst.exists():   # copy can fail on mismatched wavs — re-encode
        subprocess.run(
            [str(fasm.FFMPEG_EXE), "-y", "-loglevel", "error", "-f", "concat",
             "-safe", "0", "-i", str(lst), "-ar", "24000", str(dst)],
            capture_output=True, text=True, timeout=120)
    return dst


def _slice_wav(src: Path, start: float, end: float, dst: Path) -> Path:
    subprocess.run(
        [str(fasm.FFMPEG_EXE), "-y", "-loglevel", "error", "-i", str(src),
         "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-c", "copy", str(dst)],
        capture_output=True, text=True, timeout=60)
    if not dst.exists():   # stream-copy can't always cut cleanly — re-encode
        subprocess.run(
            [str(fasm.FFMPEG_EXE), "-y", "-loglevel", "error", "-i", str(src),
             "-ss", f"{start:.3f}", "-to", f"{end:.3f}", str(dst)],
            capture_output=True, text=True, timeout=60)
    return dst


# Spoken-only filler, prepended to the AUDIO text; the on-screen caption keeps
# the clean fact, so the reel never shows "Hmm," across the screen.
#
# This is not only flavour — it is the FIX for a real bug. Qwen swallows the
# start of a line: "Bees are amazing" came back (transcribed) as "These are
# amazing", the B simply gone, and in a batch it sometimes eats the whole first
# word. Silence padding cannot restore a phoneme the model never produced.
#
# So the spoken line is built in three parts:
#   SACRIFICE + opener + fact
# The sacrifice is a throwaway "Mm." — Qwen eats THAT, the opener stays audible,
# and the fact is never touched. Measured on the same line: "Oh, honey never
# spoils" lost its "Oh" in a batch; "Mm. Oh, honey never spoils" kept it, with
# the "Mm." gone. If it survives, a soft thinking-sound is welcome anyway.
_SACRIFICE = "Ready."

# Burned before the real lines: Qwen's first generation after a model load loses
# its opening consonant, every time, and only ever on the first line.
_WARMUP_LINE = "Ready when you are."

_SPOKEN_SYS = (
    "You rewrite a video's narration lines the way a friendly presenter would "
    "actually SAY them out loud.\n"
    'Output ONLY valid JSON: {"lines": ["...", "..."]} — exactly one line out per '
    "line in, in the same order.\n"
    "Rules:\n"
    "- Keep every FACT, NAME and NUMBER exactly as given. Never add a claim.\n"
    "- The delivery is EVEN and calm all the way through. No line is more excited "
    "than any other.\n"
    "- At most TWO lines in the whole script may open with a small, natural "
    "lead-in (And, So, Now, But here is the thing). It has to read like the "
    "sentence was always written that way.\n"
    "- NEVER append a reaction or a sign-off to the end of a line — no 'Wow', no "
    "'Pretty cool, right?', no 'imagine that'. Those sound bolted on.\n"
    "- Never use exclamation marks. End every line with a full stop.\n"
    "- Leave most lines EXACTLY as they are. Changing nothing is a good answer.\n"
    "- Keep each line about the same length. No emoji, no stage directions."
)


_OUTRO_SYS = (
    "You write the final call-to-action line of a short facts video.\n"
    'Output ONLY valid JSON: {"outro": "..."}\n'
    "Rules:\n"
    "- It must ASK THE VIEWER TO FOLLOW, and it must be a pun or an image drawn "
    "from THIS video's topic. Never the generic 'Follow for more fun facts'.\n"
    "- One short sentence, at most 12 words. Spoken out loud, friendly.\n"
    "- No emoji, no hashtags.\n"
    "Examples:\n"
    '  bees   -> {"outro": "Follow — we have got a whole hive of facts."}\n'
    '  space  -> {"outro": "Follow for more. There is a universe of this stuff."}\n'
    '  sharks -> {"outro": "Follow, and we will keep the facts circling."}'
)


def _themed_outro(topic: str, facts: list) -> Optional[str]:
    """A follow line made of the topic, not a generic sign-off.

    Rewrites the LAST beat's narration, so the caption and the voice both carry
    it. Returns None on any failure — the pipeline keeps the writer's own outro.
    """
    if not topic.strip():
        return None
    try:
        from modules.script_generator import _call_llm, _extract_json
        prompt = (f"Topic: {topic}\n\nThe video's facts:\n"
                  + "\n".join(f"- {f}" for f in facts[:6])
                  + "\n\nWrite the closing follow line.")
        got = _extract_json(_call_llm(prompt, _OUTRO_SYS, role="creative")) or {}
        outro = str(got.get("outro") or "").strip().strip('"')
        if outro and len(outro.split()) <= 16 and "follow" in outro.lower():
            log.info(f"Themed outro: {outro}")
            return outro
        log.warning(f"outro rejected: {outro!r}")
    except Exception as e:
        log.warning(f"themed outro failed ({e})")
    return None


def _spoken_lines(narrations: list, topic: str = "") -> list:
    """The narration as a presenter would say it — fillers only where they fit.

    A rotating filler on every beat sounded exactly like what it was: a loop. The
    LLM decides where a lead-in or a reaction earns its place and leaves the rest
    of the lines alone. Falls back to the plain lines, which are always safe.
    """
    lines = [(t or "").strip() for t in narrations]
    if not any(lines):
        return lines
    try:
        from modules.script_generator import _call_llm, _extract_json
        numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(lines))
        prompt = (f"Topic: {topic or '(general)'}\n\n"
                  f"Narration lines:\n{numbered}\n\n"
                  f"Rewrite them as spoken lines.")
        got = _extract_json(_call_llm(prompt, _SPOKEN_SYS, role="creative")) or {}
        out = [str(x).strip() for x in (got.get("lines") or [])]
        if len(out) == len(lines) and all(out):
            log.info(f"Spoken rewrite: {sum(1 for a, b in zip(out, lines) if a != b)}"
                     f"/{len(lines)} lines given a natural touch")
            return out
        log.warning(f"spoken rewrite returned {len(out)} lines for {len(lines)}; "
                    f"using the plain narration")
    except Exception as e:
        log.warning(f"spoken rewrite failed ({e}); using the plain narration")
    return lines


def _speech_runs(wav: Path, thresh_db: int = -38, min_sil: float = 0.06) -> list:
    """[(start, end)] of the audible runs in a WAV, via ffmpeg silencedetect."""
    r = subprocess.run(
        [str(fasm.FFMPEG_EXE), "-i", str(wav), "-af",
         f"silencedetect=n={thresh_db}dB:d={min_sil}", "-f", "null", "-"],
        capture_output=True, text=True, timeout=60)
    starts, ends = [], []
    for line in (r.stderr or "").splitlines():
        if "silence_start:" in line:
            starts.append(float(line.split("silence_start:")[1].split()[0]))
        elif "silence_end:" in line:
            ends.append(float(line.split("silence_end:")[1].split()[0]))
    total = _probe_dur(wav)
    # speech runs = the gaps BETWEEN silences
    runs, cursor = [], 0.0
    for i, s in enumerate(starts):
        if s > cursor + 0.01:
            runs.append((cursor, s))
        cursor = ends[i] if i < len(ends) else total
    if cursor < total - 0.01:
        runs.append((cursor, total))
    return runs


# The sacrifice must PROTECT the first consonant and then be INAUDIBLE. Nobody
# starts every sentence with "Hmm". So the throwaway is spoken, then cut back out
# of the waveform: find the short blip at the head, find the pause after it, and
# start the clip there.
_SAC_MAX_SEC = 0.75      # a spoken "Mm." is short; anything longer is real speech
_SAC_LEAD_SEC = 0.20     # keep a hair of lead-in so the word's attack survives
# Qwen can draw the throwaway out into a long hum (2.1 s measured). Anything
# beyond this is not a hum, it is the sentence — refuse to cut it.
_SAC_MAX_CUT_SEC = 2.6
_FILLER_TOKENS = {"mm", "mmm", "mmmm", "hm", "hmm", "hmmm", "uh", "um", "ah",
                  "er", "eh", "oh"}


def _norm_word(w) -> str:
    return re.sub(r"[^a-z]", "", str(w).lower())


def _strip_sacrifice(src: Path, first_word: str = "") -> Path:
    """Cut the throwaway off the front, leaving the real line whole.

    Silence detection was not good enough — it clipped the "B" off "Bees" on one
    beat and left an audible "Hmm" on another. WhisperX gives WORD timings, so we
    cut exactly at the start of the line's first real word. If alignment fails, or
    the word cannot be found (Qwen often swallows the throwaway completely, which
    is fine), the audio is returned untouched: never risk eating a real word.
    """
    try:
        from modules import lyric_aligner
        ok, _ = lyric_aligner.health_check()
        if not ok:
            return src
        words = lyric_aligner._words_from_audio(src) or []
    except Exception as e:
        log.warning(f"sacrifice trim: alignment unavailable ({e})")
        return src

    if not words:
        return src

    # Cut at the first REAL word. Two traps, both hit in testing:
    #  * the aligner sometimes transcribes the hum as a word ("Mmm") — so filler
    #    tokens are skipped rather than trusted as the start of the line;
    #  * it sometimes misaligns and points into the middle of the sentence — so
    #    the cut is capped, and we refuse to drop most of the line.
    # The carrier is a real word ("Ready."), so the aligner reports it. Cut at the
    # word AFTER it. A hum ("Mm") was unreliable: Qwen swallowed it about half the
    # time, and when it did, the sentence's own first consonant was eaten instead.
    sac = _norm_word(_SACRIFICE)
    idx = 0
    while idx < len(words) and _norm_word(words[idx][0]) in (_FILLER_TOKENS | {sac}):
        idx += 1
    if idx == 0 or idx >= len(words) or len(words) - idx < 2:
        return src                       # carrier not found, or nothing real left

    cut_at = float(words[idx][1]) - _SAC_LEAD_SEC
    if cut_at <= 0.05:
        return src                       # throwaway already swallowed by Qwen
    if cut_at > _SAC_MAX_CUT_SEC:
        log.warning(f"sacrifice trim: {cut_at:.2f}s is too deep to be a throwaway; "
                    f"leaving {src.name} alone")
        return src
    if cut_at >= _probe_dur(src) - 0.3:
        return src

    dst = src.with_name(src.stem + "_cut.wav")
    r = subprocess.run(
        [str(fasm.FFMPEG_EXE), "-y", "-loglevel", "error", "-ss", f"{cut_at:.3f}",
         "-i", str(src), str(dst)],
        capture_output=True, text=True, timeout=60)
    if not dst.exists() or r.returncode != 0:
        log.warning("sacrifice trim failed; keeping the head")
        return src
    dst.replace(src)
    return src


def _even_tone(text: str) -> str:
    """Flatten the punctuation that makes Qwen shout.

    Every beat is a separate generation, so its prosody is read off its own
    punctuation: a line ending in "!" came out over-excited while the next line
    was calm, and the reel lurched between the two. Full stops keep the delivery
    level. (The caption keeps the writer's original punctuation.)
    """
    text = (text or "").strip()
    if not text:
        return text
    text = text.replace("!", ".").replace("?!", "?")
    while ".." in text:
        text = text.replace("..", ".")
    return text


def _natural_speech(text: str) -> str:
    """The exact string handed to the TTS: throwaway + an even-toned line."""
    text = _even_tone(text)
    return f"{_SACRIFICE} {text}" if text else text


# Qwen starts the waveform on the first phoneme — no lead-in at all — so the very
# first consonant gets eaten ("Bees" came out "ees"). Pad silence at both ends:
# the head gives the word room to start, the tail gives the line room to land.
PAD_HEAD_SEC = 0.30
PAD_TAIL_SEC = 0.40


def _pad_wav(src: Path, head: float = PAD_HEAD_SEC, tail: float = PAD_TAIL_SEC) -> Path:
    """Add silent lead-in / lead-out so no syllable is clipped."""
    dst = src.with_name(src.stem + "_pad.wav")
    r = subprocess.run(
        [str(fasm.FFMPEG_EXE), "-y", "-loglevel", "error", "-i", str(src),
         "-af", f"adelay={int(head*1000)}:all=1,apad=pad_dur={tail:.3f}",
         str(dst)],
        capture_output=True, text=True, timeout=60)
    if not dst.exists() or r.returncode != 0:
        log.warning(f"pad failed ({r.stderr[:120]}); keeping original")
        return src
    dst.replace(src)
    return src


def _pitch_wav(src: Path, semitones: float, sr: int = 24000) -> Path:
    """Raise the pitch without changing the duration.

    No local TTS ships a child voice, so the mascot is an adult male (Eric)
    lifted a couple of semitones — asetrate does the pitch, atempo puts the
    tempo back where it was.
    """
    if abs(semitones) < 0.05:
        return src
    f = 2 ** (semitones / 12.0)
    dst = src.with_name(src.stem + "_pit.wav")
    r = subprocess.run(
        [str(fasm.FFMPEG_EXE), "-y", "-loglevel", "error", "-i", str(src),
         "-af", f"asetrate={sr}*{f:.5f},aresample={sr},atempo={1/f:.5f}",
         str(dst)],
        capture_output=True, text=True, timeout=60)
    if not dst.exists() or r.returncode != 0:
        log.warning(f"pitch shift failed ({r.stderr[:120]}); keeping original")
        return src
    dst.replace(src)
    return src


def _speed_wav(src: Path, speed: float) -> Path:
    """Speed the narration up without shifting pitch (ffmpeg atempo)."""
    if abs(speed - 1.0) < 0.01:
        return src
    dst = src.with_name(src.stem + "_spd.wav")
    r = subprocess.run(
        [str(fasm.FFMPEG_EXE), "-y", "-loglevel", "error", "-i", str(src),
         "-filter:a", f"atempo={speed:.3f}", str(dst)],
        capture_output=True, text=True, timeout=60)
    if not dst.exists() or r.returncode != 0:
        log.warning(f"speed change failed ({r.stderr[:120]}); keeping original")
        return src
    dst.replace(src)          # keep the caller's filename
    return src


# Past this, a sped-up voice stops sounding like a presenter and starts sounding
# like a chipmunk. If the beats cannot fit under the ceiling at this pace, the
# script is too long — say so out loud rather than shipping a mangled read.
_FIT_MAX_SPEED = 1.45


def _fit_to_budget(wavs: list, _p) -> list:
    """Trim the narration's PACE until the reel fits its ceiling.

    Measured, not assumed: the cloned voice read at 1.78 words/sec (a presenter
    reads at 2.5-3), which is how a 113-word script became a 63s reel. atempo is
    pitch-preserving, so speeding it up changes the pace WITHOUT touching the
    cloned timbre — the thing the reference actually decides.
    """
    budget = rs.get_facts_max_seconds()
    total = sum(_probe_dur(w) for w in wavs)
    if total <= budget:
        _p(f"⏱️ narration {total:.1f}s — under the {budget:.0f}s ceiling")
        return wavs

    factor = total / budget
    if factor > _FIT_MAX_SPEED:
        _p(f"⚠️ narration {total:.1f}s needs {factor:.2f}x to fit {budget:.0f}s; "
           f"capping at {_FIT_MAX_SPEED}x — the reel WILL run over. Write shorter "
           f"lines or fewer facts.")
        factor = _FIT_MAX_SPEED

    for w in wavs:
        _speed_wav(w, factor)
    now = sum(_probe_dur(w) for w in wavs)
    _p(f"⏱️ narration {total:.1f}s → {now:.1f}s ({factor:.2f}x) "
       f"to fit the {budget:.0f}s ceiling")
    return wavs


def _voice_beats_clone(narrations: list, out_dir: Path, _p,
                       spoken: Optional[list] = None) -> Optional[list]:
    """One WAV per fact, CLONED from a reference clip (Chatterbox).

    No local TTS ships a genuine child voice — Eric lifted +2 semitones is an adult
    timbre in a costume, and it never read as cute. Cloning a real recording is the
    only route to the mascot actually sounding like the mascot.

    Timbre AND pace come from the reference, so the speed/pitch knobs are NOT
    applied here: shifting a clone undoes the cloning.
    """
    from modules import tts_chatterbox, gpu_memory

    ref = rs.get_mascot_voice_ref()
    if not ref:
        _p("⚠️ no mascot voice reference clip on disk; using qwen.")
        return None
    ok, msg = tts_chatterbox.health_check()
    if not ok:
        _p(f"⚠️ chatterbox bridge down ({msg}); using qwen.")
        return None

    _p(f"🎙️ mascot voice: chatterbox clone / {ref.name}")
    # Chatterbox wants the card. Anything ComfyUI is still holding will OOM it.
    gpu_memory.evict_all()

    base = spoken if spoken and len(spoken) == len(narrations) else narrations
    # NO carrier word here. The "Ready." throwaway exists to absorb Qwen's eaten
    # opening consonant; Chatterbox does not have that bug (verified by
    # transcribing carrier-less output — every first word came back intact). And
    # the carrier is not free: cutting it back off relies on ASR finding it, which
    # missed often enough that "Ready, one hive can make sixty kilos" shipped.
    texts = [_even_tone(t) for t in base]
    segs = tts_chatterbox.synthesize_each(
        texts, out_dir / "voice", ref_wav=ref,
        exaggeration=rs.get_mascot_voice_exaggeration(),
    )

    # Pace IS adjusted here; timbre is NOT. The clone reads at ~1.75 words/sec
    # against a presenter's 2.5-3 — sluggish however short the script is, so a word
    # budget alone does not fix it. atempo is pitch-preserving: it changes the pace
    # without touching the cloned voice. (Pitch shifting WOULD undo the clone, and
    # is never applied on this path.)
    speed = rs.get_mascot_voice_speed()
    wavs = []
    for i, seg in enumerate(segs):
        wp = out_dir / f"beat_{i:02d}.wav"
        shutil.copy2(seg, wp)
        _speed_wav(wp, speed)     # pad LAST — padding first would be shrunk by this
        wavs.append(_pad_wav(wp))
    return wavs


def _voice_beats_mascot(narrations: list, out_dir: Path, _p,
                        spoken: Optional[list] = None) -> list:
    """One WAV per fact, in the mascot's voice.

    Qwen3-TTS is called ONCE for all the beats (the model loads in an isolated
    venv subprocess — per-beat calls would reload it every time) and the master
    is sliced back apart on the returned spans. Kokoro is the fallback so a dead
    bridge never blocks a render.
    """
    engine = rs.get_mascot_tts_engine()

    if engine == "chatterbox":
        try:
            wavs = _voice_beats_clone(narrations, out_dir, _p, spoken=spoken)
            if wavs:
                # Pace has ONE owner: the budget. The clone reads slow (1.78 w/s
                # measured, against a presenter's 2.5-3), so the fit pass is what
                # brings it up to speed — and it does it by measuring, not guessing.
                return _fit_to_budget(wavs, _p)
        except Exception as e:
            _p(f"⚠️ mascot voice (chatterbox) failed ({e}); using qwen.")
        engine = "qwen"

    if engine == "qwen":
        try:
            from modules import tts_qwen
            ok, msg = tts_qwen.health_check()
            if not ok:
                raise RuntimeError(msg)
            speaker = rs.get_mascot_voice()
            _p(f"🎙️ mascot voice: qwen3-tts / {speaker}")
            # Qwen3-TTS wants the card too. Whatever ComfyUI is still holding
            # (S2V, Qwen-Edit) will OOM it — measured, it fell straight back to
            # Kokoro. Evict first; the image stage reloads warm right after.
            from modules import gpu_memory
            gpu_memory.evict_all()
            # The LLM already decided where a filler fits (spoken); all we add
            # is the throwaway Qwen is going to eat.
            base = spoken if spoken and len(spoken) == len(narrations) else narrations
            texts = [_natural_speech(t) for t in base]
            # One WAV per beat, straight from the model — NOT a slice of a
            # concatenated master. Cutting the master on spans clipped words at
            # the seams and the narration ended abruptly.
            # Qwen's FIRST generation after a model load comes out damaged — the
            # opening consonant is eaten ("Bees" -> "These"). It was ALWAYS beat 0,
            # never a later beat. So burn one throwaway line first and drop it.
            segs = tts_qwen.synthesize_each(
                [_WARMUP_LINE] + texts, out_dir / "voice", speaker=speaker,
                instruct=rs.get_mascot_voice_instruct())[1:]
            speed = rs.get_mascot_voice_speed()
            pitch = rs.get_mascot_voice_pitch()
            wavs = []
            for i, seg in enumerate(segs):
                wp = out_dir / f"beat_{i:02d}.wav"
                shutil.copy2(seg, wp)
                # Cut the throwaway off FIRST (while the pause after it is still
                # its natural length), then speed, then pitch, then pad LAST —
                # padding earlier would just get shrunk by the tempo filters.
                first = (base[i].strip().split() or [""])[0]
                _strip_sacrifice(wp, first)
                _speed_wav(wp, speed)
                _pitch_wav(wp, pitch)
                wavs.append(_pad_wav(wp))
            return _fit_to_budget(wavs, _p)
        except Exception as e:
            _p(f"⚠️ mascot voice (qwen) failed ({e}); using kokoro.")

    from modules.tts_engine import TTSEngine
    tts = TTSEngine()
    voice = rs.get_facts_voice()
    wavs = []
    for i, text in enumerate(narrations):
        wp = out_dir / f"beat_{i:02d}.wav"
        tts.synthesize(text, output_path=wp, voice=voice)
        wavs.append(wp)
    return _fit_to_budget(wavs, _p)


def _render_facts_mascot(story, beats, aspect, music_path, _p, facts_id):
    """Every fact presented by the costumed mascot, lip-synced (Qwen still → S2V).

    VRAM discipline (one big model at a time on a 16 GB card):
      1. LLM writes all scenes (Ollama), then unloads.
      2. Qwen-Edit renders all mascot stills warm, then releases.
      3. S2V renders all talking clips warm, then releases.
      4. ffmpeg concat + music + captions (no GPU).
    """
    from modules import mascot, gpu_memory, video_backend as vb, model_registry as mr
    from modules.tts_engine import TTSEngine

    ok, why = mascot.is_available()
    if not ok:
        _p(f"mascot mode off: {why}")
        return None
    cfg = mr.get_available("video_backend", "comfyui_wan22_s2v")
    if rs.get_facts_mascot_lipsync() and not cfg:
        _p("lip-sync requested but the S2V backend is not registered; "
           "falling back to I2V + voice-over.")
        rs.set_facts_mascot_lipsync(False)

    topic = story.get("topic", "") or story.get("title", "")
    out_dir = STILLS_DIR / f"facts_{facts_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sw, sh = _S2V_DIMS.get(aspect, _S2V_DIMS["9x16"])
    ar = aspect.replace("x", ":")

    # 1. Scenes (Ollama). A user-edited mascot_scene wins; otherwise the LLM
    #    writes one and we STORE it back so it can be edited next time. Written up
    #    front so the LLM unloads before Qwen loads.
    from modules import facts_writer as _fw
    _p(f"🎭 writing {len(beats)} mascot scenes…")
    narrations = [b.get("narration", "") for b in beats]
    scenes = [(b.get("mascot_scene") or "").strip() for b in beats]
    spoken = None
    with gpu_memory.llm():
        for i, b in enumerate(beats):
            if not scenes[i]:
                scenes[i] = mascot.explainer_scene(b.get("narration", ""), topic)
        # Replace the generic "Follow for more fun facts" with one made of THIS
        # topic. It rewrites the beat itself, so the caption and the voice agree.
        outro = _themed_outro(topic, narrations[:-1])
        if outro and len(beats) > 1:
            beats[-1]["narration"] = outro
            narrations[-1] = outro
            _p(f"👋 outro: {outro}")

        # Same Ollama residency: turn the written lines into SPOKEN lines while
        # the model is already up. On-screen captions keep the clean narration.
        _p("🗣️ making the narration sound spoken…")
        spoken = _spoken_lines(narrations, topic)
    for i, sc in enumerate(scenes):       # persist for editing / reuse
        try:
            _fw.set_beat_prompt(facts_id, i, "mascot_scene", sc)
        except Exception:
            pass

    # 2. Per-beat narration WAVs. The mascot is a young jaguar cub — Kokoro reads
    #    it flat and pitch-shifting it sounds artificial, so the mascot gets
    #    Qwen3-TTS with an emotion instruct (natural, expressive, deterministic =
    #    the same voice every clip). Falls back to Kokoro if the bridge is down.
    _p("🎙️ voicing each fact…")
    wavs = _voice_beats_mascot(narrations, out_dir, _p, spoken=spoken)

    # 3. Mascot stills via Qwen-Edit (identity held), warm across all beats.
    _p(f"🖼️ rendering {len(beats)} mascot stills (Qwen-Edit)…")
    stills = []
    gpu_memory.acquire(gpu_memory.QWEN_EDIT)
    try:
        for i, sc in enumerate(scenes):
            sp = out_dir / f"still_{i:02d}.png"
            # presenter=True → waist-up framing. These stills feed Wan S2V, which
            # is a talking-head model: given legs it cannot animate, it deforms
            # them (measured — a leg bent backwards by the end of the clip).
            got = mascot.render_scene(sc, sp, aspect=aspect, seed=1000 + i,
                                      presenter=True)
            stills.append(got or _gradient_bg(i, *ASPECTS.get(aspect, ASPECTS["9x16"]),
                                              out_dir / f"still_{i:02d}.png"))
    finally:
        gpu_memory.release(gpu_memory.QWEN_EDIT)

    # 4. Animate each still.
    #
    # DEFAULT is the ordinary I2V backend with the narration as a VOICE-OVER.
    # Lip-sync (S2V) is opt-in because S2V is a talking-head model: it syncs the
    # mouth well but has no idea what hands, props or legs are, and it dissolved
    # them — a paw melted mid-clip, a leg bent backwards. It is also ~4 min a clip.
    # A presenter reel does not need a moving mouth; it needs clean motion.
    clips = []
    if rs.get_facts_mascot_lipsync():
        _p(f"🎬 animating {len(beats)} talking clips (Wan S2V, lip-sync)…")
        cfg = dict(cfg); cfg["_id"] = "comfyui_wan22_s2v"
        s2v = vb.build_backend(cfg)
        ok, msg = s2v.health_check()
        if not ok:
            _p(f"mascot mode off: S2V unhealthy ({msg}).")
            return None
        gpu_memory.acquire(gpu_memory.WAN_VIDEO)
        try:
            for i, b in enumerate(beats):
                raw = s2v.generate(
                    prompt=scenes[i], input_image=Path(stills[i]),
                    audio_path=wavs[i], aspect_ratio=ar, width=sw, height=sh,
                    output_filename=f"s2v_{facts_id}_{i:02d}.mp4", seed=2000 + i)
                clips.append(_trim_video(Path(raw), _probe_dur(wavs[i]),
                                         out_dir / f"clip_{i:02d}.mp4"))
                _p(f"  clip {i+1}/{len(beats)} done")
                # Back-to-back S2V renders fragment ComfyUI's VRAM until its weight
                # page-fault handler gives up ("Fault failed: 2" — died on clip 4,
                # twice). Drop the cache but keep S2V loaded: a full unload would
                # cost a ~4 min cold reload per clip.
                gpu_utils.soft_free_comfyui_cache()
        finally:
            gpu_memory.release(gpu_memory.WAN_VIDEO)
    else:
        _p(f"🎬 animating {len(beats)} clips (Wan I2V, narration as voice-over)…")
        from modules import horror_video
        durations = [_probe_dur(w) for w in wavs]
        for i, b in enumerate(beats):
            b["motion_prompt"] = (
                b.get("motion_prompt")
                or f"{scenes[i]}, the character moves naturally, gentle lively "
                   f"gestures, subtle camera push-in, no text")
        gpu_memory.acquire(gpu_memory.WAN_VIDEO)
        try:
            clips = list(horror_video.render_shot_clips(
                story, [Path(s) for s in stills], durations,
                aspect_ratio=ar, progress_cb=_p, fill_mode="retime"))
        finally:
            gpu_memory.release(gpu_memory.WAN_VIDEO)

    # 4b. Optional 4x upscale of each talking clip (Real-ESRGAN anime + polish),
    #     at the clip's native 480p where the detail gain is real. ComfyUI is free
    #     now (S2V released). Best-effort — a failed upscale keeps the original.
    if rs.get_upscale_enabled():
        from modules import upscaler
        _p(f"🔎 upscaling {len(clips)} clips (4x)…")
        for i, c in enumerate(clips):
            try:
                upscaler.upscale_clip(Path(c))
                _p(f"  upscaled {i+1}/{len(clips)}")
            except Exception as e:
                _p(f"  upscale {i+1} failed ({e}); keeping original")

    # 5. Narration = the exact per-beat WAVs concatenated, so the track the
    #    assembler overlays matches each clip's own audio frame-for-frame.
    narration = _concat_wavs(wavs, out_dir / "narration.wav")

    # 5b. The bed is composed LAST, to the reel's real length, so ACE-Step can end
    #     it on an outro instead of us chopping a loop wherever the reel stops.
    #     The GPU is free here — the video models have released it.
    if music_path is None:
        music_path = _music_bed(facts_id, _probe_dur(narration), _p)

    # 6. Concat clips + big captions (timed to real clip durations) + music.
    _p("🎬 assembling mascot reel…")
    return fasm.assemble_facts_clips(story, narration, clips, aspect=aspect,
                                     music_path=music_path, progress_cb=_p)


def _gradient_bg(idx: int, w: int, h: int, out_path: Path) -> Path:
    """Generate one gradient still (pure ffmpeg) — image-independent fallback."""
    c0, c1 = _GRADIENTS[idx % len(_GRADIENTS)]
    cmd = [
        str(fasm.FFMPEG_EXE), "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"gradients=s={w}x{h}:c0={c0}:c1={c1}:x0=0:y0=0:x1={w}:y1={h}",
        "-frames:v", "1", str(out_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0 or not out_path.exists():
        # last-ditch: flat colour
        subprocess.run([str(fasm.FFMPEG_EXE), "-y", "-loglevel", "error",
                        "-f", "lavfi", "-i", f"color=c={c0}:s={w}x{h}",
                        "-frames:v", "1", str(out_path)], timeout=120)
    return out_path


def narrate_facts(story: dict, progress_cb: Optional[Callable[[str], None]] = None) -> Path:
    """Voice the whole reel (kokoro per-beat) + write the spans sidecar. Returns wav."""
    facts_id = story.get("facts_id") or story.get("_id")
    beats = story.get("beats", [])
    if not beats:
        raise ValueError("Facts reel has no beats.")
    gpu_utils.ensure_vram_free()
    NARRATION_DIR.mkdir(parents=True, exist_ok=True)
    out = NARRATION_DIR / f"facts_{facts_id}.wav"
    out, spans = _voice_facts([b.get("narration", "") for b in beats], out)
    if spans:
        _spans_path(out).write_text(__import__("json").dumps(spans), encoding="utf-8")
    if progress_cb:
        progress_cb(f"narration ready (kokoro:{rs.get_facts_voice()})")
    return out


# What ACE-Step is told to write. The [outro] section is the whole point: the track
# has to LAND, not stop. MusicGen only ever gives us a 30s loop that gets chopped
# wherever the reel happens to end — which is the sudden cut.
_MOOD_TAGS = {
    "cheerful":    "upbeat playful cartoon music, bright ukulele and glockenspiel, "
                   "bouncy, sunny, family friendly",
    "calm":        "gentle soft lullaby, warm piano and strings, slow and peaceful",
    "adventurous": "adventurous orchestral, driving drums, brass, playful heroic",
    "tender":      "tender warm acoustic guitar, soft and sweet, heartfelt",
    "magical":     "magical whimsical, celesta and harp, sparkling, wondrous",
    "energetic":   "energetic upbeat pop, punchy drums, lively and fun",
}
_OUTRO_LYRICS = "[inst]\n[intro]\n[verse]\n[outro]"


def _music_bed(facts_id: str, duration_sec: float, _p) -> Optional[Path]:
    """Write the background bed, or None.

    ACE-Step FIRST, because it composes a complete piece of a REQUESTED LENGTH —
    it can be told to end on an [outro], so the music LANDS on the last frame. That
    is why the duration is passed in: it is the reel's real length, known only once
    the narration is voiced.

    MusicGen is the fallback, and it is a fallback for a reason: it maxes out at a
    30s take which then has to be looped and chopped wherever the reel ends — which
    is exactly the sudden cut this replaced.

    Best-effort by design: a dead music model must not cost a 45-minute render, so it
    degrades to no bed — but it SAYS so. Silence nobody announced is how a reel ships
    wrong.
    """
    if not rs.get_facts_music_enabled():
        return None

    mood = rs.get_facts_music_mood()
    reel = max(float(duration_sec), 8.0)
    # Ask for MORE than the reel needs. ACE-Step composes a piece shorter than the
    # length requested and pads the rest with silence — asked for 34.4s, it wrote 27s
    # of music and 7.4s of nothing, so the last fact played over dead air. With
    # headroom the music covers the reel, the padding is trimmed off, and the piece's
    # real ending is then anchored to the final frame.
    ask = reel * 1.35

    try:
        from modules import audio_backend, gpu_memory
        ab = audio_backend.get_active_backend()
        ok, why = ab.health_check()
        if not ok:
            raise RuntimeError(why)
        _p(f"🎵 composing the music bed ({mood}, {reel:.0f}s, ACE-Step)…")
        gpu_memory.evict_all()
        track = ab.generate(
            tags=_MOOD_TAGS.get(mood, _MOOD_TAGS["cheerful"]) + ", instrumental, no vocals",
            lyrics=_OUTRO_LYRICS,
            duration_sec=ask,
            output_filename=f"facts_bed_{facts_id}.mp3",
        )
        if track and Path(track).exists():
            track = fasm.trim_trailing_silence(Path(track))
            got = _probe_dur(track)
            if got < reel - 0.5:
                _p(f"⚠️ the bed is only {got:.1f}s of music for a {reel:.1f}s reel — "
                   f"it will start late so its ending still lands on the last frame")
            _p(f"🎵 music bed: {Path(track).name} ({got:.1f}s, ends on an outro)")
            return Path(track)
        raise RuntimeError("ACE-Step produced no file")
    except Exception as e:
        _p(f"⚠️ ACE-Step bed failed ({e}); falling back to MusicGen (it will fade out)")

    try:
        from modules import music_generator as mg, gpu_memory
        gpu_memory.evict_all()
        track = mg.generate_music(
            mood=mood, duration_sec=dur, script_id=facts_id,
            progress_cb=lambda m: log.info(f"music: {m}"),
        )
        if not track:
            _p("⚠️ no music track produced; continuing without a bed")
            return None
        _p(f"🎵 music bed: {Path(track).name}")
        return Path(track)
    except Exception as e:
        _p(f"⚠️ music failed ({e}); continuing without a bed")
        return None


def render_facts(
    story: dict,
    progress_cb: Optional[Callable[[str], None]] = None,
    narration_path: Optional[Path] = None,
    aspect: str = "9x16",
    music_path: Optional[Path] = None,
    animate: Optional[bool] = None,
) -> dict:
    """Full render: kokoro narration -> per-beat spans -> mood backdrops -> 9x16
    Ken Burns + big centered fact text."""
    def _p(msg: str):
        log.info(msg)
        if progress_cb:
            try:
                progress_cb(msg)
            except Exception:
                pass

    facts_id = story.get("facts_id") or story.get("_id")
    beats = story.get("beats", [])
    if not beats:
        raise ValueError("Facts reel has no beats.")

    # NOTE: the music bed is NOT written here. It is composed once the narration
    # exists, because ACE-Step needs the reel's real length to end the track on an
    # outro. Generating it up front meant guessing the length, and a guessed length
    # is what forced the old loop-and-chop ending.

    # Mascot mode: the mascot presents every fact in costume. Falls back to the
    # normal abstract-backdrop path if the mascot is unavailable, so enabling it
    # never breaks a render.
    if rs.get_facts_mascot_mode():
        try:
            out = _render_facts_mascot(story, beats, aspect, music_path, _p, facts_id)
            if out:
                # This return used to skip the publish kit entirely — a mascot
                # reel shipped with no title and no thumbnail beside it.
                stills = sorted((STILLS_DIR / f"facts_{facts_id}").glob("still_*.png"))
                out.update(_attach_publish_kit(story, out, stills))
                return out
            _p("mascot mode unavailable; using abstract backdrops.")
        except Exception as e:
            log.exception("mascot facts render failed")
            _p(f"⚠️ mascot mode failed ({e}); falling back to abstract backdrops.")

    import json as _json
    gpu_utils.ensure_vram_free()
    NARRATION_DIR.mkdir(parents=True, exist_ok=True)
    spans = None
    if narration_path and Path(narration_path).exists():
        narration_path = Path(narration_path)
        spans = _load_spans(narration_path)
        _p("🎙️ using pre-rendered narration" + (" (+spans)" if spans else ""))
    else:
        narration_path = NARRATION_DIR / f"facts_{facts_id}.wav"
        _p(f"🎙️ voicing {len(beats)} beats (kokoro:{rs.get_facts_voice()})...")
        narration_path, spans = _voice_facts(
            [b.get("narration", "") for b in beats], narration_path)
        if spans:
            _spans_path(narration_path).write_text(_json.dumps(spans), encoding="utf-8")
    total_dur = fasm._probe_duration(narration_path)
    _p(f"🎙️ narration ready: {total_dur:.1f}s")

    # Per-beat windows from the real voice spans (exact) — text flips on the voice.
    if spans and len(spans) == len(beats):
        durations = _scene_durations(spans, total_dur)
        _p(f"🎯 {len(durations)} windows from real per-beat spans")
    else:
        wsum = sum(max(1, len(b.get("narration", "").split())) for b in beats) or len(beats)
        durations = [round(max(1, len(b.get("narration", "").split())) / wsum * total_dur, 3)
                     for b in beats]

    # ---- backdrops: active backend (loose, no refs) or gradient fallback ----
    w, h = ASPECTS.get(aspect, ASPECTS["9x16"])
    out_dir = STILLS_DIR / f"facts_{facts_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ar = aspect.replace("x", ":")

    # Facts backdrops follow the conceptual/comic prompts (a bee in a chef hat),
    # so prefer Qwen-Image-Edit — it obeys the concept where Z-Image renders a
    # generic pretty photo. Fall through: Qwen -> Z-Image Turbo -> active -> gradient.
    from modules import gpu_memory
    _GPU_LABEL = {"comfyui_qwen_edit": gpu_memory.QWEN_EDIT,
                  "comfyui_zimage_turbo": gpu_memory.ZIMAGE}
    backend = None
    backend_label = None
    for _bid in ("comfyui_qwen_edit", "comfyui_zimage_turbo", None):
        try:
            b = (image_backend.get_named_backend(_bid) if _bid
                 else image_backend.get_active_backend())
            ok, _msg = b.health_check()
            if ok:
                backend = b
                backend_label = _GPU_LABEL.get(_bid, gpu_memory.FLUX_USO)
                _p(f"backdrops via {_bid or 'active backend'}")
                break
            _p(f"{_bid or 'active'} unhealthy ({_msg}); trying next…")
        except Exception as e:
            _p(f"{_bid or 'active'} unavailable ({e}); trying next…")
    if backend is None:
        _p("no image backend healthy; using gradient backdrops.")

    # Clear the card BEFORE loading the image model. Story-gen leaves Ollama
    # resident (~12.6 GB); Qwen is ~13.5 GB and the two collide on a 16 GB card,
    # which crashed the CUDA context in testing. acquire() evicts Ollama + frees
    # whatever ComfyUI held (Wan from a prior reel).
    if backend is not None:
        gpu_memory.acquire(backend_label)

    _p(f"🖼️ building {len(beats)} backdrops ({'backend' if backend else 'gradient'})...")
    backgrounds: list[Path] = []
    _reused = 0
    for i, b in enumerate(beats):
        dst = out_dir / f"bg_{i:04d}.png"
        # Reuse an already-rendered backdrop for this reel (same facts_id) — skips
        # re-paying image gen and sidesteps occasional USO hangs when iterating on
        # the video stage. Delete the bg_*.png files to force a fresh render.
        if dst.exists() and dst.stat().st_size > 0:
            backgrounds.append(dst); _reused += 1; continue
        if backend is not None:
            try:
                p = backend.generate(
                    prompt=b.get("image_prompt", "abstract cinematic background, no text"),
                    negative_prompt="text, words, letters, captions, subtitles, watermark, logo, "
                                    "blurry, low quality, distorted",
                    aspect_ratio=ar,
                    output_filename=f"facts_{facts_id}/bg_{i:04d}.png",
                )
                backgrounds.append(Path(p)); continue
            except Exception as e:
                # A backend hang/timeout usually means ComfyUI is stuck — don't keep
                # retrying it for every remaining beat (7×300s); drop to gradients.
                _p(f"backdrop {i+1} backend failed ({e}); using gradients from here.")
                backend = None
        backgrounds.append(_gradient_bg(i, w, h, dst))

    if _reused:
        _p(f"♻️ reused {_reused} existing backdrop(s)")

    # Backdrops done — hand the card back so Wan (also ~13.6 GB) starts clean.
    if backend is not None:
        gpu_memory.release(backend_label)

    # ---- music bed, composed to the narration's real length (see _music_bed) ----
    if music_path is None:
        music_path = _music_bed(facts_id, _probe_dur(narration_path), _p)

    # ---- assemble: animate each cut (Wan I2V) OR Ken Burns stills ----
    want_wan = (rs.get_facts_video_mode() == "wan") if animate is None else animate
    out = None
    if want_wan:
        _p("🎞️ animating each cut (Wan I2V)…")
        try:
            from modules import horror_video
            # facts beats carry no motion prompt — give each a gentle, upbeat move.
            for b in beats:
                b["motion_prompt"] = (b.get("motion_prompt")
                                      or "subtle cinematic motion, gentle slow push-in, "
                                         "the scene softly comes alive, no text")
            clips = horror_video.render_shot_clips(
                story, backgrounds, durations, aspect_ratio=ar, progress_cb=progress_cb,
                fill_mode="retime")
            gpu_utils.free_comfyui_vram()
            out = fasm.assemble_facts_clips(story, narration_path, clips, aspect=aspect,
                                            music_path=music_path, progress_cb=progress_cb)
        except Exception as e:
            _p(f"⚠️ Wan animation failed ({e}); falling back to Ken Burns stills.")
            out = None

    if out is None:
        _p("🎬 assembling facts reel (Ken Burns)…")
        gpu_utils.free_comfyui_vram()
        out = fasm.assemble_facts(story, narration_path, durations, backgrounds,
                                  aspect=aspect, music_path=music_path,
                                  progress_cb=progress_cb)
    out["narration_audio"] = narration_path
    # Save an upload-ready description (title + facts + hashtags) next to the reel.
    desc = (story.get("description") or "").strip()
    if desc:
        try:
            from modules.assembly import FINAL_DIR
            dfile = FINAL_DIR / f"facts_{facts_id}_description.txt"
            dfile.write_text(desc, encoding="utf-8")
            out["description"] = desc
            out["description_file"] = str(dfile)
            _p(f"📝 description saved: {dfile.name}")
        except Exception as e:
            log.warning(f"description save failed: {e}")

    # Upload kit: a pasteable title + a thumbnail you can set. Built from the
    # second backdrop (the first is the hook card) because the RENDERED reel has
    # read-along captions burned in. Never allowed to fail the render.
    _p("🖼️ building title + thumbnail…")
    out.update(_attach_publish_kit(story, out, backgrounds))
    _p("✅ facts reel complete")
    return out


def _attach_publish_kit(story: dict, out: dict, backgrounds: list) -> dict:
    """Title + thumbnails beside the finished reel. Best-effort, never raises."""
    try:
        from modules import publish_kit
        video = out.get("9x16") or out.get("16x9")
        if not video:
            return {}
        still = None
        if backgrounds:
            still = Path(backgrounds[1] if len(backgrounds) > 1 else backgrounds[0])
        context = "\n".join(
            b.get("narration", "") for b in story.get("beats", [])
        )[:1500]
        from modules import runtime_settings as rs
        kit = publish_kit.attach(
            Path(video),
            fallback_title=story.get("title", "Facts"),
            context=context,
            description=(story.get("description") or "").strip(),
            mode="facts short (true facts, fast cuts)",
            source_image=still,
            thumbnail=rs.get_facts_thumbnail_enabled(),
        )
        thumb = kit.get("thumb_9x16") or kit.get("thumb_16x9")

        # Hold the thumbnail on the FRONT of the reel. Shorts custom thumbnails are
        # not offered in every region, and where they are not the platform grabs the
        # first frame — so the thumbnail has to BE the first frame. A generated
        # thumbnail nobody can upload is just a nice file on disk.
        hold = rs.get_facts_thumb_hold_sec()
        if thumb and hold > 0:
            for key in ("9x16", "16x9", "1x1"):
                v = out.get(key)
                if v and Path(v).exists():
                    fasm.prepend_still(Path(v), Path(thumb), hold)
            out["duration"] = _probe_dur(Path(video))

        return {"publish": kit, "title": kit.get("title"), "thumbnail": thumb,
                "thumb_held_sec": hold if thumb else 0.0,
                "duration": out.get("duration")}
    except Exception as e:
        log.warning(f"publish kit skipped: {e}")
        return {}
