"""
Claw Bot — Narration Segmenter (audio-first pipeline)

Turns a rendered voiceover + its per-character timestamps into SEGMENTS — the
shots of the film. The voiceover is the source of truth: cuts land in the
natural pauses of the speech, so the visual rhythm follows the spoken rhythm
(the whole point of the audio-first rebuild).

Two inputs supported:
  1. segment_by_alignment(...)  — ElevenLabs character alignment (preferred,
     exact). The primary path.
  2. segment_by_silence(wav)    — ffmpeg silencedetect on the wav, for engines
     with no native timestamps (Chatterbox fallback).

Each segment carries TWO spans:
  - speech span  [t_start, t_end]  : where the words actually are.
  - window span  [win_start, win_end] : the slice of the timeline this shot owns.
Windows TILE [0, total_duration] with no gaps/overlaps, and each cut sits in the
MIDDLE of the silence between two sentences. So sum(window durations) ==
total audio duration exactly — assembly just concatenates silent videos of these
lengths and lays the one master narration track on top.
"""

import logging
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

_AGENT_DIR = Path(__file__).parent.parent.resolve()
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

log = logging.getLogger("claw_bot.narration_segmenter")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
FFMPEG_EXE = PROJECT_ROOT / "00_Tools" / "ffmpeg" / "bin" / "ffmpeg.exe"

# Tunables -------------------------------------------------------------------
SENTENCE_END = set(".!?")
# A gap >= this (seconds) between two characters counts as a pause we may cut on.
PAUSE_GAP_SEC = 0.22
# Shots shorter than this get merged into a neighbour (no 0.5s flash shots).
MIN_SEG_SEC = 1.4
# Video model ceiling. Segments longer than this are split at their biggest
# internal pause so the video backend (Wan ~5s) can render them in one pass.
DEFAULT_MAX_SEG_SEC = 5.0


@dataclass
class Segment:
    index: int            # 0-based segment order
    text: str             # spoken words for this shot
    t_start: float        # speech start (s)
    t_end: float          # speech end (s)
    win_start: float      # timeline window start (s) — where this shot appears
    win_end: float        # timeline window end (s)

    @property
    def speech_dur(self) -> float:
        return max(0.0, self.t_end - self.t_start)

    @property
    def window_dur(self) -> float:
        return max(0.0, self.win_end - self.win_start)

    def to_dict(self) -> dict:
        return asdict(self)


# ==============================================================================
# PRIMARY PATH — ElevenLabs character alignment
# ==============================================================================

def _is_sentence_break(chars: list, i: int) -> bool:
    """True if a sentence ends at character index i (punctuation followed by
    end-of-text or whitespace+capital). Avoids cutting on 'Mr.' mid-name."""
    if chars[i] not in SENTENCE_END:
        return False
    # look ahead past whitespace
    j = i + 1
    n = len(chars)
    while j < n and chars[j].isspace():
        j += 1
    if j >= n:
        return True                       # end of text
    nxt = chars[j]
    return nxt.isupper() or nxt in "\"'“‘"  # next sentence / quote starts


def _raw_sentence_spans(chars: list, starts: list, ends: list) -> list[tuple[int, int]]:
    """Split character stream into [start_idx, end_idx) spans at sentence ends.
    Spans cover non-space content; leading/trailing spaces are trimmed later."""
    spans = []
    n = len(chars)
    seg_start = 0
    for i in range(n):
        if _is_sentence_break(chars, i):
            spans.append((seg_start, i + 1))
            seg_start = i + 1
    if seg_start < n:
        spans.append((seg_start, n))
    return spans


def _silence_after(chars: list, starts: list, ends: list, i: int) -> float:
    """Silence (seconds) associated with cutting AFTER character i.

    ElevenLabs gives near-contiguous timestamps (end[i] ≈ start[i+1]); real
    pauses show up as a SPACE or punctuation character with a long duration —
    not as a gap between characters. So the silence is the larger of: the
    inter-character gap, and char i's own duration when it is whitespace/punct."""
    gap = (starts[i + 1] - ends[i]) if i + 1 < len(starts) else 0.0
    own = (ends[i] - starts[i]) if (chars[i].isspace() or chars[i] in ",;:") else 0.0
    return max(gap, own)


def _biggest_internal_pause(chars: list, starts: list, ends: list,
                            lo: int, hi: int) -> Optional[int]:
    """Return the char index AFTER which the largest pause occurs within
    [lo, hi), or None if no pause >= PAUSE_GAP_SEC exists."""
    best_idx, best_gap = None, PAUSE_GAP_SEC
    for i in range(lo, hi - 1):
        sil = _silence_after(chars, starts, ends, i)
        if sil >= best_gap:
            best_gap, best_idx = sil, i
    return best_idx


def _split_long_span(span: tuple, chars: list, starts: list, ends: list,
                     max_sec: float) -> list[tuple]:
    """Recursively split a span that exceeds max_sec at its biggest internal
    pause. Falls back to the whole span if no pause is found (rare)."""
    lo, hi = span
    dur = ends[hi - 1] - starts[lo]
    if dur <= max_sec or hi - lo < 2:
        return [span]
    cut = _biggest_internal_pause(chars, starts, ends, lo, hi)
    if cut is None:
        return [span]                     # no breath to split on — keep whole
    left, right = (lo, cut + 1), (cut + 1, hi)
    return _split_long_span(left, chars, starts, ends, max_sec) + \
        _split_long_span(right, chars, starts, ends, max_sec)


def _trim_spaces(chars: list, lo: int, hi: int) -> tuple[int, int]:
    """Shrink [lo, hi) to exclude leading/trailing whitespace characters."""
    while lo < hi and chars[lo].isspace():
        lo += 1
    while hi > lo and chars[hi - 1].isspace():
        hi -= 1
    return lo, hi


def segment_by_alignment(
    characters: list,
    starts: list,
    ends: list,
    max_seg_sec: float = DEFAULT_MAX_SEG_SEC,
    min_seg_sec: float = MIN_SEG_SEC,
) -> list[Segment]:
    """Build pause-bound segments from ElevenLabs character alignment.

    Accepts an Alignment.to_dict()-style trio. Returns Segments whose windows
    tile [0, total_duration]. Cuts sit in the middle of inter-sentence silences.
    """
    if not characters or not starts or not ends:
        raise ValueError("segment_by_alignment needs non-empty alignment arrays.")
    if not (len(characters) == len(starts) == len(ends)):
        raise ValueError(
            f"alignment length mismatch: chars={len(characters)} "
            f"starts={len(starts)} ends={len(ends)}"
        )

    total_dur = float(ends[-1])

    # 1) sentence spans, then split any too-long span at its biggest pause.
    spans = _raw_sentence_spans(characters, starts, ends)
    split: list[tuple] = []
    for sp in spans:
        split.extend(_split_long_span(sp, characters, starts, ends, max_seg_sec))

    # 2) trim whitespace + drop empty spans, capture speech times.
    raw: list[tuple[int, int, float, float, str]] = []
    for lo, hi in split:
        lo, hi = _trim_spaces(characters, lo, hi)
        if hi <= lo:
            continue
        text = "".join(characters[lo:hi]).strip()
        if not text:
            continue
        raw.append((lo, hi, float(starts[lo]), float(ends[hi - 1]), text))

    if not raw:
        raise ValueError("Segmentation produced no segments.")

    # 3) merge segments shorter than min_seg_sec into the previous one (or the
    #    next, if it's the first). Keeps tiny breaths from becoming flash shots.
    merged: list[list] = []
    for item in raw:
        lo, hi, ts, te, text = item
        if merged and (te - ts) < min_seg_sec:
            p = merged[-1]
            p[1], p[3], p[4] = hi, te, (p[4] + " " + text).strip()
        elif merged and (merged[-1][3] - merged[-1][2]) < min_seg_sec:
            # previous was too short — absorb this one into it
            p = merged[-1]
            p[1], p[3], p[4] = hi, te, (p[4] + " " + text).strip()
        else:
            merged.append([lo, hi, ts, te, text])

    # 4) compute window boundaries: each cut sits in the MIDDLE of the gap
    #    between consecutive segments. Windows tile [0, total_dur].
    segs: list[Segment] = []
    n = len(merged)
    for k, (lo, hi, ts, te, text) in enumerate(merged):
        if k == 0:
            win_start = 0.0
        else:
            prev_te = merged[k - 1][3]
            win_start = (prev_te + ts) / 2.0      # midpoint of the silence
        if k == n - 1:
            win_end = total_dur
        else:
            next_ts = merged[k + 1][2]
            win_end = (te + next_ts) / 2.0
        segs.append(Segment(
            index=k, text=text,
            t_start=round(ts, 3), t_end=round(te, 3),
            win_start=round(win_start, 3), win_end=round(win_end, 3),
        ))

    log.info(
        f"Segmented {total_dur:.2f}s narration into {len(segs)} shots "
        f"(windows: {[round(s.window_dur, 2) for s in segs]})"
    )
    return segs


# ==============================================================================
# SPAN PATH — exact (text, t_start, t_end) spans from per-group synthesis
# ==============================================================================

def segment_by_spans(
    spans: list[tuple[str, float, float]],
    total_dur: Optional[float] = None,
    min_seg_sec: float = MIN_SEG_SEC,
) -> list[Segment]:
    """Build segments from EXACT per-group spans (text, t_start, t_end).

    Used by local engines voiced breath-group-by-breath-group (VoxCPM): every
    group's speech times are known precisely, so no detection/guessing. Tiny
    groups merge into the previous one. Windows tile [0, total_dur] with each cut
    at the midpoint of the silence between groups.
    """
    if not spans:
        raise ValueError("segment_by_spans needs at least one span.")
    # normalize + validate ordering
    items = [(str(t), float(a), float(b)) for (t, a, b) in spans]
    for t, a, b in items:
        if b < a:
            raise ValueError(f"span end before start: {t!r} {a}->{b}")
    if total_dur is None:
        total_dur = items[-1][2]
    total_dur = float(total_dur)

    # merge groups shorter than min_seg_sec into the previous (or the next if first)
    # each entry: [t_start, t_end, _unused, text]
    merged: list[list] = []
    for t, a, b in items:
        too_short = (b - a) < min_seg_sec
        prev_short = merged and (merged[-1][1] - merged[-1][0]) < min_seg_sec
        if merged and (too_short or prev_short):
            p = merged[-1]
            p[1] = b                                  # extend window end
            p[3] = (p[3] + " " + t).strip()           # append text, keep prev start
        else:
            merged.append([a, b, None, t])

    n = len(merged)
    segs: list[Segment] = []
    for k, (ts, te, _, text) in enumerate(merged):
        win_start = 0.0 if k == 0 else (merged[k - 1][1] + ts) / 2.0
        win_end = total_dur if k == n - 1 else (te + merged[k + 1][0]) / 2.0
        segs.append(Segment(
            index=k, text=text,
            t_start=round(ts, 3), t_end=round(te, 3),
            win_start=round(win_start, 3), win_end=round(win_end, 3),
        ))
    log.info(f"Span-segmented {total_dur:.2f}s into {len(segs)} shots "
             f"(windows: {[round(s.window_dur, 2) for s in segs]})")
    return segs


# ==============================================================================
# FALLBACK PATH — ffmpeg silencedetect (engines with no native timestamps)
# ==============================================================================

def segment_by_silence(
    wav_path: Path,
    sentence_texts: Optional[list[str]] = None,
    noise_db: float = -32.0,
    min_silence_sec: float = 0.30,
    max_seg_sec: float = DEFAULT_MAX_SEG_SEC,
) -> list[Segment]:
    """Derive segments from silences in a wav using ffmpeg `silencedetect`.

    For the Chatterbox fallback (no per-character timestamps). `sentence_texts`,
    if given, is mapped onto the detected speech chunks in order; otherwise text
    is left blank and the caller assigns it. Returns windows tiling [0, dur].
    """
    if not FFMPEG_EXE.exists():
        raise FileNotFoundError(f"ffmpeg not found at {FFMPEG_EXE}")
    wav_path = Path(wav_path)
    cmd = [
        str(FFMPEG_EXE), "-hide_banner", "-i", str(wav_path),
        "-af", f"silencedetect=noise={noise_db}dB:d={min_silence_sec}",
        "-f", "null", "-",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    # silencedetect prints to stderr: silence_start / silence_end lines.
    starts, ends = [], []
    for line in r.stderr.splitlines():
        line = line.strip()
        if "silence_start:" in line:
            try:
                starts.append(float(line.split("silence_start:")[1].strip().split()[0]))
            except (IndexError, ValueError):
                pass
        elif "silence_end:" in line:
            try:
                ends.append(float(line.split("silence_end:")[1].split("|")[0].strip()))
            except (IndexError, ValueError):
                pass

    total_dur = _probe_duration(wav_path)

    # Speech chunks = the gaps BETWEEN silences. Build [chunk_start, chunk_end].
    chunks: list[tuple[float, float]] = []
    cursor = 0.0
    # pair silences in order; a silence interval is (starts[i], ends[i])
    sil = list(zip(starts, ends))
    for s_start, s_end in sil:
        if s_start > cursor:
            chunks.append((cursor, s_start))
        cursor = max(cursor, s_end)
    if cursor < total_dur:
        chunks.append((cursor, total_dur))
    if not chunks:
        chunks = [(0.0, total_dur)]

    # Windows: midpoint of each silence is the cut. Tile [0, total_dur].
    segs: list[Segment] = []
    n = len(chunks)
    for k, (cs, ce) in enumerate(chunks):
        win_start = 0.0 if k == 0 else (chunks[k - 1][1] + cs) / 2.0
        win_end = total_dur if k == n - 1 else (ce + chunks[k + 1][0]) / 2.0
        text = sentence_texts[k] if sentence_texts and k < len(sentence_texts) else ""
        segs.append(Segment(
            index=k, text=text,
            t_start=round(cs, 3), t_end=round(ce, 3),
            win_start=round(win_start, 3), win_end=round(win_end, 3),
        ))
    log.info(f"Silence-segmented {total_dur:.2f}s into {len(segs)} shots.")
    return segs


def _probe_duration(path: Path) -> float:
    ffprobe = FFMPEG_EXE.parent / "ffprobe.exe"
    cmd = [
        str(ffprobe), "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
    return float(r.stdout.strip())


# ==============================================================================
# Standalone smoke test (synthetic alignment — no API, no audio needed)
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Fake "Hi there. Run fast!" with a 0.4s pause between sentences.
    text = "Hi there. Run!"
    chars, starts, ends = [], [], []
    t = 0.0
    for ch in text:
        chars.append(ch)
        starts.append(round(t, 3))
        # a sentence-ending '.' is followed by a longer pause
        step = 0.45 if ch == "." else (0.08 if ch == " " else 0.12)
        ends.append(round(t + step, 3))
        t += step
    segs = segment_by_alignment(chars, starts, ends, min_seg_sec=0.1)
    for s in segs:
        print(f"  [{s.index}] '{s.text}' speech={s.t_start}-{s.t_end} "
              f"window={s.win_start}-{s.win_end} ({s.window_dur:.2f}s)")
    total = sum(s.window_dur for s in segs)
    print(f"sum(window)={total:.3f}  total_dur={ends[-1]:.3f}  (must match)")
