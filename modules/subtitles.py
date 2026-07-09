"""
Claw Bot — Shared subtitle/caption helpers (burned-in .ass captions)

Generalizes the ASS-writing logic that horror mode already used (sentence
chunking + proportional timing inside a window) so kids and music mode can
reuse it instead of each hand-rolling their own. All timing is derived from
data the pipelines already produce (narration spans / scene windows) — no
forced-alignment / ASR involved.
"""

import difflib
import re
from pathlib import Path
from typing import Callable, Optional

_SPLIT_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_SECTION_TAG = re.compile(r"^\[.*?\]$")
_WORD_NORM = re.compile(r"[^a-z0-9]")


def _norm_word(w: str) -> str:
    return _WORD_NORM.sub("", (w or "").lower())


def ass_time(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600); m = int((t % 3600) // 60)
    s = int(t % 60); cs = int(round((t - int(t)) * 100))
    if cs == 100:
        s += 1; cs = 0
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def ass_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("\n", " ").replace("{", "(").replace("}", ")")


def _merge_chunks(parts: list, max_chars: int) -> list:
    chunks, cur = [], ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if cur and len(cur) + 1 + len(p) > max_chars:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        chunks.append(cur)
    return chunks


def sentence_chunks(text: str, max_chars: int = 84) -> list:
    """Split narration into short on-screen cues at sentence boundaries,
    merging tiny sentences up to ~max_chars so captions stay readable."""
    parts = _SPLIT_SENTENCE.split((text or "").strip())
    return _merge_chunks(parts, max_chars) or ([(text or "").strip()] if (text or "").strip() else [])


def raw_sentences(text: str) -> list:
    """Split into sentences with NO merging — used when each sentence needs its
    own real timing (audio_segmenter.refine_windows_to_sentences), as opposed to
    sentence_chunks() which merges for on-screen readability."""
    parts = _SPLIT_SENTENCE.split((text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def merge_events(events: list, max_chars: int = 84) -> list:
    """Merge consecutive (t0, t1, text) events into fewer, more readable
    captions when their combined text fits max_chars. The merged event's time
    range is [first.t0, last.t1] — still exact real timing, just displayed
    together. Use AFTER real per-sentence timing is known (unlike
    windows_to_events, which guesses sub-splits by char length)."""
    merged: list = []
    for t0, t1, text in events:
        text = (text or "").strip()
        if not text or t1 <= t0:
            continue
        if merged and len(merged[-1][2]) + 1 + len(text) <= max_chars:
            pt0, _, ptext = merged[-1]
            merged[-1] = (pt0, t1, f"{ptext} {text}".strip())
        else:
            merged.append((t0, t1, text))
    return merged


def lyric_lines(text: str) -> list:
    """Split song lyrics into displayable lines: drop [Section] tags + blanks."""
    out = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or _SECTION_TAG.match(line):
            continue
        out.append(line)
    return out


def lyric_chunks(text: str, max_chars: int = 84) -> list:
    return _merge_chunks(lyric_lines(text), max_chars)


def windows_to_events(
    windows: list,
    chunk_fn: Callable[[str], list] = sentence_chunks,
) -> list:
    """windows = [(t0, t1, text)]. Each window's text is chunked and the chunks
    are distributed proportionally (by char length) across that window's time
    span. Returns flat [(t0, t1, text)] caption events across all windows."""
    events = []
    for t0, t1, text in windows:
        text = (text or "").strip()
        if not text or t1 <= t0:
            continue
        chunks = chunk_fn(text)
        if not chunks:
            continue
        total = sum(len(c) for c in chunks) or 1
        cur = t0
        for c in chunks:
            dur = (t1 - t0) * (len(c) / total)
            events.append((cur, min(cur + dur, t1), c))
            cur += dur
    return events


def distribute_lines_over_windows(lines: list, windows: list,
                                  min_line_sec: float = 0.5) -> list:
    """Place lyric `lines` (in order) inside real singing `windows`
    [(t_start, t_end), ...], leaving instrumental gaps between windows EMPTY.

    Lines are laid on a virtual timeline = the windows concatenated (their total
    singing time), each line getting a slice proportional to its char length,
    then mapped back to real time through the windows. A line that would straddle
    a window boundary is clamped to end at its start window's end, so no caption
    is ever shown during an instrumental gap. Returns [(t0, t1, text), ...]."""
    lines = [l.strip() for l in lines if (l or "").strip()]
    windows = [(float(a), float(b)) for a, b in windows if float(b) > float(a)]
    if not lines or not windows:
        return []
    total = sum(b - a for a, b in windows)
    total_chars = sum(len(l) for l in lines) or 1

    def _v2r(v):
        """virtual time -> (real time, window index)"""
        acc = 0.0
        for idx, (a, b) in enumerate(windows):
            d = b - a
            if v <= acc + d + 1e-9:
                return a + (v - acc), idx
            acc += d
        return windows[-1][1], len(windows) - 1

    events = []
    vt = 0.0
    for l in lines:
        share = (len(l) / total_chars) * total
        r0, i0 = _v2r(vt)
        r1, i1 = _v2r(vt + share)
        vt += share
        if i1 != i0:                       # straddles a gap — clamp to this window
            r1 = windows[i0][1]
        if r1 <= r0:
            r1 = min(windows[i0][1], r0 + min_line_sec)
        events.append((round(r0, 3), round(r1, 3), l))
    return events


def align_lines_to_words(lines: list, words: list, min_coverage: float = 0.4,
                         min_line_sec: float = 0.6, max_line_sec: float = 6.0):
    """Time each lyric line to the ACTUAL sung word timestamps.

    `words` = [(word, t_start, t_end), ...] from ASR (real singing times).
    Sequence-matches the known lyric words against the ASR words (same song, so
    they largely agree) and gives each lyric LINE the [start, end] of the ASR
    words it matched. Lines the ASR missed are interpolated between their timed
    neighbours. This is true per-line sync (vs the proportional guess).

    Returns [(t0, t1, text), ...] or None when too few lines matched
    (`min_coverage`) — e.g. ASR barely heard the vocals — so the caller can fall
    back to the window/proportional path."""
    lines = [l.strip() for l in lines if (l or "").strip()]
    if not lines or not words:
        return None

    # flatten known lyric words, remembering which line each belongs to
    known, owner = [], []
    for li, line in enumerate(lines):
        for w in line.split():
            nw = _norm_word(w)
            if nw:
                known.append(nw)
                owner.append(li)
    asr = [(_norm_word(w), float(s), float(e)) for (w, s, e) in words if _norm_word(w)]
    if not known or not asr:
        return None

    sm = difflib.SequenceMatcher(None, known, [a[0] for a in asr], autojunk=False)
    span = {}  # line_idx -> [start, end]
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            continue
        for k in range(i2 - i1):
            li = owner[i1 + k]
            s, e = asr[j1 + k][1], asr[j1 + k][2]
            cur = span.get(li)
            if cur is None:
                span[li] = [s, e]
            else:
                # Only extend within a plausible single-line duration. A word
                # matched far from the line's start is a stray hit on a repeated
                # word (e.g. a chorus "day"/"way") — ignore it so the line's span
                # doesn't balloon across an instrumental section.
                if e - cur[0] <= max_line_sec:
                    cur[1] = max(cur[1], e)

    matched = len(span)
    if matched < max(1, int(min_coverage * len(lines))):
        return None

    # interpolate lines with no matched words from timed neighbours
    total_end = asr[-1][2]
    events = []
    for li in range(len(lines)):
        if li in span:
            s, e = span[li]
        else:
            prev = next((span[k][1] for k in range(li - 1, -1, -1) if k in span), None)
            nxt = next((span[k][0] for k in range(li + 1, len(lines)) if k in span), None)
            if prev is not None and nxt is not None:
                s, e = prev, nxt
            elif prev is not None:
                s, e = prev, min(total_end, prev + min_line_sec)
            elif nxt is not None:
                s, e = max(0.0, nxt - min_line_sec), nxt
            else:
                continue
        if e - s < min_line_sec:
            e = s + min_line_sec
        elif e - s > max_line_sec:
            e = s + max_line_sec        # cap an over-long span (stray match)
        events.append((round(s, 3), round(e, 3), lines[li]))

    # enforce non-overlap / monotonic (a later line can't start before the prev ends)
    events.sort(key=lambda x: x[0])
    fixed = []
    for s, e, t in events:
        if fixed and s < fixed[-1][1]:
            s = fixed[-1][1]
            if e < s + min_line_sec:
                e = s + min_line_sec
        fixed.append((round(s, 3), round(e, 3), t))
    return fixed or None


def write_captions_ass(
    total_dur: float,
    w: int,
    h: int,
    path: Path,
    events: Optional[list] = None,
    watermark_text: Optional[str] = "Rexjaw",
) -> Path:
    """Write an .ass with an optional watermark AND optional burned-in captions
    (bottom-center, white with black outline + drop shadow). Either can be
    omitted (watermark_text=None, events=None/[])."""
    wm_size = max(14, round(h * 0.024))
    wm_margin = max(12, round(w * 0.012))
    cap_size = max(22, round(h * 0.045))
    cap_marginv = max(28, round(h * 0.06))
    cap_side = max(40, round(w * 0.08))

    style_lines = [
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, "
        "Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV\n"
    ]
    event_lines = []

    if watermark_text:
        # Premium mark: Segoe UI Semibold, ~73% opaque white, hairline outline +
        # soft shadow for depth on any background, letter-spaced for a logo feel.
        style_lines.append(
            f"Style: Mark,Segoe UI Semibold,{wm_size},&H45FFFFFF,&H60000000,&H50000000,"
            f"0,0,1,0.6,1.4,6,40,{wm_margin},40\n"
        )
        event_lines.append(
            f"Dialogue: 0,{ass_time(0)},{ass_time(total_dur)},Mark,,0,0,0,,"
            f"{{\\fsp2}}{watermark_text}\n"
        )
    if events:
        style_lines.append(
            f"Style: Cap,Arial,{cap_size},&H00FFFFFF,&H00000000,&H90000000,"
            f"0,0,1,2,1,2,{cap_side},{cap_side},{cap_marginv}\n"
        )
        for t0, t1, txt in events:
            event_lines.append(
                f"Dialogue: 0,{ass_time(t0)},{ass_time(t1)},Cap,,0,0,0,,{ass_escape(txt)}\n"
            )

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {w}\nPlayResY: {h}\n"
        "ScaledBorderAndShadow: yes\n\n"
        + "".join(style_lines) + "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    path.write_text(header + "".join(event_lines), encoding="utf-8")
    return path
