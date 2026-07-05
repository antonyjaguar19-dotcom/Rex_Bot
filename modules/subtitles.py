"""
Claw Bot — Shared subtitle/caption helpers (burned-in .ass captions)

Generalizes the ASS-writing logic that horror mode already used (sentence
chunking + proportional timing inside a window) so kids and music mode can
reuse it instead of each hand-rolling their own. All timing is derived from
data the pipelines already produce (narration spans / scene windows) — no
forced-alignment / ASR involved.
"""

import re
from pathlib import Path
from typing import Callable, Optional

_SPLIT_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_SECTION_TAG = re.compile(r"^\[.*?\]$")


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
        style_lines.append(
            f"Style: Mark,Arial,{wm_size},&H80FFFFFF,&H80000000,&H00000000,"
            f"0,0,1,1,1,6,40,{wm_margin},40\n"
        )
        event_lines.append(
            f"Dialogue: 0,{ass_time(0)},{ass_time(total_dur)},Mark,,0,0,0,,{watermark_text}\n"
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
