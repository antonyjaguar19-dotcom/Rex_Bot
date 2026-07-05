"""Tests for the shared subtitle/caption helpers (modules/subtitles.py)."""

import sys
from pathlib import Path

_AGENT = Path(__file__).parent.parent.resolve()
if str(_AGENT) not in sys.path:
    sys.path.insert(0, str(_AGENT))

from modules import subtitles as subs


def test_sentence_chunks_splits_on_boundaries():
    long_a = "A" * 80
    long_b = "B" * 80
    text = f"{long_a}. {long_b}!"
    chunks = subs.sentence_chunks(text, max_chars=84)
    assert chunks == [f"{long_a}.", f"{long_b}!"]


def test_sentence_chunks_merges_short_sentences():
    chunks = subs.sentence_chunks("Hi. Bye.", max_chars=84)
    assert chunks == ["Hi. Bye."]


def test_lyric_lines_drops_section_tags_and_blanks():
    text = "[Verse 1]\nLine one\n\n[Chorus]\nLine two\nLine three"
    assert subs.lyric_lines(text) == ["Line one", "Line two", "Line three"]


def test_windows_to_events_proportional_timing():
    windows = [(0.0, 10.0, "Short. " + "X" * 90 + ".")]
    events = subs.windows_to_events(windows, chunk_fn=subs.sentence_chunks)
    assert len(events) == 2
    assert events[0][0] == 0.0
    assert events[-1][1] == 10.0
    # longer chunk should get proportionally more time than "Short."
    assert (events[1][1] - events[1][0]) > (events[0][1] - events[0][0])


def test_windows_to_events_skips_empty_and_degenerate():
    events = subs.windows_to_events([(0.0, 5.0, "  "), (5.0, 5.0, "text"), (10.0, 12.0, "ok.")])
    assert len(events) == 1
    assert events[0][2] == "ok."


def test_write_captions_ass_watermark_only(tmp_path):
    out = tmp_path / "wm.ass"
    subs.write_captions_ass(10.0, 1920, 1080, out, events=None, watermark_text="Rexjaw")
    content = out.read_text(encoding="utf-8")
    assert "Style: Mark" in content
    assert "Style: Cap" not in content
    assert "Rexjaw" in content


def test_write_captions_ass_captions_only(tmp_path):
    out = tmp_path / "caps.ass"
    events = [(0.0, 2.0, "Hello there.")]
    subs.write_captions_ass(2.0, 1080, 1920, out, events=events, watermark_text=None)
    content = out.read_text(encoding="utf-8")
    assert "Style: Cap" in content
    assert "Style: Mark" not in content
    assert "Hello there." in content


def test_ass_escape_neutralizes_braces_and_newlines():
    escaped = subs.ass_escape("a{b}c\nd")
    assert "{" not in escaped and "}" not in escaped and "\n" not in escaped


def test_raw_sentences_no_merging():
    assert subs.raw_sentences("Hi. Bye. Go!") == ["Hi.", "Bye.", "Go!"]


def test_merge_events_combines_short_adjacent():
    events = [(0.0, 1.0, "Hi."), (1.0, 2.0, "Bye.")]
    merged = subs.merge_events(events, max_chars=84)
    assert merged == [(0.0, 2.0, "Hi. Bye.")]


def test_merge_events_keeps_real_time_bounds_when_too_long():
    long_a = "A" * 80
    long_b = "B" * 80
    events = [(0.0, 5.0, long_a), (5.0, 9.0, long_b)]
    merged = subs.merge_events(events, max_chars=84)
    assert merged == [(0.0, 5.0, long_a), (5.0, 9.0, long_b)]
