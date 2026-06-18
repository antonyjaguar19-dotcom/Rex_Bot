"""Unit tests for the audio-first narration segmenter (no API, no audio)."""

import sys
from pathlib import Path

import pytest

_AGENT = Path(__file__).parent.parent.resolve()
if str(_AGENT) not in sys.path:
    sys.path.insert(0, str(_AGENT))

from modules import narration_segmenter as ns


def _fake_alignment(text: str, pause_at_sentence: float = 0.45,
                    space: float = 0.08, char: float = 0.12):
    """Build a synthetic character alignment for `text`. A '.'/'!'/'?' is
    followed by a long pause so sentence breaks are detectable."""
    chars, starts, ends = [], [], []
    t = 0.0
    for ch in text:
        chars.append(ch)
        starts.append(round(t, 3))
        step = pause_at_sentence if ch in ".!?" else (space if ch == " " else char)
        ends.append(round(t + step, 3))
        t += step
    return chars, starts, ends


def test_windows_tile_total_duration():
    """sum(window durations) must equal total audio duration exactly — this is
    the invariant that keeps the video timeline == the narration length."""
    chars, starts, ends = _fake_alignment("Hello there friend. Run away fast! Why now?")
    segs = ns.segment_by_alignment(chars, starts, ends, min_seg_sec=0.1)
    total = ends[-1]
    assert abs(sum(s.window_dur for s in segs) - total) < 1e-6
    # windows are contiguous: each starts where the previous ended
    for a, b in zip(segs, segs[1:]):
        assert abs(a.win_end - b.win_start) < 1e-6
    assert segs[0].win_start == 0.0
    assert abs(segs[-1].win_end - total) < 1e-6


def test_one_segment_per_sentence():
    chars, starts, ends = _fake_alignment("First line here. Second line now.")
    segs = ns.segment_by_alignment(chars, starts, ends, min_seg_sec=0.1)
    assert len(segs) == 2
    assert segs[0].text == "First line here."
    assert segs[1].text == "Second line now."


def test_cut_sits_in_the_pause():
    chars, starts, ends = _fake_alignment("Alpha beta. Gamma delta.")
    segs = ns.segment_by_alignment(chars, starts, ends, min_seg_sec=0.1)
    # the cut between shot 0 and 1 is the midpoint of the inter-sentence silence
    gap_start = segs[0].t_end
    gap_end = segs[1].t_start
    assert gap_start < segs[0].win_end < gap_end


def test_long_sentence_splits_under_ceiling():
    """A single long sentence with internal breaths must split so each piece
    fits the video model ceiling."""
    # One long sentence with a clear breath (0.5s space) every 3rd word, so
    # every breath is a valid split point and each chunk can fit the ceiling.
    chars, starts, ends = [], [], []
    t = 0.0
    words = ("the quick fox jumps over hills and runs along rivers past "
             "tall trees toward home").split()
    for w_i, w in enumerate(words):
        for ch in w:
            chars.append(ch); starts.append(round(t, 3)); ends.append(round(t + 0.18, 3)); t += 0.18
        gap = 0.5 if (w_i + 1) % 2 == 0 else 0.1   # breath every 2 words
        chars.append(" "); starts.append(round(t, 3)); ends.append(round(t + gap, 3)); t += gap
    chars[-1] = "."  # end on punctuation
    total = ends[-1]
    segs = ns.segment_by_alignment(chars, starts, ends, max_seg_sec=2.0, min_seg_sec=0.1)
    assert len(segs) >= 3                       # the long sentence was split
    assert abs(sum(s.window_dur for s in segs) - total) < 1e-6   # still tiles
    for s in segs:
        assert s.speech_dur <= 2.0 + 0.3        # each chunk fits the ceiling


def test_tiny_segment_merges():
    chars, starts, ends = _fake_alignment("Hi. This is a much longer sentence here.")
    segs = ns.segment_by_alignment(chars, starts, ends, min_seg_sec=1.0)
    # "Hi." is ~0.3s — below 1.0s floor — so it merges with the next shot.
    assert all(s.window_dur >= 0.4 for s in segs)
    assert any("Hi." in s.text for s in segs)


def test_spans_tile_total_duration():
    """segment_by_spans windows tile [0, total_dur] exactly (VoxCPM path)."""
    spans = [("First.", 0.0, 2.0), ("Second.", 2.4, 4.5), ("Third.", 4.9, 7.0)]
    total = 7.3
    segs = ns.segment_by_spans(spans, total_dur=total)
    assert len(segs) == 3
    assert segs[0].win_start == 0.0
    assert abs(segs[-1].win_end - total) < 1e-6
    assert abs(sum(s.window_dur for s in segs) - total) < 1e-6
    for a, b in zip(segs, segs[1:]):
        assert abs(a.win_end - b.win_start) < 1e-6
    # cut between groups sits in the inter-group silence
    assert spans[0][2] < segs[0].win_end < spans[1][1]
    assert [s.text for s in segs] == ["First.", "Second.", "Third."]


def test_spans_tiny_group_merges():
    spans = [("Hi.", 0.0, 0.4), ("A longer line follows here now.", 0.8, 3.5)]
    segs = ns.segment_by_spans(spans, total_dur=3.5, min_seg_sec=1.0)
    assert len(segs) == 1
    assert segs[0].text == "Hi. A longer line follows here now."


def test_spans_empty_raises():
    with pytest.raises(ValueError):
        ns.segment_by_spans([])


def test_spans_default_total_dur():
    spans = [("One.", 0.0, 2.0), ("Two.", 2.3, 4.1)]
    segs = ns.segment_by_spans(spans)         # total_dur inferred from last end
    assert abs(segs[-1].win_end - 4.1) < 1e-6


def test_mismatched_arrays_raise():
    with pytest.raises(ValueError):
        ns.segment_by_alignment(["a", "b"], [0.0], [0.1, 0.2])


def test_empty_raises():
    with pytest.raises(ValueError):
        ns.segment_by_alignment([], [], [])
