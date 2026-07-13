"""A beat's audio must contain the beat's line.

Chatterbox collapses now and then: handed "Get ready to be amazed by these game
secrets." it returned **0.12 seconds** of audio. The fact beats in the same batch
were fine, so nothing downstream noticed — the reel shipped with a hook that was a
blip and an outro cut in half, and the video was cut to those lengths.

It is always the SHORT lines (the hook, the outro), and it is stochastic: the same
lines and the same voice worked on five earlier reels.

Narration is the source of truth in this pipeline. A take that cannot possibly
contain its line is not narration.
"""
from pathlib import Path

import pytest

from modules import facts_pipeline as fp

HOOK = "Get ready to be amazed by these game secrets."           # 9 words
FACT = "The longest Tetris session lasted over 3 days straight"  # 8 words
OUTRO = "Follow for more mind-blowing facts."                    # 5 words


@pytest.fixture
def take(tmp_path, monkeypatch):
    """Make a real file that probes as `secs` seconds."""
    table = {}
    monkeypatch.setattr(fp, "_probe_dur", lambda p: table.get(Path(p).name, 0.0))

    def _make(name: str, secs: float) -> Path:
        p = tmp_path / name
        p.write_bytes(b"RIFF" + b"\0" * 64)
        table[name] = secs
        return p
    return _make


def test_a_blip_is_not_a_spoken_line(take):
    blip = take("a.wav", 0.12)        # what actually shipped
    real = take("b.wav", 4.76)        # a real take of a similar line
    assert fp._short_takes([HOOK, FACT], [blip, real]) == [0]


def test_the_outro_case(take):
    """0.60s of raw audio for "Follow for more mind-blowing facts!" — cut in half,
    and it shipped."""
    assert fp._short_takes([OUTRO], [take("o.wav", 0.60)]) == [0]


def test_a_real_take_is_never_rejected(take):
    """The floor is a PHYSICAL one (4 words/sec is auctioneer pace), not this
    voice's measured 1.7-1.9. Anchoring it to the measured pace would have made a
    faster-reading clone re-roll every line and then lose its voice to a preset."""
    assert fp._short_takes([HOOK, FACT],
                           [take("a.wav", 4.4), take("b.wav", 4.76)]) == []
    # A brisk-but-real read (9 words in 2.5s = 3.6 w/s) is kept.
    assert fp._short_takes([HOOK], [take("c.wav", 2.5)]) == []


def test_a_genuinely_tiny_line_is_allowed(take):
    """"Wow." is short because it IS short. The floor must not fail it."""
    assert fp._short_takes(["Wow."], [take("w.wav", 0.9)]) == []


def test_an_unmeasurable_take_is_not_called_bad(take):
    """UNKNOWN IS NOT BROKEN. _probe_dur returns 0.0 when it cannot measure (a
    missing ffprobe, an odd container). Treating that as a zero-second take would
    re-roll every line and then drop the mascot's voice for a preset — on every
    reel — because of a broken probe."""
    assert fp._short_takes([HOOK], [take("a.wav", 0.0)]) == []


def test_a_missing_or_empty_file_IS_bad(take, tmp_path, monkeypatch):
    """No audio is not "unknown" — it is nothing."""
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    assert fp._short_takes([HOOK], [empty]) == [0]
    assert fp._short_takes([HOOK], [tmp_path / "gone.wav"]) == [0]


def test_the_clone_re_rolls_the_clipped_lines_then_gives_up(monkeypatch, tmp_path):
    """Re-roll only the failures (a fresh sample usually lands), and if the voice
    keeps clipping, ship NOTHING from this path — the caller falls back to a preset
    voice, whole and consistent. A reel whose hook is a 0.1s blip is not a reel."""
    # facts_pipeline imports these INSIDE the function, so patch them at source.
    from modules import gpu_memory, mascot_library, tts_chatterbox

    monkeypatch.setattr(mascot_library, "voice_ref", lambda: tmp_path / "ref.wav")
    monkeypatch.setattr(tts_chatterbox, "health_check", lambda: (True, "ok"))
    monkeypatch.setattr(gpu_memory, "evict_all", lambda: None)

    batches = []

    def fake_each(texts, out_dir, **k):
        batches.append(list(texts))
        return [Path(out_dir) / f"seg_{i:02d}.wav" for i in range(len(texts))]
    monkeypatch.setattr(tts_chatterbox, "synthesize_each", fake_each)

    # Beat 0 is always clipped; beat 1 is always fine.
    monkeypatch.setattr(fp, "_short_takes",
                        lambda texts, wavs: [i for i, t in enumerate(texts)
                                             if t.startswith("Get ready")])

    out = fp._voice_beats_clone([HOOK, FACT], tmp_path, lambda m: None)

    assert out is None, "a reel must not ship with a clipped hook"
    assert len(batches) == 3, "one full pass, then two re-rolls of the bad line"
    assert batches[1] == [HOOK] and batches[2] == [HOOK], "re-roll ONLY what failed"
