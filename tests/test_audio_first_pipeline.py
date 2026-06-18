"""Offline tests for the audio-first structurer mapping (mock LLM, no API/GPU).

Verifies the invariant that matters most: narration text in the script ALWAYS
matches what was voiced (verbatim from the segments), shot count == segment
count, timing windows are carried, and the pipeline survives an LLM failure.
"""

import json
import sys
from pathlib import Path

import pytest

_AGENT = Path(__file__).parent.parent.resolve()
if str(_AGENT) not in sys.path:
    sys.path.insert(0, str(_AGENT))

from modules import audio_first_pipeline as afp
from modules import narration_segmenter as nseg
from modules import script_generator as sg


def _segments():
    # Plain third-person lines (no speech verbs) so the validator's attribution
    # scrub leaves them untouched and we can assert verbatim equality.
    texts = [
        "Morning light spilled across the quiet meadow.",
        "A small brown fox crept toward the sparkling stream.",
        "The fox paused and watched the water ripple gently.",
    ]
    segs, t = [], 0.0
    for i, txt in enumerate(texts):
        dur = 2.0
        segs.append(nseg.Segment(
            index=i, text=txt,
            t_start=round(t, 3), t_end=round(t + dur, 3),
            win_start=round(t, 3), win_end=round(t + dur, 3),
        ))
        t += dur
    return segs


def _fake_meta(n: int) -> str:
    shots = [{
        "segment": i + 1, "beat": "hook", "shot_type": "medium",
        "speaker": "narrator", "visual_description": f"scene {i+1}",
        "first_frame_prompt": "f", "last_frame_prompt": "l", "motion_prompt": "m",
    } for i in range(n)]
    return json.dumps({
        "title": "Meadow Morning", "culture": "animal-kingdom", "style": "storybook",
        "characters": [{"name": "Rusty", "type": "animal",
                        "appearance": "a young brown fox with a bushy tail",
                        "locked_visual_token": "a young brown fox with a bushy white-tipped tail"}],
        "setting": "a meadow at dawn", "shots": shots,
        "moral": "curiosity is gentle", "music_mood": "calm",
    })


def test_structure_matches_segments(monkeypatch):
    segs = _segments()
    monkeypatch.setattr(sg, "_call_llm", lambda *a, **k: _fake_meta(len(segs)))
    script = afp._structure_segments(
        title="Meadow Morning", theme="a curious fox", prose="...", segments=segs,
    )
    assert len(script["shots"]) == len(segs)
    for i, (shot, seg) in enumerate(zip(script["shots"], segs)):
        assert shot["narration"] == seg.text          # VERBATIM
        assert shot["shot_number"] == i + 1            # sequential
        assert shot["win_start"] == seg.win_start
        assert shot["win_end"] == seg.win_end
    assert script["_audio_first"] is True
    assert any(c.get("name") == "Rusty" for c in script["characters"])


def test_narration_verbatim_survives_validator_scrub(monkeypatch):
    """Audio-first invariant: even when a segment carries dialogue-attribution
    tags (which the classic validator would scrub), the script narration must
    stay EXACTLY what was voiced — text == audio."""
    texts = [
        "Brrr, whispered Squeak to himself, feeling shy. Hey there!",
        "Why so silent? chirped Bella, the friendly robin.",
        "The meadow was calm and the sun rose slowly over the hills.",
    ]
    segs, t = [], 0.0
    for i, txt in enumerate(texts):
        segs.append(nseg.Segment(index=i, text=txt,
                                 t_start=t, t_end=t + 3.0,
                                 win_start=t, win_end=t + 3.0))
        t += 3.0
    monkeypatch.setattr(sg, "_call_llm", lambda *a, **k: _fake_meta(len(segs)))
    script = afp._structure_segments(title="T", theme="x", prose="p", segments=segs)
    assert len(script["shots"]) == len(segs)
    for shot, seg in zip(script["shots"], segs):
        assert shot["narration"] == seg.text       # verbatim, NOT scrubbed


def test_wrong_shot_count_forced_to_segments(monkeypatch):
    """LLM returns too few shots — output must still be one shot per segment,
    narration verbatim, with deterministic beats filling the gaps."""
    segs = _segments()
    monkeypatch.setattr(sg, "_call_llm", lambda *a, **k: _fake_meta(1))  # only 1
    script = afp._structure_segments(
        title="T", theme="x", prose="...", segments=segs,
    )
    assert len(script["shots"]) == len(segs)
    for shot, seg in zip(script["shots"], segs):
        assert shot["narration"] == seg.text
    # first shot is the LLM one (medium); later shots got fallback metadata
    assert all(s["shot_type"] in ("wide", "medium", "closeup", "insert")
               for s in script["shots"])


def test_llm_failure_falls_back(monkeypatch):
    segs = _segments()
    def _boom(*a, **k):
        raise RuntimeError("ollama down")
    monkeypatch.setattr(sg, "_call_llm", _boom)
    script = afp._structure_segments(title="T", theme="x", prose="p", segments=segs)
    assert len(script["shots"]) == len(segs)
    for shot, seg in zip(script["shots"], segs):
        assert shot["narration"] == seg.text
    # first shot defaults to a wide atmosphere establishing frame
    assert script["shots"][0]["shot_type"] == "wide"
    assert script["shots"][0]["beat"] == "atmosphere"


def test_voice_ids_per_speaker_distinct_and_consistent():
    """Narrator keeps one voice across all its groups; each character gets a
    distinct voice by gender."""
    script = {
        "shots": [
            {"shot_number": 1, "speaker": "narrator"},
            {"shot_number": 2, "speaker": "Momo"},
            {"shot_number": 3, "speaker": "Benny"},
            {"shot_number": 4, "speaker": "narrator"},
        ],
        "characters": [
            {"name": "Momo", "type": "animal",
             "appearance": "a cheerful female mouse, she squeaks"},
            {"name": "Benny", "type": "animal",
             "appearance": "a shy male bear, he grumbles"},
        ],
    }
    vids = afp._voice_ids_for_groups(script, 4)
    assert len(vids) == 4
    assert vids[0] == vids[3]                       # narrator consistent
    assert vids[1] != vids[2]                       # two chars differ
    assert vids[0] not in (vids[1], vids[2])        # narrator distinct from chars


def test_retime_overwrites_timing_and_duration():
    script = {"shots": [{"shot_number": i + 1, "narration": "x",
                         "win_dur": 0.0} for i in range(3)]}
    segs = [
        nseg.Segment(0, "One.", 0.0, 2.0, 0.0, 2.2),
        nseg.Segment(1, "Two.", 2.4, 4.0, 2.2, 4.3),
        nseg.Segment(2, "Three.", 4.5, 6.0, 4.3, 6.0),
    ]
    afp._retime_shots(script, segs)
    assert script["shots"][0]["win_dur"] == 2.2
    assert script["shots"][2]["win_end"] == 6.0
    assert script["shots"][1]["narration"] == "Two."      # verbatim from span
    assert script["duration_seconds"] == 6


def test_split_sentences():
    out = afp._split_sentences("Hello there. How are you? Run!")
    assert out == ["Hello there.", "How are you?", "Run!"]


def test_breath_groups_short_sentences_pass_through():
    out = afp._breath_groups("Hi there. Run fast.")
    assert out == ["Hi there.", "Run fast."]


def test_breath_groups_split_long_sentence_at_clauses():
    long = ("The tiny fox crept through the tall grass, past the old oak tree, "
            "over the cold stream, and into the dark woods beyond the hills.")
    out = afp._breath_groups(long, max_chars=40)
    assert len(out) >= 2                          # the long sentence was split
    assert all(len(g) <= 80 for g in out)         # no runaway groups
    # reassembled text preserves the words (split only added boundaries)
    joined = " ".join(out).replace(" ,", ",")
    assert "tiny fox" in joined and "dark woods" in joined
