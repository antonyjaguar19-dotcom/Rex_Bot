"""The shared shot grammar: one framing vocabulary + crop ladder for every pipeline.

The lesson ladder and the kids grammar used to be two copies of this; this module is the
single source, and lesson mode + shot_framing now delegate to it. These tests pin the
vocabulary (incl. the new establishing/montage), the by-kind assignment + neighbour de-dup,
and the sequence grouping.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import shot_grammar as sg          # noqa: E402


def test_the_vocabulary_is_the_superset_including_establishing_and_montage():
    for name in ("establishing", "wide", "medium", "closeup", "insert", "montage"):
        assert name in sg.FRAMINGS, name
        assert sg.is_framing(name)
    assert not sg.is_framing("banana")
    assert sg.DEFAULT_FRAMING == "medium"


def test_every_framing_has_a_crop_window():
    for name in sg.FRAMINGS:
        assert name in sg.FRAME_WINDOW, name
    # establishing and montage are the whole frame (a montage is composed, never cropped)
    assert sg.window_factors("establishing") == sg.NO_CROP
    assert sg.window_factors("montage") == sg.NO_CROP
    # closeup is the tightest of the ladder; insert drops to the hands band
    assert sg.window_factors("closeup")[1] < sg.window_factors("medium")[1]
    assert sg.window_factors("insert")[0] > 0.0        # anchored below the face
    # an unknown framing never crops — a broken name must not break the picture
    assert sg.window_factors("nonsense") == sg.NO_CROP


def test_framing_is_assigned_by_kind_then_neighbours_are_de_duped():
    kind_map = {"intro": "establishing", "teach": "medium", "example": "wide",
                "check": "closeup", "recap": "wide", "outro": "medium"}
    nudge = {"medium": "wide", "wide": "medium"}
    # two 'teach' in a row -> the second workhorse is nudged off its neighbour
    got = sg.framing_for(["teach", "teach", "check"], kind_map, "medium", nudge)
    assert got == ["medium", "wide", "closeup"]
    # closeup/establishing are NOT nudged even when they repeat (rare, load-bearing)
    got = sg.framing_for(["check", "check"], kind_map, "medium", nudge)
    assert got == ["closeup", "closeup"]
    # an unknown kind falls back to the default
    assert sg.framing_for(["mystery"], kind_map, "medium", nudge) == ["medium"]


def test_sequences_group_consecutive_same_titles():
    beats = [{"k": "intro"}, {"k": "teach"}, {"k": "teach"}, {"k": "recap"}]
    titles = {"intro": "Introduction", "recap": "Recap"}
    seq = sg.sequences(beats, lambda b: titles.get(b["k"], "Topic"))
    assert seq == [
        {"title": "Introduction", "shots": [0]},
        {"title": "Topic", "shots": [1, 2]},
        {"title": "Recap", "shots": [3]},
    ]
    assert sg.sequences([], lambda b: "x") == []
