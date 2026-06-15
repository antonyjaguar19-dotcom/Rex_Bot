"""P1 dialogue: per-shot `speaker` defaulting/validation in script_generator."""
import sys
from pathlib import Path

_AGENT = Path(__file__).parent.parent.resolve()
if str(_AGENT) not in sys.path:
    sys.path.insert(0, str(_AGENT))

from modules import script_generator as sg


def _pad(shots):
    # Validator floors at 3 shots; pad with narrator filler so the shot under
    # test is shots[0] but the floor is satisfied.
    while len(shots) < 3:
        n = len(shots) + 1
        shots.append({"shot_number": n, "speaker": "narrator",
                      "narration": f"Filler line {n}."})
    return shots


def _script(shots, chars=None):
    return {
        "title": "T", "shots": _pad(shots),
        "characters": chars if chars is not None else [
            {"name": "Terry", "type": "animal", "appearance": "a young green tortoise",
             "locked_visual_token": "a young green tortoise"},
        ],
    }


def test_missing_speaker_defaults_narrator():
    out = sg._validate_and_default(_script([
        {"shot_number": 1, "beat": "hook", "shot_type": "wide", "narration": "Dawn broke."},
    ]))
    assert out["shots"][0]["speaker"] == "narrator"


def test_explicit_narrator_kept():
    out = sg._validate_and_default(_script([
        {"shot_number": 1, "speaker": "narrator", "narration": "Dawn broke."},
    ]))
    assert out["shots"][0]["speaker"] == "narrator"


def test_known_character_canonical_casing():
    out = sg._validate_and_default(_script([
        {"shot_number": 1, "speaker": "terry", "narration": "Wait for me!"},
    ]))
    # case-insensitive match → restored to canonical cast name
    assert out["shots"][0]["speaker"] == "Terry"


def test_unknown_speaker_falls_back_to_narrator():
    out = sg._validate_and_default(_script([
        {"shot_number": 1, "speaker": "Ghost", "narration": "Boo."},
    ]))
    assert out["shots"][0]["speaker"] == "narrator"


def test_quote_marks_stripped_from_dialogue():
    out = sg._validate_and_default(_script([
        {"shot_number": 1, "speaker": "Terry", "narration": '"Wait for me!"'},
    ]))
    assert '"' not in out["shots"][0]["narration"]
    assert out["shots"][0]["narration"] == "Wait for me!"
