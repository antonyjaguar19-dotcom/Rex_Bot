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


def test_strip_attribution_trailing_pronoun():
    assert sg._strip_attribution(
        "I'll wake everyone with my crow, he said.", "Redcomb"
    ) == "I'll wake everyone with my crow."


def test_strip_attribution_trailing_keeps_address():
    # ", she replied" stripped; the dialogue address ", Red" stays.
    assert sg._strip_attribution(
        "But you're so loud, Red, she replied.", "Featherlite"
    ) == "But you're so loud, Red."


def test_looks_like_narration_third_person():
    assert sg._looks_like_narration("Little Fox watched the river quietly.", "Little Fox")
    assert sg._looks_like_narration("He started hopping across the stones.", "Little Fox")


def test_looks_like_narration_keeps_dialogue():
    assert not sg._looks_like_narration("I want to cross the river.", "Fox")     # first person
    assert not sg._looks_like_narration("Why so quiet?", "Fox")                  # question
    assert not sg._looks_like_narration("Thanks, Squirrely!", "Fox")            # exclamation
    assert not sg._looks_like_narration("Over here.", "Fox")                     # not name/pronoun start


def test_validator_retags_narration_to_narrator():
    out = sg._validate_and_default(_script([
        {"shot_number": 1, "speaker": "Terry", "narration": "Terry watched the water for a while."},
        {"shot_number": 2, "speaker": "Terry", "narration": "I can do this!"},
    ]))
    assert out["shots"][0]["speaker"] == "narrator"   # third-person → retagged
    assert out["shots"][1]["speaker"] == "Terry"      # real dialogue kept


def test_strip_attribution_with_trailing_action():
    # Attribution followed by stage-action — cut everything from the tag to end.
    assert sg._strip_attribution(
        "I don't know if I can do it, said Shy Fox nervously, tugging his tail.",
        "Shy Fox",
    ) == "I don't know if I can do it."


def test_strip_attribution_inverted():
    assert sg._strip_attribution(
        "Let's race, said Featherlite.", "Featherlite"
    ) == "Let's race."


def test_strip_attribution_leading():
    assert sg._strip_attribution(
        "Featherlite said, What if we take turns?", "Featherlite"
    ) == "What if we take turns?"


def test_strip_attribution_no_tag_unchanged():
    assert sg._strip_attribution("No way!", "Redcomb") == "No way!"


def test_validator_scrubs_attribution_on_all_shots():
    out = sg._validate_and_default(_script([
        {"shot_number": 1, "speaker": "Terry", "narration": "I can do it, he said."},
        # leaked-dialogue narrator shot: tail scrubbed too
        {"shot_number": 2, "speaker": "narrator", "narration": "Wait for me, she whispered softly."},
    ]))
    assert out["shots"][0]["narration"] == "I can do it."
    assert out["shots"][1]["narration"] == "Wait for me."


def test_strip_attribution_keeps_said_as_content():
    # "said" used as description, not attribution → not stripped.
    assert sg._strip_attribution("He said nothing and walked away.", "narrator") \
        == "He said nothing and walked away."
