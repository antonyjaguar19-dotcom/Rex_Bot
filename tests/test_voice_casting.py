"""P2 voice casting: assignment, gender guess, speaker→voice resolution."""
import sys
from pathlib import Path

_AGENT = Path(__file__).parent.parent.resolve()
if str(_AGENT) not in sys.path:
    sys.path.insert(0, str(_AGENT))

from modules import voice_casting as vc


def test_assign_gives_distinct_voices():
    script = {"characters": [
        {"name": "Leo", "appearance": "a 6-year-old boy in a red shirt"},
        {"name": "Mia", "appearance": "a 7-year-old girl with pigtails"},
        {"name": "Pip", "appearance": "a young rabbit"},
    ]}
    vc.assign_voices(script, narrator_voice="af_heart")
    voices = [c["voice"] for c in script["characters"]]
    assert all(voices)                       # every char got one
    assert len(set(voices)) == 3             # all distinct
    assert "af_heart" not in voices          # distinct from narrator


def test_assign_is_idempotent():
    script = {"characters": [{"name": "Leo", "appearance": "a boy", "voice": "am_eric"}]}
    vc.assign_voices(script, narrator_voice="af_heart")
    assert script["characters"][0]["voice"] == "am_eric"   # not overridden


def test_gender_guess():
    assert vc.guess_gender({"name": "Leo", "appearance": "a young boy, he runs"}) == "male"
    assert vc.guess_gender({"name": "Mia", "appearance": "a little girl, she smiles"}) == "female"
    assert vc.guess_gender({"name": "Rock", "appearance": "a grey stone"}) == "neutral"


def test_male_gets_male_voice():
    script = {"characters": [{"name": "Dad", "appearance": "a tall man, father of two"}]}
    vc.assign_voices(script, narrator_voice="af_heart")
    assert script["characters"][0]["voice"] in vc.MALE_VOICES


def test_resolve_voice_narrator_and_unknown():
    script = {"characters": [{"name": "Leo", "appearance": "a boy", "voice": "am_eric"}]}
    assert vc.resolve_voice(script, "narrator") == vc._narrator_voice()
    assert vc.resolve_voice(script, None) == vc._narrator_voice()
    assert vc.resolve_voice(script, "Ghost") == vc._narrator_voice()  # not in cast


def test_resolve_voice_character_case_insensitive():
    script = {"characters": [{"name": "Leo", "appearance": "a boy", "voice": "am_eric"}]}
    assert vc.resolve_voice(script, "leo") == "am_eric"
    assert vc.resolve_voice(script, "Leo") == "am_eric"


def test_resolve_invalid_voice_falls_back():
    script = {"characters": [{"name": "Leo", "appearance": "a boy", "voice": "bogus_voice"}]}
    assert vc.resolve_voice(script, "Leo") == vc._narrator_voice()
