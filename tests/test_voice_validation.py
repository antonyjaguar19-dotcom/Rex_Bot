"""Voice ids are validated on write and self-healed on read.

Why this exists: `!set_facts_voice Af_bella` (capital A) persisted to
runtime_settings.json, and the next dashboard page build died with
"Invalid value: Af_bella" -> HTTP 500. Nothing validated the write, nothing
guarded the read, and the select had its own hardcoded copy of the voice list.
"""

import json

import pytest

from modules import voices


# ---------------------------------------------------------------- voices module

def test_normalize_forgives_case_and_whitespace():
    assert voices.normalize("  Af_Bella ") == "af_bella"
    assert voices.normalize("AF_HEART") == "af_heart"


def test_normalize_rejects_unknown():
    assert voices.normalize("af_nope") is None
    assert voices.normalize("") is None
    assert voices.normalize(None) is None


def test_normalize_respects_the_allowed_subset():
    # af_heart is a real Kokoro voice but not offered for facts
    assert voices.normalize("af_heart") == "af_heart"
    assert voices.normalize("af_heart", voices.FACTS_VOICES) is None


def test_coerce_never_fails():
    assert voices.coerce("Af_bella", "af_heart") == "af_bella"
    assert voices.coerce("garbage", "af_heart") == "af_heart"
    assert voices.coerce(None, "af_heart") == "af_heart"


def test_suggest_offers_near_matches():
    assert "af_bella" in voices.suggest("af_bel")
    assert "Valid voices" in voices.suggest("zzz")


def test_facts_voices_are_a_subset_of_kokoro():
    assert set(voices.FACTS_VOICES) <= set(voices.KOKORO_VOICES)
    assert voices.DEFAULT_VOICE in voices.KOKORO_VOICES
    assert voices.DEFAULT_FACTS_VOICE in voices.FACTS_VOICES


# ---------------------------------------------------------------- runtime_settings

@pytest.fixture
def rs(tmp_path, monkeypatch):
    from modules import runtime_settings as _rs
    monkeypatch.setattr(_rs, "SETTINGS_PATH", tmp_path / "runtime_settings.json")
    return _rs


def _write_raw(rs, **kv):
    rs.SETTINGS_PATH.write_text(json.dumps(kv), encoding="utf-8")


def test_set_facts_voice_normalizes_case(rs):
    rs.set_facts_voice("Af_Bella")
    assert rs.get_facts_voice() == "af_bella"
    assert json.loads(rs.SETTINGS_PATH.read_text())["facts_voice"] == "af_bella"


def test_set_facts_voice_rejects_unknown(rs):
    with pytest.raises(ValueError, match="not a facts voice"):
        rs.set_facts_voice("af_heart")        # valid Kokoro, not a facts voice
    with pytest.raises(ValueError):
        rs.set_facts_voice("Af_bogus")
    assert not rs.SETTINGS_PATH.exists() or \
        "facts_voice" not in json.loads(rs.SETTINGS_PATH.read_text())


def test_poisoned_facts_voice_self_heals_on_read(rs):
    """A config already holding 'Af_bella' must not brick the dashboard."""
    _write_raw(rs, facts_voice="Af_bella")
    assert rs.get_facts_voice() == "af_bella"      # coerced, not raised


def test_completely_bogus_facts_voice_falls_back(rs):
    _write_raw(rs, facts_voice="nonsense")
    assert rs.get_facts_voice() == voices.DEFAULT_FACTS_VOICE


def test_set_voice_override_validates(rs):
    rs.set_voice_override("AM_Adam")
    assert rs.get_voice_override() == "am_adam"
    with pytest.raises(ValueError, match="not a Kokoro voice"):
        rs.set_voice_override("am_bogus")


def test_poisoned_voice_override_is_ignored_on_read(rs):
    _write_raw(rs, voice="Af_bella")
    assert rs.get_voice_override() == "af_bella"
    _write_raw(rs, voice="totally_invalid")
    assert rs.get_voice_override() is None                  # treated as no override
    assert rs.get_effective_voice() == voices.DEFAULT_VOICE  # always valid


def test_effective_voice_is_always_a_real_voice(rs):
    _write_raw(rs, voice="!!!")
    assert rs.get_effective_voice() in voices.KOKORO_VOICES
