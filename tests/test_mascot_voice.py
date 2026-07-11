"""The mascot's voice: Qwen3-TTS (expressive, deterministic) with Kokoro fallback.

Kokoro reads flat and pitch-shifting it to fake a young voice sounds artificial,
so the mascot presenter gets Qwen with an emotion instruct. Vivian was chosen for
the jaguar cub.
"""
import pytest
import modules.runtime_settings as rs
from modules import facts_pipeline as fp


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "SETTINGS_PATH", tmp_path / "runtime_settings.json")


def test_defaults_to_qwen_vivian():
    assert rs.get_mascot_tts_engine() == "qwen"
    assert rs.get_mascot_voice() == "Vivian"
    assert "cartoon kid" in rs.get_mascot_voice_instruct()


def test_voice_round_trips_case_insensitively():
    rs.set_mascot_voice("dylan")
    assert rs.get_mascot_voice() == "Dylan"


def test_bad_voice_and_engine_rejected():
    with pytest.raises(ValueError):
        rs.set_mascot_voice("Gandalf")
    with pytest.raises(ValueError):
        rs.set_mascot_tts_engine("elevenlabs")


def test_unknown_stored_voice_self_heals(tmp_path):
    rs._save({"mascot_voice": "Bogus"})
    assert rs.get_mascot_voice() == "Vivian"


def test_qwen_master_is_sliced_into_one_wav_per_beat(tmp_path, monkeypatch):
    """Qwen loads in an isolated venv subprocess, so it is called ONCE for all
    beats and the master is cut back apart on the returned spans."""
    import modules.tts_qwen as tq
    calls = []

    def fake(texts, output_path=None, speaker=None, instruct=None, **k):
        calls.append({"texts": list(texts), "speaker": speaker})
        return output_path, [(t, i * 2.0, i * 2.0 + 1.5) for i, t in enumerate(texts)]

    monkeypatch.setattr(tq, "health_check", lambda: (True, "ok"))
    monkeypatch.setattr(tq, "synthesize_segments", fake)
    cut = []
    monkeypatch.setattr(fp, "_slice_wav",
                        lambda src, a, b, dst: (cut.append((a, b)), dst)[1])

    wavs = fp._voice_beats_mascot(["one", "two", "three"], tmp_path, lambda m: None)

    assert len(calls) == 1, "one model load for the whole reel, not one per beat"
    assert calls[0]["speaker"] == "Vivian"
    assert len(wavs) == 3
    assert cut == [(0.0, 1.5), (2.0, 3.5), (4.0, 5.5)]


def test_falls_back_to_kokoro_when_qwen_is_down(tmp_path, monkeypatch):
    import modules.tts_qwen as tq
    monkeypatch.setattr(tq, "health_check", lambda: (False, "venv missing"))

    made = []

    class FakeTTS:
        def synthesize(self, text, output_path=None, voice=None):
            made.append(output_path)
            return output_path
    import modules.tts_engine as te
    monkeypatch.setattr(te, "TTSEngine", lambda: FakeTTS())

    wavs = fp._voice_beats_mascot(["a", "b"], tmp_path, lambda m: None)
    assert len(wavs) == 2 and len(made) == 2, "a dead bridge must not block a render"
