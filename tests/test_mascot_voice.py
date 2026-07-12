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


def test_defaults_to_qwen_boy_voice():
    """No local TTS has a child voice: Eric (adult male) lifted +2 semitones is
    the jaguar cub. The instruct also demands an EVEN delivery — punctuation-led
    excitement made single beats shout while the next stayed calm."""
    assert rs.get_mascot_tts_engine() == "qwen"
    assert rs.get_mascot_voice() == "Eric"
    assert rs.get_mascot_voice_pitch() == 2.0
    assert rs.get_mascot_voice_speed() == 1.2
    ins = rs.get_mascot_voice_instruct()
    assert "young boy" in ins and "never excited" in ins


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
    assert rs.get_mascot_voice() == "Eric"


def test_qwen_voices_each_beat_and_burns_a_warmup(tmp_path, monkeypatch):
    """Two bugs, one call:

    * Slicing a concatenated master on spans clipped words at the seams, so every
      beat is now its own generation (the CLI already voiced them separately —
      it just threw the individual audio away).
    * Qwen's FIRST generation after a model load loses its opening consonant
      ("Bees" -> "These"), and it was ALWAYS the first line. So a throwaway line
      is burned ahead of the real ones and dropped.
    """
    import modules.tts_qwen as tq
    calls = []

    def fake(texts, out_dir, speaker=None, instruct=None, **k):
        calls.append(list(texts))
        return [Path(out_dir) / f"seg_{i:02d}.wav" for i in range(len(texts))]

    monkeypatch.setattr(tq, "health_check", lambda: (True, "ok"))
    monkeypatch.setattr(tq, "synthesize_each", fake)
    monkeypatch.setattr(fp.shutil, "copy2", lambda a, b: Path(b).write_bytes(b""))
    monkeypatch.setattr(fp, "_strip_sacrifice", lambda w, *a: w)
    monkeypatch.setattr(fp, "_speed_wav", lambda w, s: w)
    monkeypatch.setattr(fp, "_pitch_wav", lambda w, s: w)
    monkeypatch.setattr(fp, "_pad_wav", lambda w, **k: w)
    monkeypatch.setattr(fp, "gpu_utils", fp.gpu_utils)

    wavs = fp._voice_beats_mascot(["one", "two", "three"], tmp_path, lambda m: None)

    assert len(calls) == 1, "one model load for the whole reel"
    sent = calls[0]
    assert sent[0] == fp._WARMUP_LINE, "a throwaway is burned first"
    assert len(sent) == 4                      # warm-up + 3 beats
    assert len(wavs) == 3, "the warm-up is dropped, one WAV per beat"


def test_the_carrier_word_is_prefixed_to_every_spoken_line():
    """Qwen eats the start of a line, so a carrier takes the damage. It is cut
    back out of the audio afterwards — nobody says 'Hmm' before every sentence."""
    out = fp._natural_speech("Bees are amazing.")
    assert out.startswith(fp._SACRIFICE)
    assert "Bees are amazing." in out


def test_exclamation_marks_are_flattened_for_speech():
    """A line ending in '!' came out shouted while the next beat stayed calm —
    each beat is its own generation and reads prosody off its own punctuation."""
    assert "!" not in fp._even_tone("Bees are amazing! Really!")
    assert fp._even_tone("Bees are amazing!") == "Bees are amazing."


def test_a_cut_deeper_than_a_throwaway_is_refused(tmp_path, monkeypatch):
    """The aligner sometimes misaligns and points into the middle of the
    sentence. It once ate 'Bees dance' — cap the cut."""
    wav = tmp_path / "b.wav"
    wav.write_bytes(b"")
    import modules.lyric_aligner as la
    monkeypatch.setattr(la, "health_check", lambda: (True, "ok"))
    monkeypatch.setattr(la, "_words_from_audio",
                        lambda p: [["ready", 0.1, 0.4], ["bees", 5.0, 5.4],
                                   ["dance", 5.5, 5.9]])
    monkeypatch.setattr(fp, "_probe_dur", lambda p: 8.0)
    cut = []
    monkeypatch.setattr(fp.subprocess, "run",
                        lambda *a, **k: cut.append(a) or type("R", (), {"returncode": 1})())
    fp._strip_sacrifice(wav)
    assert not cut, "a 5s cut is the sentence, not a throwaway"


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


# --- presenter framing (S2V breaks legs it can't animate) --------------------

def test_presenter_style_locks_proportions_and_frees_the_pose():
    """The feet were once nailed to the floor — but that rule only ever existed to
    stop Wan-S2V mangling legs it could not animate, and the video stage no longer
    uses S2V. What must STAY is the proportion lock (Qwen quietly slims the cub
    into a lanky humanoid whenever the costume changes) and the anti-intersection
    negative (a paw went straight through a hat brim)."""
    from modules import mascot as mas
    assert "full body visible" in mas.STYLE_PRESENTER
    assert "dynamic playful action pose" in mas.STYLE_PRESENTER
    assert "feet planted" not in mas.STYLE_PRESENTER, "the pose is free again"
    for kept in ("short chunky cub", "chibi proportions"):
        assert kept in mas.STYLE_PRESENTER, kept
    for banned in ("thin legs", "lanky", "human proportions",
                   "hand passing through object", "limbs intersecting props"):
        assert banned in mas.NEGATIVE_PRESENTER, banned


def test_render_scene_presenter_locks_the_build_and_skips_the_headline(tmp_path, monkeypatch):
    from PIL import Image
    from modules import mascot as mas
    from modules.image_backends import comfyui_qwen_edit as qwen
    monkeypatch.setattr(mas, "ASSETS_DIR", tmp_path)
    Image.new("RGB", (64, 64)).save(tmp_path / "mascot.png")
    monkeypatch.setattr(mas, "active_backend_id", lambda: mas.BACKEND_ID)
    monkeypatch.setattr(mas, "backend_healthy", lambda: (True, "ok"))
    seen = {}

    class R:
        success = True
        image_path = None
    def fake(**kw):
        seen.update(kw)
        Image.new("RGB", (8, 8)).save(kw["output_path"])
        R.image_path = kw["output_path"]
        return R
    monkeypatch.setattr(qwen, "generate", fake)

    mas.render_scene("a scene", tmp_path / "o.png", headline="BEE FACTS",
                     presenter=True)
    assert "full body visible" in seen["prompt"]
    assert "chibi proportions" in seen["prompt"], "the cub's build is restated"
    assert "headline text" not in seen["prompt"], "no baked title on a video still"
    assert "hand passing through object" in seen["negative_prompt"]


def test_explainer_scene_prompt_asks_for_fun_and_clean_props():
    """With S2V gone the mascot may leap again, but the prop must not pass through
    it — a paw once went through a hat brim."""
    from modules import mascot as mas
    assert "FULL BODY in frame" in mas._EXPLAINER_SYS
    assert "FUN and VISUALLY INTERESTING" in mas._EXPLAINER_SYS
    assert "never overlap or pass through" in mas._EXPLAINER_SYS
    assert "STANDS ON BOTH FEET" not in mas._EXPLAINER_SYS


def test_s2v_negative_guards_limbs():
    from modules.video_backends import comfyui_wan_s2v as s2v
    assert "deformed legs" in s2v.DEFAULT_NEGATIVE
    assert "broken limbs" in s2v.DEFAULT_NEGATIVE


# ---------------------------------------------------------------- prose salvage

def test_a_prose_answer_is_salvaged_not_dropped():
    """Measured on a live reel: the model replied with prose instead of JSON —
    "Note: To meet your requirement strictly ..." — the JSON parse came back
    empty and that shot silently fell back to a generic 'standing next to
    honeybees' pose. The scene was still in the text."""
    from modules import mascot as mas
    raw = ("Note: To meet your requirement strictly with no background.\n"
           "the mascot character in a beekeeper suit lifting a dripping "
           "honeycomb, proud grin")
    got = mas._salvage_scene(raw)
    assert "beekeeper suit" in got
    assert not got.lower().startswith("note")


def test_salvage_ignores_the_models_own_chatter():
    from modules import mascot as mas
    assert mas._salvage_scene("Sure! Here is the JSON you asked for.") == ""
    assert mas._salvage_scene("") == ""


def test_explainer_falls_back_to_prose_when_json_is_missing(monkeypatch):
    from modules import mascot as mas
    import modules.script_generator as sg
    monkeypatch.setattr(sg, "_call_llm",
                        lambda *a, **k: "Note: no JSON.\nthe mascot character "
                                        "in a bee costume waving a honeycomb")
    monkeypatch.setattr(sg, "_extract_json", lambda r: {})
    got = mas.explainer_scene("Bees make honey.", topic="bees")
    assert "bee costume" in got, "the costumed scene must survive a prose answer"
