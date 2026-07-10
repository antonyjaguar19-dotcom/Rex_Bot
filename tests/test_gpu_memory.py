"""gpu_memory — one owner for "which model holds the 16 GB card".

Each test here is a regression for a failure that actually happened in one day:

  1. Ollama stayed loaded during a Wan render -> a 90s clip took 16 minutes.
  2. A warm batch evicted its own resident model before every item -> 14 cold
     loads of ~4 minutes each.
  3. Qwen ran with 211 MB free -> ComfyUI's offload streams killed the CUDA
     context mid-batch, and 12 videos silently got the wrong thumbnails.
"""

import pytest

from modules import gpu_memory as gm


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    gm.forget()
    calls = []
    monkeypatch.setattr(gm, "unload_llm", lambda: calls.append("llm"))
    monkeypatch.setattr(gm, "unload_image_models", lambda: calls.append("comfy"))
    monkeypatch.setattr(gm, "free_gb", lambda: 15.0)
    yield calls
    gm.forget()


# ---------------------------------------------------------------- residency

def test_acquire_evicts_then_marks_resident(clean):
    gm.acquire(gm.QWEN_EDIT)
    assert clean == ["llm", "comfy"]
    assert gm.resident_model() == gm.QWEN_EDIT


def test_acquiring_the_same_model_twice_evicts_nothing(clean):
    """Bug 2: a warm batch must not evict the model it is about to use."""
    gm.acquire(gm.QWEN_EDIT)
    clean.clear()
    gm.acquire(gm.QWEN_EDIT)
    gm.acquire(gm.QWEN_EDIT)
    assert clean == [], "re-acquiring the resident model must be a no-op"


def test_a_different_model_does_evict(clean):
    """Bug 1: Wan must not start while Ollama's 12.6 GB is still loaded."""
    gm.acquire(gm.OLLAMA)
    clean.clear()
    gm.acquire(gm.WAN_VIDEO)
    assert clean == ["llm", "comfy"]
    assert gm.resident_model() == gm.WAN_VIDEO


def test_release_frees_and_clears(clean):
    gm.acquire(gm.QWEN_EDIT)
    clean.clear()
    gm.release(gm.QWEN_EDIT)
    assert clean == ["comfy"]
    assert gm.resident_model() is None


def test_release_does_not_evict_someone_elses_model(clean):
    """A stale release() must not pull the card out from under the next job."""
    gm.acquire(gm.WAN_VIDEO)
    clean.clear()
    gm.release(gm.QWEN_EDIT)          # we no longer hold it
    assert clean == []
    assert gm.resident_model() == gm.WAN_VIDEO


def test_release_without_a_label_always_frees(clean):
    gm.acquire(gm.WAN_VIDEO)
    clean.clear()
    gm.release()
    assert clean == ["comfy"]
    assert gm.resident_model() is None


def test_forget_drops_state_without_freeing(clean):
    """A dead CUDA context is not a resident model, and freeing it is pointless."""
    gm.acquire(gm.QWEN_EDIT)
    clean.clear()
    gm.forget()
    assert clean == []
    assert gm.resident_model() is None


# ---------------------------------------------------------------- headroom

def test_critical_threshold_sits_between_the_two_observed_runs():
    """0.2 GB killed the context; 13.4 GB ran eight Wan clips cleanly."""
    assert 0.2 < gm.CRITICAL_FREE_GB < 13.4


def test_needs_gb_is_weights_only():
    """ComfyUI reserves its own working room via --reserve-vram, so adding it
    here too would warn on every healthy load and teach you to ignore the log."""
    assert gm.needs_gb(gm.QWEN_EDIT) == pytest.approx(13.5)
    assert gm.needs_gb("something_unknown") == pytest.approx(8.0)


@pytest.mark.parametrize("free", [14.4, 13.4, 2.5])
def test_a_normal_load_does_not_warn(clean, monkeypatch, caplog, free):
    """13.4 GB free loaded a 13.5 GB model fine in a real 8-clip Wan run:
    ComfyUI streams and offloads. 'free >= weights' was never the right test,
    and a card capped at ~14.4 GB free could never satisfy it."""
    monkeypatch.setattr(gm, "free_gb", lambda: free)
    monkeypatch.setattr(gm.time, "sleep", lambda s: None)
    with caplog.at_level("WARNING"):
        gm.acquire(gm.QWEN_EDIT)
    assert "no working room" not in caplog.text


def test_no_working_room_warns_but_proceeds(clean, monkeypatch, caplog):
    """The 211 MB case: this is what actually killed the CUDA context."""
    monkeypatch.setattr(gm, "free_gb", lambda: 0.2)
    monkeypatch.setattr(gm.time, "sleep", lambda s: None)
    with caplog.at_level("WARNING"):
        gm.acquire(gm.QWEN_EDIT)
    assert "no working room" in caplog.text
    assert gm.resident_model() == gm.QWEN_EDIT           # ComfyUI can still offload


# ---------------------------------------------------------------- context managers

def test_resident_context_releases_on_success(clean):
    with gm.resident(gm.QWEN_EDIT):
        assert gm.resident_model() == gm.QWEN_EDIT
    assert gm.resident_model() is None


def test_resident_context_releases_on_exception(clean):
    with pytest.raises(RuntimeError):
        with gm.resident(gm.QWEN_EDIT):
            raise RuntimeError("render blew up")
    assert gm.resident_model() is None, "a crash must not leak 13.5 GB"


def test_resident_context_is_warm_inside(clean):
    with gm.resident(gm.QWEN_EDIT):
        clean.clear()
        gm.acquire(gm.QWEN_EDIT)      # a second render in the same batch
        gm.acquire(gm.QWEN_EDIT)
        assert clean == [], "renders inside the context must stay warm"


def test_llm_context_unloads_the_model_afterwards(clean):
    with gm.llm():
        assert gm.resident_model() == gm.OLLAMA
    assert clean[-1] == "llm"
    assert gm.resident_model() is None


def test_every_known_model_has_a_size():
    for label in (gm.OLLAMA, gm.WAN_VIDEO, gm.QWEN_EDIT, gm.FLUX_USO,
                  gm.ZIMAGE, gm.SDXL):
        assert gm.MODEL_VRAM_GB[label] > 0


def test_two_big_models_cannot_coexist_on_16gb():
    """The arithmetic behind the whole module."""
    card = 16.0
    assert gm.MODEL_VRAM_GB[gm.WAN_VIDEO] + gm.MODEL_VRAM_GB[gm.OLLAMA] > card
    assert gm.MODEL_VRAM_GB[gm.QWEN_EDIT] + gm.MODEL_VRAM_GB[gm.WAN_VIDEO] > card


def test_a_slow_free_is_remeasured_before_warning(clean, monkeypatch, caplog):
    """Freeing is async: ComfyUI returns from /free before CUDA hands the memory
    back. Measuring once reported '0.6 GB free' on a card that settled to 14.4.
    """
    readings = iter([0.6, 14.4])          # immediately after evict, then settled
    monkeypatch.setattr(gm, "free_gb", lambda: next(readings))
    monkeypatch.setattr(gm.time, "sleep", lambda s: None)
    with caplog.at_level("WARNING"):
        gm.acquire(gm.QWEN_EDIT)
    assert "no working room" not in caplog.text, "must re-measure before warning"


def test_a_genuinely_full_card_still_warns(clean, monkeypatch, caplog):
    monkeypatch.setattr(gm, "free_gb", lambda: 0.2)   # stays full after settling
    monkeypatch.setattr(gm.time, "sleep", lambda s: None)
    with caplog.at_level("WARNING"):
        gm.acquire(gm.QWEN_EDIT)
    assert "no working room" in caplog.text
