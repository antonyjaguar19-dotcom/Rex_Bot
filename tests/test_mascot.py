"""mascot — branded thumbnails rendered by USO from a single reference image.

Two properties matter above all:
  * Until a mascot image exists this is a SILENT no-op — no errors, no cost,
    thumbnails keep working exactly as before.
  * Nothing here may raise into a pipeline. A GPU hiccup means an ordinary
    thumbnail, never a lost render.
"""

import pytest
from PIL import Image

from modules import mascot as mas
from modules import publish_kit as pk


@pytest.fixture
def assets(tmp_path, monkeypatch):
    monkeypatch.setattr(mas, "ASSETS_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def installed(assets):
    p = assets / "mascot.png"
    Image.new("RGB", (512, 512), (200, 90, 40)).save(p)
    return p


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    import modules.script_generator as sg
    monkeypatch.setattr(sg, "_call_llm",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))


# ---------------------------------------------------------------- availability

def test_no_mascot_image_means_unavailable(assets):
    assert mas.mascot_path() is None
    ok, why = mas.is_available()
    assert ok is False
    assert "no mascot image" in why


def test_mascot_found_once_installed(installed):
    assert mas.mascot_path() == installed


def test_empty_file_is_not_a_mascot(assets):
    (assets / "mascot.png").write_bytes(b"")
    assert mas.mascot_path() is None


def test_extension_priority(assets):
    (assets / "mascot.jpg").write_bytes(b"x")
    assert mas.mascot_path().name == "mascot.jpg"
    Image.new("RGB", (8, 8)).save(assets / "mascot.png")
    assert mas.mascot_path().name == "mascot.png"   # png wins


def test_render_scene_without_mascot_returns_none(assets):
    assert mas.render_scene("a scene", assets / "o.png") is None


def test_render_for_video_without_mascot_is_a_no_op(assets, tmp_path):
    assert mas.render_for_video("T", "ctx", tmp_path, "stem") == {}


# ---------------------------------------------------------------- scene prompt

def test_fallback_scene_names_the_mascot_and_a_real_noun():
    scene = mas.fallback_scene("Amazing Goldfish Facts",
                               "Goldfish recognize their owners.")
    assert "mascot character" in scene
    assert "goldfish" in scene.lower()


def test_fallback_scene_skips_filler_words():
    scene = mas.fallback_scene("Surprising Facts About Bees", "These are amazing things.")
    assert "surprising" not in scene.lower()
    assert "bees" in scene.lower()


def test_scene_prompt_falls_back_when_llm_is_down():
    scene = mas.scene_prompt("Goldfish Facts", "Goldfish recognize owners.")
    assert "mascot character" in scene


def test_scene_prompt_uses_llm(monkeypatch):
    import modules.script_generator as sg
    monkeypatch.setattr(sg, "_call_llm", lambda *a, **k:
                        '{"scene": "the mascot character holding a glass bowl '
                        'with an orange goldfish"}')
    scene = mas.scene_prompt("Goldfish Facts", "ctx about goldfish")
    assert scene.startswith("the mascot character holding")


def test_scene_prompt_forces_the_mascot_in(monkeypatch):
    import modules.script_generator as sg
    monkeypatch.setattr(sg, "_call_llm",
                        lambda *a, **k: '{"scene": "a goldfish in a bowl"}')
    scene = mas.scene_prompt("Goldfish Facts", "ctx")
    assert "mascot" in scene.lower()


def test_scene_prompt_strips_requests_for_text_in_the_image(monkeypatch):
    import modules.script_generator as sg
    monkeypatch.setattr(sg, "_call_llm", lambda *a, **k:
                        '{"scene": "the mascot character holding a sign with text saying hi"}')
    scene = mas.scene_prompt("T", "ctx")
    assert "text" not in scene.lower()


def test_scene_prompt_truncates_a_rambling_answer(monkeypatch):
    import modules.script_generator as sg
    monkeypatch.setattr(sg, "_call_llm",
                        lambda *a, **k: '{"scene": "the mascot character ' + "word " * 80 + '"}')
    assert len(mas.scene_prompt("T", "ctx").split()) <= 40


# ---------------------------------------------------------------- render path

def test_render_for_video_uses_one_scene_and_seed_for_every_aspect(installed, tmp_path, monkeypatch):
    calls = []

    def fake_render(scene, out_png, aspect="9x16", seed=None):
        calls.append((scene, aspect, seed))
        Image.new("RGB", (64, 64)).save(out_png)
        return out_png

    monkeypatch.setattr(mas, "render_scene", fake_render)
    monkeypatch.setattr(mas, "backend_healthy", lambda: (True, "ok"))

    art = mas.render_for_video("Goldfish Facts", "goldfish recognize owners",
                               tmp_path, "reel", aspects=("9x16", "16x9"))
    assert set(art) >= {"9x16", "16x9", "_scene", "_seed"}
    scenes = {c[0] for c in calls}
    seeds = {c[2] for c in calls}
    assert len(scenes) == 1, "each aspect must render the SAME scene"
    assert len(seeds) == 1, "each aspect must use the SAME seed"


def test_render_for_video_survives_a_failing_backend(installed, tmp_path, monkeypatch):
    monkeypatch.setattr(mas, "backend_healthy", lambda: (True, "ok"))
    monkeypatch.setattr(mas, "render_scene", lambda *a, **k: None)
    assert mas.render_for_video("T", "ctx", tmp_path, "s") == {}


def test_render_scene_never_raises_when_backend_explodes(installed, tmp_path, monkeypatch):
    import modules.image_backend as ib
    monkeypatch.setattr(ib, "get_named_backend",
                        lambda *_: (_ for _ in ()).throw(RuntimeError("comfy down")))
    assert mas.render_scene("scene", tmp_path / "o.png") is None


# ---------------------------------------------------------------- publish_kit wiring

def test_publish_kit_ignores_mascot_when_disabled(monkeypatch, tmp_path):
    import modules.runtime_settings as rs
    monkeypatch.setattr(rs, "get_mascot_thumbnails_enabled", lambda: False)
    called = []
    monkeypatch.setattr(mas, "render_for_video", lambda **k: called.append(1) or {})
    assert pk._mascot_art(tmp_path / "v.mp4", "T", "c", "facts", ("9x16",)) == {}
    assert not called, "mascot must not render when the toggle is off"


def test_publish_kit_falls_back_to_still_without_mascot(assets, tmp_path, monkeypatch):
    import modules.runtime_settings as rs
    monkeypatch.setattr(rs, "get_mascot_thumbnails_enabled", lambda: True)
    still = tmp_path / "bg.png"
    Image.new("RGB", (720, 1280), (10, 60, 90)).save(still)
    video = tmp_path / "reel_9x16.mp4"
    video.write_bytes(b"not really a video")     # attach() only needs it to exist

    kit = pk.attach(video, "Goldfish Facts", context="ctx", source_image=still)
    assert kit["thumb_source"] == "still"        # no mascot image installed
    assert "mascot_scene" not in kit


def test_publish_kit_uses_mascot_art_when_available(tmp_path, monkeypatch):
    import modules.runtime_settings as rs
    monkeypatch.setattr(rs, "get_mascot_thumbnails_enabled", lambda: True)

    art_9 = tmp_path / "art9.png"
    Image.new("RGB", (768, 1344), (120, 40, 30)).save(art_9)
    monkeypatch.setattr(mas, "render_for_video",
                        lambda **k: {"9x16": art_9, "_scene": "mascot with a goldfish",
                                     "_seed": 7})

    video = tmp_path / "reel_9x16.mp4"
    video.write_bytes(b"x")
    kit = pk.attach(video, "Goldfish Facts", context="ctx", aspects=("9x16",))
    assert kit["thumb_source"] == "mascot"
    assert kit["mascot_scene"] == "mascot with a goldfish"
    assert Image.open(kit["thumb_9x16"]).size == pk.THUMB_9X16


# ---------------------------------------------------------------- backend choice

def test_qwen_is_preferred_when_healthy(monkeypatch):
    import modules.image_backend as ib

    class Ok:
        def health_check(self): return True, "alive"
    monkeypatch.setattr(ib, "get_named_backend", lambda bid: Ok())
    assert mas.active_backend_id() == mas.BACKEND_ID == "comfyui_qwen_edit"


def test_falls_back_to_uso_when_qwen_unusable(monkeypatch):
    import modules.image_backend as ib

    def loader(bid):
        if bid == mas.BACKEND_ID:
            raise RuntimeError("qwen model missing")
        class Ok:
            def health_check(self): return True, "alive"
        return Ok()
    monkeypatch.setattr(ib, "get_named_backend", loader)
    assert mas.active_backend_id() == mas.FALLBACK_BACKEND_ID == "comfyui_uso"


def test_render_scene_routes_to_qwen(installed, tmp_path, monkeypatch):
    from modules.image_backends import comfyui_qwen_edit as qwen
    seen = {}

    class R:
        success = True
        image_path = tmp_path / "o.png"
    def fake(**kw):
        seen.update(kw)
        Image.new("RGB", (8, 8)).save(kw["output_path"])
        R.image_path = kw["output_path"]
        return R
    monkeypatch.setattr(qwen, "generate", fake)
    monkeypatch.setattr(mas, "active_backend_id", lambda: mas.BACKEND_ID)
    monkeypatch.setattr(mas, "backend_healthy", lambda: (True, "ok"))

    out = mas.render_scene("a scene", tmp_path / "o.png", aspect="9x16", seed=7)
    assert out
    assert seen["aspect_ratio"] == "9:16"
    assert seen["seed"] == 7
    assert seen["negative_prompt"] == mas.NEGATIVE     # qwen honours negatives
    assert "lora_strength" not in seen                 # that is a USO-only knob


def test_render_scene_routes_to_uso_fallback(installed, tmp_path, monkeypatch):
    from modules.image_backends import comfyui_uso as uso
    seen = {}

    class R:
        success = True
        image_path = tmp_path / "o.png"
    def fake(**kw):
        seen.update(kw)
        Image.new("RGB", (8, 8)).save(kw["output_path"])
        return R
    monkeypatch.setattr(uso, "generate", fake)
    monkeypatch.setattr(mas, "active_backend_id", lambda: mas.FALLBACK_BACKEND_ID)
    monkeypatch.setattr(mas, "backend_healthy", lambda: (True, "ok"))

    mas.render_scene("a scene", tmp_path / "o.png", seed=7)
    assert seen["lora_strength"] == mas.POSE_LORA_STRENGTH


# ---------------------------------------------------------------- memory policy

def test_render_for_video_writes_scene_before_touching_the_gpu(installed, tmp_path, monkeypatch):
    """Ollama must be unloaded AFTER the scene is written, not before."""
    order = []
    monkeypatch.setattr(mas, "scene_prompt", lambda *a, **k: order.append("llm") or "scene")
    monkeypatch.setattr(mas, "prepare_gpu", lambda: order.append("evict"))
    monkeypatch.setattr(mas, "release", lambda: order.append("release"))
    monkeypatch.setattr(mas, "backend_healthy", lambda: (True, "ok"))

    def fake_render(scene, png, aspect="9x16", seed=None, reference_images=None):
        order.append(f"render:{aspect}")
        Image.new("RGB", (8, 8)).save(png)
        return png
    monkeypatch.setattr(mas, "render_scene", fake_render)

    mas.render_for_video("T", "ctx", tmp_path, "s", aspects=("9x16", "16x9"))
    assert order == ["llm", "evict", "render:9x16", "render:16x9", "release"]


def test_release_after_false_keeps_the_model_warm(installed, tmp_path, monkeypatch):
    released = []
    monkeypatch.setattr(mas, "scene_prompt", lambda *a, **k: "scene")
    monkeypatch.setattr(mas, "prepare_gpu", lambda: None)
    monkeypatch.setattr(mas, "release", lambda: released.append(1))
    monkeypatch.setattr(mas, "backend_healthy", lambda: (True, "ok"))
    monkeypatch.setattr(mas, "render_scene",
                        lambda s, p, **k: (Image.new("RGB", (8, 8)).save(p), p)[1])

    mas.render_for_video("T", "ctx", tmp_path, "s", aspects=("9x16",),
                         release_after=False)
    assert not released, "bulk runs must keep the model resident"


def test_release_runs_even_when_a_render_raises(installed, tmp_path, monkeypatch):
    released = []
    monkeypatch.setattr(mas, "scene_prompt", lambda *a, **k: "scene")
    monkeypatch.setattr(mas, "prepare_gpu", lambda: None)
    monkeypatch.setattr(mas, "release", lambda: released.append(1))
    monkeypatch.setattr(mas, "backend_healthy", lambda: (True, "ok"))
    monkeypatch.setattr(mas, "render_scene",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        mas.render_for_video("T", "ctx", tmp_path, "s", aspects=("9x16",))
    assert released == [1], "a crash must not leak 13.5 GB of VRAM"
