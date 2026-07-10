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

    def fake_render(scene, out_png, aspect="9x16", seed=None, headline=""):
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

    def fake_render(scene, png, aspect="9x16", seed=None, reference_images=None,
                    headline=""):
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


# ---------------------------------------------------------------- residency

def test_prepare_gpu_delegates_to_the_residency_manager(monkeypatch):
    from modules import gpu_memory as gm
    calls = []
    monkeypatch.setattr(gm, "acquire", lambda label: calls.append(label))
    monkeypatch.setattr(mas, "active_backend_id", lambda: mas.BACKEND_ID)
    mas.prepare_gpu()
    assert calls == [gm.QWEN_EDIT]


def test_release_hands_the_card_back(monkeypatch):
    from modules import gpu_memory as gm
    calls = []
    monkeypatch.setattr(gm, "release", lambda label=None: calls.append(label))
    monkeypatch.setattr(mas, "active_backend_id", lambda: mas.BACKEND_ID)
    mas.release()
    assert calls == [gm.QWEN_EDIT]


def test_uso_fallback_reserves_its_own_label(monkeypatch):
    from modules import gpu_memory as gm
    calls = []
    monkeypatch.setattr(gm, "acquire", lambda label: calls.append(label))
    monkeypatch.setattr(mas, "active_backend_id", lambda: mas.FALLBACK_BACKEND_ID)
    mas.prepare_gpu()
    assert calls == [gm.FLUX_USO]


# ---------------------------------------------------------------- multi-angle refs

def test_mascot_refs_defaults_to_the_front_view_only(assets):
    """Measured: extra angles make Qwen copy a reference stance instead of
    acting out the scene, and the comic expression is what sells a thumbnail."""
    for n in mas.MASCOT_ANGLE_NAMES:
        Image.new("RGB", (64, 64)).save(assets / n)
    assert [p.name for p in mas.mascot_refs()] == ["mascot_front.png"]


def test_mascot_refs_prefers_the_angle_set(assets):
    for n in mas.MASCOT_ANGLE_NAMES:
        Image.new("RGB", (64, 64)).save(assets / n)
    refs = mas.mascot_refs(3)
    assert [p.name for p in refs] == ["mascot_front.png",
                                      "mascot_threequarter.png",
                                      "mascot_side.png"]
    assert all("back" not in p.name for p in refs), "back view has no face or logo"


def test_mascot_refs_falls_back_to_the_single_image(installed):
    assert [p.name for p in mas.mascot_refs(3)] == ["mascot.png"]


def test_mascot_refs_respects_the_cap(assets):
    for n in mas.MASCOT_ANGLE_NAMES:
        Image.new("RGB", (64, 64)).save(assets / n)
    assert len(mas.mascot_refs(1)) == 1
    assert len(mas.mascot_refs(3)) == 3      # never 4: Qwen takes image1..image3


def test_render_scene_sends_the_front_ref_to_qwen(assets, tmp_path, monkeypatch):
    for n in mas.MASCOT_ANGLE_NAMES:
        Image.new("RGB", (64, 64)).save(assets / n)
    Image.new("RGB", (64, 64)).save(assets / "mascot.png")

    from modules.image_backends import comfyui_qwen_edit as qwen
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
    monkeypatch.setattr(mas, "active_backend_id", lambda: mas.BACKEND_ID)
    monkeypatch.setattr(mas, "backend_healthy", lambda: (True, "ok"))

    mas.render_scene("scene", tmp_path / "o.png", seed=1)
    assert len(seen["reference_images"]) == 1     # front only, by default
    assert "reference_image" not in seen          # Qwen wants the list form

    seen.clear()
    mas.render_scene("scene", tmp_path / "o2.png", seed=1,
                     reference_images=[assets / n for n in mas.MASCOT_ANGLE_NAMES[:3]])
    assert len(seen["reference_images"]) == 3     # explicit multi-ref still works


def test_uso_fallback_gets_only_one_reference(assets, tmp_path, monkeypatch):
    for n in mas.MASCOT_ANGLE_NAMES:
        Image.new("RGB", (64, 64)).save(assets / n)

    from modules.image_backends import comfyui_uso as uso
    seen = {}

    class R:
        success = True
        image_path = None
    def fake(**kw):
        seen.update(kw)
        Image.new("RGB", (8, 8)).save(kw["output_path"])
        R.image_path = kw["output_path"]
        return R
    monkeypatch.setattr(uso, "generate", fake)
    monkeypatch.setattr(mas, "active_backend_id", lambda: mas.FALLBACK_BACKEND_ID)
    monkeypatch.setattr(mas, "backend_healthy", lambda: (True, "ok"))

    mas.render_scene("scene", tmp_path / "o.png", seed=1)
    assert "reference_images" not in seen     # USO is single-reference only
    assert seen["reference_image"].name == "mascot_front.png"


def test_angle_set_alone_is_a_valid_install(assets):
    """Dropping in only mascot_front.png must not silently disable the feature."""
    Image.new("RGB", (64, 64)).save(assets / "mascot_front.png")
    assert mas.mascot_path().name == "mascot_front.png"
    ok, why = mas.is_available()
    assert "no mascot image" not in why


def test_prewritten_scene_skips_the_llm(installed, tmp_path, monkeypatch):
    """Bulk runs write every scene up front, then unload Ollama. Passing the
    scene back in must not re-invoke the LLM (which would reload 12.6 GB)."""
    called = []
    monkeypatch.setattr(mas, "scene_prompt", lambda *a, **k: called.append(1) or "x")
    monkeypatch.setattr(mas, "prepare_gpu", lambda: None)
    monkeypatch.setattr(mas, "release", lambda: None)
    monkeypatch.setattr(mas, "backend_healthy", lambda: (True, "ok"))
    monkeypatch.setattr(mas, "render_scene",
                        lambda s, p, **k: (Image.new("RGB", (8, 8)).save(p), p)[1])

    art = mas.render_for_video("T", "ctx", tmp_path, "s", aspects=("9x16",),
                               scene="the mascot character eating cake")
    assert not called, "a pre-written scene must not call the LLM again"
    assert art["_scene"] == "the mascot character eating cake"


def test_missing_scene_still_calls_the_llm(installed, tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(mas, "scene_prompt", lambda *a, **k: called.append(1) or "written")
    monkeypatch.setattr(mas, "prepare_gpu", lambda: None)
    monkeypatch.setattr(mas, "release", lambda: None)
    monkeypatch.setattr(mas, "backend_healthy", lambda: (True, "ok"))
    monkeypatch.setattr(mas, "render_scene",
                        lambda s, p, **k: (Image.new("RGB", (8, 8)).save(p), p)[1])
    art = mas.render_for_video("T", "ctx", tmp_path, "s", aspects=("9x16",))
    assert called == [1]
    assert art["_scene"] == "written"


# ---------------------------------------------------------------- scene safety

def test_scene_violation_catches_the_real_offenders():
    """Both of these were actually produced by the LLM on real reels."""
    assert mas.scene_violation(
        "the mascot character caught with its tongue stuck to a forbidden sign "
        "while trying to type hate speech on a laptop") == "hate speech"
    assert mas.scene_violation(
        "the mascot character stuffing a heart-shaped liver into its mouth, "
        "cheeks bulging") == "liver"


@pytest.mark.parametrize("scene,bad", [
    ("the mascot character holding a gun", "gun"),
    ("the mascot character covered in blood", "blood"),
    ("the mascot character smoking a cigarette", "cigarette"),
    ("the mascot character holding a knife", "knife"),
])
def test_scene_violation_blocks_unbrandable_imagery(scene, bad):
    assert mas.scene_violation(scene) == bad


@pytest.mark.parametrize("scene", [
    "the mascot character eating a slice of chocolate cake",
    "the mascot character dodging a flying goldfish",
    "the mascot character in a hangar next to an organic apple",   # hang/organ
])
def test_scene_violation_allows_normal_scenes(scene):
    assert mas.scene_violation(scene) is None, "whole-word matching only"


def test_unsafe_llm_scene_is_rerolled_then_falls_back(monkeypatch):
    import modules.script_generator as sg
    calls = []
    monkeypatch.setattr(sg, "_call_llm", lambda *a, **k: calls.append(1) or
                        '{"scene": "the mascot character holding a gun"}')
    out = mas.scene_prompt("Hatred", "a video about hatred", "hatred")
    assert len(calls) == 2, "one re-roll before giving up"
    assert mas.scene_violation(out) is None
    assert "gun" not in out


def test_unsafe_first_try_then_safe_second_is_accepted(monkeypatch):
    import modules.script_generator as sg
    seq = ['{"scene": "the mascot character covered in blood"}',
           '{"scene": "the mascot character hugging a giant heart balloon"}']
    monkeypatch.setattr(sg, "_call_llm", lambda *a, **k: seq.pop(0))
    out = mas.scene_prompt("Love", "a video about love", "love")
    assert "heart balloon" in out


def test_abstract_topic_falls_back_to_a_neutral_scene():
    """'the mascot standing next to hatred' is not a thumbnail."""
    scene = mas.fallback_scene("Hatred's Hidden Costs", "hate is costly", "hate")
    assert mas.scene_violation(scene) is None
    assert scene == mas._NEUTRAL_SCENE


def test_concrete_topic_keeps_the_object_in_the_fallback():
    scene = mas.fallback_scene("Goldfish Facts", "goldfish recognize owners", "goldfish")
    assert "goldfish" in scene


# ---------------------------------------------------------------- fatal GPU fault

def test_fatal_signs_classify_a_dead_cuda_context():
    from modules.image_backends import comfyui_qwen_edit as qwen
    assert qwen._is_fatal("CUDA error: an illegal memory access was encountered")
    assert qwen._is_fatal("torch.OutOfMemoryError: CUDA out of memory")
    assert qwen._is_fatal("device-side assert triggered")
    assert not qwen._is_fatal("Qwen produced no image")
    assert not qwen._is_fatal("connection refused")


def test_fatal_render_raises_instead_of_using_a_still(installed, tmp_path, monkeypatch):
    """The bug: a dead CUDA context made 12 of 14 videos quietly fall back to a
    still-frame thumbnail while the run reported '14 kits built, 0 failures'."""
    from modules.image_backends import comfyui_qwen_edit as qwen

    class R:
        success = False
        fatal = True
        error = "CUDA error: an illegal memory access was encountered"
    monkeypatch.setattr(qwen, "generate", lambda **k: R)
    monkeypatch.setattr(mas, "active_backend_id", lambda: mas.BACKEND_ID)
    monkeypatch.setattr(mas, "backend_healthy", lambda: (True, "ok"))
    from modules import gpu_memory as gm
    gm._set_resident(gm.QWEN_EDIT)

    with pytest.raises(mas.MascotGpuFault):
        mas.render_scene("scene", tmp_path / "o.png", seed=1)
    assert gm.resident_model() is None, "a dead context is not a resident model"


def test_non_fatal_render_failure_still_falls_back(installed, tmp_path, monkeypatch):
    from modules.image_backends import comfyui_qwen_edit as qwen

    class R:
        success = False
        fatal = False
        error = "Qwen produced no image"
    monkeypatch.setattr(qwen, "generate", lambda **k: R)
    monkeypatch.setattr(mas, "active_backend_id", lambda: mas.BACKEND_ID)
    monkeypatch.setattr(mas, "backend_healthy", lambda: (True, "ok"))
    assert mas.render_scene("scene", tmp_path / "o.png", seed=1) is None


def test_publish_kit_surfaces_the_fatal_fault(tmp_path, monkeypatch):
    import modules.runtime_settings as rs
    monkeypatch.setattr(rs, "get_mascot_thumbnails_enabled", lambda: True)
    monkeypatch.setattr(mas, "render_for_video",
                        lambda **k: (_ for _ in ()).throw(
                            mas.MascotGpuFault("CUDA context lost")))
    video = tmp_path / "reel_9x16.mp4"
    video.write_bytes(b"x")
    kit = pk.attach(video, "T", context="ctx", aspects=("9x16",))
    assert "CUDA context lost" in kit["mascot_fatal"]
    assert kit["thumb_source"] != "mascot"      # still degrades for ONE video
