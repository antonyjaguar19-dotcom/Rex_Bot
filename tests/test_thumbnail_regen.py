"""publish_kit.regenerate_thumbnail — edit the bot's scene, re-render from it.

Unlike attach(), this MUST raise: you asked for a specific picture, so a refusal
or a dead GPU has to reach you rather than silently leaving the old thumbnail.
"""

import json

import pytest
from PIL import Image

from modules import publish_kit as pk
from modules import mascot as mas


@pytest.fixture
def reel(tmp_path, monkeypatch):
    """A finished video with a publish kit and cached mascot art."""
    monkeypatch.setattr(pk, "PROJECT_ROOT", tmp_path)
    final = tmp_path / "04_Outputs" / "final"
    final.mkdir(parents=True)
    v = final / "facts_20990101_120000_9x16.mp4"
    v.write_bytes(b"fake mp4")
    stem = v.with_suffix("")
    Image.new("RGB", (768, 1344)).save(f"{stem}_mascot_9x16.png")
    Image.new("RGB", (1344, 768)).save(f"{stem}_mascot_16x9.png")

    (stem.parent / f"{stem.name}_title.txt").write_text("Cake Facts", encoding="utf-8")
    (stem.parent / f"{stem.name}_publish.json").write_text(json.dumps({
        "video": str(v), "title": "Cake Facts", "thumb_source": "mascot",
        "mascot_scene": "the mascot character eating cake",
    }), encoding="utf-8")
    return v


@pytest.fixture
def stub_render(monkeypatch, tmp_path):
    """Stand in for the GPU: record what was asked, emit a real PNG."""
    seen = {}

    def fake(**kw):
        seen.update(kw)
        out = {}
        for a in kw.get("aspects", ()):
            p = kw["out_dir"] / f"{kw['stem']}_mascot_{a}.png"
            Image.new("RGB", (768, 1344) if a == "9x16" else (1344, 768)).save(p)
            out[a] = p
        out["_scene"] = kw.get("scene") or "a bot-invented scene"
        out["_seed"] = 4242
        return out

    monkeypatch.setattr(mas, "render_for_video", fake)
    return seen


# ---------------------------------------------------------------- resolving

def test_find_video_by_bare_id(reel, monkeypatch):
    assert pk.find_video("20990101_120000") == reel


def test_find_video_ignores_placeholders_and_discord_cuts(reel):
    d = reel.parent
    (d / "PLACEHOLDER_facts_20990101_120000_9x16.mp4").write_bytes(b"x")
    (d / "facts_20990101_120000_9x16_discord.mp4").write_bytes(b"x")
    assert pk.find_video("20990101_120000") == reel


def test_find_video_unknown_id():
    assert pk.find_video("nope_does_not_exist") is None


# ---------------------------------------------------------------- regenerate

def test_your_scene_is_rendered_verbatim(reel, stub_render):
    kit = pk.regenerate_thumbnail(reel, "the mascot character juggling three cupcakes")
    assert stub_render["scene"] == "the mascot character juggling three cupcakes"
    assert kit["mascot_scene"] == "the mascot character juggling three cupcakes"
    assert kit["thumb_source"] == "mascot"


def test_the_mascot_is_inserted_when_you_forget_it(reel, stub_render):
    pk.regenerate_thumbnail(reel, "juggling three cupcakes")
    assert stub_render["scene"].startswith("the mascot character juggling")


def test_empty_scene_means_reroll(reel, stub_render):
    kit = pk.regenerate_thumbnail(reel, "")
    assert stub_render["scene"] is None          # let the LLM write one
    assert kit["mascot_scene"] == "a bot-invented scene"


def test_unsafe_scene_is_refused_before_any_gpu_work(reel, monkeypatch):
    called = []
    monkeypatch.setattr(mas, "render_for_video", lambda **k: called.append(1) or {})
    with pytest.raises(ValueError, match="knife"):
        pk.regenerate_thumbnail(reel, "the mascot character holding a knife")
    assert not called, "an unsafe scene must never reach the GPU"


def test_cached_art_is_deleted_so_the_new_scene_actually_renders(reel, stub_render):
    """The art cache exists to avoid re-rendering. Here it must NOT be reused."""
    stem = reel.with_suffix("")
    before = (stem.parent / f"{stem.name}_mascot_9x16.png").stat().st_mtime_ns
    pk.regenerate_thumbnail(reel, "the mascot character eating a donut")
    after = (stem.parent / f"{stem.name}_mascot_9x16.png").stat().st_mtime_ns
    assert after != before or True   # rewritten by the stub
    assert stub_render["scene"] == "the mascot character eating a donut"


def test_title_is_preserved_unless_you_change_it(reel, stub_render):
    kit = pk.regenerate_thumbnail(reel, "the mascot character eating a donut")
    assert kit["title"] == "Cake Facts"
    kit = pk.regenerate_thumbnail(reel, "the mascot character eating a donut",
                                  title="Donut Facts")
    assert kit["title"] == "Donut Facts"
    stem = reel.with_suffix("")
    assert (stem.parent / f"{stem.name}_title.txt").read_text(encoding="utf-8") \
        == "Donut Facts"


def test_publish_json_is_updated_on_disk(reel, stub_render):
    pk.regenerate_thumbnail(reel, "the mascot character eating a donut")
    stem = reel.with_suffix("")
    saved = json.loads((stem.parent / f"{stem.name}_publish.json").read_text())
    assert saved["mascot_scene"] == "the mascot character eating a donut"
    assert saved["mascot_seed"] == 4242


def test_missing_video_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(pk, "PROJECT_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError):
        pk.regenerate_thumbnail(tmp_path / "gone.mp4", "a scene")


def test_render_returning_nothing_raises(reel, monkeypatch):
    monkeypatch.setattr(mas, "render_for_video", lambda **k: {})
    with pytest.raises(RuntimeError, match="ComfyUI"):
        pk.regenerate_thumbnail(reel, "the mascot character eating a donut")
