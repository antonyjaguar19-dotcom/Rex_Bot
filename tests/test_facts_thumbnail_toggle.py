"""Facts mode can skip the thumbnail — a whole extra stage (LLM headline + Qwen).

Off writes the paste-ready title and description, but no cover image.
"""

import pytest
from PIL import Image

from modules import publish_kit as pk
import modules.runtime_settings as rs


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Never touch the real 05_Config/runtime_settings.json."""
    monkeypatch.setattr(rs, "SETTINGS_PATH", tmp_path / "runtime_settings.json")
    yield


def test_default_is_on():
    # Nothing stored -> a Short gets a cover by default.
    assert rs.get_facts_thumbnail_enabled() is True


def test_set_round_trips():
    rs.set_facts_thumbnail_enabled(False)
    assert rs.get_facts_thumbnail_enabled() is False
    rs.set_facts_thumbnail_enabled(True)
    assert rs.get_facts_thumbnail_enabled() is True


def test_attach_thumbnail_false_writes_title_but_no_image(tmp_path, monkeypatch):
    monkeypatch.setattr(pk, "build_title", lambda *a, **k: "Bee Facts")
    # If the headline or mascot ran, these would blow up — proving they don't.
    monkeypatch.setattr(pk, "build_headline",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("headline ran")))
    monkeypatch.setattr(pk, "_mascot_art",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("mascot ran")))

    v = tmp_path / "facts_1_9x16.mp4"
    v.write_bytes(b"\x00")
    kit = pk.attach(v, "Bee Facts", description="d", thumbnail=False)

    assert kit["title"] == "Bee Facts"
    assert kit["thumb_source"] == "disabled"
    assert "thumb_9x16" not in kit and "thumb_16x9" not in kit
    assert (tmp_path / "facts_1_9x16_title.txt").exists()
    assert (tmp_path / "facts_1_9x16_description.txt").exists()
    assert not (tmp_path / "facts_1_9x16_thumb_9x16.jpg").exists()


def test_attach_thumbnail_true_still_renders(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "get_mascot_thumbnails_enabled", lambda: False)
    monkeypatch.setattr(pk, "build_title", lambda *a, **k: "T")
    monkeypatch.setattr(pk, "build_headline", lambda *a, **k: ("T", "T"))
    frame = tmp_path / "still.png"
    Image.new("RGB", (1080, 1920)).save(frame)
    v = tmp_path / "facts_1_9x16.mp4"
    v.write_bytes(b"\x00")
    kit = pk.attach(v, "T", source_image=frame, thumbnail=True)
    assert "thumb_9x16" in kit
