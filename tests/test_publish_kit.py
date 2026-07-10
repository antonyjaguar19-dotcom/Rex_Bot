"""publish_kit — title + thumbnail beside every finished video.

The rule that matters most: a thumbnail is cosmetic, the render cost an hour of
GPU. Nothing in here may raise into a pipeline.
"""

import json
import subprocess

import pytest
from PIL import Image

from modules import publish_kit as pk


@pytest.fixture
def video(tmp_path):
    from modules.assembly import FFMPEG_EXE
    out = tmp_path / "facts_20990101_120000_9x16.mp4"
    subprocess.run(
        [str(FFMPEG_EXE), "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "testsrc=size=360x640:rate=12:duration=2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)],
        check=True, timeout=120)
    return out


@pytest.fixture
def still(tmp_path):
    p = tmp_path / "bg_0001.png"
    Image.new("RGB", (720, 1280), (30, 80, 140)).save(p)
    return p


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    """Default: no LLM. build_title must fall back, not hang or raise."""
    import modules.script_generator as sg
    monkeypatch.setattr(sg, "_call_llm",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))


@pytest.fixture(autouse=True)
def no_mascot(monkeypatch, tmp_path):
    """Never touch the real mascot asset or the GPU from these tests.

    Once 02_Agent/assets/mascot.png existed, attach() started calling USO for
    real: the suite went from seconds to minutes and depended on ComfyUI being
    up. Tests must not depend on installed assets.
    """
    import modules.mascot as mas
    monkeypatch.setattr(mas, "ASSETS_DIR", tmp_path / "no_assets")
    monkeypatch.setattr(mas, "render_for_video", lambda **k: {})


# ---------------------------------------------------------------- title

def test_clean_title_strips_quotes_hashtags_and_newlines():
    assert pk._clean_title('"Wow: Bees"\nsecond line') == "Wow: Bees"
    assert pk._clean_title("Bees Rule #shorts #facts") == "Bees Rule"
    assert pk._clean_title("  spaced   out  ") == "spaced out"


def test_clean_title_truncates_on_a_word_boundary():
    long = "word " * 40
    t = pk._clean_title(long)
    assert len(t) <= pk.TITLE_MAX + 1     # +1 for the ellipsis
    assert t.endswith("…")
    assert "wor…" not in t                # never cut mid-word


def test_build_title_falls_back_when_llm_is_down():
    assert pk.build_title("Bee Facts", "some context", "facts") == "Bee Facts"


def test_build_title_skips_llm_without_context():
    assert pk.build_title("Bee Facts", "", "facts") == "Bee Facts"


def test_build_title_uses_llm_when_it_answers(monkeypatch):
    import modules.script_generator as sg
    monkeypatch.setattr(sg, "_call_llm", lambda *a, **k: '{"title": "Bees Are Wild"}')
    assert pk.build_title("Bee Facts", "ctx", "facts") == "Bees Are Wild"


def test_build_title_rejects_empty_llm_answer(monkeypatch):
    import modules.script_generator as sg
    monkeypatch.setattr(sg, "_call_llm", lambda *a, **k: '{"title": "   "}')
    assert pk.build_title("Bee Facts", "ctx", "facts") == "Bee Facts"


# ---------------------------------------------------------------- thumbnail

def test_render_thumbnail_sizes_and_format(still, tmp_path):
    out = tmp_path / "t.jpg"
    assert pk.render_thumbnail(still, "A Readable Title", out, pk.THUMB_16X9)
    assert Image.open(out).size == pk.THUMB_16X9
    out2 = tmp_path / "t2.jpg"
    assert pk.render_thumbnail(still, "A Readable Title", out2, pk.THUMB_9X16)
    assert Image.open(out2).size == pk.THUMB_9X16


def test_thumbnail_stays_under_youtubes_2mb_cap(still, tmp_path):
    out = tmp_path / "t.jpg"
    pk.render_thumbnail(still, "Title", out, pk.THUMB_16X9)
    assert out.stat().st_size < pk.THUMB_MAX_BYTES


def test_very_long_title_still_renders(still, tmp_path):
    out = tmp_path / "t.jpg"
    assert pk.render_thumbnail(still, "word " * 30, out, pk.THUMB_16X9)


def test_render_thumbnail_returns_none_on_bad_input(tmp_path):
    assert pk.render_thumbnail(tmp_path / "missing.png", "T",
                               tmp_path / "o.jpg", pk.THUMB_16X9) is None


def test_grab_frame_from_video(video, tmp_path):
    out = tmp_path / "f.png"
    assert pk.grab_frame(video, out)
    assert out.exists()


# ---------------------------------------------------------------- attach

def test_attach_writes_the_whole_kit(video, still):
    kit = pk.attach(video, "Goldfish Facts", context="ctx",
                    description="a description", mode="facts", source_image=still)
    stem = video.with_suffix("")
    assert kit["title"] == "Goldfish Facts"
    assert kit["thumb_source"] == "still"
    for suffix in ("_title.txt", "_description.txt", "_publish.json",
                   "_thumb_16x9.jpg", "_thumb_9x16.jpg"):
        assert (stem.parent / f"{stem.name}{suffix}").exists(), suffix
    saved = json.loads((stem.parent / f"{stem.name}_publish.json").read_text())
    assert saved["title"] == "Goldfish Facts"


def test_attach_uses_video_frame_when_no_still(video):
    kit = pk.attach(video, "T", description="", mode="facts")
    assert kit["thumb_source"] == "video frame"
    assert Path_exists(kit.get("thumb_16x9"))
    # the intermediate frame must be cleaned up
    assert not (video.with_suffix("").parent /
                (video.stem + "_frame.png")).exists()


def Path_exists(p):
    from pathlib import Path
    return bool(p) and Path(p).exists()


def test_attach_never_raises_on_a_missing_video(tmp_path):
    kit = pk.attach(tmp_path / "gone.mp4", "T")      # must not raise
    assert "thumb_16x9" not in kit


def test_attach_never_raises_when_thumbnail_fails(video, monkeypatch):
    monkeypatch.setattr(pk, "render_thumbnail",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    kit = pk.attach(video, "Still Fine")             # render must survive
    assert kit["title"] == "Still Fine"


def test_attach_skips_description_file_when_empty(video, still):
    kit = pk.attach(video, "T", description="", source_image=still)
    stem = video.with_suffix("")
    assert not (stem.parent / f"{stem.name}_description.txt").exists()
    assert "description_file" not in kit


def test_attach_does_not_delete_a_caller_supplied_still(video, still):
    pk.attach(video, "T", source_image=still)
    assert still.exists(), "publish_kit deleted the pipeline's own still!"


# ---------------------------------------------------------------- grounding

CTX = ("Goldfish can actually recognize their owners and respond to them.\n"
       "Some fancy goldfish can have up to 5 eyes, growing extra ones.\n"
       "In the wild, goldfish can live up to 25 years.")
FB = "Amazing Goldfish Facts"


@pytest.mark.parametrize("title,grounded", [
    ("Goldfish Can Recognize Their Owners", True),      # straight from narration
    ("Goldfish Grow Extra Eyes", True),                 # unusual, but the script says it
    ("Amazing Goldfish Facts Revealed", True),          # hype words are allowed
    ("Goldfish Can Solve Calculus", False),             # invented claim
    ("Goldfish Live in Volcanoes", False),              # invented noun
    ("6 Surprising Water Facts", False),                # wrong subject entirely
])
def test_title_grounding(title, grounded):
    assert pk._title_is_grounded(title, CTX, FB) is grounded


def test_grounding_allows_simple_plurals():
    assert pk._title_is_grounded("The Owner Remembers", "owners remember", "t")


def test_ungrounded_llm_title_is_rejected_and_falls_back(monkeypatch):
    import modules.script_generator as sg
    monkeypatch.setattr(sg, "_call_llm",
                        lambda *a, **k: '{"title": "Goldfish Pilot Spaceships"}')
    assert pk.build_title(FB, CTX, "facts") == FB      # never ships the invention


def test_grounded_llm_title_is_kept(monkeypatch):
    import modules.script_generator as sg
    monkeypatch.setattr(sg, "_call_llm",
                        lambda *a, **k: '{"title": "Goldfish Recognize Their Owners"}')
    assert pk.build_title(FB, CTX, "facts") == "Goldfish Recognize Their Owners"
