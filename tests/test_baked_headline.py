"""The image model renders the title INTO the artwork.

Qwen is a text-rendering model. Measured on the bee reel, one seed:
    "BEE FACTS"                       -> perfect
    "Bees Have 5 Eyes"                -> perfect, numeral included
    "Worker Bees Flap Wings 230x/sec" -> perfect, wrapped to two lines by itself
It sizes and places the type as part of the composition, which a PIL overlay
cannot. USO cannot spell, so it always gets the overlay.
"""

import pytest
from PIL import Image

from modules import mascot as mas
from modules import publish_kit as pk


@pytest.fixture
def installed(tmp_path, monkeypatch):
    monkeypatch.setattr(mas, "ASSETS_DIR", tmp_path)
    Image.new("RGB", (512, 512)).save(tmp_path / "mascot.png")
    return tmp_path


# ---------------------------------------------------------------- can_bake

@pytest.mark.parametrize("title,ok", [
    ("BEE FACTS", True),
    ("Bees Have 5 Eyes", True),
    ("Worker Bees Flap Wings 230x/sec", True),          # 31 chars, verified
    ("Goldfish Can Recognize Their Owners 🐟", True),    # emoji is stripped first
    ("", False),
    ("x" * 60, False),                                   # long titles garble
])
def test_can_bake(title, ok):
    assert mas.can_bake(title) is ok


def test_bake_clause_strips_emoji_and_quotes_the_title():
    c = mas.bake_clause("Bees Have 5 Eyes 🐝")
    assert '"Bees Have 5 Eyes"' in c
    assert "🐝" not in c
    assert "perfectly spelled" in c


def test_bake_clause_asks_for_letters_in_the_scene_not_on_it():
    """"White sans-serif across the bottom" is the overlay we are escaping."""
    c = mas.bake_clause("Bee Facts")
    for want in ("three-dimensional", "casting soft shadows", "same art style"):
        assert want in c, want
    assert "sans-serif" not in c


def test_bake_clause_keeps_the_letters_clear_of_the_character():
    """First live render: the mascot stood across the words and '230 FLAPS'
    read as '2?0 F??S'. The type is the message; it cannot be occluded."""
    c = mas.bake_clause("230 Flaps A Second")
    assert "does not cover any letter" in c
    assert "completely visible and unobstructed" in c
    assert "character overlaps" not in c


def test_bake_clause_gives_each_aspect_its_own_layout():
    """Prose ("off to one side") did not move Qwen off centre; naming the region
    each element owns did. Portrait stacks, landscape splits."""
    tall = mas.bake_clause("BEE FACTS", "9x16")
    wide = mas.bake_clause("BEE FACTS", "16x9")
    assert "upper half" in tall and "lower half" in tall
    assert "right half" in wide and "far left" in wide
    # Squeezing the phrase into a region made Qwen delete words, then duplicate them.
    assert "no word is left out" in tall
    assert "appears exactly once" in tall
    assert mas.bake_clause("X", "unknown_aspect") == mas.bake_clause("X", "9x16")


def test_style_does_not_also_dictate_placement():
    """Two placement rules in one prompt fight each other."""
    assert "never centred" not in mas.STYLE_BAKED
    assert "fill the rest of the frame" not in mas.STYLE_BAKED


def test_the_baked_negative_bans_the_overlay_look():
    assert "flat text overlay" in mas.NEGATIVE_BAKED
    assert "text pasted on top" in mas.NEGATIVE_BAKED


# ---------------------------------------------------------------- routing

def test_qwen_gets_the_baked_prompt_and_no_text_ban(installed, tmp_path, monkeypatch):
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

    mas.render_scene("a scene", tmp_path / "o.png", headline="Bees Have 5 Eyes")
    assert '"Bees Have 5 Eyes"' in seen["prompt"]
    assert "no text" not in seen["prompt"]
    # the text bans would fight the headline
    assert "letters, words" not in seen["negative_prompt"]
    assert "misspelled text" in seen["negative_prompt"]


def test_a_long_headline_falls_back_to_the_overlay(installed, tmp_path, monkeypatch):
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

    mas.render_scene("a scene", tmp_path / "o.png", headline="y" * 80)
    assert "headline text" not in seen["prompt"]
    # back to the ordinary negative, which bans text outright
    assert "letters, words" in seen["negative_prompt"]


def test_uso_never_bakes(installed, tmp_path, monkeypatch):
    """Flux garbles small text; only Qwen may draw the headline."""
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

    mas.render_scene("a scene", tmp_path / "o.png", headline="BEE FACTS")
    assert "headline text" not in seen["prompt"]


def test_render_for_video_reports_whether_it_baked(installed, tmp_path, monkeypatch):
    monkeypatch.setattr(mas, "scene_prompt", lambda *a, **k: "a scene")
    monkeypatch.setattr(mas, "prepare_gpu", lambda: None)
    monkeypatch.setattr(mas, "release", lambda: None)
    monkeypatch.setattr(mas, "backend_healthy", lambda: (True, "ok"))
    monkeypatch.setattr(mas, "active_backend_id", lambda: mas.BACKEND_ID)
    monkeypatch.setattr(mas, "render_scene",
                        lambda s, p, **k: (Image.new("RGB", (8, 8)).save(p), p)[1])

    art = mas.render_for_video("Bee Facts", "ctx", tmp_path, "s", aspects=("9x16",))
    assert art["_baked_headline"] is True

    art = mas.render_for_video("z" * 80, "ctx", tmp_path, "s", aspects=("9x16",))
    assert art["_baked_headline"] is False


# ---------------------------------------------------------------- no double title

def test_save_thumbnail_paints_nothing(tmp_path):
    src = tmp_path / "art.png"
    Image.new("RGB", (768, 1344), (10, 20, 30)).save(src)
    out = tmp_path / "t.jpg"
    assert pk.save_thumbnail(src, out, pk.THUMB_9X16)
    assert Image.open(out).size == pk.THUMB_9X16


def test_baked_art_is_not_overlaid(tmp_path, monkeypatch):
    """Painting a title over baked type would show the headline twice."""
    import modules.runtime_settings as rs
    monkeypatch.setattr(rs, "get_mascot_thumbnails_enabled", lambda: True)

    art = tmp_path / "art.png"
    Image.new("RGB", (768, 1344)).save(art)
    monkeypatch.setattr(mas, "render_for_video",
                        lambda **k: {"9x16": art, "_scene": "s", "_seed": 1,
                                     "_baked_headline": True})
    painted = []
    monkeypatch.setattr(pk, "render_thumbnail",
                        lambda *a, **k: painted.append(1) or a[2])

    video = tmp_path / "reel_9x16.mp4"
    video.write_bytes(b"x")
    kit = pk.attach(video, "Bee Facts", context="ctx", aspects=("9x16",))
    assert not painted, "baked art must not be painted over"
    assert kit["headline"] == "baked into the art"


def test_unbaked_art_is_still_overlaid(tmp_path, monkeypatch):
    import modules.runtime_settings as rs
    monkeypatch.setattr(rs, "get_mascot_thumbnails_enabled", lambda: True)
    art = tmp_path / "art.png"
    Image.new("RGB", (768, 1344)).save(art)
    monkeypatch.setattr(mas, "render_for_video",
                        lambda **k: {"9x16": art, "_scene": "s", "_seed": 1,
                                     "_baked_headline": False})
    video = tmp_path / "reel_9x16.mp4"
    video.write_bytes(b"x")
    kit = pk.attach(video, "Bee Facts", context="ctx", aspects=("9x16",))
    assert kit["headline"] == "overlaid"


# ---------------------------------------------------------------- the hook

def test_headline_is_the_hook_not_the_title():
    """A 47-character title baked into art renders as tiny type."""
    long_title = "Worker Bees Flap Wings 230x/sec & More Bee Facts"
    assert not mas.can_bake(long_title)
    h, short = pk.build_headline(long_title)   # no context -> deterministic
    assert mas.can_bake(h)
    assert len(h) <= pk.HEADLINE_MAX
    assert len(short) <= pk.HEADLINE_SHORT_MAX


def test_headline_falls_back_when_the_llm_invents_a_fact(monkeypatch):
    import modules.script_generator as sg
    monkeypatch.setattr(sg, "_call_llm",
                        lambda *a, **k: '{"headline": "Bees Have 9 Eyes", "short": "9 Eyes"}')
    monkeypatch.setattr(sg, "_extract_json", lambda r: __import__("json").loads(r))
    h, _ = pk.build_headline("Bee Facts And More", context="Bees have five eyes in total.")
    assert "9" not in h, "an invented number must be rejected"


def test_headline_accepts_a_grounded_hook(monkeypatch):
    import modules.script_generator as sg
    monkeypatch.setattr(sg, "_call_llm",
                        lambda *a, **k: '{"headline": "Bees Have 5 Eyes", "short": "5 Eyes"}')
    monkeypatch.setattr(sg, "_extract_json", lambda r: __import__("json").loads(r))
    h, short = pk.build_headline("Bee Facts", context="Bees have 5 eyes in total.")
    assert h == "Bees Have 5 Eyes"
    assert short == "5 Eyes"


def test_render_for_video_bakes_the_hook_not_the_long_title(installed, tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(mas, "scene_prompt", lambda *a, **k: "a scene")
    monkeypatch.setattr(mas, "prepare_gpu", lambda: None)
    monkeypatch.setattr(mas, "release", lambda: None)
    monkeypatch.setattr(mas, "active_backend_id", lambda: mas.BACKEND_ID)
    monkeypatch.setattr(mas, "backend_healthy", lambda: (True, "ok"))
    monkeypatch.setattr(mas, "render_scene",
                        lambda s, p, **k: (seen.update(k),
                                           Image.new("RGB", (8, 8)).save(p), p)[2])
    art = mas.render_for_video("Worker Bees Flap Wings 230x/sec & More Bee Facts",
                               "ctx", tmp_path, "s", aspects=("9x16",),
                               headline="Bees Have 5 Eyes")
    assert art["_baked_headline"] is True
    assert seen["headline"] == "Bees Have 5 Eyes"


# ---------------------------------------------------------------- portrait fit

def test_portrait_uses_the_short_form_not_a_chopped_headline():
    """A 9:16 frame holding a full-body mascot fits about two words of giant
    type. Given four, Qwen silently DROPS the last two — so we choose which
    words go, and we choose ones that still mean something."""
    assert mas.fit_headline("230 Flaps A Second", "9x16", "230 Flaps") == "230 Flaps"
    assert mas.fit_headline("230 Flaps A Second", "16x9", "230 Flaps") == "230 Flaps A Second"


def test_portrait_keeps_a_short_headline_whole():
    assert mas.fit_headline("BEE FACTS", "9x16", "BEE FACTS") == "BEE FACTS"


def test_portrait_prefers_a_long_headline_over_a_mutilated_one():
    """With no usable short form, a smaller point size beats "Bees Have"."""
    assert mas.fit_headline("Goldfish Remember You", "9x16", None) == "Goldfish Remember You"


@pytest.mark.parametrize("headline,expected", [
    ("Bees Have 5 Eyes", "5 Eyes"),          # payload at the end
    ("230 Flaps A Second", "230 Flaps"),     # payload at the start
    ("Worker Bees Flap Wings", "Flap Wings"),
    ("BEE FACTS", "BEE FACTS"),              # already short
])
def test_short_fallback_never_ends_or_starts_on_filler(headline, expected):
    assert pk._short_fallback(headline) == expected


def test_render_for_video_records_what_each_aspect_actually_says(installed, tmp_path, monkeypatch):
    monkeypatch.setattr(mas, "scene_prompt", lambda *a, **k: "a scene")
    monkeypatch.setattr(mas, "prepare_gpu", lambda: None)
    monkeypatch.setattr(mas, "release", lambda: None)
    monkeypatch.setattr(mas, "active_backend_id", lambda: mas.BACKEND_ID)
    monkeypatch.setattr(mas, "backend_healthy", lambda: (True, "ok"))
    monkeypatch.setattr(mas, "render_scene",
                        lambda s, p, **k: (Image.new("RGB", (8, 8)).save(p), p)[1])
    art = mas.render_for_video("Bee Facts", "ctx", tmp_path, "s",
                               aspects=("9x16", "16x9"),
                               headline="230 Flaps A Second",
                               headline_short="230 Flaps")
    assert art["_headline_shown"] == {"9x16": "230 Flaps",
                                      "16x9": "230 Flaps A Second"}
