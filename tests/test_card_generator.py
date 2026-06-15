"""Card generator unit tests — pure PIL/logic, no GPU/TTS/ffmpeg."""
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

_AGENT = Path(__file__).parent.parent.resolve()
if str(_AGENT) not in sys.path:
    sys.path.insert(0, str(_AGENT))

from modules import card_generator as cg


def test_effective_card_duration_floor():
    # Short speech clamps to the minimum on-screen length.
    assert cg.effective_card_duration(0.0) == cg.MIN_CARD_SEC
    assert cg.effective_card_duration(0.5) == cg.MIN_CARD_SEC


def test_effective_card_duration_pads_tail():
    # Long speech = audio + tail pad (above the floor).
    raw = cg.MIN_CARD_SEC + 2.0
    assert cg.effective_card_duration(raw) == pytest.approx(raw + cg.TAIL_PAD_SEC)


def test_wrap_text_splits_to_fit():
    img = Image.new("RGB", (400, 200))
    draw = ImageDraw.Draw(img)
    font = cg._load_font(40)
    lines = cg._wrap_text(draw, "the quick brown fox jumps over", font, 200)
    assert len(lines) > 1
    for ln in lines:
        assert draw.textlength(ln, font=font) <= 200 or " " not in ln


def test_wrap_text_empty():
    img = Image.new("RGB", (400, 200))
    draw = ImageDraw.Draw(img)
    assert cg._wrap_text(draw, "", cg._load_font(20), 200) == [""]


def test_render_card_image_solid_fallback(tmp_path):
    # No frame → solid background, still produces a canvas-sized PNG.
    out = tmp_path / "card.png"
    cg._render_card_image(None, "Hello World", 1080, 1920, out, label="The moral")
    assert out.exists()
    assert Image.open(out).size == (1080, 1920)


def test_synth_card_audio_empty_text_noop():
    path, dur = cg.synth_card_audio("", Path("nope.wav"))
    assert path is None and dur == 0.0
