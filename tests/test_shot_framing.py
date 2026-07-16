"""Framing is delivered by cropping the render.

Qwen-Edit anchors to the full-body reference and ignores the prompt's framing clause, so the
camera distance is a deterministic crop instead (measured: a closeup beat came back
full-body). These guard the crop maths — a monotonic zoom ladder, centred, always inside the
frame, headroom on the closeup — and the no-op that keeps a full-frame shot untouched.
"""
from PIL import Image

from modules import shot_framing as sf


def test_the_zoom_ladder_is_monotonic():
    """Each framing reads closer than the one before it — establishing is the whole scene,
    closeup is the face. If wide were not tighter than establishing they would look
    identical (they render identical)."""
    heights = {f: sf.window(f, 1920, 1080)[3] - sf.window(f, 1920, 1080)[1]
               for f in ("establishing", "wide", "medium", "closeup")}
    assert heights["establishing"] > heights["wide"] > heights["medium"] > heights["closeup"]


def test_the_window_is_centred_and_inside_the_frame():
    for f in sf.FRAME_WINDOW:
        x0, y0, x1, y1 = sf.window(f, 1920, 1080)
        assert 0 <= x0 and x1 <= 1920 and 0 <= y0 and y1 <= 1080
        assert abs((1920 - (x1 - x0)) - 2 * x0) <= 1        # centred horizontally
        # source 16:9 aspect kept, so the child is not stretched by the re-fit
        assert abs((x1 - x0) / (y1 - y0) - 1920 / 1080) < 0.02


def test_the_closeup_keeps_headroom_and_is_the_tightest():
    x0, y0, x1, y1 = sf.window("closeup", 1920, 1080)
    assert y0 > 0, "the closeup starts below the top edge — headroom over the hair"
    assert (y1 - y0) < 1080 * 0.6, "a closeup is head-and-shoulders, not most of the body"


def test_establishing_and_an_unknown_framing_do_not_touch_the_picture():
    assert sf.is_noop("establishing") is True
    assert sf.is_noop("banana") is True         # unknown => full frame, never broken
    assert sf.is_noop("closeup") is False


def test_crop_rewrites_only_when_it_tightens_and_keeps_the_size(tmp_path):
    p = tmp_path / "still.png"
    Image.new("RGB", (1280, 720), "white").save(p)

    assert sf.crop_to_framing(p, "establishing") is False    # full frame — untouched
    assert sf.crop_to_framing(p, "closeup") is True          # tighter — rewritten
    assert Image.open(p).size == (1280, 720)                 # re-fit to the frame size
