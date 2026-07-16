"""Deliver a lesson shot's FRAMING by cropping the rendered still.

The framing (establishing/wide/medium/closeup) is chosen per beat and swapped into the
image prompt — but the render backend does not honour it. Qwen-Image-Edit anchors to the
FULL-BODY reference photo and a text clause ("her face fills the frame") is too weak to
override it: measured on the real Class-1 lesson (nakshu, 2026-07-15), a closeup beat came
back full-body, and every framing rendered at the same distance. The prompt swap is
necessary but not sufficient.

So the framing is DELIVERED here, deterministically, after the render: a closer framing is
a tighter 16:9 crop of the same picture, re-fit to the frame. The lesson presenter is a
centred, camera-facing character with the head near the top, so a proportional top-centre
window is reliable — and it is the pipeline that composes the shot that way, not a guess.

A monotonic zoom ladder, so each step reads closer than the last:
    establishing  full frame        (the whole scene)
    wide          a touch tighter   (so wide ≠ establishing, which render identically)
    medium        waist-up
    closeup       head and shoulders — the face, for lip-sync (wide = ~35px mouth)

`medium`/unknown with a full window == the picture unchanged (the old behaviour).
"""
import logging
from pathlib import Path

from modules import shot_grammar

log = logging.getLogger("claw_bot.shot_framing")

# (top_fraction, height_fraction) of the full frame. Width is derived to keep the source
# 16:9 aspect and centred. Head sits near the top, so closeup keeps a little headroom and
# the rest anchor at the top edge. Sourced from the shared shot grammar (which adds
# establishing/wide/medium/closeup/insert/montage) so every pipeline crops to one ladder.
FRAME_WINDOW = shot_grammar.FRAME_WINDOW
_DEFAULT = shot_grammar.NO_CROP     # unknown framing => no crop, never a broken picture


def window(framing: str, w: int, h: int) -> tuple:
    """The crop box (x0, y0, x1, y1) for this framing on a w×h frame — centred, source
    aspect preserved, clamped inside the frame. Pure maths, so it is unit-testable."""
    top_f, h_f = FRAME_WINDOW.get((framing or "").strip().lower(), _DEFAULT)
    ch = max(1, min(h, round(h * h_f)))
    cw = max(1, min(w, round(ch * w / h)))          # keep the source aspect
    x0 = (w - cw) // 2
    y0 = min(round(h * top_f), h - ch)
    return (x0, y0, x0 + cw, y0 + ch)


def is_noop(framing: str) -> bool:
    """A full-frame window changes nothing — skip the re-encode."""
    return FRAME_WINDOW.get((framing or "").strip().lower(), _DEFAULT) == (0.0, 1.0)


def crop_to_framing(png_path, framing: str) -> bool:
    """Crop `png_path` IN PLACE to its framing and re-fit to the original size.

    Run ONCE, right after the still is rendered — a second pass would crop the crop. Never
    raises: a framing that cannot be applied leaves the full picture, which is a worse shot
    but not a broken one. Returns True when the file was rewritten."""
    if is_noop(framing):
        return False
    p = Path(png_path)
    try:
        from PIL import Image
        im = Image.open(p).convert("RGB")
        w, h = im.size
        box = window(framing, w, h)
        if box == (0, 0, w, h):
            return False
        im.crop(box).resize((w, h), Image.LANCZOS).save(p)
        log.info(f"{p.name}: framed to {framing} {box}")
        return True
    except Exception as e:
        log.warning(f"could not frame {p.name} to {framing} ({e}); kept the full picture")
        return False
