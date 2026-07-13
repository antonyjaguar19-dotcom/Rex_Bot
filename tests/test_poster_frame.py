"""A re-rolled thumbnail must reach the VIDEO, not just the .jpg beside it.

The reel opens on its thumbnail (Shorts grabs the first frame in regions where
custom thumbnails are not offered). Rerolling the art rewrote the jpg and left
the video opening on the poster you had just rejected — prepend_still refused to
touch a reel that already carried one, which is exactly what stopped the holds
from stacking. Both must be true at once: never stack, always replace.

Real ffmpeg here — the bug was in the ffmpeg call, so mocking it proves nothing.
"""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.assembly import FFMPEG_EXE, FFPROBE_EXE   # noqa: E402
from modules import facts_assembly as fasm             # noqa: E402

pytestmark = pytest.mark.skipif(not Path(FFMPEG_EXE).exists(),
                                reason="ffmpeg not installed")


def _dur(p: Path) -> float:
    r = subprocess.run([str(FFPROBE_EXE), "-v", "error", "-show_entries",
                        "format=duration", "-of", "default=nw=1:nk=1", str(p)],
                       capture_output=True, text=True, timeout=60)
    return float((r.stdout or "0").strip())


def _first_frame_rgb(video: Path, out: Path) -> tuple:
    from PIL import Image
    subprocess.run([str(FFMPEG_EXE), "-y", "-loglevel", "error", "-i", str(video),
                    "-vframes", "1", str(out)], capture_output=True, timeout=60)
    return Image.open(out).convert("RGB").resize((1, 1)).getpixel((0, 0))


def _solid(path: Path, color: str, seconds=1.0, audio=True):
    cmd = [str(FFMPEG_EXE), "-y", "-loglevel", "error",
           "-f", "lavfi", "-i", f"color=c={color}:s=240x426:r=16:d={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i",
                f"anullsrc=channel_layout=stereo:sample_rate=44100:d={seconds}"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-t", str(seconds)]
    if audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd += [str(path)]
    subprocess.run(cmd, capture_output=True, timeout=120)


def _still(path: Path, color: str):
    subprocess.run([str(FFMPEG_EXE), "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", f"color=c={color}:s=240x426", "-vframes", "1", str(path)],
                   capture_output=True, timeout=60)


@pytest.fixture()
def reel(tmp_path, monkeypatch):
    monkeypatch.setattr(fasm, "POSTERLESS_DIR", tmp_path / "_posterless")
    v = tmp_path / "facts_TEST_9x16.mp4"
    _solid(v, "green", seconds=2.0)
    return v


def test_reroll_replaces_the_poster_instead_of_stacking(reel, tmp_path):
    base_dur = _dur(reel)

    red, blue = tmp_path / "red.png", tmp_path / "blue.png"
    _still(red, "red")
    _still(blue, "blue")

    fasm.prepend_still(reel, red, hold_sec=0.5)
    held = _dur(reel)
    assert held == pytest.approx(base_dur + 0.5, abs=0.12)
    r, g, b = _first_frame_rgb(reel, tmp_path / "f1.png")
    assert r > 150 and g < 90, "the reel must open on the poster it was given"

    # The re-roll: same reel, new art.
    fasm.prepend_still(reel, blue, hold_sec=0.5, replace=True)
    assert _dur(reel) == pytest.approx(held, abs=0.12), "hold must not stack"
    r, g, b = _first_frame_rgb(reel, tmp_path / "f2.png")
    assert b > 150 and r < 90, "the reel must open on the NEW poster"


def test_publish_kit_rerun_still_does_not_stack(reel, tmp_path):
    """Without `replace`, a second kit pass leaves the reel alone (the guard that
    the re-roll fix must not break)."""
    red = tmp_path / "red.png"
    _still(red, "red")
    fasm.prepend_still(reel, red, hold_sec=0.5)
    once = _dur(reel)
    fasm.prepend_still(reel, red, hold_sec=0.5)          # no replace=
    assert _dur(reel) == pytest.approx(once, abs=0.05)


def test_hold_is_recorded_and_readable(reel, tmp_path):
    red = tmp_path / "red.png"
    _still(red, "red")
    assert fasm.thumb_hold_of(reel) is None
    fasm.prepend_still(reel, red, hold_sec=0.4)
    assert fasm.thumb_hold_of(reel) == pytest.approx(0.4, abs=0.01)
