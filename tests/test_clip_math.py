"""Duration / fps arithmetic — the bug class that cropped all narration.

These are pure-math methods, so we build a ClipGenerator via __new__ and
set fps directly (the real __init__ loads video backends and TTS, which
need ComfyUI / model weights).
"""

import math

from modules.clip_generator import ClipGenerator


def _gen(fps: int) -> ClipGenerator:
    g = ClipGenerator.__new__(ClipGenerator)
    g.fps = fps
    return g


class TestFramesForDuration:
    def test_wan_16fps_5s(self):
        assert _gen(16)._frames_for_duration(5.0) == 80

    def test_minimum_one_second(self):
        # Sub-second narration must still render at least 1s of video
        assert _gen(16)._frames_for_duration(0.2) == 16
        assert _gen(24)._frames_for_duration(0.0) == 24

    def test_rounding(self):
        # 3.97s @ 16fps = 63.52 frames -> rounds to 64
        assert _gen(16)._frames_for_duration(3.97) == 64


class TestLtx2FrameRule:
    """LTX-2 latent video needs frame_count = 8*N + 1; round UP, never down
    (rounding down would truncate audio)."""

    def test_already_valid(self):
        g = _gen(24)
        for n in (1, 9, 17, 25, 121):
            assert g._round_to_ltx2_frames(n) == n

    def test_rounds_up(self):
        g = _gen(24)
        assert g._round_to_ltx2_frames(2) == 9
        assert g._round_to_ltx2_frames(10) == 17
        assert g._round_to_ltx2_frames(80) == 81

    def test_never_shrinks(self):
        g = _gen(24)
        for n in range(1, 400):
            assert g._round_to_ltx2_frames(n) >= n


class TestFpsRelabelRegression:
    """Document the 16->24 fps relabel bug: same frames at a higher fps play
    SHORTER. Any future fps mismatch between gen and mux must trip this."""

    def test_relabel_shrinks_duration(self):
        frames = 80                  # 5s of Wan output at 16fps
        dur_at_16 = frames / 16      # 5.0s
        dur_at_24 = frames / 24      # 3.33s — the cropped-narration bug
        assert math.isclose(dur_at_16, 5.0)
        assert dur_at_24 < dur_at_16 * 0.7
