"""Pure-string tests for the audio-first assembly filter (no ffmpeg run)."""

import sys
from pathlib import Path

_AGENT = Path(__file__).parent.parent.resolve()
if str(_AGENT) not in sys.path:
    sys.path.insert(0, str(_AGENT))

from modules import assembly


def test_vfilter_single_clip():
    f = assembly._build_audio_first_vfilter(1, 1080, 1920)
    assert "[v0]null[vout]" in f
    assert "concat" not in f          # one clip needs no concat


def test_vfilter_multi_concat():
    f = assembly._build_audio_first_vfilter(3, 1080, 1920)
    assert "concat=n=3:v=1:a=0[vout]" in f
    # one scale/pad chain per clip
    assert f.count("scale=1080:1920") == 3
    assert "[v0][v1][v2]concat" in f
