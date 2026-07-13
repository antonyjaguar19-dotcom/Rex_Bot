"""The music bed: found by its body, never looped, and CHECKED in the finished file.

Three real bugs, all of which shipped a reel that looked fine at every step:
  1. ACE-Step under-fills — asked for 49s it composed 29s of music, padded 10s of
     digital silence, then left 8s of junk. Trimming only the TAIL left the hole in
     the middle, and fitting a long track by cutting from the FRONT then threw the
     music away and kept the dead air.
  2. Filling the gap by replaying the bed's opening is heard as a stutter — the
     same eight bars twice. Silence at the head beats a loop.
  3. Nothing ever checked the bed was audible IN THE REEL. So the reel is measured.

Real ffmpeg throughout: every one of these bugs was in an ffmpeg call.
"""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.assembly import FFMPEG_EXE                # noqa: E402
from modules import facts_assembly as fasm             # noqa: E402

pytestmark = pytest.mark.skipif(not Path(FFMPEG_EXE).exists(),
                                reason="ffmpeg not installed")


def _tone(path: Path, spec: str, seconds: float):
    """A wav built from an ffmpeg lavfi expression."""
    subprocess.run([str(FFMPEG_EXE), "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", f"{spec}:d={seconds}", "-t", str(seconds), str(path)],
                   capture_output=True, timeout=120)


def _ace_style_track(path: Path):
    """What ACE-Step actually handed us: music, then dead air, then quiet junk."""
    music, silence, junk = path.parent / "m.wav", path.parent / "s.wav", path.parent / "j.wav"
    _tone(music, "sine=f=220:r=44100", 29.0)
    _tone(silence, "anullsrc=r=44100:cl=mono", 10.0)
    _tone(junk, "sine=f=180:r=44100", 8.0)
    subprocess.run([str(FFMPEG_EXE), "-y", "-loglevel", "error",
                    "-i", str(music), "-i", str(silence), "-i", str(junk),
                    "-filter_complex",
                    "[2:a]volume=-40dB[q];[0:a][1:a][q]concat=n=3:v=0:a=1[a]",
                    "-map", "[a]", str(path)], capture_output=True, timeout=180)
    return path


# ----------------------------------------------------------------- the trim --

def test_bed_is_cut_at_its_music_body_not_its_tail(tmp_path):
    track = _ace_style_track(tmp_path / "bed.wav")
    assert fasm._probe_duration(track) == pytest.approx(47.0, abs=0.5)

    end = fasm._music_body_end(track)
    assert end == pytest.approx(29.0, abs=0.6), "the music stops at 29s; find it there"

    trimmed = fasm.trim_trailing_silence(track)
    got = fasm._probe_duration(trimmed)
    assert got == pytest.approx(29.0, abs=0.7), f"kept {got:.1f}s — dead air survived"


def test_a_track_that_plays_all_the_way_out_is_left_alone(tmp_path):
    track = tmp_path / "solid.wav"
    _tone(track, "sine=f=220:r=44100", 30.0)
    assert fasm._music_body_end(track) is None
    assert fasm._probe_duration(fasm.trim_trailing_silence(track)) == pytest.approx(30.0, abs=0.3)


def test_an_early_pause_does_not_truncate_the_bed(tmp_path):
    """A rest in the first bars is music, not the end of the piece."""
    a, gap, b = tmp_path / "a.wav", tmp_path / "g.wav", tmp_path / "b.wav"
    _tone(a, "sine=f=220:r=44100", 3.0)
    _tone(gap, "anullsrc=r=44100:cl=mono", 2.0)
    _tone(b, "sine=f=330:r=44100", 25.0)
    track = tmp_path / "paused.wav"
    subprocess.run([str(FFMPEG_EXE), "-y", "-loglevel", "error",
                    "-i", str(a), "-i", str(gap), "-i", str(b), "-filter_complex",
                    "[0:a][1:a][2:a]concat=n=3:v=0:a=1[o]", "-map", "[o]", str(track)],
                   capture_output=True, timeout=180)
    assert fasm._music_body_end(track) is None, "a 3s-in pause must not end the bed"


# ---------------------------------------------------------------- the align --

def test_a_short_bed_is_delayed_never_looped(tmp_path):
    """Replaying the opening to fill the reel is the stutter the user reported."""
    bed = tmp_path / "short.wav"
    _tone(bed, "sine=f=220:r=44100", 20.0)
    out = fasm._align_music_tail(bed, 30.0)
    assert fasm._probe_duration(out) == pytest.approx(30.0, abs=0.15)
    # Delayed, not looped: the reel OPENS on silence and the music plays once.
    r = subprocess.run([str(FFMPEG_EXE), "-hide_banner", "-i", str(out), "-af",
                        "silencedetect=noise=-50dB:d=1.0", "-f", "null", "-"],
                       capture_output=True, text=True, timeout=120)
    starts = [ln for ln in r.stderr.splitlines() if "silence_start" in ln]
    assert starts and "silence_start: 0" in starts[0], "a short bed must open on silence"


def test_a_long_bed_is_trimmed_from_the_front_keeping_its_ending(tmp_path):
    bed = tmp_path / "long.wav"
    _tone(bed, "sine=f=220:r=44100", 60.0)
    out = fasm._align_music_tail(bed, 36.0)
    assert fasm._probe_duration(out) == pytest.approx(36.0, abs=0.15)


# ---------------------------------------------------------------- the audit --

def _reel(path: Path, voice: Path, bed: Path | None, secs=20.0):
    """A 'reel': voice at unity, bed loudnorm'd under it — the pipeline's own mix."""
    if bed is None:
        subprocess.run([str(FFMPEG_EXE), "-y", "-loglevel", "error",
                        "-f", "lavfi", "-i", f"color=c=black:s=64x64:r=16:d={secs}",
                        "-i", str(voice), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-t", str(secs), str(path)],
                       capture_output=True, timeout=300)
        return path
    fc = (f"[2:a]loudnorm=I={fasm.MUSIC_LUFS}:TP={fasm.MUSIC_TP}:LRA=11[m];"
          f"[1:a][m]amix=inputs=2:duration=first:dropout_transition=0:"
          f"weights='1 1':normalize=0[a]")
    subprocess.run([str(FFMPEG_EXE), "-y", "-loglevel", "error",
                    "-f", "lavfi", "-i", f"color=c=black:s=64x64:r=16:d={secs}",
                    "-i", str(voice), "-i", str(bed), "-filter_complex", fc,
                    "-map", "0:v", "-map", "[a]", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-t", str(secs), str(path)],
                   capture_output=True, timeout=300)
    return path


@pytest.fixture()
def voice(tmp_path):
    v = tmp_path / "voice.wav"
    _tone(v, "sine=f=140:r=44100", 20.0)
    return v


def test_audit_hears_a_healthy_bed(tmp_path, voice):
    bed = tmp_path / "bed.wav"
    _tone(bed, "sine=f=440:r=44100", 20.0)
    a = fasm.audit_music_bed(_reel(tmp_path / "good.mp4", voice, bed), voice)
    assert a is not None
    assert a["dead_blocks"] == 0
    assert a["mean_db"] > fasm.BED_FLOOR_DB


def test_audit_catches_a_bed_that_dies_halfway(tmp_path, voice):
    """The exact shipped bug: music for the first half, dead air for the rest."""
    half, dead, bed = tmp_path / "h.wav", tmp_path / "d.wav", tmp_path / "dying.wav"
    _tone(half, "sine=f=440:r=44100", 10.0)
    _tone(dead, "anullsrc=r=44100:cl=mono", 10.0)
    subprocess.run([str(FFMPEG_EXE), "-y", "-loglevel", "error",
                    "-i", str(half), "-i", str(dead), "-filter_complex",
                    "[0:a][1:a]concat=n=2:v=0:a=1[o]", "-map", "[o]", str(bed)],
                   capture_output=True, timeout=180)
    a = fasm.audit_music_bed(_reel(tmp_path / "bad.mp4", voice, bed), voice)
    assert a is not None
    assert a["dead_blocks"] >= 3, f"dropout not detected: {a}"


def test_audit_catches_a_bed_that_is_not_there_at_all(tmp_path, voice):
    a = fasm.audit_music_bed(_reel(tmp_path / "none.mp4", voice, None), voice)
    assert a is not None
    assert a["dead_blocks"] >= 5, f"a missing bed must not read as present: {a}"
