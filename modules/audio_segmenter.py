"""
Claw Bot — Audio Segmenter (silence-driven scene splits)

For the horror pipeline: one continuous narration is rendered, then the visuals
are cut on the AUDIO's natural pauses. We detect silences with ffmpeg
`silencedetect`, then snap each beat's boundary to the nearest real pause so
every still change lands on a breath/sentence gap (not mid-word).

Pure ffmpeg + parsing — no model.
"""

import logging
import re
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modules.assembly import FFMPEG_EXE, _probe_duration

log = logging.getLogger("claw_bot.audio_segmenter")

_SIL_START = re.compile(r"silence_start:\s*([0-9.]+)")
_SIL_END = re.compile(r"silence_end:\s*([0-9.]+)")


def detect_silence_midpoints(wav: Path, noise_db: int = -30,
                             min_silence: float = 0.4) -> list[float]:
    """Return the midpoint time (s) of each detected silence — candidate cut
    points. `noise_db` lower (e.g. -35) = stricter; `min_silence` = shortest
    pause that counts as a cut."""
    cmd = [
        str(FFMPEG_EXE), "-hide_banner", "-i", str(Path(wav).resolve()),
        "-af", f"silencedetect=n={noise_db}dB:d={min_silence}",
        "-f", "null", "-",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    log_txt = (r.stderr or "") + (r.stdout or "")
    starts, ends = [], []
    for line in log_txt.splitlines():
        m = _SIL_START.search(line)
        if m:
            starts.append(float(m.group(1)))
        m = _SIL_END.search(line)
        if m:
            ends.append(float(m.group(1)))
    mids = []
    for i, s in enumerate(starts):
        e = ends[i] if i < len(ends) else s + min_silence
        mids.append(round((s + e) / 2.0, 3))
    mids.sort()
    log.info(f"silencedetect: {len(mids)} pauses in {wav.name}")
    return mids


def _nearest(target: float, candidates: list[float], tol: float) -> float | None:
    best, bestd = None, tol
    for c in candidates:
        d = abs(c - target)
        if d <= bestd:
            best, bestd = c, d
    return best


def plan_windows_from_silence(
    wav: Path,
    weights: list[float],
    *,
    noise_db: int = -30,
    min_silence: float = 0.4,
    tol: float = 2.5,
) -> list[float]:
    """Tile the narration into per-beat windows whose boundaries snap to real
    pauses. `weights` = per-beat size hint (e.g. word counts); boundaries are
    placed proportionally then snapped to the nearest detected silence within
    `tol` seconds. Returns per-beat DURATIONS summing to the audio length."""
    total = _probe_duration(wav)
    n = len(weights)
    if n <= 1:
        return [total]
    wsum = sum(weights) or float(n)
    mids = detect_silence_midpoints(wav, noise_db, min_silence)

    # proportional target boundary time for each internal split
    bounds = []
    cum = 0.0
    used = set()
    for i in range(n - 1):
        cum += weights[i]
        target = cum / wsum * total
        snap = _nearest(target, [m for m in mids if m not in used], tol)
        b = snap if snap is not None else target
        # keep strictly increasing
        if bounds and b <= bounds[-1]:
            b = min(total, bounds[-1] + 0.2)
        bounds.append(round(b, 3))
        if snap is not None:
            used.add(snap)

    starts = [0.0] + bounds
    ends = bounds + [total]
    durs = [max(0.3, round(e - s, 3)) for s, e in zip(starts, ends)]
    # normalize tiny drift so sum == total
    drift = round(total - sum(durs), 3)
    if abs(drift) > 0.01:
        durs[-1] = max(0.3, round(durs[-1] + drift, 3))
    log.info(f"planned {n} windows from silence (sum={sum(durs):.1f}s vs {total:.1f}s)")
    return durs
