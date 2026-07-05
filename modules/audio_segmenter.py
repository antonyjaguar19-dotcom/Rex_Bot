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
import tempfile
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


def _extract_slice(wav_path: Path, t0: float, t1: float, out_path: Path) -> Path:
    """Cut [t0, t1) out of wav_path into out_path (output-side seek = sample
    accurate, cheap for the few-second slices this is used on)."""
    cmd = [
        str(FFMPEG_EXE), "-y", "-loglevel", "error",
        "-i", str(Path(wav_path).resolve()),
        "-ss", f"{max(0.0, t0):.3f}", "-to", f"{t1:.3f}",
        "-c", "copy", str(out_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if r.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"slice extract failed ({t0}-{t1}): {r.stderr.strip()}")
    return out_path


def refine_windows_to_sentences(
    wav_path: Path,
    windows: list,
    tmp_dir: "Path | None" = None,
) -> list:
    """Upgrade beat/shot-level (t0, t1, text) windows to real per-sentence
    timing for burned-in captions.

    A window whose text is ONE sentence already has exact timing — passed
    through unchanged. A window with 2+ sentences currently only has an exact
    OUTER boundary; where the sentences split inside it was previously a
    char-length guess. This slices that window's real audio out of wav_path
    and re-runs plan_windows_from_silence on JUST that slice (same mechanism
    used for the beat/shot boundaries themselves) — snapping the internal
    split to the real pause instead of guessing. Falls back to a proportional
    char-length split if ffmpeg/detection fails for a window.

    Returns a flat list of (t0, t1, text) events, one per sentence.
    """
    from modules.subtitles import raw_sentences

    out: list = []
    tmp_dir = Path(tmp_dir) if tmp_dir else Path(tempfile.mkdtemp(prefix="capref_"))
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for t0, t1, text in windows:
        text = (text or "").strip()
        if not text or t1 <= t0:
            continue
        sentences = raw_sentences(text)
        if len(sentences) <= 1:
            out.append((t0, t1, sentences[0] if sentences else text))
            continue
        try:
            slice_wav = tmp_dir / f"slice_{round(t0, 3)}_{round(t1, 3)}.wav"
            _extract_slice(wav_path, t0, t1, slice_wav)
            weights = [max(1, len(s.split())) for s in sentences]
            durs = plan_windows_from_silence(slice_wav, weights)
            cur = t0
            for s, d in zip(sentences, durs):
                out.append((cur, min(t1, cur + d), s))
                cur += d
        except Exception as e:
            log.warning(f"caption refine failed for window {t0}-{t1} ({e}); "
                       f"falling back to proportional split.")
            total_chars = sum(len(s) for s in sentences) or 1
            cur = t0
            for s in sentences:
                dur = (t1 - t0) * (len(s) / total_chars)
                out.append((cur, min(t1, cur + dur), s))
                cur += dur
    return out
