"""WhisperX forced-alignment CLI — runs INSIDE the isolated venv
(03_Models/venv_whisperx). Aligns KNOWN lyrics text against rendered song
audio (no ASR transcription — we already know the words; this only finds
WHERE they land in time), returning per-line (text, t_start, t_end) spans.
Driven by modules/lyric_aligner.py over subprocess, mirrors qwen_tts_cli.py.

job JSON: {"audio_path": "...", "lines": ["line one", "line two", ...],
           "out_json": "..."}
"""
import json
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).parent.parent
os.environ.setdefault("HF_HOME", str(PROJECT / "03_Models" / "hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(PROJECT / "03_Models" / "hf_cache" / "hub"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# whisperx.load_audio shells out to a bare "ffmpeg" on PATH — point it at our
# contained copy instead of requiring a system-wide install.
_FFMPEG_BIN = PROJECT / "00_Tools" / "ffmpeg" / "bin"
if _FFMPEG_BIN.is_dir():
    os.environ["PATH"] = str(_FFMPEG_BIN) + os.pathsep + os.environ.get("PATH", "")

# CPU-only: this is a batch/offline step (not realtime), and staying off CUDA
# sidesteps any torch/CUDA-build compatibility question in this isolated venv
# entirely (the main venv's torch build is pinned to the host GPU already).
DEVICE = os.environ.get("WHISPERX_DEVICE", "cpu")


def main(job_path: str):
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    audio_path = job["audio_path"]
    lines = [ln for ln in job.get("lines", []) if (ln or "").strip()]
    out_json = Path(job["out_json"])
    if not lines:
        raise ValueError("no non-empty lyric lines to align")

    import whisperx

    audio = whisperx.load_audio(audio_path)
    total_dur = len(audio) / 16000.0  # whisperx.load_audio resamples to 16kHz

    # One flat segment spanning the whole song — we supply the ALREADY-KNOWN
    # text (the real lyrics), not an ASR hypothesis, so there's no
    # transcription mismatch to worry about. The wav2vec2 CTC aligner finds
    # where each word actually lands inside this span.
    full_text = " ".join(lines)
    segments = [{"text": full_text, "start": 0.0, "end": total_dur}]

    align_model, metadata = whisperx.load_align_model(language_code="en", device=DEVICE)
    result = whisperx.align(segments, align_model, metadata, audio, DEVICE,
                            return_char_alignments=False)

    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            if "start" in w and "end" in w:
                words.append((w["word"], float(w["start"]), float(w["end"])))

    if not words:
        raise RuntimeError("alignment produced no word timestamps")

    # Distribute aligned words back across the original LINE boundaries (word
    # count per line), taking each line's span as [first word start, last word
    # end] among the words assigned to it.
    spans = []
    wi = 0
    for line in lines:
        n = max(1, len(line.split()))
        chunk = words[wi:wi + n]
        wi += n
        if not chunk:
            continue
        spans.append([line, round(chunk[0][1], 3), round(chunk[-1][2], 3)])
    # Any leftover words (word-count drift vs the aligner's own tokenization)
    # get folded into the last line's end time.
    if wi < len(words) and spans:
        spans[-1][2] = round(words[-1][2], 3)

    out_json.write_text(json.dumps({"spans": spans, "total_dur": total_dur}), encoding="utf-8")
    print("WHISPERX_ALIGN_DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
