"""WhisperX CLI — runs INSIDE the isolated venv (03_Models/venv_whisperx).

Finds WHERE a song actually sings. ACE-Step songs have instrumental intros,
outros, and breaks; the lyrics are NOT sung 1:1-in-order across the whole file.
So we do NOT force-fit the known lyrics across [0, dur] (that put captions on
the intro music). Instead we TRANSCRIBE with voice-activity detection: whisper's
VAD only emits words where singing actually happens, so the returned word
timestamps are real singing regions with instrumental gaps naturally excluded.
The python side then places the CLEAN known lyrics inside those real windows.

Driven by modules/lyric_aligner.py over subprocess, mirrors qwen_tts_cli.py.

job JSON: {"audio_path": "...", "out_json": "..."}
out JSON: {"words": [[word, start, end], ...], "total_dur": float}
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

# CPU-only: offline batch step (not realtime); sidesteps any torch/CUDA-build
# question in this isolated venv. int8 keeps the small model fast on CPU.
DEVICE = os.environ.get("WHISPERX_DEVICE", "cpu")
COMPUTE = os.environ.get("WHISPERX_COMPUTE", "int8")
MODEL = os.environ.get("WHISPERX_MODEL", "small")


def main(job_path: str):
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    audio_path = job["audio_path"]
    out_json = Path(job["out_json"])

    import whisperx

    audio = whisperx.load_audio(audio_path)
    total_dur = len(audio) / 16000.0  # whisperx.load_audio resamples to 16kHz

    # 1) ASR + VAD: transcribe the ACTUAL vocals. VAD gives us segments only
    #    where singing happens (intro/outro/instrumental breaks excluded).
    model = whisperx.load_model(MODEL, DEVICE, compute_type=COMPUTE)
    result = model.transcribe(audio, batch_size=8)
    segments = result.get("segments", [])
    lang = result.get("language", "en")

    words = []
    if segments:
        # 2) Forced-align the transcription for tighter word-level timestamps.
        try:
            align_model, metadata = whisperx.load_align_model(language_code=lang, device=DEVICE)
            aligned = whisperx.align(segments, align_model, metadata, audio, DEVICE,
                                     return_char_alignments=False)
            for seg in aligned.get("segments", []):
                for w in seg.get("words", []):
                    if "start" in w and "end" in w:
                        words.append([w["word"], float(w["start"]), float(w["end"])])
        except Exception as e:
            print(f"[align skipped: {e}]", flush=True)

    # Fallback to segment-level times if word alignment produced nothing.
    if not words:
        for seg in segments:
            if "start" in seg and "end" in seg:
                words.append([seg.get("text", "").strip(),
                              float(seg["start"]), float(seg["end"])])

    out_json.write_text(json.dumps({"words": words, "total_dur": total_dur}),
                        encoding="utf-8")
    print(f"WHISPERX_ALIGN_DONE words={len(words)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
