"""Chatterbox voicing CLI — runs INSIDE the isolated chatterbox venv (its own
torch 2.6). The main-venv adapter (modules/tts_chatterbox.py) calls this via
subprocess so the two torch installs never collide.

Job JSON (arg 1):
{
  "groups": [{"text": "...", "ref_wav": "<path|null>",
              "exaggeration": 0.6, "cfg_weight": 0.4}, ...],
  "gap_sec": 0.34,
  "out_wav": "<master wav path>",
  "spans_out": "<json path>",
  "segments_dir": "<dir|null>"      # optional: also write one wav per group
}
Writes the master wav (groups stitched with silence gaps) and a spans JSON:
{"sr": 24000, "spans": [["text", t_start, t_end], ...], "files": [...]}

`segments_dir` exists because the facts mascot needs one clip per beat (each beat
is cut to its own video shot). Slicing them back out of the master lost fricatives
at the seams, and re-running the CLI per beat reloaded the model every time.
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from chatterbox.tts import ChatterboxTTS


def main():
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    groups = job["groups"]
    gap_sec = float(job.get("gap_sec", 0.34))
    out_wav = Path(job["out_wav"])
    spans_out = Path(job["spans_out"])
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    seg_dir = job.get("segments_dir")
    seg_dir = Path(seg_dir) if seg_dir else None
    if seg_dir:
        seg_dir.mkdir(parents=True, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChatterboxTTS.from_pretrained(device=dev)
    sr = int(model.sr)
    gap = np.zeros(int(round(gap_sec * sr)), dtype=np.float32)

    pieces, spans, files, cursor = [], [], [], 0
    for i, g in enumerate(groups):
        text = (g.get("text") or "").strip()
        if not text:
            continue
        kw = dict(
            exaggeration=float(g.get("exaggeration", 0.6)),
            cfg_weight=float(g.get("cfg_weight", 0.4)),
        )
        rw = g.get("ref_wav")
        if rw and Path(rw).exists():
            kw["audio_prompt_path"] = str(rw)
        wav = model.generate(text, **kw)
        a = wav.squeeze().detach().cpu().numpy().astype(np.float32)
        if seg_dir:
            seg = seg_dir / f"seg_{len(spans):02d}.wav"
            sf.write(str(seg), a, sr)
            files.append(str(seg))
        t0 = cursor / sr
        pieces.append(a)
        cursor += len(a)
        spans.append([text, round(t0, 3), round(cursor / sr, 3)])
        if i < len(groups) - 1:
            pieces.append(gap)
            cursor += len(gap)

    master = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
    sf.write(str(out_wav), master, sr)
    spans_out.write_text(
        json.dumps({"sr": sr, "spans": spans, "files": files}), encoding="utf-8"
    )
    print(f"OK {out_wav} groups={len(spans)} dur={len(master)/sr:.2f}s dev={dev}")


if __name__ == "__main__":
    main()
