"""Qwen3-TTS CLI — runs INSIDE the isolated venv (03_Models/venv_qwen_tts,
transformers 4.56.1). Reads a job JSON, voices each text with a preset Custom
Voice (deterministic = consistent), stitches with silence gaps, writes the
master wav + spans JSON. Driven by modules/tts_qwen.py over subprocess.

job JSON: {"texts": [...], "speaker": "eric", "language": "english",
           "gap_sec": 0.4, "out_wav": "...", "spans_out": "..."}
"""
import json, os, sys
from pathlib import Path

PROJECT = Path(__file__).parent.parent
os.environ.setdefault("HF_HOME", str(PROJECT / "03_Models" / "hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(PROJECT / "03_Models" / "hf_cache" / "hub"))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

IMPL = (PROJECT / "01_ComfyUI" / "ComfyUI_windows_portable" / "ComfyUI" / "custom_nodes"
        / "TTS-Audio-Suite" / "engines" / "qwen3_tts" / "impl")
sys.path.insert(0, str(IMPL))

REPO = os.environ.get("QWEN_TTS_REPO", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")


def main(job_path: str):
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    texts = [t for t in job["texts"] if (t or "").strip()]
    speaker = job.get("speaker", "eric")
    language = job.get("language", "english")
    gap_sec = float(job.get("gap_sec", 0.4))
    instruct = job.get("instruct") or None  # style/emotion steering for Custom Voice

    import numpy as np, soundfile as sf, torch

    # transformers <5.10 keeps ROPE 'default'; guard-patch in case of a newer pin.
    from transformers import modeling_rope_utils as _mru
    if "default" not in _mru.ROPE_INIT_FUNCTIONS:
        def _def(config=None, device=None, seq_len=None, **kw):
            base = getattr(config, "rope_theta", 10000.0)
            partial = getattr(config, "partial_rotary_factor", 1.0)
            hd = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
            dim = int(hd * partial)
            inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
            return inv.to(device), 1.0
        _mru.ROPE_INIT_FUNCTIONS["default"] = _def

    import qwen_tts  # noqa
    from qwen_tts import Qwen3TTSModel

    tts = Qwen3TTSModel.from_pretrained(REPO, device_map="cuda:0",
                                        dtype=torch.bfloat16, attn_implementation=None)

    # Each text is voiced on its own pass anyway, so writing it to its own file
    # costs nothing — and it is the ONLY way to get a clean per-beat WAV. Cutting
    # the concatenated master back apart on spans clipped words at the seams.
    seg_dir = job.get("segments_dir")
    if seg_dir:
        Path(seg_dir).mkdir(parents=True, exist_ok=True)

    sr = None
    pieces, spans, files, cursor = [], [], [], 0
    for i, text in enumerate(texts):
        wavs, this_sr = tts.generate_custom_voice(
            text, speaker=speaker, language=language, instruct=instruct)
        wav = np.asarray(wavs[0], dtype="float32").reshape(-1)
        sr = sr or this_sr
        t0 = cursor / sr
        pieces.append(wav)
        cursor += len(wav)
        spans.append([text, round(t0, 3), round(cursor / sr, 3)])
        if seg_dir:
            fp = Path(seg_dir) / f"seg_{i:02d}.wav"
            sf.write(str(fp), wav, sr)
            files.append(str(fp))
        if i < len(texts) - 1:
            gap = np.zeros(int(round(gap_sec * sr)), dtype="float32")
            pieces.append(gap)
            cursor += len(gap)
        print(f"[qwen] {i+1}/{len(texts)}", flush=True)

    master = np.concatenate(pieces) if pieces else np.zeros(1, dtype="float32")
    sf.write(job["out_wav"], master, sr or 24000)
    Path(job["spans_out"]).write_text(
        json.dumps({"spans": spans, "sr": sr, "files": files}), encoding="utf-8")
    print("QWEN_CLI_DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
