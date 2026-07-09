# RESTORE.md — rebuild the Rexjaw bot on a fresh / broken PC

This repo holds the **code + config snapshot** only. The heavy parts (Python
packages, ComfyUI, AI model weights, Ollama models) are NOT in git — you
re-download them once. Follow these steps top to bottom.

Think of it like re-linking a Nuke script on a new workstation: the script (this
repo) is safe in git; you just have to re-point it at the plugins (ComfyUI),
the render engine (models), and the license keys (secrets).

---

## 0. What you need first (prerequisites)
- **Windows 10/11**, an **NVIDIA GPU** (project built on RTX 5080, 16 GB) + recent driver.
- **Python 3.11** (the bot's venv). Install to a known path, e.g. `E:\Rexjaw_VFX\00_Tools\python311`.
- **Git**, and (optional) **GitHub CLI** `gh` for auth.
- **ffmpeg** (static build) at `E:\Rexjaw_VFX\00_Tools\ffmpeg\bin\`.
- **Ollama** at `E:\Rexjaw_VFX\00_Tools\ollama\ollama.exe`.
- **ComfyUI** (portable) at `E:\Rexjaw_VFX\01_ComfyUI\ComfyUI_windows_portable\`.
- Disk: **plan ~150–250 GB** for all model weights + venvs.

## 1. Recreate the folder tree
```
E:\Rexjaw_VFX\
├── 00_Tools\      ffmpeg, ollama, python311  (installed, not in git)
├── 01_ComfyUI\    ComfyUI portable            (installed, not in git)
├── 02_Agent\      <-- THIS REPO (git clone here)
├── 03_Models\     model weights + side venvs  (downloaded, not in git)
├── 04_Outputs\    renders (auto-created)
├── 05_Config\     live config + secrets       (restored from config_snapshot)
└── 06_Logs\       logs (auto-created)
```

## 2. Clone the repo into `02_Agent`
```
cd E:\Rexjaw_VFX
git clone https://github.com/antonyjaguar19-dotcom/Rex_Bot.git 02_Agent
cd 02_Agent
git checkout feat/horror-story-mode
```

## 3. Restore `05_Config` from the snapshot
The bot reads config from `E:\Rexjaw_VFX\05_Config\` (OUTSIDE the repo). Copy the
snapshot back:
```
robocopy 02_Agent\config_snapshot ..\05_Config /E
```
Then create the real secrets file from the template and fill in your values:
```
copy 02_Agent\config_snapshot\secrets.env.template ..\05_Config\secrets.env
notepad ..\05_Config\secrets.env
```
Keys: `DISCORD_BOT_TOKEN`, `DASHBOARD_PASSWORD`, `NGROK_AUTHTOKEN`, `NGROK_DOMAIN`,
`CLAW_DASHBOARD_URL`, `HF_TOKEN`. **secrets.env must be BOM-free** — edit in plain
Notepad, don't `Out-File -Encoding utf8` (adds a BOM that drops the first line).

## 4. Main Python venv + packages
```
cd E:\Rexjaw_VFX\02_Agent
E:\Rexjaw_VFX\00_Tools\python311\python.exe -m venv venv
venv\Scripts\python -m pip install --upgrade pip
venv\Scripts\pip install -r requirements.txt
```
For an EXACT match of the working machine (incl. the CUDA torch build), use the
full freeze instead: `venv\Scripts\pip install -r requirements.lock.txt`
(torch is pinned to `+cu128` — install the matching CUDA wheel).

## 5. Side venvs (optional features)
Each is a separate isolated env under `03_Models\` so it can't break ComfyUI:
- `venv_whisperx`  — music lyric alignment (WhisperX).
- `venv_qwen_tts`  — Qwen3-TTS horror narrator bridge.
- `venv_chatterbox`— expressive TTS (optional).
Create each with Python 3.11 and `pip install` the package it wraps (whisperx /
chatterbox-tts / the qwen tts cli). The bot degrades gracefully if a side venv is
missing (falls back to Kokoro etc.).

## 6. ComfyUI + custom nodes
Install the portable ComfyUI at `01_ComfyUI\ComfyUI_windows_portable\`. Add the
custom nodes the workflows use (via ComfyUI-Manager): **USO**, **IP-Adapter
Plus**, **ACE-Step 1.5**, **WanVideo**, **LTX-2**, **Z-Image**, **GGUF** loader.
Point ComfyUI at the shared models via `extra_model_paths.yaml` so it also reads
`03_Models\ComfyUI\`.

## 7. Model weights  → see **MODELS.md**
Download every file listed in `MODELS.md` into the matching ComfyUI folder
(`checkpoints` / `diffusion_models` / `vae` / `text_encoders` / `loras`). Keep
filenames identical — `config_snapshot/models.json` resolves them by name.

## 8. Ollama models
```
set OLLAMA_MODELS=E:\Rexjaw_VFX\03_Models\ollama
ollama pull qwen2.5:14b-instruct-q6_K
ollama pull llama3.1:8b-instruct-q8_0
ollama create qwen3story -f 02_Agent\config_snapshot\qwen3-30b-modelfile
```
(Full list + sizes in MODELS.md.)

## 9. Launch + verify
```
# start ComfyUI
01_ComfyUI\ComfyUI_windows_portable\run_nvidia_gpu.bat
# start Ollama (env set as above)
ollama serve
# start the bot (also launches the dashboard on http://127.0.0.1:7860)
cd E:\Rexjaw_VFX\02_Agent
venv\Scripts\python claw_bot.py
```
Healthy = bot logs "connected to Gateway", dashboard answers on :7860, Discord
control panel appears. Run the test suite too: `venv\Scripts\python -m pytest tests -q`.

## 10. Quick smoke tests
- Discord: `!facts the deep ocean` → reel + description posts to #videos.
- Standalone runners: `run_facts_e2e.py "topic"`, `run_music_e2e.py "theme" 30`,
  `run_horror_e2e.py "theme" 2`, `run_kids_e2e.py "theme"`.

---

### What is / isn't in git
- **IN:** all Python source (`claw_bot.py`, `modules/`, runners, `tests/`),
  `assets/watermark.png`, `requirements*.txt`, `config_snapshot/` (models.json,
  styles.json, workflows, voice_refs, ollama modelfile, secrets **template**),
  and the docs (README, RESTORE, MODELS, CLAUDE.md).
- **OUT (re-download / re-create):** `venv/`, `01_ComfyUI/`, `03_Models/`,
  `04_Outputs/`, `06_Logs/`, and the real `05_Config/secrets.env`.

See `CLAUDE.md` (root + 02_Agent) for the full architecture, decisions, and the
per-mode pipeline details.
