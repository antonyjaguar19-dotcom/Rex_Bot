# Claw Bot Agent (Rex VFX Bot)

A locally-run AI animation pipeline that generates **30-second educational YouTube shorts for kids under 10**, controlled entirely through Discord.

Built by a VFX artist for daily content creation, with strict containment: every model, dependency, and output stays inside the project folder. No cloud APIs, no SaaS lock-in, no data leaving your machine without permission.

---

## What it does

1. **Generates a daily script** about a good habit (sharing, honesty, hygiene, etc.)
2. **Storyboards each shot** as a single starting keyframe
3. **(Coming) Animates each frame** into a video clip
4. **(Coming) Assembles the shots, adds audio, uploads to YouTube**

Every stage pauses for **human approval via Discord** before moving on.

---

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Contained Python environment + ComfyUI install | ✅ Done |
| 2 | Discord bot communication layer | ✅ Done |
| 3 | Daily script generation (Ollama / Llama) | ✅ Done |
| 4 | Storyboard generation (Z-Image, single frame per shot) | ✅ Done |
| Session 1A | Creative expansion (6 culture modes, 5 visual styles) | ✅ Done |
| 5 | Video generation per shot | 🚧 Next |
| 6 | Video assembly + YouTube upload automation | ⏳ Planned |

---

## Architecture

```
Discord (control surface)
    │
    └──► Discord Bot (discord_bot.py)
            │
            ├──► Script Generator ──► Ollama / Llama
            │
            ├──► Storyboard Generator ──► ComfyUI ──► Z-Image Base/Turbo
            │
            ├──► (Phase 5) Video Generator ──► ComfyUI ──► Video model TBD
            │
            ├──► (Phase 6) Assembler ──► ffmpeg ──► YouTube Data API
            │
            └──► Health Monitor ──► #status channel (auto-refresh 30s)
```

**Pluggable model registry:** All models declared in `05_Config/models.json`. Adding a new image or video backend means writing one adapter file, no core rewrites.

---

## Hardware requirements

- **GPU:** NVIDIA RTX 3070 or better (8 GB VRAM minimum)
- **RAM:** 32 GB minimum, 64 GB recommended
- **Storage:** ~50 GB for models + outputs
- **OS:** Windows 10 / 11

---

## Project structure

```
E:\Rexjaw_VFX\
├── 01_Tools\          # ComfyUI, Python venv, Ollama (gitignored)
├── 02_Agent\          # All Python source code (this repo)
├── 03_Models\         # Image/video/LLM model weights (gitignored)
├── 05_Config\         # models.json, styles.json
├── 06_Logs\           # ComfyUI + bot logs (gitignored)
├── 07_Outputs\        # Generated scripts, frames, videos (gitignored)
└── .env               # Discord token, API keys (NEVER commit)
```

---

## Discord commands

| Command | Action |
|---------|--------|
| `!commands` | List every available command |
| `!current_settings` | Show active model, style, culture mode |
| `!list_styles` | Show all available visual styles |
| `!switch_model <name>` | Swap active image backend |
| `!generate_storyboard` | Generate a fresh storyboard from approved script |
| `!list_storyboards` | Browse past storyboards |
| `!regen_shot <shot_number>` | Surgically regenerate one shot |

Approval flow uses Discord reactions (✅ / ❌) on bot messages.

---

## Setup (high level)

> Detailed install steps live in `/docs/SETUP.md` (coming soon). This is the overview.

1. Clone this repo into `E:\Rexjaw_VFX\02_Agent`
2. Create a Python virtual environment inside `E:\Rexjaw_VFX\01_Tools\venv`
3. Install dependencies: `pip install -r requirements.txt`
4. Install ComfyUI into `E:\Rexjaw_VFX\01_Tools\ComfyUI`
5. Download Z-Image Base/Turbo weights into `E:\Rexjaw_VFX\03_Models\`
6. Install Ollama, pull a Llama model
7. Create a Discord application + bot, copy token into `.env`
8. Run: `python 02_Agent/discord_bot.py`

---

## Configuration

**`05_Config/models.json`** — registry of available image/video/LLM backends with per-model settings.

**`05_Config/styles.json`** — visual style presets (storybook, cartoon, anime, watercolor, pixelart). Each style maps to a prompt prefix and ComfyUI workflow tweaks.

**`.env`** — secrets only. Discord token, future YouTube OAuth credentials. **Never commit this file.**

---

## Design decisions worth knowing

- **Single frame per shot** is deliberate. Phase 5 video models animate from one starting frame, so generating two keyframes per shot would create throwaway work.
- **Z-Image over SDXL** because it accepts paragraph-style natural language prompts (Qwen 3 4B text encoder). Better fit than comma-tag prompt engineering for a non-coder workflow.
- **Pluggable adapters** mean future video models drop in without touching the bot or generator core.
- **VRAM management** via ComfyUI's `/free` endpoint and optional Ollama unload between jobs — critical for 8 GB cards juggling LLM + image + video models in one pipeline.
- **Containment** keeps the project portable. Move the `E:\Rexjaw_VFX` folder to another machine, reinstall venv, and you're back up.

---

## Roadmap

- **Phase 5:** Choose video model (AnimateDiff, LTX-Video, Wan2.1, or CogVideoX), build adapter, add Discord approval
- **Phase 6:** ffmpeg-based shot assembly, TTS narration, background music, YouTube Data API upload
- **Future:** Character consistency (LoRA / IP-Adapter), multi-language Indian regional support, voice cloning for consistent narrator

---

## License

Personal project. Not for redistribution while in active development.

---

## Acknowledgements

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — image generation backend
- [Z-Image](https://github.com/Tongyi-MAI) — Alibaba Tongyi-MAI text-to-image
- [Ollama](https://ollama.com) — local LLM runtime
- [discord.py](https://discordpy.readthedocs.io) — Discord bot framework

Built with help from Claude (Anthropic) as a coding mentor.
