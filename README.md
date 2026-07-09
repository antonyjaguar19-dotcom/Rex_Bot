# Claw Bot Agent (Rex VFX Bot)

> **Rebuilding on a new / broken PC?** Read **[RESTORE.md](RESTORE.md)** (full
> step-by-step) and **[MODELS.md](MODELS.md)** (every model weight + where it
> goes). Config (models.json, styles, workflows, voice refs, secrets template)
> is snapshotted in **`config_snapshot/`**. Modes: **Facts · Music · Story ·
> Horror** — details in `CLAUDE.md`.

A locally-run AI animation pipeline that turns a theme into a narrated, multi-aspect
video for YouTube — driven by an **audio-first** workflow where the voiceover is
rendered first and its natural pauses decide where every scene cuts.

Controlled through a **Discord bot** *and* a **NiceGUI web dashboard** (shared on-disk
state, synced both ways). Strict containment: every model, dependency, and output stays
inside the project folder. The only optional cloud touch is gone — TTS now runs locally.

---

## The audio-first idea

Old pipeline: write script → storyboard → animate → bolt narration on at the end.

New pipeline: **render the voiceover first.** The pauses in the speech *are* the edit.
Shot count floats with the spoken rhythm, so cuts always land in natural breaths — the
thing that actually holds viewer attention.

```
!generate_script <theme>
   → story_writer (Qwen) writes the prose
   → VoxCPM voices it per breath-group → one master VO wav + exact pause timestamps
   → segmenter tiles cut-windows on the pauses → shot count FLOATS with the rhythm
   → structurer (Qwen) annotates each segment: beat / shot_type / camera / cast
   → standard script JSON  (+ _audio_first, _master_audio, per-shot win_dur)

🎭 casting gate            → confirm / edit the cast
approve prompts            → image + motion prompt per shot (edit / reseed / approve-all)
!generate_storyboard       → one keyframe per shot (Z-Image)
!generate_video            → one SILENT clip per shot, sized to its window (Wan I2V)
!assemble                  → hard-cut silent clips + lay the master VO over the timeline
                             → cuts land in pauses → 9x16 + 16x9 + 1x1 MP4, music under
```

Every stage pauses for **human approval** (Discord buttons or dashboard). Narration is
**locked** to the voiced track — editing the text would desync the captions from the
audio, so to change wording you regenerate the script.

---

## What's local / offline

| Stage | Engine | Notes |
|-------|--------|-------|
| Story + structuring | **Ollama** (Qwen 2.5 14B, creative routes to a stronger model) | local LLM |
| Voiceover | **VoxCPM** (0.5B, Apache-2.0) | local, free, monetizable; **no cloud, no API key, no quota** |
| Pause timestamps | per-breath-group synthesis | exact spans, no external transcription service |
| Storyboard image | **ComfyUI + Z-Image Base/Turbo** (Flux.2 optional) | one keyframe per shot |
| Video per shot | **ComfyUI + Wan 2.2 14B I2V** @16fps | silent clips; master VO added at assembly |
| Assembly | **ffmpeg** | 9x16 / 16x9 / 1x1, crossfade or cut, music mix |

VoxCPM falls back to local **Kokoro + ffmpeg silencedetect** if it's unavailable, so the
pipeline never hard-stops. Weights auto-download into the contained `03_Models/hf_cache`.

---

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Contained Python environment + ComfyUI install | ✅ Done |
| 2 | Discord bot + NiceGUI dashboard (shared state) | ✅ Done |
| 3 | Script generation (Ollama / Qwen, 2-stage) | ✅ Done |
| 4 | Storyboard generation (Z-Image, single frame per shot) | ✅ Done |
| 5 | Video generation per shot (Wan 2.2 14B I2V) | ✅ Done |
| 6 | Assembly (multi-aspect MP4, TTS, music) | ✅ Done |
| **Audio-first** | Voiceover-driven cuts (VoxCPM) — primary pipeline | ✅ Wired, validated |
| 7 | YouTube upload automation | ⏳ Planned |

---

## Architecture

```
Discord bot  ─┐                            ┌─►  Ollama / Qwen        (story + structure)
              ├─► shared on-disk JSON ◄─────┤
NiceGUI web  ─┘   (synced both ways)        ├─►  VoxCPM               (voiceover, local)
                                            ├─►  ComfyUI ─► Z-Image   (storyboard frames)
                                            ├─►  ComfyUI ─► Wan 2.2   (silent shot clips)
                                            └─►  ffmpeg               (assemble + music)
```

**Pluggable model registry:** all models declared in `05_Config/models.json`. A new image
or video backend is one adapter file in `modules/image_backends/` or
`modules/video_backends/` — no core rewrites. The audio-first router reads the
`_audio_first` flag on a script and routes assembly + per-shot video automatically;
classic (per-shot TTS) functions stay on disk as a fallback.

---

## Hardware requirements

- **GPU:** NVIDIA RTX 5080 (16 GB VRAM) is the reference card; 8 GB works for image-only
- **RAM:** 32 GB minimum, 64 GB recommended
- **Storage:** ~50 GB for models + outputs
- **OS:** Windows 10 / 11

> **VRAM note:** Ollama keeps its model resident (~12 GB for Qwen 14B). Free it before
> Wan video jobs (`gpu_utils.free_ollama_vram`) or ComfyUI can OOM — the pipeline does this
> per shot, but a long batch run should unload Ollama up front.

---

## Project structure

```
E:\Rexjaw_VFX\
├── 00_Tools\        # ffmpeg, ollama.exe, python311
├── 01_ComfyUI\      # ComfyUI portable install
├── 02_Agent\        # Python source (this repo) — modules\, venv\
├── 03_Models\       # weights + Ollama models + hf_cache (gitignored)
├── 04_Outputs\      # scripts/ storyboards/ clips/ audio/ final/ approved_prompts/
├── 05_Config\       # models.json, styles.json, runtime_settings.json, secrets.env
└── 06_Logs\         # gitignored
```

---

## Discord commands (core)

| Command | Action |
|---------|--------|
| `!generate_script <theme>` | Audio-first: voice the narration, pauses set the shots |
| `!generate_storyboard <id>` | One keyframe per shot (after prompt approval) |
| `!generate_video <id>` | One silent clip per shot, sized to its window |
| `!assemble <id>` | Stitch clips + master VO → 9x16 / 16x9 / 1x1 |
| `!add_shot <id> <before\|after> <shot#> [brief]` | Insert a new shot, renumber artifacts |
| `!regen_shot <shot#>` / `!regen_video_shot <shot#>` | Surgically regenerate one shot |
| `!switch_model <name>` | Swap active image/video backend |
| `!set_transition <crossfade\|cut>` | Final-cut transition mode |
| `!current_settings` / `!list_styles` / `!commands` | Inspect state |

Approval uses Discord **buttons** (Approve / Edit / Reject) plus a pinned control panel.
The full set is available on the web dashboard too (`http://127.0.0.1:7860`).

---

## Setup (high level)

1. Clone into `E:\Rexjaw_VFX\02_Agent`
2. Create a venv at `02_Agent\venv`, then `pip install -r requirements.txt`
3. `pip install voxcpm` (audio-first voice; weights auto-download on first use)
4. Install ComfyUI into `01_ComfyUI`, add Z-Image + Wan 2.2 weights to `03_Models`
5. Install Ollama, pull a Qwen model
6. Create a Discord app + bot, put the token in `05_Config/secrets.env` (BOM-free)
7. Start Ollama + ComfyUI, then run `python 02_Agent/claw_bot.py` (dashboard auto-launches)

---

## Configuration

- **`05_Config/models.json`** — image/video/LLM backends with per-model settings
  (`max_clip_seconds`, `default_fps`, recommended cfg/steps).
- **`05_Config/styles.json`** — visual style presets (storybook, cartoon, anime,
  watercolor, pixelart) → prompt prefix + workflow tweaks.
- **`05_Config/runtime_settings.json`** — style / voice / aspect / transition / music
  overrides, editable from either front-end.
- **`05_Config/secrets.env`** — Discord token, dashboard password. **Never commit.**

---

## Design decisions worth knowing

- **Audio-first** — the voiceover is the source of truth. Video freeze-pads if a shot
  renders short; narration is never trimmed (no `-shortest` in any mux).
- **Shot count floats** — pauses decide the cut count, not a fixed N. The structurer only
  *annotates* the voiced segments; narration text is injected verbatim, never rewritten.
- **VoxCPM over cloud TTS** — local, free, Apache-2.0 (monetizable output), and per-group
  synthesis gives exact pause timestamps without any external transcription service.
- **fps must match** across gen → upscale → assemble (Wan = 16fps). Relabeling fps is not
  resampling and silently shrinks duration — a bug class we explicitly guard.
- **Pluggable adapters** — new models drop in via `models.json` + one adapter file.
- **Containment** — move the `E:\Rexjaw_VFX` folder, rebuild the venv, and you're back up.

---

## Roadmap

- **YouTube upload automation** (Data API, OAuth).
- **Long-form faceless mode (proposed):** 5–15 min stickman/doodle storytelling videos on
  the same audio-first engine — static doodle frames + ffmpeg Ken Burns pan instead of Wan
  video (≈100× faster for 100+ scenes), plus auto title/description/tags/thumbnail prompts.
- **Per-character consistency** (LoRA / reference anchoring).

---

## License

Personal project. Not for redistribution while in active development.

---

## Acknowledgements

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — image/video generation backend
- [Z-Image](https://github.com/Tongyi-MAI) — Alibaba Tongyi-MAI text-to-image
- [VoxCPM](https://github.com/OpenBMB/VoxCPM) — OpenBMB local tokenizer-free TTS
- [Wan 2.2](https://github.com/Wan-Video) — image-to-video model
- [Ollama](https://ollama.com) — local LLM runtime
- [discord.py](https://discordpy.readthedocs.io) — Discord bot framework
- [NiceGUI](https://nicegui.io) — web dashboard

Built with help from Claude (Anthropic) as a coding mentor.
