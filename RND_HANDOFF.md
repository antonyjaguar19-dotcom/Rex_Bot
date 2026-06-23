# Rexjaw / Claw Bot — R&D Handoff (2026-06-23)

Briefing for an R&D chat to design an **optimized plan**. Self-contained: paste this whole file into the chat.

## North-star goal
Build toward an **AI that automatically makes short films**. Today it's a local, fully-offline animation pipeline driven by a Discord bot + an offline NiceGUI dashboard. Strict containment: everything under `E:\Rexjaw_VFX`. No cloud, no API keys, no paid services (monetizable output).

## Hardware (the binding constraint)
- **RTX 5080, 16 GB VRAM**, 64 GB RAM, Windows 11. CUDA 12.8, torch 2.11+cu128.
- 16 GB VRAM is the bottleneck for FLUX training/inference (heavy offload).

## Pipelines (3 modes, switchable)
1. **Kids story** (original): theme → Qwen 2.5 14B story → Z-Image storyboard → Wan 2.2 I2V video per shot → Kokoro/VoxCPM narration → assemble 9x16/16x9/1x1. Mature, works. Consistency = `locked_visual_token` text only (drifts).
2. **Music video**: theme → Qwen lyrics+style → **ACE-Step** song (vocals) → Z-Image stills → ffmpeg Ken Burns → 3 aspects + watermark. Full e2e passed.
3. **Horror story** (newest, the active R&D): see below.

## Horror mode v2 — current architecture
`LLM story → continuous narration (audio-first) → silence-split → consistent storyboard → animated video → assemble (16x9 + ambient drone + "Rexjaw" watermark)`

- **Writer**: chunked Qwen (premise+bible → chapters → beats). True-horror tone (creepypasta/nosleep). Capped ~5 min. Each beat: narration + image_prompt + motion_prompt + characters + location. Character/location **bible** with `locked_token`.
- **Narration**: **Qwen3-TTS** (open local weights, isolated venv `03_Models/venv_qwen_tts`, transformers 4.56.1 — 5.x breaks it). Custom Voice **"eric"** (preset = deterministic = consistent), `instruct` for serious horror tone, pitch 0.90 via ffmpeg asetrate+atempo (deeper, no slow-mo). Driven over subprocess bridge (`modules/tts_qwen.py` + `qwen_tts_cli.py`) — no ComfyUI dependency. Other engines available: Kokoro (fast, consistent), Chatterbox (isolated venv), VoxCPM (drifts).
- **Splits**: ffmpeg `silencedetect` on the continuous narration; per-beat image windows snap to real pauses.
- **Storyboard consistency**: **per-character FLUX LoRA** (chosen as strongest). `lora_autotrain` = hero portraits → qwen-multiangle dataset → ai-toolkit train → register; `comfyui_flux_lora` auto-stacks a LoRA when the character NAME is in the prompt. Falls back to prompt-tokens if no LoRA.
- **Video**: one **Wan 2.2 I2V** clip per beat from the still (+motion_prompt), freeze-padded to the beat's audio window. `horror_video_mode` = wan | kenburns.

## What's proven working (validated this session)
- Writer (scary, 5-min), Qwen3-TTS eric narration (serious/deep/consistent).
- **Per-character LoRA trained + registered** (`rexj_ethan`, 800 steps).
- **flux_lora storyboard** producing LoRA-face-locked stills (10/10) — after fixing a /history poll-timeout crash on the 16 GB cold-load.
- Earlier Ken-Burns horror full e2e produced a complete video.
- Last run was **stopped by choice right before the Wan stage** (so Wan+assemble for v2 not yet rendered, but the chain up to consistent storyboard is proven).

## THE HARD PERFORMANCE FINDINGS (core R&D material)
On the 16 GB 5080, the quality path is **brutally slow**:

| Stage | Measured cost | Notes |
|---|---|---|
| FLUX LoRA train (ai-toolkit, low-VRAM quantized) | **~7 hr for 800 steps** (~15-35 s/step; ~35 min just to load the 24 GB model) | Per character. The single biggest blocker. Offload-bound, not attention-bound (SageAttention won't fix). |
| Qwen3-TTS narration | **~1 min per beat** (~3.5 min audio took ~12 min) | 55-beat story ≈ 45-60 min just to voice. |
| Wan 2.2 I2V clip | **~4-7 min per ~5 s clip** | 10-15 shots = 1-2.5 hr. |
| flux_lora still | ~30-60 s after a ~3-5 min first cold-load | OK. |
| Ken Burns still (Z-Image) | ~2 min | The "fast" visual path. |

**Net per horror video at film quality: LoRA train (hrs/char) + narration (tens of min) + Wan clips (1-2.5 hr) = many hours, dominated by LoRA training.** This makes **LoRA-per-story impractical** on this GPU.

## Consistency options (decided + alternatives)
- **FLUX LoRA** (current horror choice): strongest identity + best realism, but ~7 hr/char train. Generic infra → reusable by any mode.
- **Flux Kontext** (installed): reference-image identity on FLUX, no training, FLUX quality, lighter. Single-ref (weaker for multi-angle).
- **IP-Adapter (cubiq, SDXL/SD1.5 only — NO FLUX)**: no training, multi-ref face lock, mature, fast — but SDXL realism < FLUX. Planned for **kids** mode (fast/cheap fits).
- Architecture intent: **pluggable consistency per mode** — FLUX-LoRA for film-grade, IP-Adapter/Kontext for fast.

## Installed / ready
- ComfyUI (ACE-Step, Z-Image, Wan 2.2 14B, Flux Kontext, **flux1-dev-fp8** for LoRA render).
- ai-toolkit (07_Training/ai-toolkit, own venv, torch+torchaudio cu128) + **flux1-dev base 24 GB** (HF token has gated access).
- Qwen3-TTS isolated venv. Chatterbox isolated venv. Kokoro, VoxCPM.
- TTS-Audio-Suite cloned in custom_nodes (NOT installed — needs transformers 5.3, would break ComfyUI; Qwen3-TTS runs isolated instead).
- Dashboard is now **offline-only** (ngrok/Cloudflare tunnels removed).

## OPEN QUESTIONS for the optimized plan
1. **Kill the LoRA-train bottleneck.** Options to weigh: (a) drop steps to ~250-400 (faster, weaker); (b) abandon per-story LoRA → use Kontext/IP-Adapter no-train consistency even for horror; (c) pre-train a reusable cast library instead of per-story; (d) cloud-GPU the training only (breaks "fully local" — acceptable?); (e) accept multi-hour and queue/batch overnight.
2. **Faster narration**: Qwen3-TTS is slow per-beat. Batch larger chunks? Different engine for drafts vs final? Kokoro for speed?
3. **Faster video**: Wan 4-step lightx2v LoRA (3-4× faster, quality cost)? SageAttention for ComfyUI gens (Wan/Flux)? Fewer/longer shots?
4. **Consistency strategy per mode** — finalize: horror = LoRA vs Kontext; kids = IP-Adapter (SDXL); music = none.
5. **Story length vs shot count** — LLM under-produces (asks 5 min, writes ~2-3). Expand pass? And shot count drives total render time linearly.
6. **Pipeline orchestration for auto-short-film** — approval gates, batching, resumability, a job queue, and which stages can run in parallel vs the single 16 GB GPU forcing serialization.
7. **Target: realistic time-per-finished-minute of film** on this hardware, and what quality/speed tradeoffs hit it.

## Key constraints to respect in any plan
- Fully local/offline, contained in `E:\Rexjaw_VFX`, no paid APIs.
- Single 16 GB GPU → stages serialize; VRAM discipline (free between stages).
- Narration = source of truth for timing (no `-shortest`); fps consistency across gen/upscale/assemble.
- Pluggable backends via models.json (new model = one adapter file).
