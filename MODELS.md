# MODELS.md — every model weight the bot needs

None of these live in git (too large). After a rebuild, download each into the
ComfyUI models folder shown, or the Ollama store. The exact filenames are what
`config_snapshot/models.json` expects — keep them identical.

ComfyUI models root on this machine: `E:\Rexjaw_VFX\03_Models\ComfyUI\` (and/or
`E:\Rexjaw_VFX\01_ComfyUI\ComfyUI_windows_portable\ComfyUI\models\`). ComfyUI
searches both via `extra_model_paths.yaml` — mirror that on the new PC.

## Ollama LLMs (text) — `ollama pull` / custom modelfiles
| Model | ~Size | Use |
|---|---|---|
| `qwen2.5:14b-instruct-q6_K` | 12 GB | structurer / facts / song JSON |
| `qwen3story:latest` | 13 GB | creative story/lyrics writer (custom, see `config_snapshot/qwen3-30b-modelfile`) |
| `qwen3.6-thinker:latest` | 13 GB | creative role (alt) |
| `llama3.1:8b-instruct-q8_0` | 8.5 GB | fallback |
| `llama3.1:8b-instruct-q4_K_M` | 4.9 GB | fallback (light) |

Custom models: `ollama create qwen3story -f config_snapshot/qwen3-30b-modelfile`
(after pulling its base). Ollama store dir here = `03_Models\ollama` (env
`OLLAMA_MODELS`).

## ComfyUI — checkpoints/  (`models/checkpoints`)
- `sd_xl_base_1.0.safetensors`  — SDXL base (kids IP-Adapter backend)
- `sd_xl_turbo_1.0_fp16.safetensors` — SDXL turbo

## ComfyUI — diffusion_models/  (unet; `models/diffusion_models` or `unet`)
- `flux1-dev-fp8.safetensors` — Flux.1-dev fp8 (USO + flux_lora; **primary image**)
- `z_image_turbo_bf16.safetensors` — Z-Image Turbo (**facts + music stills**)
- `z_image_bf16.safetensors` — Z-Image base
- `flux-2-klein-base-9b-fp8.safetensors` — Flux.2 Klein
- `flux1-dev-kontext_fp8_scaled.safetensors` — Flux Kontext
- `wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors` — Wan 2.2 14B I2V (high) **primary video**
- `wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors` — Wan 2.2 14B I2V (low)
- `wan2.2_ti2v_5B_fp16.safetensors` — Wan 2.2 5B
- `wan2.2_s2v_14B_fp8_scaled.safetensors` — Wan 2.2 S2V 14B
- `ltx-2-19b-dev-Q4_K_M.gguf` — LTX-2 19B (GGUF, optional video)

## ComfyUI — vae/  (`models/vae`)
- `ae.safetensors` — Flux VAE
- `wan2.2_vae.safetensors`, `wan_2.1_vae.safetensors` — Wan VAEs
- `LTX2_video_vae_bf16.safetensors` — LTX-2 VAE
- `full_encoder_small_decoder.safetensors` — Z-Image VAE

## ComfyUI — text_encoders/ (clip; `models/text_encoders` or `clip`)
- `clip_l.safetensors`
- `t5xxl_fp8_e4m3fn_scaled.safetensors`
- `umt5_xxl_fp8_e4m3fn_scaled.safetensors` — Wan text encoder
- `qwen_3_4b.safetensors`, `qwen_3_8b_fp8mixed.safetensors` — Z-Image/Flux2 encoders
- `gemma-3-12b-it-qat-UD-Q4_K_XL.gguf` — LTX-2 text encoder
- `ltx-2-19b-embeddings_connector_dev_bf16.safetensors` — LTX-2 connector
- `wav2vec2_large_english_fp16.safetensors` — Wan S2V audio encoder

## ComfyUI — loras/  (`models/loras`)
- `uso-flux1-dit-lora-v1.safetensors` — **USO** char-consistency LoRA (story stills)
- `StorybookRedmondV2-KidsBook-KidsRedmAF.safetensors` — kids storybook style
- `wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors` — Wan 4-step (OFF by default)
- `wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors`

## Other model stacks (separate installers/venvs — see RESTORE.md)
- **ACE-Step 1.5 turbo** (song audio) — ComfyUI custom node + its weights
  (acestep_v1.5_turbo, qwen 0.6b/1.7b ace15, ace_1.5_vae).
- **Kokoro-82M** (TTS) — auto-downloads to `03_Models/hf_cache` on first run
  (repo `hexgrad/Kokoro-82M`); voices af_bella/af_nicole/af_sky/am_adam/am_michael.
- **Qwen3-TTS** — bridge in `venv_qwen_tts` (horror voice, optional).
- **Chatterbox** — `venv_chatterbox` (expressive TTS, optional).
- **WhisperX** — `venv_whisperx` (music lyric alignment).
- **IP-Adapter PLUS** + **CLIP-Vision** — for the SDXL kids backend.

> Sources: most weights are on Hugging Face (search the exact filename). Wan 2.2 /
> Z-Image / Flux / LTX-2 are all published; USO LoRA = `bytedance-research/USO`.
> Keep filenames byte-identical to the list above so `models.json` resolves them.
