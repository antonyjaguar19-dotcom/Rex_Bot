# CLAUDE.md — Claw Bot project context

> Auto-loaded every session. Dense. Current.

## Owner
**Jeffy** — VFX artist (Maya, Nuke, Redshift, Shotgun), Chennai. Zero coding. Use VFX analogies. Windows 10/11. **RTX 5080 (16 GB VRAM), 64 GB RAM.** (Upgraded from RTX 3070 8GB 2026-05-29.) VRAM no longer hard bottleneck — full-quality paths viable.

## Core Rules (NEVER violate)
1. **Containment** — everything inside `E:\Rexjaw_VFX`. No system pip.
2. **No secrets in repo** — `.env` + `secrets.env` gitignored.
3. **Approval gates** — pause at script, prompts, storyboard, video, final.
4. **VRAM discipline** — call ComfyUI `/free`, unload Ollama between stages.
5. **Narration = source of truth** — NEVER `-shortest` in muxes (crops last word). Cap by `-t audio_duration` or map full audio.
6. **fps consistency** — Wan native 16fps. gen + upscale + assemble must agree on fps. Relabel ≠ resample (relabel shrinks duration 16/24=0.667x).
7. **Restart bot to load .py edits** — Python no hot-reload.

## Folder Structure
```
E:\Rexjaw_VFX\
├── 00_Tools\       # ffmpeg, ollama.exe, python311
├── 01_ComfyUI\     # ComfyUI install
├── 02_Agent\       # Python source (this repo) — modules\, modules\image_backends\, modules\video_backends\, venv\
├── 03_Models\      # weights + Ollama models (gitignored)
├── 04_Outputs\     # scripts/ storyboards/ clips/ (clips/_temp = raw parts) final/ approved_prompts/
├── 05_Config\      # models.json, styles.json, runtime_settings.json, secrets.env, pending_state.json, workflows/
├── 06_Logs\        # gitignored
```

---

## 1. What This Bot Does
Local AI animation pipeline. Theme → 30-sec kids story (Qwen 2.5 14B via Ollama) → per-shot storyboard image (ComfyUI + Z-Image Turbo) → per-shot video (ComfyUI + Wan 2.2 14B I2V @16fps) → Kokoro TTS narration → assemble 9x16 + 16x9 + 1x1 MP4s for YouTube Shorts. Driven by Discord bot **and** NiceGUI browser dashboard (localhost:7860, left-nav tabbed UI), shared on-disk JSON state synced both ways. User approves every stage; prompts + narration editable inline (manual or AI rewrite). Dashboard reachable from phone via ngrok tunnel behind a password gate. Strict containment in `E:\Rexjaw_VFX`.

---

## 2. File Structure (every .py, one line)

### `02_Agent/` root — mostly STALE duplicates of `modules/` (TODO #1). Live edits go to `modules/`.
- `claw_bot.py` — **MAIN**. Discord bot, command handlers, approval-button wiring, persistent state, dashboard auto-launch.
- `agent.py` — channel-scoped chat memory.
- `agent_router.py` — Qwen router: chat → tool calls.
- `status_check.py` — phase-by-phase install/health report.
- `inspect_workflow.py` / `preflight_phase4.py` — one-off tools.
- `test_*.py` — one-off sanity tests.
- STALE dups: `approval_buttons.py` `clip_generator.py` `comfyui_*.py` `control_panel.py` `image_backend.py` `video_backend.py` `music_generator.py` `pending_feedback.py` `prompt_polisher.py` `runtime_settings.py` `safety_filter.py` `script_generator.py` `storyboard_generator.py` `storyboard_workflow.py` `tts_engine.py` `upscaler.py` `video_workflow.py`.

### `02_Agent/modules/` — LIVE code
- `__init__.py` — empty pkg marker.
- `agent.py` — per-channel memory (turns, script, stage).
- `agent_router.py` — Qwen router → `{reply, tool_call}`.
- `approval_buttons.py` — Discord button views (Script/Storyboard/Video/Shot/Clip), custom_ids, modal prompt edit.
- `assembly.py` — final stitch. Per-shot clips → 9x16+16x9+1x1. Transition crossfade (silent-padded) OR cut (instant +0.4s breath). Music mix under narration.
- `beat_policy.py` — per-beat knobs (cfg, negative augment). lora_4step now False all beats. Frame-cap field unused.
- `channel_cleanup.py` — bulk-delete bot msgs.
- `clip_generator.py` — **CORE**. TTS → split narration into ≤`max_clip_seconds` parts → video/part (last-frame chained) → mux freeze-pad → concat one shot clip. `max_clip_seconds` + fps read PER-MODEL from active backend config (Wan 5s→splits; LTX 20s→single pass, no chaining). Fallbacks: DEFAULT_MAX_SECONDS=5, DEFAULT_FPS=16.
- `comfyui_kontext_base.py` — Kontext img2img (char anchor, optional).
- `contact_sheet.py` — PIL grid; one PNG of all frames.
- `control_panel.py` — persistent pinned button panel in #claw-bot.
- `dashboard.py` — Gradio dashboard (legacy fallback).
- `dashboard_nicegui.py` — **ACTIVE** NiceGUI dashboard. localhost:7860. Full pipeline wizard + control parity (settings/models/queue/tools/per-shot regen). Signature-cached refresh (no flicker).
- `embed_styles.py` — themed Discord embed factory.
- `feedback_thinker.py` — Qwen: vague feedback → surgical instructions.
- `generation_meta.py` — per-job metrics → generation_history.json.
- `gpu_utils.py` — VRAM stats, ComfyUI /free, Ollama unload, ensure_vram_free.
- `health_monitor.py` — live status-channel embed.
- `image_backend.py` — pluggable image-adapter loader.
- `model_registry.py` — reads models.json; switch backends.
- `music_generator.py` — MusicGen BG music.
- `pending_feedback.py` — JSON paused-revisions queue.
- `sync_bridge.py` — Web→Discord on-disk event queue (emit + cursor files); drained by claw_bot poller.
- `progress_bar.py` — Unicode bar + rolling-window ETA.
- `prompt_approval.py` — batch image+motion prompt gen → per-shot embed Edit/Reseed/Approve + master Approve-All + JSON state.
- `prompt_assembler.py` — Qwen builds final Z-Image prompts + Wan motion prompts (beat-aware char scrub).
- `prompt_polisher.py` — legacy single-pass; `!repolish` only.
- `runtime_settings.py` — JSON overrides: style/voice/aspect/cfg/sync/upscale/music/reference/**transition_mode**.
- `safety_filter.py` — block unsafe words.
- `script_generator.py` — 2-stage story gen: story_writer (prose) → structurer (schema JSON). Targets ~30s / 70-85 words.
- `shot_tailor.py` — per-shot beat tailor; NOT in auto pipeline (superseded by prompt_assembler).
- `storyboard_generator.py` — one image/shot via active backend; reads approved prompts/seeds.
- `storyboard_workflow.py` — Discord workflow: gen frames, per-shot embeds, contact sheet, approval.
- `story_writer.py` — stage-1 free prose writer.
- `theme_bank.py` — theme-of-day / random theme.
- `tts_engine.py` — Kokoro-82M wrapper, multi-voice.
- `upscaler.py` — Real-ESRGAN 4x/clip via ComfyUI. Injects source fps into output (no duration shrink), no -shortest re-mux.
- `video_backend.py` — pluggable video-adapter loader.
- `video_workflow.py` — Discord workflow: per-shot TTS+video, post clips w/ ClipControlView, approval.

### `02_Agent/modules/image_backends/`
- `comfyui_flux2.py` — Flux.2 Klein 9B fp8 (best quality, slow).
- `comfyui_kontext_base.py` — Kontext img2img (char anchor).
- `comfyui_zimage_base.py` — **ACTIVE** Z-Image Base/Turbo; paragraph prompts; injects seed/dims/cfg.

### `02_Agent/modules/video_backends/`
- `__init__.py` — empty.
- `comfyui_ltx_video.py` — legacy LTX-2.
- `comfyui_wan22.py` — Wan 2.2 5B.
- `comfyui_wan22_14B.py` — **ACTIVE** Wan 2.2 14B I2V dual-UNet fp8; optional lightx2v 4-step LoRA (default OFF).

---

## 3. What We Changed Today (2026-06-10) — production hardening pass
- **Git repo FIXED.** Ignore file was named `gitignore` (no dot) → git never read it; venv (3,496 files) + .pyc were tracked, and most live modules were NEVER committed. Now: `.gitignore` proper (adds secrets.env, .nicegui/, *.zip), venv/pycache untracked, ALL live source committed + pushed to GitHub `origin` (Rex_Bot repo). 85 tracked files.
- **Atomic JSON writes** — new `modules/file_utils.py` (`atomic_write_json`: tmp + fsync + rename). All 16 state-write sites converted (stats, pending_state, scripts, approved prompts, runtime settings, registry, panel state, seeds, gen history, feedback, agent memory). Crash mid-write keeps old file.
- **Shared GPU job lock** — new `modules/job_lock.py`. Same-thread reentrant (Discord pipeline chains), cross-thread release (dashboard UI-thread acquire → worker release), on-disk marker `05_Config/job_lock.json` (blocks 2nd bot process; stale/dead-pid markers stolen; 4h staleness). Discord entrypoints via `_gpu_job(label)` decorator; `!upscale`/`!assemble` inline; dashboard via `_try_begin/_end` (keeps S.busy). NOTE: two Discord commands on the same event loop can still interleave (same thread = reentrant) — status quo kept.
- **subprocess timeouts** — every ffmpeg/ffprobe `subprocess.run` now has timeout (60s probes, 300–900s encodes): assembly, clip_generator, music_generator, upscaler.
- **Dashboard login hardened** — `secrets.compare_digest` + 60s lockout after 5 failed attempts (public tunnel exposure).
- **Log rotation** — claw_bot.log rotates at 10 MB, 5 backups.
- **Crash auto-restart** — `00_Tools/launch_clawbot.ps1` runs bot in a loop: exit 0 (`!shutdown`/`!restart_bot`) = stop; non-zero = relaunch after 10s, max 5 crashes/5 min.
- **Boot config validation** — new `modules/config_check.py`: models.json (active∈available per backend), styles.json, runtime_settings parse; refuses boot w/ clear error. Warns on secrets.env BOM + <10 GB disk. Disk also checked per GPU job (decorator + _try_begin).
- **Test suite** — `tests/` pytest, 72 tests, no GPU: import smoke (all modules), clip fps/duration math + LTX 8N+1 rule + fps-relabel regression, atomic writes, job-lock semantics, sync-bridge cursor, config validation. Run: `venv\Scripts\python -m pytest tests -q`.
- **requirements.txt re-pinned to venv reality** (discord.py 2.7.1, aiohttp 3.13.5, nicegui 3.12.1…); dropped unused ollama/watchdog/nvidia-ml-py/loguru pkgs; full snapshot in `requirements.lock.txt`. pytest added.
- **22 stale root dups DELETED** (recoverable from git history). Root keeps only claw_bot, agent, agent_router, status_check, inspect_workflow, preflight, test_*.
- **Sync cursor per-event** (no batch re-posts after crash). **pending_state capped** 50/category on load. **`CLAW_OWNER_IDS`** optional in secrets.env = lock all !commands to listed user ids (unset = open).

## Earlier (2026-06-04)
- **Dashboard UI redesign** (`dashboard_nicegui.py`): long scroll → left-nav drawer + vertical tabs (Pipeline/Settings/Models/Queue/Tools). Section cards built at page level then `.move()`'d into tab panels (avoids mass re-indent). Live log → collapsible footer. CSS toned down (static gradient hero, less glow). Mobile: header ☰ button toggles drawer (Quasar auto-hides drawer on phones).
- **Web↔Discord sync.** Discord→Web already free (dashboard polls disk every 1.5s). Web→Discord NEW: `modules/sync_bridge.py` on-disk event queue (`sync_events.json` web-writes, `sync_cursor.json` bot-writes). Bot poller `_sync_bridge_loop` (claw_bot, 5s) re-posts updated clip/frame to #videos/#storyboards. Emits: video_regen, storyboard_regen, video_done, final_done. Bot skips backlog (cursor=latest_id at startup).
- **Per-shot narration edit** (dashboard Prompts tab): 🎙️ textarea + Save → writes script JSON `shots[].narration` (`update_shot_narration`). Read from disk each render → synced across front-ends.
- **AI rewrite narration**: `script_generator.rewrite_narration` (Qwen, one clean line). Web 🪄 dialog button; Discord `!rewrite_narration <id> <shot> [hint]` + control-panel `NarrationRewriteModal`.
- **Dashboard auth**: Starlette `_AuthMiddleware` + `/login` page. ON when `DASHBOARD_PASSWORD` set in secrets.env; unset = open + warning. Logout button. Bypass `/_nicegui*` + `/login`. Uses `app.storage.user` (storage_secret already set).
- **Remote mobile access**: ngrok fixed static URL (`*.ngrok-free.dev`). `02_Agent/start_dashboard_ngrok.ps1` (reads NGROK_AUTHTOKEN/NGROK_DOMAIN/DASHBOARD_PASSWORD). Cloudflare quick-tunnel fallback `start_dashboard_tunnel.ps1` (random URL, no account). `start_rex.bat` one-click (bot+tunnel). Wired into `00_Tools/launch_clawbot.ps1` (boot shortcut now starts tunnel + kills stale ngrok; `stop_clawbot.ps1` kills ngrok). `control_panel.py` now load_dotenv()s secrets.env so `CLAW_DASHBOARD_URL` feeds the Discord "Open Dashboard" link.

### Earlier (2026-05-29)
- **GPU upgrade** RTX 3070 8GB → RTX 5080 16GB. Updated both CLAUDE.md owner lines.
- **`clip_generator.py`**: `DEFAULT_FPS` 24→16 (matched Wan native, killed lighting flicker). `_mux` else-branch `-shortest`→`-t audio_duration` (narration never trimmed). `_concat_clips` verifies output dur vs sum(parts), re-encodes if short >0.2s. `generate_clip` logs `dur=X vs narration=Y` + ⚠️ if short.
- **`beat_policy.py`**: all `lora_4step` True→False (kill 4-step distill artifacts; 5080 affords full 20-step).
- **`dashboard_nicegui.py`**: added `_rcache`/`_changed()` signature guard. Media + prompt containers rebuild only on content change — stops 1.5s timer destroying ui.image/ui.video/textarea each tick (flicker + textarea wipe).
- **`runtime_settings.py`**: `transition_mode` (crossfade|cut) get/set/clear + `get_effective_transition_mode` (default crossfade).
- **`assembly.py`**: import rs. Crossfade now silent-pads each non-last clip (tpad/apad CROSSFADE_SEC) so xfade overlap lands on silence, not narration. New `_build_cut_filter` (instant hard cut, concat). `CUT_GAP_SEC=0.4` breath (freeze last frame + silent audio) per shot except last so last word finishes before cut. `_assemble_one_aspect(transition=)`. total_dur mode-aware.
- **`control_panel.py`**: VideosView 🎞️ Transition button → set_transition.
- **`claw_bot.py`**: `cmd_set_transition` (aliases transition/set_xfade) + `_CMDS["set_transition"]`.
- **`upscaler.py`**: **ROOT CAUSE FIX** for cropped narration. `_probe_fps` reads source r_frame_rate; `_build_workflow(source_fps)` injects it into `VHS_VideoCombine.frame_rate` (was hardcoded 24 vs 16fps source → 0.667x shrink). `_reattach_audio` dropped `-shortest`.
- **`dashboard_nicegui.py`**: control parity with Discord. Added Settings card (style/aspect/voice/music/sync/transition/steps/cfg + upscale & reference toggles + show-current/reset-all), Models card (image/video backend switch), Queue card (pending feedback list/load/drop), Tools card (upscale/re-assemble/suggest theme), and per-shot 🔁 Regen buttons on storyboard + video cards. New actions: `regen_storyboard_shot`, `regen_video_shot`, `run_upscale_action`, `render_queue`. Settings write straight to runtime_settings.json (shared state, no Discord coupling).

---

## 4. In Progress / Half-Done
- **Bot restart needed (2026-06-10)** — hardening-pass edits (atomic writes, job lock, login lockout, config validation, owner gate) on disk, NOT live until restart. Tests pass (72), but live Discord + dashboard flows unverified.
- **Bot restart needed (2026-06-04)** — today's .py edits (UI tabs, sync_bridge, auth, narration edit, AI rewrite) on disk, NOT live until restart.
- **Mobile nav ☰** — verified by server build only, not a live phone browser. Confirm on device after restart + hard-refresh.
- **Discord→Web settings widgets** — web reads runtime settings live on render, but the select WIDGETS show stale value until page reload.
- **AI rewrite** — LLM output not exercised live (needs Ollama up).
- **Bot not yet restarted** — today's .py fixes on disk, NOT live. Must restart + regenerate clips for script `20260529_080533` (its on-disk clips still old cropped 24fps).
- **Recovery option unused** — `04_Outputs/clips/_temp` holds full 16fps raw Wan part videos; clips rebuildable without re-render (user chose clean regenerate instead).
- **`assembly.py` FPS=24** while gen now 16 — assembly re-encodes (resamples, harmless dup), but inconsistent. Consider 16.
- **Dashboard ↔ Discord parity** — MOSTLY DONE. Dashboard now has settings/models/queue/tools/per-shot regen. Still missing: queue RESUME-with-feedback (only load+drop), multi-script generation queue, repolish, stats counters (Discord-specific). Untested in live browser — verify after restart.
- **Multi-part shot UX** — clip splits long narration into parts; UI shows ONE clip/shot; no per-part regen.

---

## 5. Known Issues / TODOs
**New (2026-06-04):**
- **secrets.env must be BOM-free** — PS `Out-File -Encoding utf8` adds a BOM → dotenv drops the first line. Edit with a plain editor / append only.
- **ngrok free = 1 tunnel/agent**, fixed `.ngrok-free.dev` URL. Second run collides (launcher kills stale ngrok first). Custom-hostname errors (ERR_NGROK_314) = wrong value pasted (use the `*.ngrok-free.dev` hostname, not `rd_/ep_` IDs).
- **Mobile drawer** hidden < Quasar breakpoint → needs header ☰ toggle (added).
- **`sync_events.json`** capped at 200 events; cursor in `sync_cursor.json`.

1. ~~Stale root .py duplicates~~ DELETED 2026-06-10 (in git history if needed).
2. ~~discord.py version mismatch~~ RE-PINNED 2026-06-10 (requirements match venv).
3. **audiocraft 1.3.0 dep conflict** — torch pin vs installed. Music gen may crash; verify.
4. **Continuation part motion prompt** — Part B+ reuses Part A motion; Wan may stall. Split per part via LLM.
5. **Continuation seed bump +7919** — heuristic; may jump at part boundary. Try frame-0 latent init.
6. **No per-part regen** — UI regens whole shot.
7. **Dashboard no watchdog** — NiceGUI thread death = silent UI loss. (Bot process itself now auto-restarts on crash via launcher loop.)
8. **`CUT_GAP_SEC=0.4` hardcoded** in assembly.py — make runtime (`!set_cut_gap`) if needed.
9. **beat_policy frame-cap field** unused — delete or repurpose.
10. **Per-character LoRA** — chars drift across shots; needs per-character LoRA train.
11. **YouTube uploader** — not built; upload manual.
12. ~~Pending state JSON unbounded~~ CAPPED 2026-06-10 (newest 50/category kept on load).

---

## 6. Important Decisions (don't redo)
- **Sync directions split (2026-06-04).** Discord→Web = disk poll (dashboard 1.5s, free, no code). Web→Discord = `sync_bridge` event queue + bot poller. Separate files (events web-writes, cursor bot-writes) = no cross-process write contention. Skip backlog on boot (cursor=latest_id).
- **Narration source of truth = script JSON** (`shots[].narration`), NOT the prompts JSON. Edit/AI-rewrite reads disk first (no clobber) + writes back → both front-ends sync. Consumed at render time.
- **Public dashboard ⇒ password gate MANDATORY.** Bind stays `127.0.0.1`; tunnel runs on same PC and reaches it. NEVER bind 0.0.0.0 to expose. Gate = Starlette middleware, off when `DASHBOARD_PASSWORD` unset (localhost dev).
- **ngrok free static domain over Cloudflare named tunnel.** CF named tunnel needs an owned domain; ngrok free gives 1 fixed `.ngrok-free.dev`. CF quick tunnel kept as no-account random-URL fallback.
- **Dashboard layout = left-nav vertical tabs, not long scroll.** Cards built then `.move()`'d into panels (avoids re-indenting the whole UI). Mobile MUST keep the ☰ toggle (drawer auto-hides).
- **Narration = duration source of truth.** Video freeze-pads if short; audio never truncated. NO `-shortest` anywhere in muxes — it crops the last word.
- **fps must match across gen→upscale→assemble.** Wan = 16fps. Upscale output fps PROBED from source + injected (not hardcoded). Relabeling frames to another fps shrinks duration (16/24=0.667x) — the bug that cropped all narration.
- **4-step lightx2v LoRA OFF.** Distill = quality ceiling + artifacts. 5080 has VRAM for full 20-step. cfg 3.5 full path; beat overrides 2.0-2.8 valid.
- **Transition modes:** crossfade (0.3s dissolve, silent-tail padded so narration never overlaps) vs cut (instant + 0.4s breath so last word finishes). Default crossfade. Switchable: control panel 🎞️ / `!set_transition`.
- **Dashboard refresh guarded by content signature** — never blind `.clear()` on the 1.5s timer (caused media flicker + textarea wipe).
- **Approve prompts BEFORE render** (edit-then-render). Single Approve-All. Image + motion approved separately per shot. Modal popups for edits. Batch posting + master button.
- **Dropped shot_tailor + prompt_polisher** from auto pipeline (prompt_assembler does it once, beat-aware).
- **Removed QA + continuity VLM passes** — Qwen2.5-VL hallucinated, net negative. Model + code deleted.
- **NiceGUI over Gradio** (Material Design, Quasar). Both Discord + Dashboard run, shared on-disk JSON. Localhost-only dashboard.
- **Clip ceiling is PER-MODEL** via `max_clip_seconds` in models.json (Wan=5, LTX=20). `n_parts=ceil(narration/max)`. Wan splits + last-frame chains; LTX single pass (chaining auto-skipped when n_parts=1). One MP4/shot. fps also per-model (`default_fps`).
- **Switching video/image model RESETS cfg+steps overrides** (`rs.reset_overrides_for_model_switch`) → each model runs at its own models.json recommended settings. Wired in `cmd_switch_model` + dashboard model selects.
- **LTX adapter workflow file is configurable** (`workflow_file` in models.json; default `ltx2_video_only_v1_api.json`). LTX-2.3 not yet runnable: only UI-format workflow exists (`LTX-2.3_-_I2V_T2V_Basic_GGUF.json`, 111 nodes) — needs API-format export + models.json entry.
- **`approved_prompts/{script_id}.json`** = shared Discord↔Dashboard state, read every render.
- **Persistent approval views** — custom_id + timeout=None + _PersistentDict → pending_state.json + restore on_ready. Survives crash.
- **Contact sheet before approval** — one composite PNG > scrolling 10 embeds.
- **2-stage script gen** — story_writer prose → structurer schema. Single frame per shot (Wan I2V from one start frame).
- **Z-Image over SDXL** — paragraph prompts for non-coder. `!` prefix commands kept.
- **Pluggable backends via models.json** — new model = one adapter file in modules/{image,video}_backends/. Don't rewrite cores.
- **Never `python -m modules.X`** — dual-import bug. Standalone scripts + sys.path.insert.
- **`PENDING_APPROVALS = {}`** module-level — absence = NameError. Same for siblings.

---

## Known Bug Patterns
- **fps relabel ≠ resample** — re-containering N frames at a different fps changes DURATION (16→24 = 0.667x shorter), and -shortest re-mux then crops audio. Always match output fps to source.
- **`-shortest` crops narration** — with `-c:v copy` a frame-timing quirk stops on the short video stream. Use `-t audio_duration` or map full audio.
- **Dual-import bug** — never `python -m modules.X`.
- **Asyncio threading** — progress callbacks from non-async threads need `asyncio.run_coroutine_threadsafe`.
- **aiohttp ClientOSError [WinError 64]** during Discord upload — `_send_with_retry`; per-retry fresh `discord.File`.
- **Dashboard blind `.clear()` on timer** — destroys/reloads media = flicker. Guard with content signature.

## Git Hygiene
- Commits: lowercase imperative ("add video adapter").
- Never commit `.env`, `01_ComfyUI/`, `03_Models/`, `06_Logs/`, `04_Outputs/`, `venv/`.
- Branch off `main`; merge only when tested.

## Test Before Push
```powershell
cd E:\Rexjaw_VFX
.\02_Agent\venv\Scripts\activate
python 02_Agent\claw_bot.py  # bot connects + dashboard at http://127.0.0.1:7860
```
Bot online + dashboard loads + Discord panel updates = safe.

## How to Help
- Read this first. New concept → one-paragraph VFX analogy. Short responses. Confirm destructive actions.
- Phase 5/6 → pluggable adapter pattern.
- **Caveman mode** active per session — terse fragments; code/commits/security normal.
