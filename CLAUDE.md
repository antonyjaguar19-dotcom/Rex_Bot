# CLAUDE.md — Claw Bot project context

> Auto-loaded every session. Dense. Current.

## Owner
**Jeffy** — VFX artist (Maya, Nuke, Redshift, Shotgun), Chennai. Zero coding. Use VFX analogies. Windows 10/11. **RTX 5080 (16 GB VRAM), 64 GB RAM.** (Upgraded from RTX 3070 8GB 2026-05-29.) VRAM no longer hard bottleneck — full-quality paths viable.

## Core Rules (NEVER violate)
1. **Containment** — all inside `E:\Rexjaw_VFX`. No system pip.
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
Local AI animation pipeline. Theme → 30-sec kids story (Qwen 2.5 14B via Ollama) → per-shot storyboard image (ComfyUI + Z-Image Turbo) → per-shot video (ComfyUI + Wan 2.2 14B I2V @16fps) → Kokoro TTS narration → assemble 9x16 + 16x9 + 1x1 MP4s for YouTube Shorts. Driven by Discord bot **and** NiceGUI browser dashboard (localhost:7860, left-nav tabbed UI), shared on-disk JSON state synced both ways. User approves every stage; prompts + narration editable inline (manual or AI rewrite). Dashboard reachable from phone via ngrok tunnel behind password gate. Strict containment in `E:\Rexjaw_VFX`.

---

## 2. File Structure (every .py, one line)

### `02_Agent/` root — entrypoints + harnesses only. Live module edits go to `modules/`.
- `claw_bot.py` — **MAIN**. Discord bot, command handlers, approval-button wiring, persistent state, dashboard auto-launch.
- `agent.py` — channel-scoped chat memory.
- `agent_router.py` — Qwen router: chat → tool calls.
- `status_check.py` — phase-by-phase install/health report.
- `inspect_workflow.py` / `preflight_phase4.py` — one-off tools.
- CLI bridges (isolated venvs, driven over subprocess): `qwen_tts_cli.py`, `chatterbox_cli.py`, `whisperx_align_cli.py`.
- Render harnesses: `run_kids_e2e.py` `run_horror_e2e.py` `run_music_e2e.py` `run_facts_e2e.py` `run_voice_samples.py` `run_voice_music_samples.py` `qa_styles_run.py` `test_lora_e2e_run.py` (LoRA resume path).
- Real test suite = `tests/` (pytest, no GPU). 13 legacy root `test_*.py` one-offs deleted 2026-07-10 (in git history).

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
- `sync_bridge.py` — Web→Discord on-disk event queue (emit + cursor files); drained by claw_bot poller. Carries job start/progress/end, finished media, and the approval views.
- `job_feed.py` — Discord→Web live job snapshot (`05_Config/bot_job.json`; bot writes, dashboard reads). Fed by `_gpu_job` tapping the log stream.
- `mascot.py` — mascot thumbnails + presenter stills (scene prompt → Qwen-Edit/USO render).
- `mascot_library.py` — the mascot SHELF: `assets/mascots/<id>/` (art + angles + optional `voice.wav` + `meta.json`); `rs.active_mascot` picks who presents. `mascot.mascot_path()/refs()` resolve through it.
- `progress_bar.py` — Unicode bar + rolling-window ETA.
- `prompt_approval.py` — batch image+motion prompt gen → per-shot embed Edit/Reseed/Approve + master Approve-All + JSON state.
- `prompt_assembler.py` — Qwen builds final Z-Image prompts + Wan motion prompts (beat-aware char scrub).
- `prompt_polisher.py` — legacy single-pass; `!repolish` only.
- `runtime_settings.py` — JSON overrides: style/voice/aspect/cfg/sync/upscale/music/reference/**transition_mode**.
- `safety_filter.py` — block unsafe words.
- `script_generator.py` — 2-stage story gen: story_writer (prose) → structurer (schema JSON). Targets ~30s / 70-85 words.
- `shot_editor.py` — insert new shot anywhere (LLM-written), renumber + shift all on-disk artifacts.
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
- `comfyui_sdxl_ipadapter.py` — **KIDS** backend. SDXL + per-style LoRA + IP-Adapter PLUS (cast-sheet character lock). Pinned for kids via `storyboard_generator._kids_backend()`.
- `comfyui_flux_lora.py` — **global active** (music + horror fallback). Flux.1-dev + char-LoRA stacking.
- `comfyui_flux2.py` — Flux.2 Klein 9B fp8 (best quality, slow; has Node-100 reference path).
- `comfyui_kontext_base.py` — Kontext img2img (char anchor).
- `comfyui_zimage_base.py` — Z-Image Base/Turbo; paragraph prompts; injects seed/dims/cfg.

### `02_Agent/modules/video_backends/`
- `__init__.py` — empty.
- `comfyui_ltx_video.py` — legacy LTX-2.
- `comfyui_wan22.py` — Wan 2.2 5B.
- `comfyui_wan22_14B.py` — **ACTIVE** Wan 2.2 14B I2V dual-UNet fp8; optional lightx2v 4-step LoRA (default OFF).

---

## 3F. FACTS MODE — **FROZEN 2026-07-13**. Spec: `02_Agent/FACTS_MODE.md`. Don't tune it without being asked.
Facts Shorts is finished and locked: mascot presents 5 true facts, cloned voice, Wan 720p shots, read-along captions, music bed, thumbnail held as the first frame, under 40s. Frozen settings + reasons live in **`FACTS_MODE.md`**; `config_snapshot/` mirrors `05_Config` (which is outside git). 577 tests green. Commits: 6623667, 5e6d99c, 30582d6, ad1c8bf, d4d9fd0, c787651.
- **Music bed (3 shipped bugs, all silent).** ACE-Step under-fills — asked 49s it wrote 29s of music + 10s of digital silence + 8s of junk. Fixes: ask **2x** and *measure the composed body*, re-ask longer if short (`_MUSIC_ASK_HEADROOM=2.0`, `_MUSIC_TRIES=2`); cut the bed at its **music body** (`_music_body_end`, first gap ≥1.5s under −45dB after ≥8s of music) not its tail — trimming the tail left the hole mid-track and `_align_music_tail` (which fits a long track by cutting from the FRONT to keep the outro) then kept the dead air; **never loop** a short bed (replaying its opening = the same 8 bars twice, heard instantly) — it starts late instead.
- **`fasm.audit_music_bed()`** — subtracts the narration back out of the **finished file**; what remains IS the bed. Logs `bed verified … no dropouts` or `THE MUSIC DROPS OUT`, to dashboard + Discord. Every music bug looked right at every intermediate step and was wrong in the file.
- **No silent downgrade** — `facts_pipeline.preflight()` proves the backend answers BEFORE the story is written; mascot mode ON now RAISES instead of quietly rendering abstract backdrops (a dead ComfyUI used to cost a 40-min render of the wrong film).
- **Poster frame replaceable** — `prepend_still(replace=True)` rebuilds from a pristine copy in `final/_posterless/`; a re-rolled thumbnail now reaches the VIDEO (it used to rewrite only the .jpg, leaving the reel opening on the poster you just rejected). `publish_kit._refresh_poster_frame` does it for every aspect.
- **Kokoro removed from all facts UI** (survives as a silent in-pipeline fallback). Reel **description** now shows in the dashboard (it looked for `facts_<id>_description.txt`; publish_kit writes `<video-stem>_description.txt`, so mascot reels showed none).

## 3P. FACTS LIBRARY + FACT MEMORY (2026-07-13)
Two features on the Facts tab. Spec: `FACTS_MODE.md` § The reel library + the fact memory.
- **`modules/facts_library.py`** — outliner of every FINISHED reel (newest first) with video, the facts it told, upload kit, and **delete**. Delete takes the whole footprint (story JSON + `storyboards/facts_{id}/` + `final/facts_{id}_*` incl. kit + `_posterless/`); a reel lives in 5 places and deleting by hand meant remembering all 5 (that's why `final/` held 212 facts files). Id is validated `^\d{8}_\d{6}$` BEFORE it goes into a delete glob (`*` would take the lot). Discord: `!facts_list`, `!facts_delete <id> [forget]`.
- **`modules/facts_memory.py`** (`04_Outputs/facts/_memory.json`) — every shipped fact is remembered per topic; the writer is asked for facts NOT on the list. **The prompt is not the feature — the CHECK is**: a model told "don't repeat these" rewords the fact instead. `is_repeat()` = content-word Jaccard (plurals folded, NUMBERS KEPT so "three hearts" ≠ "nine brains"), threshold 0.55 (measured: reworded 1.00/0.75/0.71 vs new 0.25/0.20). Repeats are dropped; if the reel goes short it re-rolls and TELLS the model what it repeated. Exhausted topic fails loudly. A THIN roll (few facts, no repeats) must not be blamed on the memory. `backfill_from_reels()` runs once (31 reels predated it). Placeholder reels never recorded.
- **Deleting a reel does NOT forget its facts** — a bad reel is a reason to want a different one next time. Release: delete-dialog switch / `!facts_delete <id> forget` / `!facts_forget <topic>`. Inspect: `!facts_memory [topic]`.
- Watch: `_singular()` — naive `/(es|s)$/` turned 'octopuses'→'octopus' but 'octopus'→'octopu', so the two never matched and repeats slipped through. Fold plurals, leave -us/-ss/-is/-as alone.
- `tests/conftest.py` isolates `facts_memory.MEMORY_PATH` — the writer records now, so a test would poison the live do-not-repeat list. 624 tests green.

## 3O. FACTS UI STRIPPED (2026-07-13) — one control: WHO presents
The dashboard Facts card is now Topic + **Mascot dropdown** + Generate. Every other widget is gone (video mode, thumbnail, mascot toggle, voice/emotion/pace, lip-sync, music+mood, max-sec, thumb-hold). Guarded by `tests/test_dashboard_build.py::test_the_facts_card_offers_nothing_but_the_mascot`.
- **Three settings became CONSTANTS in `runtime_settings`** (they ignore the file — a toggle removed from the UI but still read from disk is a silent downgrade with nothing on screen to explain it): `get_facts_video_mode()` → `"wan"` (facts are ANIMATED; Ken Burns only survives as the in-pipeline OOM rescue + the `animate=False` harness override), `get_facts_mascot_mode()` → `True`, `get_mascot_tts_engine()` → `"chatterbox"` (every mascot speaks in ITS OWN cloned clip; Qwen→Kokoro remain the fallback cascade when there's no clip). Setters `set_facts_video_mode` / `set_facts_mascot_mode` / `set_mascot_tts_engine` DELETED, with their Discord commands (`!facts_video`, `!facts_mascot`); `!mascot_voice` is now a clip-installer for the ACTIVE mascot (no preset picking).
- Still tunable from Discord only (frozen defaults, not on the page): thumbnail, music+mood, max seconds, thumb hold, lip-sync, `!mascot_tone`.
- **`tests/conftest.py` now also blanks `rs.DEFAULT_MASCOT_VOICE_REF`** — with the clone as the default engine, voice tests were reaching for the owner's REAL recording and the live Chatterbox bridge (suite 165s → 110s once stopped).

## 3M. MASCOT SHELF (2026-07-13) — more than one mascot
A mascot used to be ONE file (`assets/mascot.png`), so facts mode could only ever star one character. Now: `modules/mascot_library.py` + `assets/mascots/<id>/` (primary art, optional angles, optional `voice.wav`, `meta.json` name). `rs.active_mascot` = who presents; `mascot.mascot_path()/mascot_refs()` resolve through the shelf, so switching mascot switches every presenter still, thumbnail and (with its own clip) the cloned voice. **Dashboard: new left-nav "Mascots" tab** (add / delete / set active / upload angles + voice) and a **Mascot dropdown on the Facts card**. **Discord: `!mascot list|use <id>|new <name>|rename <id> <name>|rm <id>`.** Details in `FACTS_MODE.md` § The mascot shelf.
- **Migration is a copy, once** — `migrate()` copies the old flat `assets/mascot*.png` + `mascot_voice.wav` into `mascots/default/` and leaves the originals. Once `assets/mascots/` exists the flat layout is never read again (deleting every mascot must NOT resurrect the old one).
- **Intake = name + 3 views (front/threequarter/side) + a ~10s voice clip** (no BACK view — no face, no chest logo, so identity transfer never keyed on it) (`create_from_intake`). Only FRONT is required (it's also installed as the primary; `mascot_refs()` still passes ONE ref by default — three made Qwen copy a reference's stance). Files are STAGED and only committed on Create, so a failed intake leaves nothing behind. The voice is **cloned, not trained** (Chatterbox reads the clip); 5-30s accepted, ~10s ideal, outside that = warning not refusal (`voice_warning()`).
- **A mascot with no `voice.wav` uses the shared `mascot_voice_ref` clip** — give a new character its own clip or it speaks in the cub's voice.
- **`tests/test_dashboard_build.py` is new and load-bearing** — it builds the real page inside a manual NiceGUI `Client`. A dashboard HTTP 200 proves NOTHING: NiceGUI runs a `@ui.page` builder on websocket CONNECT, so a bad widget arg served a healthy 200 and only exploded in the browser. That is how `e.content`/`e.name` (NiceGUI 2.x upload API) nearly shipped — 3.x gives `e.file`, and `await e.file.save(path)`.
- 596 tests green.

## 3S. Sync — BOTH directions now (2026-07-13)
Discord ↔ Web are interchangeable mid-job.
- **Web → Discord** (`sync_bridge`, extended): `_bg_gpu` emits `job_start`/`job_end`, `State.push` emits throttled `job_progress`; the bot keeps ONE live embed per job (progress coalesced — newest line per job per poll), uploads the finished media, and re-posts the **same approval view** for script/storyboard/video stages, so approving from either side drives the pipeline.
- **Discord → Web** (`modules/job_feed.py`, NEW): `05_Config/bot_job.json` holds the running Discord job — **bot writes, dashboard reads** (mirror image of `sync_events.json`; one writer per file, no cross-process locking). `_gpu_job` publishes the job and **taps the `claw_bot` log stream** for its duration, so it covers every mode without per-command wiring. Dashboard shows it in the status strip (it used to say "✓ Idle — ready" while the GPU was pinned), the live log, and the Facts stepper.

## 3c. Repo cleanup (2026-07-10)
Swept `E:\Rexjaw_VFX` for dead files. **Nothing hard-deleted except pure caches** — everything else moved to `_quarantine_20260710/` (137 files, 19.8 MB). Delete that folder when you're happy; restore from it if something breaks.
- **Hard-deleted (regenerable):** `.nicegui/` root + `02_Agent/` (22k session-storage files), `.pytest_cache` ×2, repo `__pycache__` ×8. Dashboard sessions reset → re-login once.
- **Quarantined:** `Rexvfx_Bot-main/` (stale GitHub zip extract of this repo), `gitignore` (the dotless dead file from the old bug), `AI_Training/` (superseded by `07_Training/`), `test/` (a Topaz Video AI project), `assets/espeak-ng.msi` (spent installer — TTS uses `00_Tools/espeak-ng`), `flux2*_current.json`, `bot_tool_code.txt` + `PROJECT_CODE_DUMP.txt` + `PROJECT_SNAPSHOT.md` (regenerable via `generate_tree.py`), `qwen-context.md`, `venc activation.txt`, `02_Agent/.qwen-history.json`, `polish_test.log`, `ERexjaw_VFX_tmp_sig.txt`, `02_Agent/04_Outputs/` (stray nested test wavs), and 90 one-off logs from `06_Logs/` (kept claw_bot/launcher/comfyui/ollama).
- **git rm'd (recoverable from history):** 13 legacy root `test_*.py` one-offs. Kept `run_*_e2e.py`, `qa_styles_run.py`, `test_lora_e2e_run.py`.
- **Deliberately kept:** `03_Assets/music/` (empty but is `music_generator`'s fallback lib), `06_Memory/` (live agent memory), `07_Training/` (6.5 GB, LoRA datasets — referenced by `modules/lora/*`), `00_Tools/` (4.9 GB toolchain), `02_Agent/assets/watermark.png`, `tools/status_post.py`, `CLAUDE.original.md`, `claw_bot_cheatsheet.md`, root `requirements.txt`.
- **Verified after:** 241 tests green, `claw_bot` imports, `config_check.validate_configs()` returns `[]`, music/agent dirs resolve.

## 3m. What We Changed (2026-07-10) — MANUAL MODE (5th pipeline)
Direct-drive mode: user prompts ComfyUI straight, no LLM chain. Commit 5160b80 on `feat/horror-story-mode`. 241 tests green. **Bot restart needed.**
- **`modules/manual_mode.py`** — project store `04_Outputs/manual/{id}/project.json` (atomic), board ops, freeform image gen (active/named backend, ref-image aware w/ TypeError fallback, adult-profile safety), `animate_shot` (active video backend, capped at model `max_clip_seconds`, no chaining), Kokoro narration, ACE-Step music bed, ffmpeg assembly (clips normalized OR Ken Burns stills → uniform 30fps segs → concat → music amix 0.30; narration-wins durations per Rule 5; hard cuts). Current-project marker `_current.txt` = Discord↔web sync.
- **Dashboard** — new **Manual** tab: project picker/new, image-model select + seed/steps/cfg (writes rs overrides), prompt/negative, reference upload + upload-straight-to-board (`ui.upload`), per-shot cards (motion/narration/duration, Animate/TTS/reorder/replace-with-last-gen/remove), music gen, multi-aspect assemble. Renderer signature-cached (finals mtimes folded in — stable filenames).
- **Discord** — `!mhelp` `!mnew` `!mlist` `!muse` `!mboard` `!mgen` (attachment=reference) `!mupload` (attachment=shot) `!mmotion` `!manim` `!mnarr` `!mdur` `!mmove` `!mrm` `!mmusic` `!massemble`. GPU-lock via `_gpu_job`. Model/param switching reuses `!switch_model`/`!set_steps`/`!set_cfg`.
- **Verified**: 15 new tests; assemble e2e w/ real ffmpeg (narration-wins math exact), Kokoro TTS live, dashboard headless boot HTTP 200. NOT live-verified (ComfyUI was down): image gen / animate / music adapters — glue only, adapters production-proven.

## 3z. What We Changed (2026-06-26) — kids mode → production (LLM + SDXL+IP-Adapter character consistency)
Deep pass on KIDS story mode (audio-first pipeline). Restart needed each .py edit; bot restarted + verified single instance. 201 tests green. Commits on `feat/horror-story-mode`: 7b83031, 0183636, 68c8e88, c924822, 2210edb, cd38419, c41b653, 08f4c09, f15b706.
- **Dialogue-aware narration split** (`audio_first_pipeline._split_sentences`) — old splitter broke on `!`/`?` inside quotes + didn't split after `."` → dangling fragments + two-speaker shots + wrong speaker→voice. New walker: never breaks inside a quote, keeps trailing close-quote, newlines = hard breaks. `_breath_groups` keeps a quoted line whole even if >70c. Structurer prompt rules 6+7: continuation fragments keep same subject/place, never pull later plot in.
- **Story length loop** (`story_writer.write_story`) — one-shot retry accepted 47→48 as "fixed". Now 3-pass loop keeping the draft closest to 75w, stops in-window [65,90]. Stories land 65-85w / 7-9 shots (~30s) instead of thin 45-60w / 4-5 shots (~20s).
- **Cast correctness** — `_normalize_char_types` (talking mouse stays animal, was typed human → rendered people), structurer only extracts NAMED chars (killed phantom 3rd character), title-label leak stripped (`_split_title_body`).
- **Render-path fixes** (`storyboard_generator`) — backfilled approved prompts now enriched via `_rewrite_as_paragraph` (were bare → rendered HUMANS); empty-frame anti-ghost only on `wide` shots (was blanking closeups); char-sheet injected on closeup/medium/insert even when prompt names no one.
- **NEW kids backend** `comfyui_sdxl_ipadapter.py` — SDXL + real per-style LoRA (pixar/cartoon_saloon/claymation/stickman, no-op on flux) + IP-Adapter PLUS on the cast sheet. PINNED kids-only (`_kids_backend()`); global active stays flux_lora so music/horror untouched. Workflow `05_Config/workflows/sdxl_ipadapter_api.json`.
- **Cast-sheet hardening** — SDXL+Pixar is human-biased + "reference sheet"+wide → drew a HUMAN on a contact-sheet collage that IPA propagated everywhere. Fixed: species-forward + anti-human (animal casts) + anti-collage + 1:1 aspect.
- **Per-character references (mixed-species fix)** — one combined cast sheet collapsed 2 different animal species into 1 (rabbit+tortoise → 2 rabbits). Now `_generate_character_sheets` renders ONE solo ref per char; `_shot_reference` uses this shot's char(s): 1→its sheet, 2+→`_composite_refs` stitches solos side-by-side as the IPA ref. Verified rabbit+tortoise both correct.
- **Style guard** (`audio_first._guard_style_for_cast`) — humanoid-art styles (cartoon_saloon/stickman) render animals as humanoids; non-human casts auto-switch to pixar.
- **CAST MATRIX — what kids mode is good at:** ✅ solo any type · same-species pair/trio · two humans · human+animal · two-different-animal-species (now, via per-char refs). Note: `05_Config` (models.json + workflow) is OUTSIDE the git repo.

## 3a. What We Changed (2026-06-25) — kids-story LLM QA hardening (13 fixes)
Deep QA of the kids LLM chain (story → image+motion prompts → storyboard stills) across 7 render rounds (robots/animals/humans/two-same-species). Harness: `scratchpad/kids_qa.py` (gen story → run real `prompt_assembler` per shot → render N stills → `kids_qa_report.json`). 187 tests green. Files touched: `story_writer.py`, `script_generator.py`, `prompt_assembler.py`, `voice_casting.py`. **Bot restart needed.**
- **Story length** — stage-1 wrote 206w → 13 shots/108w (~43s). `story_writer`: ~75w window + over-length retry (>110w shorten) + under-length retry (<62w enrich). Now lands ~70-85w / 6-8 shots.
- **Story arc protected** — brevity pressure was cutting the turning point (hare's nap dropped, rabbit talked like the underdog). Now: the 3 load-bearing beats (spark / turning-point / payoff) are never trimmed, story must END on payoff, characters stay in-role. + quote-format guard (name/"said" OUTSIDE quotes).
- **Beat order** — structurer spammed `consequence` / used `choice` for a plain answer. Structurer rule: each turning-point beat once, **choice before consequence, consequence = last shot only**. Deterministic guard in `_validate_and_default` demotes mid-story consequence→observation, forces final shot→consequence.
- **shot_type variety / `insert`** — walls of `medium`, `insert` never chosen despite object handling. Deterministic nudge: if story has manipulation verbs and ZERO inserts, promote one medium→insert.
- **Character tokens (consistency)** — vague tokens drifted ("energetic robot with yellow lights" → rendered a yellow CHICK). Structurer rule 4a: tokens lead with concrete form+color, BAN personality/glow words, always name species/form; humans must state gender word + hair length/style. Deterministic token cleaners: strip the char's OWN name from its token, strip trailing action/expression gerunds ("…, wagging its tail"), strip article-led names ("the puppy"→"Puppy").
- **Gender (voice + visual)** — a boy was voiced FEMALE because story context ("Mom" in his dialogue) outranked his own "boy" appearance. `voice_casting.guess_gender`: SELF fields decisive, context only tiebreaker; persist guessed `gender` onto the char for `prompt_assembler._pronoun`. Token "boy + short hair" nudge fixes the visual girl-drift on flux.
- **Prompt hygiene** — stage-direction leak ("Hello, quietly but clearly." voiced by TTS) stripped on spoken lines via `_strip_trailing_stage_direction`. Breathing-beat motion no longer hallucinates outdoor leaves indoors (animates the actual setting).
- **Validated visually** — strong continuity (look/clothing/scale/species/gender) across all character types incl. two-same-species mice. Residual: beat *labels* still loose (order correct); full visual style/gender fidelity wants the planned SDXL+IP-Adapter (flux prompt-only wall, see [[kids-ipadapter-planned]]).

## 3b. What We Changed (2026-06-24) — kids styles on flux + horror cleanup + QA loop
- **Kids = 4 prompt-only styles** (`pixar` default, `cartoon_saloon`, `claymation`, `stickman`) in `05_Config/styles.json`, tagged `modes:["story"]`. `spectrum`/`photoreal` kept for music/horror. New `script_generator.get_style_ids_for_mode(mode)` filters the LLM style menu + all pickers. Old styles removed (storybook/cartoon/anime/watercolor/pixelart/stopmotion); all `storybook` fallbacks → `pixar`.
- **Active image backend `comfyui_zimage_turbo → comfyui_flux_lora`** (models.json) — kids + music now render on flux1-dev. The user's 4 style LoRAs + Horrorstyle are **SDXL = no-op on flux** (`lora key not loaded`) → styles are PROMPT-ONLY. For real style LoRAs need Base Model "Flux.1 D".
- **Horror: per-char LoRA DROPPED → prompt-token consistency** (training setup kept). Stills on flux + fixed `Horrorstyle` style LoRA (runtime `horror_style_lora`/`_weight`; `comfyui_flux_lora` gained `use_char_lora`/`extra_loras` kwargs). Length **5min→2min** (DEFAULT_MINUTES=2, MAX_BEATS=10). Voice **pitch 0.90→1.0** = raw Qwen3-TTS, never sped/slowed.
- **Dashboard "Server error" fixed** — stale `style=storybook` override crashed NiceGUI select; cleared override + selects fall back to `(auto)` on unknown value.
- **QA loop** (`qa_styles_run.py`, env `QA_ONLY=`): per style → story → storyboard → stills, judge story/prompt/image. R1: cartoon_saloon+stickman ✅; pixar consistency bug (shot1 = human girl not robot) + claymation too glossy. Fixed: `storyboard_generator._characters_description` injects locked token when char in shot narration/speaker (not only prompt); structurer rule #10 + schema now require NAMING the char in first_frame_prompt; claymation suffix forced to matte plasticine. R2: pixar shots 2&3 fixed, shot1 wide still drifts; claymation re-render was still running at handoff. 187 tests green. **Bot restart needed.**

## 3. What We Changed Today (2026-06-11) — dramatic storytelling pass
- **Insert-shot feature** — new `modules/shot_editor.py` `insert_shot(script_id, position, brief)`: LLM writes new shot from surrounding-shot context (fallback builds from brief if Ollama down), script renumbers, on-disk artifacts shift DESCENDING so rendered work stays attached (approved_prompts keys, `shot{N}_*.png` + storyboard.json manifest, `clip_{sid}_shot{N}[_vK].mp4` incl. revisions, seeds json). New shot gets UNAPPROVED prompts entry. Discord: `!add_shot <id> <before|after> <shot#> [brief]`. Dashboard: per-shot "➕ Add shot" button + before/after dialog. After insert: `!regen_shot` + `!regen_video_shot` that shot only, then `!assemble`. Tests: `tests/test_shot_editor.py` (3 tests).
- **Shot-type grammar** — required `shot_type` field (wide|medium|closeup|insert) in BOTH script-gen prompts (single-stage AND 2-stage structurer): insert REQUIRED when character handles object, wide REQUIRED for new locations, closeup for reactions; never two consecutive same types. `_validate_and_default` derives shot_type from beat when LLM omits. `prompt_assembler.SHOT_TYPE_FRAMING` injects concrete spatial framing into image prompts per shot_type.
- **Age-first appearance** — character `appearance` + `locked_visual_token` must START with explicit age (number for humans, life stage for animals); assembler never drops age words.
- **Simpler motion prompts** — motion system prompt rewritten: ONE main action + ONE camera move, 25-50 words, camera matched to shot_type (insert/closeup → static/push-in, wide → pan/drift).
- **Casting confirmation gate** — after script approval + health gate, 🎭 casting embed (name/type/appearance per char) with Continue/Edit cast/Cancel. Edit modal (≤5 chars) updates BOTH appearance and locked_visual_token in script JSON. View NOT crash-persistent — re-run `!generate_storyboard <id>` to re-post. Skipped when no structured characters.
- **NOT YET LIVE** — bot needs restart to load these edits. 75 tests green.

## Earlier (2026-06-10) — production hardening pass
- **Git repo FIXED.** Ignore file named `gitignore` (no dot) → git never read it; venv (3,496 files) + .pyc tracked, most live modules NEVER committed. Now: `.gitignore` proper (adds secrets.env, .nicegui/, *.zip), venv/pycache untracked, ALL live source committed + pushed to GitHub `origin` (Rex_Bot repo). 85 tracked files.
- **Atomic JSON writes** — new `modules/file_utils.py` (`atomic_write_json`: tmp + fsync + rename). All 16 state-write sites converted (stats, pending_state, scripts, approved prompts, runtime settings, registry, panel state, seeds, gen history, feedback, agent memory). Crash mid-write keeps old file.
- **Shared GPU job lock** — new `modules/job_lock.py`. Same-thread reentrant (Discord pipeline chains), cross-thread release (dashboard UI-thread acquire → worker release), on-disk marker `05_Config/job_lock.json` (blocks 2nd bot process; stale/dead-pid markers stolen; 4h staleness). Discord entrypoints via `_gpu_job(label)` decorator; `!upscale`/`!assemble` inline; dashboard via `_try_begin/_end` (keeps S.busy). NOTE: two Discord commands on same event loop can still interleave (same thread = reentrant) — status quo kept.
- **subprocess timeouts** — every ffmpeg/ffprobe `subprocess.run` now has timeout (60s probes, 300–900s encodes): assembly, clip_generator, music_generator, upscaler.
- **Dashboard login hardened** — `secrets.compare_digest` + 60s lockout after 5 failed attempts (public tunnel exposure).
- **Log rotation** — claw_bot.log rotates at 10 MB, 5 backups.
- **Crash auto-restart** — `00_Tools/launch_clawbot.ps1` runs bot in loop: exit 0 (`!shutdown`/`!restart_bot`) = stop; non-zero = relaunch after 10s, max 5 crashes/5 min.
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
- **Remote mobile access**: ngrok fixed static URL (`*.ngrok-free.dev`). `02_Agent/start_dashboard_ngrok.ps1` (reads NGROK_AUTHTOKEN/NGROK_DOMAIN/DASHBOARD_PASSWORD). Cloudflare quick-tunnel fallback `start_dashboard_tunnel.ps1` (random URL, no account). `start_rex.bat` one-click (bot+tunnel). Wired into `00_Tools/launch_clawbot.ps1` (boot shortcut now starts tunnel + kills stale ngrok; `stop_clawbot.ps1` kills ngrok). `control_panel.py` now load_dotenv()s secrets.env so `CLAW_DASHBOARD_URL` feeds Discord "Open Dashboard" link.

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
- **`dashboard_nicegui.py`**: control parity with Discord. Added Settings card (style/aspect/voice/music/sync/transition/steps/cfg + upscale & reference toggles + show-current/reset-all), Models card (image/video backend switch), Queue card (pending feedback list/load/drop), Tools card (upscale/re-assemble/suggest theme), per-shot 🔁 Regen buttons on storyboard + video cards. New actions: `regen_storyboard_shot`, `regen_video_shot`, `run_upscale_action`, `render_queue`. Settings write straight to runtime_settings.json (shared state, no Discord coupling).

---

## 4. In Progress / Half-Done
- **RESTARTED + VERIFIED 2026-07-13 16:39** — one bot (launcher → venv stub → interpreter; the stub means ONE bot shows as TWO pids — do not "kill the duplicate"), dashboard 7860, ComfyUI 8188, Ollama 11434. Facts mode frozen and live. Known cosmetic: `Could not pin panel: 403 Forbidden` — the bot lacks Manage Messages in #claw-bot (panel posts, won't pin).
- **USO wired (2026-07-02) + 2 TODOs for 2026-07-03.** USO single-image char-consistency backend built (`modules/image_backends/comfyui_uso.py`), registered in models.json, and wired into KIDS (`storyboard_generator._kids_backend()` prefers USO) + MUSIC (`musicvideo_pipeline` scene-0 anchor cascade). Toggle `rs.get/set_uso_mode_enabled()` (default True). `image_backend.get_named_backend(id)` added. models.json `image_backend.active` now = `comfyui_uso` (global default). 201 tests pass; real story→USO storyboard verified (cartoon_saloon, 2-char, ref×2 multi-char held). **Bot restart needed** to load .py edits. TODO tomorrow: (1) wire USO into photoreal HORROR mode (`horrorstory_pipeline.py`); (2) add narration-edit UI to web dashboard for ALL modes (kids has it; add music+horror in `dashboard_nicegui.py`; backend `script_generator.update_shot_narration`/`rewrite_narration` exists).
- **RESTARTED 2026-06-10 23:08** — hardening pass + watchdog LIVE. Verified: 1 bot instance, dashboard answers HTTP 307 (login gate active), status embed posted, sync poller up, no stale lock. Still unverified by hand: Discord busy-notice when dashboard renders, login lockout, mobile ☰.
- **Launcher restart-loop gotcha (fixed)** — first version of crash-restart loop in `00_Tools/launch_clawbot.ps1` caused launcher windows to ping-pong: new launcher kills old launcher's bot → old loop sees non-zero exit → respawns → 2-3 simultaneous bots. Loop now exits if another claw_bot.py already running. Launcher is OUTSIDE the repo — back up manually if edited.
- **Bot restart needed (2026-06-04)** — today's .py edits (UI tabs, sync_bridge, auth, narration edit, AI rewrite) on disk, NOT live until restart.
- **Mobile nav ☰** — verified by server build only, not live phone browser. Confirm on device after restart + hard-refresh.
- **Discord→Web settings widgets** — web reads runtime settings live on render, but select WIDGETS show stale value until page reload.
- **AI rewrite** — LLM output not exercised live (needs Ollama up).
- **Bot not yet restarted** — today's .py fixes on disk, NOT live. Must restart + regenerate clips for script `20260529_080533` (its on-disk clips still old cropped 24fps).
- **Recovery option unused** — `04_Outputs/clips/_temp` holds full 16fps raw Wan part videos; clips rebuildable without re-render (user chose clean regenerate instead).
- **`assembly.py` FPS=24** while gen now 16 — assembly re-encodes (resamples, harmless dup), but inconsistent. Consider 16.
- **Dashboard ↔ Discord parity** — MOSTLY DONE. Dashboard now has settings/models/queue/tools/per-shot regen. Still missing: queue RESUME-with-feedback (only load+drop), multi-script generation queue, repolish, stats counters (Discord-specific). Untested in live browser — verify after restart.
- **Multi-part shot UX** — clip splits long narration into parts; UI shows ONE clip/shot; no per-part regen.

---

## 5. Known Issues / TODOs
**New (2026-06-04):**
- **secrets.env must be BOM-free** — PS `Out-File -Encoding utf8` adds BOM → dotenv drops first line. Edit with plain editor / append only.
- **ngrok free = 1 tunnel/agent**, fixed `.ngrok-free.dev` URL. Second run collides (launcher kills stale ngrok first). Custom-hostname errors (ERR_NGROK_314) = wrong value pasted (use `*.ngrok-free.dev` hostname, not `rd_/ep_` IDs).
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
- **Narration source of truth = script JSON** (`shots[].narration`), NOT prompts JSON. Edit/AI-rewrite reads disk first (no clobber) + writes back → both front-ends sync. Consumed at render time.
- **Public dashboard ⇒ password gate MANDATORY.** Bind stays `127.0.0.1`; tunnel runs on same PC and reaches it. NEVER bind 0.0.0.0 to expose. Gate = Starlette middleware, off when `DASHBOARD_PASSWORD` unset (localhost dev).
- **ngrok free static domain over Cloudflare named tunnel.** CF named tunnel needs owned domain; ngrok free gives 1 fixed `.ngrok-free.dev`. CF quick tunnel kept as no-account random-URL fallback.
- **Dashboard layout = left-nav vertical tabs, not long scroll.** Cards built then `.move()`'d into panels (avoids re-indenting whole UI). Mobile MUST keep ☰ toggle (drawer auto-hides).
- **Narration = duration source of truth.** Video freeze-pads if short; audio never truncated. NO `-shortest` anywhere in muxes — crops last word.
- **fps must match across gen→upscale→assemble.** Wan = 16fps. Upscale output fps PROBED from source + injected (not hardcoded). Relabeling frames to another fps shrinks duration (16/24=0.667x) — bug that cropped all narration.
- **4-step lightx2v LoRA OFF.** Distill = quality ceiling + artifacts. 5080 has VRAM for full 20-step. cfg 3.5 full path; beat overrides 2.0-2.8 valid.
- **Transition modes:** crossfade (0.3s dissolve, silent-tail padded so narration never overlaps) vs cut (instant + 0.4s breath so last word finishes). Default crossfade. Switchable: control panel 🎞️ / `!set_transition`.
- **Dashboard refresh guarded by content signature** — never blind `.clear()` on 1.5s timer (caused media flicker + textarea wipe).
- **Approve prompts BEFORE render** (edit-then-render). Single Approve-All. Image + motion approved separately per shot. Modal popups for edits. Batch posting + master button.
- **Dropped shot_tailor + prompt_polisher** from auto pipeline (prompt_assembler does it once, beat-aware).
- **Removed QA + continuity VLM passes** — Qwen2.5-VL hallucinated, net negative. Model + code deleted.
- **NiceGUI over Gradio** (Material Design, Quasar). Both Discord + Dashboard run, shared on-disk JSON. Localhost-only dashboard.
- **Clip ceiling is PER-MODEL** via `max_clip_seconds` in models.json (Wan=5, LTX=20). `n_parts=ceil(narration/max)`. Wan splits + last-frame chains; LTX single pass (chaining auto-skipped when n_parts=1). One MP4/shot. fps also per-model (`default_fps`).
- **Switching video/image model RESETS cfg+steps overrides** (`rs.reset_overrides_for_model_switch`) → each model runs at its own models.json recommended settings. Wired in `cmd_switch_model` + dashboard model selects.
- **LTX adapter workflow file configurable** (`workflow_file` in models.json; default `ltx2_video_only_v1_api.json`). LTX-2.3 not yet runnable: only UI-format workflow exists (`LTX-2.3_-_I2V_T2V_Basic_GGUF.json`, 111 nodes) — needs API-format export + models.json entry.
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
- **fps relabel ≠ resample** — re-containering N frames at different fps changes DURATION (16→24 = 0.667x shorter), and -shortest re-mux then crops audio. Always match output fps to source.
- **`-shortest` crops narration** — with `-c:v copy` a frame-timing quirk stops on short video stream. Use `-t audio_duration` or map full audio.
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