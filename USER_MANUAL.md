# Claw Bot — User Manual

> Plain-English guide for Jeffy. Think of Claw Bot as a tiny in-house animation studio:
> you give it a theme, it writes the script, draws the shots, voices the narration,
> films each shot, and edits the final video — pausing for your approval at every stage,
> like dailies in a VFX pipeline.

---

## 1. What Claw Bot Does

A local AI animation studio that runs entirely on your PC (RTX 5080). Give it a theme →
it makes a short vertical video (YouTube Shorts style, 9x16 + 16x9 + 1x1). Three modes:

| Mode | What it makes | Voice / look |
|------|---------------|--------------|
| **Kids story** | ~30s illustrated kids fable (animals/kids learn a lesson) | warm narrator + character voices; cartoon styles (pixar / cartoon_saloon / claymation / stickman) |
| **Music video** | a song + Ken-Burns photo montage | AI song (ACE-Step) + still images that pan/zoom |
| **Horror story** | ~2min true-horror short | deep "eric" narrator voice; dark photoreal look |

It's driven two ways at once (they share the same files, so you can use either):
- **Discord bot** — type `!commands` in the `#claw-bot` channel.
- **Dashboard** — a web page at **http://127.0.0.1:7860** on this PC (buttons, no typing).

---

## 2. Starting & Stopping

**Start:** double-click the launcher shortcut (runs `00_Tools\launch_clawbot.ps1`). It boots
Ollama (the writer), ComfyUI (the renderer), and the bot. Wait ~1 minute. The dashboard is
ready when http://127.0.0.1:7860 loads.

**Stop:** type `!shutdown` in Discord, or close the launcher window.

**Restart after a code change:** `!restart_bot` (Python doesn't hot-reload — edits only take
effect after a restart).

> ⚠️ Only run ONE launcher window. Starting a second one makes duplicate bots fight each
> other (dashboard flickers). If that happens, close all launcher windows and start one.

---

## 3. The Pipeline (kids story) — stage by stage

Every stage PAUSES for your approval. Nothing renders until you say go.

1. **Theme → Script.** `!generate_script <theme>` (alias `!gs`). The bot writes a ~30s story
   (70-85 words, 7-9 shots), picks a style, casts the characters. → You approve / edit / regenerate.
2. **Casting confirm.** It shows the cast (name, type, look). → Continue / Edit cast / Cancel.
3. **Prompts.** It writes an image prompt + a motion prompt per shot. → Edit any, then **Approve All**.
4. **Storyboard.** `!generate_storyboard <id>` — draws one still per shot (+ a character
   reference sheet so everyone stays on-model). → Approve / regenerate single shots.
5. **Video.** `!generate_video <id>` — voices the narration, films each shot (Wan), per shot.
   → Approve / regenerate.
6. **Final.** `!assemble <id>` — stitches all shots + narration + music into the 3 final MP4s.

Files land in `04_Outputs\` (scripts / storyboards / clips / final).

**Other modes:** `!music_video <theme>` / `!make_song`, and `!horror_story <theme>` /
`!make_horror`. Same approve-as-you-go idea.

---

## 4. Handy Commands (type in #claw-bot)

| Command | Does |
|---------|------|
| `!commands` (`!h`) | full command list |
| `!generate_script <theme>` (`!gs`) | start a kids story |
| `!today_script` | story from today's theme |
| `!suggest_theme` | get a theme idea |
| `!list_scripts` (`!ls`) / `!show_script <id>` | browse / view scripts |
| `!generate_storyboard <id>` | draw the stills |
| `!regen_shot <id> <shot#>` | redraw one still |
| `!generate_video <id>` | film the shots |
| `!regen_video_shot <id> <shot#>` | refilm one shot |
| `!assemble <id>` | build the final MP4s |
| `!add_shot <id> <before\|after> <shot#> [brief]` | insert a new shot |
| `!rewrite_narration <id> <shot#> [hint]` | AI-rewrite one narration line |
| `!list_styles` / `!set_style <style>` | see / set the look |
| `!list_voices` / `!set_voice <voice>` | see / set the narrator voice |
| `!set_transition <crossfade\|cut>` | shot-to-shot transition |
| `!upscale <id>` | 4x upscale the clips |
| `!stats` / `!status` | counters / health |

(The **dashboard** has buttons for all of these — no typing needed.)

---

## 5. What the Bot Is GOOD At (kids story characters)

Character consistency uses a "cast sheet" reference (IP-Adapter) so a character looks the
same in every shot. Tested and reliable:

- ✅ **One character**, any kind (animal, human, or robot/creature)
- ✅ **Several of the SAME species** (e.g. two or three mice) — stay distinct
- ✅ **Two humans** (boy + girl) — faces/hair/clothes stay separate
- ✅ **Human + a pet animal** (girl + dog)
- ✅ **Two DIFFERENT animal species** (the classic rabbit vs. tortoise) — fixed via separate
  per-character reference sheets

Styles: **pixar** (best for animals), **cartoon_saloon**, **claymation**, **stickman**.

---

## 6. Limitations (know these before you rely on it)

**Story / writing**
- Kids stories target ~30s only. Longer = multiple stories.
- Beat *labels* (hook/spark/etc.) are loosely assigned — the story ORDER is correct, the tags are rough.
- Dialogue attributions ("Pip said") are spoken by the narrator/voice as written (audio-first design).

**Look / characters**
- **Style suitability:** `cartoon_saloon` and `stickman` LoRAs draw human-shaped figures, so the
  bot now auto-switches an **animal cast** off those to **pixar**. Use them mainly for human casts.
- **Same-species twins** can look very similar if their descriptions don't differ enough — give
  each a distinct color/clothing in the cast.
- Wide establishing shots can render characters small or as scenery; closeups/mediums are tightest.
- The IP-Adapter "lock strength" is fixed (0.75). If a character drifts, it's tunable in config
  (`ipadapter_weight`) — ask before changing.

**Video**
- Wan films in ~5s chunks; long narration is split into chained parts (one clip per shot). Very
  long lines can show a brief frozen tail.
- 16 fps throughout. Narration is the source of truth — video never crops the last word.

**System**
- **Local only.** Dashboard is `127.0.0.1` (this PC). No phone/remote access (tunnel removed).
- One render at a time (GPU lock). Music gen (audiocraft) may be flaky — verify.
- Restart the bot after any code edit. Run only ONE launcher.
- Style LoRAs are **SDXL** for kids — they do nothing on the flux backends (kids is pinned to the
  SDXL+IP-Adapter backend; music/horror stay on flux).

**Not built yet**
- YouTube auto-upload (upload manually).
- Per-part (sub-shot) regeneration — regen is whole-shot.

---

## 7. Quick Troubleshooting

| Symptom | Fix |
|---------|-----|
| Dashboard won't load | wait ~1 min after launch; ComfyUI is slow to boot. Check the launcher window. |
| Characters render as the wrong species / humans | bot must be running CURRENT code — `!restart_bot`. Animal stories: use **pixar**. |
| Dashboard flickers / two bots | close ALL launcher windows, start ONE. |
| Last word of narration cut | shouldn't happen by design; if it does, re-assemble (`!assemble`). |
| Render stuck | check ComfyUI window; `!status` for health; worst case `!restart_bot`. |
| Out of VRAM | bot frees models between stages automatically; if stuck, restart. |

---

## 8. Golden Rules (don't break these)

1. Everything stays inside `E:\Rexjaw_VFX`.
2. Approve each stage — script, prompts, storyboard, video, final.
3. One launcher, one bot.
4. Restart the bot after editing any `.py`.
5. Narration is the source of truth — never trim it.

---

*Generated 2026-06-26. For technical/dev detail see `02_Agent/CLAUDE.md`.*
