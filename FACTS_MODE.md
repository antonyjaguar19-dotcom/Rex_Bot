# Facts Shorts — FROZEN (2026-07-13)

The facts reel is **done**. This file is the spec it was frozen at: the settings,
the invariants, and the reasons. Treat every number here as load-bearing — each one
replaced something that shipped broken.

To unfreeze, say so explicitly. Otherwise: **do not tune these.**

---

## What it makes

A 9x16 vertical reel for Shorts / Instagram: the Rexjaw mascot presents 5 true facts
about a topic, in its own cloned voice, over Wan-animated shots, with read-along
captions, a music bed under the narration, and the thumbnail held on the front.

Under 40 seconds, every time.

Start it from Discord (`!facts <topic>`) or the dashboard **Facts** tab. Either
front-end shows the other's progress.

---

## The frozen config

Live values in `05_Config/runtime_settings.json`; mirrored in `config_snapshot/`
(that folder is the rebuild reference — `05_Config` is outside git).

**The dashboard's Facts card has exactly ONE control: which mascot presents.** Everything
below is frozen, and three of them are no longer settings at all — `get_facts_video_mode()`,
`get_facts_mascot_mode()` and `get_mascot_tts_engine()` return constants and ignore the
file. A toggle that is gone from the UI but still read from disk is a silent downgrade
waiting to happen: a stale `kenburns` would have shipped a slideshow, a stale
`facts_mascot_mode: false` would have shipped backdrops, and nothing on screen would
have said why.

| Setting | Value | Why this value |
|---|---|---|
| `facts_mascot_mode` | **constant `true`** | The mascot presents. Backdrops are the OOM/rescue path, not a product. Not read from disk. |
| `active_mascot` | `default` | **Which** mascot presents (a folder under `02_Agent/assets/mascots/`). The one remaining choice. See "The mascot shelf" below. |
| `mascot_tts_engine` | **constant `chatterbox`** | Every mascot speaks in its OWN cloned voice (the clip in its folder). Picking a preset meant picking the wrong voice for the character on screen. Qwen3-TTS → Kokoro survive as the fallback cascade when there is no clip to clone. Not read from disk. |
| `mascot_voice_ref` | `02_Agent/assets/mascot_voice.wav` | The SHARED clone clip, used by any mascot that carries none of its own. **Gitignored — a real recording of the owner's voice. Never commit it.** |
| `mascot_voice_exaggeration` | `0.35` | Calm and even. Higher gets theatrical; every earlier voice failed by being too excited. |
| `mascot_voice_speed` | `1.20` | Pitch-preserving (`atempo`). The clone reads slow on its own. **Never pitch-shift a cloned voice** — it stops being the voice. |
| `facts_mascot_lipsync` | `false` | Wan S2V moves the mouth but distorts hands and props. Voice-over wins. |
| `facts_video_mode` | **constant `wan`** | Wan 2.2 14B I2V per shot — a facts reel is ANIMATED. Ken Burns survives inside the pipeline as the OOM rescue (`render_facts(animate=False)` still forces it for a harness). Not read from disk. |
| video resolution | *(unset → 720p)* | Wan honours `rs.VIDEO_RES_PRESETS`. A stale `480p` override once silently dragged every reel down. |
| `facts_max_seconds` | `40.0` | Hard ceiling, enforced by **measuring** the voiced beats and trimming the pace to fit (max 1.45x). Not a hope. |
| `facts_thumbnail` | `true` | Qwen draws the headline INTO the art (baked, not overlaid). |
| `facts_thumb_hold_sec` | `0.5` | The thumbnail is held as the reel's **first frame** — Shorts custom thumbnails are not offered in every region, and where they are not, the platform grabs frame one. |
| `facts_music_enabled` | `true` | |
| `facts_music_mood` | `cheerful` | |
| `DEFAULT_N_FACTS` | `5` | 6 facts + a hook + an outro would not fit under 40s without rushing. |

---

## Invariants — the things that must stay true

**Narration is the source of truth.** Video freeze-pads if short; audio is never
truncated. No `-shortest` in any mux.

**The music bed** (`facts_pipeline._music_bed`, `facts_assembly`):
- ACE-Step does **not** compose for as long as it is asked. Told 49s it wrote 29s of
  music, padded 10s of digital silence, then left 8s of junk. So the **ask is never
  the answer** — ask 2x (`_MUSIC_ASK_HEADROOM`), measure the composed body, ask again
  longer if it under-filled (`_MUSIC_TRIES = 2`).
- The bed is cut at its **music body** (`_music_body_end`: first gap >= 1.5s under
  -45 dB, after at least 8s of music) — not at its tail. Trimming only the tail left
  the hole in the middle, and fitting a long track from the FRONT (to keep the
  composed outro) then kept the dead air and threw the music away.
- **Never loop a short bed.** Replaying its opening to cover the reel is heard as the
  same eight bars twice. A short bed starts late; silence at the head beats a stutter.
- The bed sits at **-32 LUFS** (~13 dB under the voice), set by `loudnorm` — never by
  a fixed `volume=` factor, because the source level varies. `amix` runs with
  `normalize=0`, or it halves the **voice** too.
- After every mux, `audit_music_bed()` subtracts the narration back out of the
  **finished file** and checks what remains. It reports `bed verified … no dropouts`
  or shouts `THE MUSIC DROPS OUT`. Every music bug so far looked correct at every
  intermediate step and was wrong in the file.

**Failures are loud.** `facts_pipeline.preflight()` proves the renderer answers
*before* the story is written. Mascot mode ON + anything broken **raises** — it never
quietly falls back to backdrops and hands you a different film.

**The poster frame is replaceable, not stackable.** `prepend_still` keeps a pristine
copy in `04_Outputs/final/_posterless/` and rebuilds the front from it, so a re-rolled
thumbnail replaces the old poster instead of bolting a second one on.

**Kokoro is not offered** for the mascot anywhere in the UI. It survives inside the
pipeline as a silent fallback only.

---

## The reel library + the fact memory (2026-07-13)

**Past reels** (`modules/facts_library.py`) — the Facts tab lists every finished reel,
newest first; open one for its video, the facts it told, and its upload kit. Delete
removes the WHOLE footprint, which is why this lives in one module: a reel is scattered
over `04_Outputs/facts/facts_{id}.json`, `storyboards/facts_{id}/`, `final/facts_{id}_*`
(video, thumbnails, title, description, publish.json, mascot art, Discord preview) and
`final/_posterless/facts_{id}_*`. Deleting by hand meant remembering all five, which is
why `final/` had 212 facts files in it. The id is validated against `^\d{8}_\d{6}$`
before it is interpolated into a delete glob — `*` would take the lot.
Discord: `!facts_list` · `!facts_delete <id> [forget]`.

**Fact memory** (`modules/facts_memory.py`, `04_Outputs/facts/_memory.json`) — every fact
that ships is written down, keyed by topic, and the next reel about that topic is asked
for facts that are NOT on the list. Ask Qwen about octopuses three times and you get the
three famous ones three times: the second reel is a re-upload with new pictures.
- **The prompt is not the feature; the CHECK is.** A model told "don't repeat these"
  rewords the fact instead, and a reworded fact is the same fact. `is_repeat()` compares
  content-word sets (order-free, plurals folded, **numbers kept** — "three hearts" and
  "nine brains" differ only by the number). Measured: reworded repeat 1.00 / 0.75 / 0.71,
  genuinely new 0.25 / 0.20 → threshold **0.55**.
- Repeats are dropped from the reel; if that leaves it short, the roll is re-rolled and
  the model is **told what it repeated** ("those are burned") — "try again" alone just
  returns the same famous facts. An exhausted topic fails LOUDLY rather than shipping
  four repeats.
- A **thin** roll (too few facts, none repeated) is a different failure and must not be
  blamed on the memory — it says "usable facts", not "already used (0 on record)".
- `backfill_from_reels()` runs once and learns from the reels already on disk (31 of them
  when this shipped), or the first repeat topic would repeat itself.
- **Deleting a reel does NOT forget its facts.** A bad reel is a reason to want a
  *different* one next time. Release them explicitly: the delete dialog's switch,
  `!facts_delete <id> forget`, or `!facts_forget <topic>`.
- Placeholder reels are never remembered — "here is an interesting thing about bees
  number 1" is not a fact, and it would poison the topic forever.
- Discord: `!facts_memory [topic]` · `!facts_forget <topic>`.

---

## The mascot shelf (2026-07-13)

There can be more than one mascot, and the reel stars whichever is **active**.

```
02_Agent/assets/mascots/<id>/
    mascot.png              primary reference (required)
    mascot_front.png        optional angles — same filenames as the old flat layout
    mascot_threequarter.png
    mascot_side.png         (no BACK view: no face, no chest logo — it was never
                             what identity transfer keyed on, so it is not asked for)
    voice.wav               optional: THIS mascot's cloned voice (gitignored)
    meta.json               {"name": "Jaguar Cub"}
```

**Adding one** (dashboard Mascots tab → *New mascot*, or `!mascot new <name>` with the
files attached): a name, **three views** (front / three-quarter / side) and a
**~10s voice clip**. Only the FRONT view is required — it is the reference the renderer
conditions on, and it is installed as the primary too (`mascot_refs()` still hands over
ONE reference by default; three of them made Qwen copy a reference's stance instead of
acting out the scene). The clip is **cloned, not trained** — Chatterbox reads it and
speaks in its timbre, so a noisy clip is a noisy mascot. 5–30s is accepted, ~10s is the
mark; outside that you get a warning, never a refusal. Files are staged and only become
a mascot on **Create**, so a failed intake leaves nothing behind.

- `modules/mascot_library.py` owns the shelf; `rs.active_mascot` holds the id.
  `mascot.mascot_path()` / `mascot_refs()` resolve through it, so switching mascot
  switches every presenter still, every thumbnail and (when it carries a `voice.wav`)
  the cloned narration — not just a label.
- **Dashboard → Mascots tab**: add (name + image), delete, set active, upload extra
  angles or a voice clip. **Facts tab → Mascot dropdown** picks who presents.
  **Discord**: `!mascot list` · `!mascot use <id>` · `!mascot new <name>` (attach an
  image) · `!mascot rm <id>`.
- A mascot with no `voice.wav` of its own falls back to the shared
  `mascot_voice_ref` clip. Give a new character its own clip or it speaks in the
  cub's voice.
- **Back-compat:** the old flat `assets/mascot.png` still works while no shelf exists.
  `mascot_library.migrate()` copies it in as `default` on first use (the originals are
  left alone). Once `assets/mascots/` exists it is the only truth — deleting every
  mascot leaves you with none rather than resurrecting the flat file, and
  `preflight()` then raises instead of quietly rendering backdrops.

---

## How to change it safely

1. **Measure the artifact, not the code.** Every bug in this pipeline was invisible in
   the filtergraph and obvious in the file: an inaudible bed, a ducked voice, a bed
   that died at 16s, a thumbnail that never reached the video, a hold that silently
   failed. Decode the output and look at it.
2. **Tests use real ffmpeg** (`tests/test_music_bed.py`, `tests/test_poster_frame.py`).
   These bugs live *inside* ffmpeg calls, where a mock proves nothing.
3. **Restart the bot** — Python has no hot-reload. A `.py` edit is not live until then.

Run the suite: `venv\Scripts\python -m pytest tests -q`
