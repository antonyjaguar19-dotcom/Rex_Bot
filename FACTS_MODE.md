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

| Setting | Value | Why this value |
|---|---|---|
| `facts_mascot_mode` | `true` | The mascot presents. Backdrops are the fallback path, not the product. |
| `active_mascot` | `default` | **Which** mascot presents (a folder under `02_Agent/assets/mascots/`). See "The mascot shelf" below. |
| `mascot_tts_engine` | `chatterbox` | Voice **clone**. No local TTS has a child voice; every preset was an adult timbre pitch-shifted, and every one sounded wrong. |
| `mascot_voice_ref` | `02_Agent/assets/mascot_voice.wav` | The SHARED clone clip, used by any mascot that carries none of its own. **Gitignored — a real recording of the owner's voice. Never commit it.** |
| `mascot_voice_exaggeration` | `0.35` | Calm and even. Higher gets theatrical; every earlier voice failed by being too excited. |
| `mascot_voice_speed` | `1.20` | Pitch-preserving (`atempo`). The clone reads slow on its own. **Never pitch-shift a cloned voice** — it stops being the voice. |
| `facts_mascot_lipsync` | `false` | Wan S2V moves the mouth but distorts hands and props. Voice-over wins. |
| `facts_video_mode` | `wan` | Wan 2.2 14B I2V per shot. Ken Burns stills are the fast fallback. |
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

## The mascot shelf (2026-07-13)

There can be more than one mascot, and the reel stars whichever is **active**.

```
02_Agent/assets/mascots/<id>/
    mascot.png              primary reference (required)
    mascot_front.png        optional angles — same filenames as the old flat layout
    mascot_threequarter.png
    mascot_side.png
    mascot_back.png
    voice.wav               optional: THIS mascot's cloned voice (gitignored)
    meta.json               {"name": "Jaguar Cub"}
```

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
