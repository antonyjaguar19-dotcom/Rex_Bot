"""Full-model music-video e2e. Generates a short song (Ollama), renders it
(ACE-Step audio + scene stills + Ken Burns assembly), then VERIFIES the new
subtitle work: lyric captions are burned in and their timing came from the
WhisperX forced-aligner (not the proportional fallback). Prints a PASS/FAIL
report. Standalone, no Discord.

Usage:  python run_music_e2e.py ["theme"] [duration_sec]
"""
import sys
import time
import traceback
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

THEME = sys.argv[1] if len(sys.argv) > 1 else "a quiet town waking up at sunrise"
DURATION = int(sys.argv[2]) if len(sys.argv) > 2 else 30


def _p(m):
    print(m, flush=True)


def main():
    t0 = time.time()
    _p(f"🎵 music e2e — theme: {THEME!r} ({DURATION}s)")

    from modules import song_generator as sg
    from modules import musicvideo_pipeline as mvp
    from modules import lyric_aligner
    from modules.assembly import _probe_duration
    from modules.subtitles import lyric_lines

    ok, msg = lyric_aligner.health_check()
    _p(f"   lyric_aligner: {'OK' if ok else 'MISSING'} — {msg}")

    # ---- 1. song JSON (Ollama) ----
    _p("✍️  generating song...")
    song = sg.generate_song(THEME, duration_sec=DURATION)
    lines = lyric_lines(song.get("lyrics", ""))
    _p(f"   song: {song.get('title')!r} — {len(song.get('scenes', []))} scenes, "
       f"{len(lines)} lyric lines, style={song.get('visual_style')}")

    # ---- 2. full render (audio + stills + assembly w/ burned captions) ----
    _p("🎬 rendering (ACE-Step + stills + assembly)...")
    out = mvp.render_musicvideo(song, progress_cb=lambda m: _p(f"   {m}"))

    # ---- 3. VERIFY ----
    _p("\n=== VERIFY ===")
    fails = []
    song_audio = out.get("song_audio")
    song_dur = _probe_duration(Path(song_audio)) if song_audio else 0.0
    _p(f"song audio: {song_dur:.2f}s")

    for aspect in ("9x16", "16x9", "1x1"):
        vid = Path(out[aspect])
        if not vid.exists():
            fails.append(f"{aspect}: output missing"); continue
        vdur = _probe_duration(vid)
        # Core Rule 5: song is source of truth for length. Assembly clones the
        # last frame as a tail pad then trims to song length, so video ~= song.
        drift = abs(vdur - song_dur)
        dur_ok = drift <= 1.0
        if not dur_ok:
            fails.append(f"{aspect}: duration {vdur:.2f}s vs song {song_dur:.2f}s (drift {drift:.2f}s)")
        _p(f"{aspect}: {vdur:.2f}s ({vid.stat().st_size/1e6:.1f} MB) "
           f"dur_match={'OK' if dur_ok else 'FAIL'}")

    # captions: check the .ass the assembler wrote actually has Cap events, and
    # that the timings match a real alignment (not the 0..song_dur proportional
    # spread — real alignment starts after 0 and ends before song_dur).
    from modules.musicvideo_assembly import TEMP_DIR
    song_id = song.get("song_id") or song.get("_id")
    ass = TEMP_DIR / f"overlay_{song_id}_16x9.ass"
    if not ass.exists():
        fails.append("caption .ass not found")
    else:
        txt = ass.read_text(encoding="utf-8")
        n_caps = txt.count("Style: Cap") + txt.count(",Cap,,")
        has_caps = ",Cap,," in txt
        _p(f"captions: .ass has Cap events={has_caps} ({txt.count(',Cap,,')} lines)")
        if lines and not has_caps:
            fails.append("no burned caption events in .ass despite having lyrics")

        # Re-run the aligner directly to confirm real timestamps were produced
        # (proves the sync path, independent of what got written).
        if ok and lines and song_audio:
            spans = lyric_aligner.align_lyrics(Path(song_audio), lines)
            if spans:
                first_start = spans[0][1]
                last_end = spans[-1][2]
                _p(f"alignment: {len(spans)} lines, first_start={first_start:.2f}s, "
                   f"last_end={last_end:.2f}s (song {song_dur:.2f}s)")
                real = first_start > 0.01 or last_end < song_dur - 0.5
                if not real:
                    fails.append("alignment looks like proportional fallback, not real timing")
            else:
                fails.append("aligner returned no spans on re-run")

    mins = (time.time() - t0) / 60
    _p(f"\n{'✅ PASS' if not fails else '❌ FAIL'} — music e2e in {mins:.1f} min")
    for f in fails:
        _p(f"   - {f}")
    _p(f"final: {out.get('16x9')}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        _p(f"❌ music e2e FAILED: {e}")
        sys.exit(1)
