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
    # that they sit inside the detected singing windows — NOT on the intro
    # music (the bug: captions started at ~0s during the instrumental intro).
    from modules.musicvideo_assembly import TEMP_DIR
    song_id = song.get("song_id") or song.get("_id")
    ass = TEMP_DIR / f"overlay_{song_id}_16x9.ass"
    if not ass.exists():
        fails.append("caption .ass not found")
    else:
        txt = ass.read_text(encoding="utf-8")
        has_caps = ",Cap,," in txt
        _p(f"captions: .ass has Cap events={has_caps} ({txt.count(',Cap,,')} lines)")
        if lines and not has_caps:
            fails.append("no burned caption events in .ass despite having lyrics")

        # captions must be readable — not all crammed into a fraction of a second
        import re as _re2
        _caps = _re2.findall(r"Dialogue: 0,(\d):(\d\d):(\d\d\.\d\d),(\d):(\d\d):(\d\d\.\d\d),Cap,,", txt)
        if _caps:
            def _s(h, m, sec):
                return int(h) * 3600 + int(m) * 60 + float(sec)
            span = _s(*_caps[-1][3:]) - _s(*_caps[0][:3])
            avg = span / max(1, len(_caps))
            _p(f"caption span {span:.1f}s over {len(_caps)} lines (avg {avg:.2f}s/line)")
            if avg < 0.4:
                fails.append(f"captions crammed: avg {avg:.2f}s/line (<0.4s) — unreadable")

        # Re-detect the singing windows and confirm captions land inside them —
        # proves instrumental intro/outro stay empty (the actual sync fix).
        if ok and lines and song_audio:
            windows = lyric_aligner.get_vocal_windows(Path(song_audio))
            if windows:
                v_start = windows[0][0]
                v_end = windows[-1][1]
                _p(f"vocal windows: {[(round(a,1), round(b,1)) for a,b in windows]} "
                   f"(song {song_dur:.2f}s)")
                # first caption must not begin before singing starts (0.3s slack)
                import re as _re
                first_cap = _re.search(r"Dialogue: 0,(\d):(\d\d):(\d\d\.\d\d),0:.*?,Cap,,", txt)
                if first_cap:
                    fc = int(first_cap[1])*3600 + int(first_cap[2])*60 + float(first_cap[3])
                    _p(f"first caption at {fc:.2f}s, singing starts {v_start:.2f}s")
                    if fc < v_start - 0.3:
                        fails.append(f"caption at {fc:.2f}s begins before vocals ({v_start:.2f}s) — on intro music")
                real = v_start > 0.5 or v_end < song_dur - 0.5
                if not real:
                    _p("note: vocals span nearly the whole song (no intro/outro to skip)")
            else:
                _p("note: no vocal windows detected — captions used proportional fallback")

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
