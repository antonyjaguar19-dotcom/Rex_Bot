"""Render the SAME song in every vocal type (female/male/duet/choir/rap/
instrumental) and post each audio clip to Discord #videos for comparison.
Skips stills/assembly — voice is the only thing under test.

Run: venv\Scripts\python run_voice_music_samples.py ["theme"] [duration_sec]
"""
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT.parent / "05_Config" / "secrets.env")

from modules import song_generator as sg
from modules import audio_backend
from modules.musicvideo_pipeline import _vocal_tags, _effective_lyrics

TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "").strip()
API = "https://discord.com/api/v10"
H = {"Authorization": f"Bot {TOKEN}"}

THEME = sys.argv[1] if len(sys.argv) > 1 else "a rainy night drive"
DUR = int(sys.argv[2]) if len(sys.argv) > 2 else 20
VOCALS = ["female", "male", "duet", "instrumental"]


def chan(names):
    for gu in requests.get(f"{API}/users/@me/guilds", headers=H, timeout=15).json():
        for c in requests.get(f"{API}/guilds/{gu['id']}/channels", headers=H, timeout=15).json():
            if c.get("type") == 0 and c.get("name", "").lower() in names:
                return c["id"]
    return None


def main():
    cid = chan(["videos", "status"])
    print("writing one song (Ollama)...", flush=True)
    song = sg.generate_song(THEME, duration_sec=DUR)
    lyrics = song.get("lyrics", "")
    title = song.get("title", "Untitled")
    print(f"song '{title}', style={song.get('song_style')}", flush=True)
    if cid:
        requests.post(f"{API}/channels/{cid}/messages", headers=H,
                      json={"content": f"🎤 **Music voice-type samples** — same song "
                                       f"*{title}* ({song.get('song_style')}, {DUR}s), "
                                       f"{len(VOCALS)} vocal types. Reply with your pick."},
                      timeout=30)

    ab = audio_backend.get_active_backend()
    for vt in VOCALS:
        s = dict(song); s["vocal_type"] = vt
        tags = _vocal_tags(s)
        print(f"rendering {vt} ({tags})...", flush=True)
        t0 = time.time()
        try:
            audio = ab.generate(
                tags=tags, lyrics=_effective_lyrics(s), duration_sec=float(DUR),
                bpm=song.get("bpm"), keyscale=song.get("keyscale"),
                language=song.get("language", "en"),
                output_filename=f"voicetest_{vt}.mp3")
        except Exception as e:
            print(f"  {vt} FAILED: {e}", flush=True)
            if cid:
                requests.post(f"{API}/channels/{cid}/messages", headers=H,
                              json={"content": f"❌ **{vt}** failed: `{e}`"}, timeout=30)
            continue
        print(f"  {vt} done in {time.time()-t0:.0f}s", flush=True)
        if cid and Path(audio).exists():
            with open(audio, "rb") as f:
                requests.post(f"{API}/channels/{cid}/messages", headers=H,
                              data={"content": f"🎤 **{vt}** vocal — _{tags}_"},
                              files={"files[0]": (Path(audio).name, f, "audio/mpeg")},
                              timeout=180)
    if cid:
        requests.post(f"{API}/channels/{cid}/messages", headers=H,
                      json={"content": "✅ All voice samples posted."}, timeout=30)
    print("done", flush=True)


if __name__ == "__main__":
    main()
