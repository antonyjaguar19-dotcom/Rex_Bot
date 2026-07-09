"""Full-model horror e2e. Writes a 30-min horror story, renders it, and posts
milestone updates to the Discord #status channel via REST (bot token, no gateway
— does not clash with the running bot). Runs standalone/detached.
"""
import sys, time, traceback
from pathlib import Path

import requests
from dotenv import load_dotenv
import os

# Windows console is cp1252 — emojis in print() crash it. Force utf-8 (Discord
# REST posts are utf-8 JSON regardless).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT.parent / "05_Config" / "secrets.env")

TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "").strip()
API = "https://discord.com/api/v10"
HDRS = {"Authorization": f"Bot {TOKEN}", "Content-Type": "application/json"}
THEME = sys.argv[1] if len(sys.argv) > 1 else "an abandoned lighthouse that calls people into the sea"
MINUTES = int(sys.argv[2]) if len(sys.argv) > 2 else 30
REUSE_ID = sys.argv[3] if len(sys.argv) > 3 else None  # reuse an existing horror_id (skip writing)

_cid = None
_last = {"msg": None, "t": 0.0}


def _status_channel_id():
    g = requests.get(f"{API}/users/@me/guilds", headers=HDRS, timeout=15).json()
    for guild in g:
        chs = requests.get(f"{API}/guilds/{guild['id']}/channels", headers=HDRS, timeout=15).json()
        for c in chs:
            if c.get("name", "").lower() == "status" and c.get("type") == 0:
                return c["id"]
    return None


def post(msg: str, force: bool = False):
    """Post to #status, throttled (skip dup within 8s) to avoid rate limits."""
    global _cid
    now = time.time()
    if not force and msg == _last["msg"] and now - _last["t"] < 8:
        return
    _last.update(msg=msg, t=now)
    print(msg, flush=True)
    if not _cid:
        return
    try:
        requests.post(f"{API}/channels/{_cid}/messages", headers=HDRS,
                      json={"content": msg[:1900]}, timeout=15)
    except Exception as e:
        print(f"(discord post failed: {e})", flush=True)


def main():
    global _cid
    _cid = _status_channel_id()
    post(f"🎃 **Horror e2e started** — theme: *{THEME}* (~{MINUTES} min target). "
         f"Posting milestones here.", force=True)
    t0 = time.time()

    from modules import horror_writer as hw
    from modules import horrorstory_pipeline as hsp

    # ---- write (chunked) — or reuse an existing story ----
    if REUSE_ID:
        story = hw.load_horror(REUSE_ID)
        if not story:
            post(f"❌ horror_id {REUSE_ID} not found", force=True); return
        post(f"♻️ Reusing story *{story.get('title')}* ({REUSE_ID})", force=True)
    else:
        def write_cb(m):
            if any(k in m for k in ("chapter", "outline", "story written", "saved")):
                post(f"✍️ {m}")
        story = hw.generate_horror_story(THEME, MINUTES, progress_cb=write_cb)
    nbeats = len(story["beats"])
    words = sum(len(b["narration"].split()) for b in story["beats"])
    post(f"📖 **Story written:** *{story['title']}* — {nbeats} beats, ~{words} words "
         f"(~{words//150} min). Voicing the full narration first...", force=True)

    # ---- narration FIRST: render full story audio + upload it (as mp3) ----
    narration = hsp.narrate_story(story, progress_cb=lambda m: post(f"⏳ {m}"))
    try:
        from modules.assembly import _probe_duration, FFMPEG_EXE
        ndur = _probe_duration(narration)
    except Exception:
        ndur = 0; FFMPEG_EXE = None
    # compress wav -> mp3 so it fits Discord's 25MB limit
    import subprocess
    mp3 = Path(narration).with_suffix(".mp3")
    try:
        subprocess.run([str(FFMPEG_EXE), "-y", "-loglevel", "error", "-i", str(narration),
                        "-b:a", "128k", str(mp3)], timeout=300, check=True)
    except Exception as e:
        print("mp3 convert failed:", e); mp3 = None
    upload = mp3 if (mp3 and mp3.exists()) else Path(narration)
    usize = upload.stat().st_size / 1e6 if upload.exists() else 999
    if _cid and usize < 24:
        with open(upload, "rb") as f:
            requests.post(f"{API}/channels/{_cid}/messages", headers=HDRS,
                          data={"content": f"🎙️ **Full story narration** — *{story['title']}* "
                                           f"({ndur:.0f}s, Qwen3-TTS eric). Now rendering {nbeats} photoreal stills..."},
                          files={"files[0]": (upload.name, f, "audio/mpeg")}, timeout=300)
    else:
        post(f"🎙️ **Full narration ready** ({ndur:.0f}s, {usize:.0f}MB — too big; "
             f"in 04_Outputs/audio). Rendering {nbeats} stills...", force=True)

    # ---- render — reuse narration, post stage + every 10th still ----
    def render_cb(m):
        if "still" in m:
            # m like "🖼️ still 12/63..."
            try:
                n = int(m.split("still")[1].split("/")[0])
                if n == 1 or n % 10 == 0:
                    post(f"🖼️ {m}")
            except Exception:
                pass
        elif any(k in m for k in ("voicing", "narration ready", "assembling",
                                  "ken burns", "drone", "muxing", "complete")):
            post(f"⏳ {m}")
    out = hsp.render_horror(story, progress_cb=render_cb, narration_path=narration)

    mins = (time.time() - t0) / 60
    vid = Path(out["16x9"])
    size_mb = vid.stat().st_size / 1e6 if vid.exists() else 0
    post(f"✅ **Horror e2e DONE** in {mins:.0f} min — *{story['title']}*\n"
         f"16x9: `{vid}` ({size_mb:.0f} MB, {out.get('duration',0):.0f}s)\n"
         f"(too large to upload here — file is in 04_Outputs/final)", force=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        try:
            post(f"❌ **Horror e2e FAILED:** `{e}`", force=True)
        except Exception:
            pass
        sys.exit(1)
