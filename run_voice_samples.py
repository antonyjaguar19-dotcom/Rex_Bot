"""Generate the SAME facts script in every installed kokoro voice x 3 pacings
(calm / lively / excited) and post them to the Discord #status channel via REST
(bot token; no gateway, doesn't clash with a running bot). Listen on phone, pick
one, then set it with rs.set_facts_voice() / rs.set_facts_voice_speed().
"""
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT.parent / "05_Config" / "secrets.env")

from modules.tts_engine import TTSEngine

TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "").strip()
API = "https://discord.com/api/v10"
HDRS = {"Authorization": f"Bot {TOKEN}"}

SCRIPT = ("Prepare to be amazed! Did you know an octopus has three hearts and "
          "blue blood? Follow for more mind-blowing facts.")

VOICES = ["af_bella", "af_nicole", "af_sky", "am_adam", "am_michael"]
# kokoro has no emotion control — "expression" = pacing (speed).
PACINGS = [("calm", 0.92), ("lively", 1.06), ("excited", 1.20)]

OUT = ROOT.parent / "04_Outputs" / "audio" / "_voice_samples"
OUT.mkdir(parents=True, exist_ok=True)


def status_channel_id():
    g = requests.get(f"{API}/users/@me/guilds", headers=HDRS, timeout=15).json()
    for guild in g:
        chs = requests.get(f"{API}/guilds/{guild['id']}/channels", headers=HDRS, timeout=15).json()
        for c in chs:
            if c.get("name", "").lower() == "status" and c.get("type") == 0:
                return c["id"]
    return None


def main():
    cid = status_channel_id()
    if not cid:
        print("no #status channel found"); return
    requests.post(f"{API}/channels/{cid}/messages", headers=HDRS,
                  json={"content": f"🎙️ **Facts voice samples** — same script, "
                                   f"{len(VOICES)} voices × 3 pacings.\n"
                                   f"> _{SCRIPT}_\n"
                                   f"Reply with the winner (voice + calm/lively/excited)."},
                  timeout=30)

    for voice in VOICES:
        files, payload = {}, []
        for i, (label, speed) in enumerate(PACINGS):
            wav = OUT / f"{voice}_{label}.wav"
            try:
                TTSEngine(voice=voice, speed=speed).synthesize(
                    SCRIPT, output_path=wav, voice=voice, speed=speed)
            except Exception as e:
                print(f"{voice}/{label} failed: {e}"); continue
            files[f"files[{i}]"] = (wav.name, open(wav, "rb"), "audio/wav")
            payload.append(f"{label} ({speed})")
        if not files:
            continue
        print(f"posting {voice}: {', '.join(payload)}", flush=True)
        r = requests.post(f"{API}/channels/{cid}/messages", headers=HDRS,
                          data={"content": f"**{voice}** — {' · '.join(payload)}"},
                          files=files, timeout=120)
        for _k, fh in files.items():
            try: fh[1].close()
            except Exception: pass
        if r.status_code >= 300:
            print(f"  post failed {r.status_code}: {r.text[:200]}")
        time.sleep(1)

    requests.post(f"{API}/channels/{cid}/messages", headers=HDRS,
                  json={"content": "✅ All voice samples posted."}, timeout=30)
    print("done")


if __name__ == "__main__":
    main()
