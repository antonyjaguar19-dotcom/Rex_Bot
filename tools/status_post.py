"""
status_post.py — one-shot Discord #status poster (REST, no gateway).

Used by the LoRA-pipeline build to stream progress into the #status channel
while the main bot keeps running. Uses the bot token via REST only (no second
gateway login, so it never conflicts with the live bot).

Usage:
    python tools/status_post.py "message text"
    python tools/status_post.py --channel claw-bot "message text"

Reads DISCORD_BOT_TOKEN from 05_Config/secrets.env. Never prints the token.
Caches resolved channel id in 05_Config/.status_channel_cache.json.
"""

import json
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
import os

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
SECRETS = PROJECT_ROOT / "05_Config" / "secrets.env"
CACHE = PROJECT_ROOT / "05_Config" / ".status_channel_cache.json"
API = "https://discord.com/api/v10"

# Preferred channel names, first hit wins.
PREFERRED = ["status", "claw-bot", "general"]


def _headers(token: str) -> dict:
    return {"Authorization": f"Bot {token}", "Content-Type": "application/json"}


def _resolve_channel(token: str, want: str | None) -> str:
    """Find a text channel id by name across the bot's guilds. Cache it."""
    names = [want] if want else PREFERRED
    # cache only valid for default preference (not an explicit override)
    if not want and CACHE.exists():
        try:
            c = json.loads(CACHE.read_text())
            if c.get("id"):
                return c["id"]
        except Exception:
            pass

    h = _headers(token)
    guilds = requests.get(f"{API}/users/@me/guilds", headers=h, timeout=15).json()
    found = None
    for g in guilds:
        chans = requests.get(f"{API}/guilds/{g['id']}/channels", headers=h, timeout=15).json()
        text = {c["name"]: c["id"] for c in chans if c.get("type") == 0}
        for n in names:
            if n in text:
                found = text[n]
                break
        if found:
            break
    if not found:
        raise RuntimeError(f"No channel found matching {names}")
    if not want:
        CACHE.write_text(json.dumps({"id": found, "ts": time.time()}))
    return found


def post(message: str, channel: str | None = None) -> bool:
    load_dotenv(SECRETS)
    token = os.environ.get("DISCORD_BOT_TOKEN")
    if not token:
        print("ERROR: DISCORD_BOT_TOKEN not in secrets.env", file=sys.stderr)
        return False
    try:
        cid = _resolve_channel(token, channel)
        r = requests.post(
            f"{API}/channels/{cid}/messages",
            headers=_headers(token),
            json={"content": message[:1990]},
            timeout=15,
        )
        if r.status_code in (200, 201):
            return True
        print(f"ERROR: post failed HTTP {r.status_code}: {r.text[:200]}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return False


if __name__ == "__main__":
    args = sys.argv[1:]
    chan = None
    if args and args[0] == "--channel":
        chan = args[1]
        args = args[2:]
    msg = " ".join(args) if args else "(empty)"
    ok = post(msg, chan)
    sys.exit(0 if ok else 1)
