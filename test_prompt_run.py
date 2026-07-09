"""
test_prompt_run.py — one-shot pipeline dry-run up to the prompt stage.

Runs the REAL modules: script_generator (Qwen) -> prompt_assembler (Qwen) to
produce, for a given theme, the per-shot IMAGE prompt (Z-Image) and MOTION
prompt (Wan I2V) — exactly what the approval gate would show. NO ComfyUI render.

Prints everything to stdout (for inspection) and posts a readable summary to
the Discord #scripts channel via the REST API (Bot token from secrets.env) so
it is visible in Discord. Uses REST only (no gateway) so it does not collide
with a live claw_bot instance already holding the token.

Run:
    venv\Scripts\python test_prompt_run.py "two forest friends learn to share"
"""

import logging
import sys
from pathlib import Path

AGENT = Path(__file__).parent.resolve()
if str(AGENT) not in sys.path:
    sys.path.insert(0, str(AGENT))

# Windows console is cp1252 — emojis in print() crash. Force utf-8.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv
import os

SECRETS = AGENT.parent / "05_Config" / "secrets.env"
load_dotenv(SECRETS)

import requests
from modules import script_generator as sg
from modules import prompt_assembler as pa

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test_prompt_run")

API = "https://discord.com/api/v10"
TOKEN = os.getenv("DISCORD_BOT_TOKEN")
POST_CHANNEL = "scripts"   # NOT claw-bot, so live bots don't react to it


# ---------------------------------------------------------------------------
# Discord REST helpers (no gateway)
# ---------------------------------------------------------------------------
def _headers():
    return {"Authorization": f"Bot {TOKEN}", "Content-Type": "application/json"}


def _find_channel(name: str):
    if not TOKEN:
        log.warning("No DISCORD_BOT_TOKEN; skipping Discord post.")
        return None
    g = requests.get(f"{API}/users/@me/guilds", headers=_headers(), timeout=30)
    g.raise_for_status()
    for guild in g.json():
        ch = requests.get(f"{API}/guilds/{guild['id']}/channels",
                          headers=_headers(), timeout=30)
        ch.raise_for_status()
        for c in ch.json():
            if c.get("type") == 0 and c.get("name") == name:
                return c["id"]
    log.warning(f"Channel #{name} not found in any guild.")
    return None


def _post(channel_id: str, content: str):
    # Discord hard limit 2000 chars/message — chunk on newlines.
    if not channel_id:
        return
    for chunk in _chunks(content, 1900):
        r = requests.post(f"{API}/channels/{channel_id}/messages",
                          headers=_headers(), json={"content": chunk}, timeout=30)
        if r.status_code >= 300:
            log.warning(f"Discord post failed {r.status_code}: {r.text[:200]}")


def _chunks(text: str, limit: int):
    buf = ""
    for line in text.split("\n"):
        if len(buf) + len(line) + 1 > limit:
            if buf:
                yield buf
            buf = line
        else:
            buf = f"{buf}\n{line}" if buf else line
    if buf:
        yield buf


def _style_suffix(script: dict) -> str:
    style_id = script.get("style") or sg.get_default_style()
    return sg.get_style_description(style_id).get("prompt_suffix", "")


# ---------------------------------------------------------------------------
def main():
    theme = sys.argv[1] if len(sys.argv) > 1 else "two forest friends learn to share"
    log.info(f"THEME: {theme}")

    cid = _find_channel(POST_CHANNEL)
    _post(cid, f"🧪 **PROMPT DRY-RUN** — theme: *{theme}*\n_(script + image/motion prompts, no render)_")

    # --- Stage 1+2: script -------------------------------------------------
    script = sg.generate_script(theme)
    title = script.get("title", "?")
    shots = script.get("shots", []) or []
    chars = script.get("characters", []) or []

    cast = "\n".join(
        f"  • **{c.get('name','?')}** ({c.get('type','?')}) "
        f"voice=`{c.get('voice','-')}` — {str(c.get('appearance',''))[:80]}"
        for c in chars
    ) or "  (none)"
    head = (f"🎬 **{title}**  ({len(shots)} shots)\n"
            f"Setting: {str(script.get('setting',''))[:160]}\n"
            f"**Cast:**\n{cast}")
    print("\n" + head + "\n")
    _post(cid, head)

    style_suffix = _style_suffix(script)

    # --- Per-shot prompts --------------------------------------------------
    for idx, shot in enumerate(shots, start=1):
        n = shot.get("shot_number", idx)
        beat = (shot.get("beat") or "").strip().lower() or "hook"
        stype = shot.get("shot_type", "?")
        spk = shot.get("speaker", "narrator")
        narr = (shot.get("narration") or "").strip()

        try:
            img_p = pa.assemble_image_prompt(
                script=script, shot=shot, style_suffix=style_suffix, frame_type="first")
        except Exception as e:
            img_p = f"[IMG FAIL: {e}]"
        try:
            mot_p = pa.assemble_motion_prompt(
                script=script, shot=shot, approved_image_prompt=img_p)
        except Exception as e:
            mot_p = f"[MOTION FAIL: {e}]"

        block = (
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"**SHOT {n}** · `{beat}` · `{stype}` · speaker: **{spk}**\n"
            f"🎙️ _{narr}_\n\n"
            f"**🖼️ IMAGE:** {img_p}\n\n"
            f"**🎞️ MOTION:** {mot_p}"
        )
        print(block + "\n")
        _post(cid, block)

    done = f"✅ Done — {len(shots)} shots, script_id `{script.get('_id') or script.get('script_id','?')}`"
    print(done)
    _post(cid, done)


if __name__ == "__main__":
    main()
