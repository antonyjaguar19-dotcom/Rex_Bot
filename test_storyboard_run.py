"""
test_storyboard_run.py — render a real storyboard from an existing script.

Pipeline:
  1. load script_{id}.json
  2. prompt_approval.generate_all_prompts(script)  (Qwen image + motion prompts)
  3. mark EVERY prompt approved=True  (auto-approve as-given, no edits)
  4. prompt_approval._save(state)     -> approved_prompts/{id}.json
  5. storyboard_generator.generate_storyboard(id, "16:9")  (ComfyUI Z-Image render)
  6. build a contact sheet of all shots
  7. post the contact sheet (+ per-shot summary) to Discord #storyboards via REST

Run:
    venv\Scripts\python test_storyboard_run.py 20260615_194841
"""

import logging
import sys
from pathlib import Path

AGENT = Path(__file__).parent.resolve()
ROOT = AGENT.parent
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

load_dotenv(ROOT / "05_Config" / "secrets.env")

import requests
from modules import prompt_approval as pap
from modules import storyboard_generator as sg
from modules import contact_sheet as cs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test_storyboard_run")

API = "https://discord.com/api/v10"
TOKEN = os.getenv("DISCORD_BOT_TOKEN")
POST_CHANNEL = "storyboards"


# --------------------------------------------------------------------------
# Discord REST (no gateway)
# --------------------------------------------------------------------------
def _headers():
    return {"Authorization": f"Bot {TOKEN}"}


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
    log.warning(f"Channel #{name} not found.")
    return None


def _post_text(cid: str, content: str):
    if not cid:
        return
    for chunk in _chunks(content, 1900):
        r = requests.post(f"{API}/channels/{cid}/messages",
                          headers={**_headers(), "Content-Type": "application/json"},
                          json={"content": chunk}, timeout=30)
        if r.status_code >= 300:
            log.warning(f"Discord text post failed {r.status_code}: {r.text[:200]}")


def _post_file(cid: str, path: Path, content: str = ""):
    if not cid or not path.exists():
        return
    with open(path, "rb") as f:
        r = requests.post(
            f"{API}/channels/{cid}/messages",
            headers=_headers(),
            data={"content": content[:1900]},
            files={"files[0]": (path.name, f, "image/png")},
            timeout=120,
        )
    if r.status_code >= 300:
        log.warning(f"Discord file post failed {r.status_code}: {r.text[:200]}")


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


# --------------------------------------------------------------------------
def main():
    script_id = sys.argv[1] if len(sys.argv) > 1 else "20260615_194841"
    log.info(f"SCRIPT: {script_id}")

    script_path = ROOT / "04_Outputs" / "scripts" / f"script_{script_id}.json"
    if not script_path.exists():
        log.error(f"Script not found: {script_path}")
        sys.exit(1)

    import json
    script = json.loads(script_path.read_text(encoding="utf-8"))
    script.setdefault("_id", script_id)
    script.setdefault("script_id", script_id)
    title = script.get("title", "?")
    shots = script.get("shots", []) or []

    cid = _find_channel(POST_CHANNEL)
    _post_text(cid, f"🎨 **STORYBOARD RENDER** — *{title}* (`{script_id}`, {len(shots)} shots)\n_auto-approving all prompts as given, then rendering Z-Image…_")

    # --- 1+2: generate all prompts (LLM) -----------------------------------
    log.info("Generating image+motion prompts for all shots (Qwen)…")

    def _pp(msg, i, n):
        log.info(f"  prompt {i}/{n}: {msg}")

    state = pap.generate_all_prompts(script, progress=_pp)

    # --- 3: auto-approve everything ----------------------------------------
    for k, v in state.get("prompts", {}).items():
        v["approved"] = True
    pap._save(state)
    log.info(f"Approved + saved {len(state.get('prompts', {}))} shot prompts.")

    # --- 5: render storyboard ---------------------------------------------
    log.info("Rendering storyboard via ComfyUI (Z-Image)…")

    def _rp(stage, i, n):
        log.info(f"  render {i}/{n}: {stage}")

    result = sg.generate_storyboard(
        script_id, aspect_ratio="16:9", progress_callback=_rp,
    )
    log.info(f"Render done: success={result.success} frames={result.total_frames} err={result.error}")

    # --- 6: contact sheet --------------------------------------------------
    first_frames = [f for f in result.frames if f.frame_type == "first"]
    first_frames.sort(key=lambda f: f.shot_number)
    paths = [ROOT / f.image_path for f in first_frames]
    paths = [p for p in paths if p.exists()]
    labels = [f"Shot {f.shot_number} · {f.beat}" for f in first_frames if (ROOT / f.image_path).exists()]

    if not paths:
        _post_text(cid, f"❌ No frames rendered. error={result.error}")
        log.error("No frames rendered.")
        return

    sheet = ROOT / "04_Outputs" / "storyboards" / script_id / "_contact_sheet.png"
    cs.build_contact_sheet(paths, sheet, shot_labels=labels)
    log.info(f"Contact sheet: {sheet}")

    _post_file(cid, sheet, f"🎬 **{title}** — {len(paths)}/{len(shots)} shots rendered. Check consistency (character look, style, depth).")

    # also post each frame with its narration for shot-by-shot eval
    sid_shot = {f.shot_number: f for f in first_frames}
    for shot in shots:
        n = shot.get("shot_number")
        fr = sid_shot.get(n)
        if not fr:
            continue
        p = ROOT / fr.image_path
        if not p.exists():
            continue
        narr = (shot.get("narration") or "").strip()
        spk = shot.get("speaker", "narrator")
        _post_file(cid, p, f"**Shot {n}** · `{fr.beat}` · {shot.get('shot_type','?')} · 🗣 {spk}\n🎙️ _{narr}_")

    _post_text(cid, f"✅ Storyboard render complete — `{script_id}`")
    log.info("DONE.")


if __name__ == "__main__":
    main()
