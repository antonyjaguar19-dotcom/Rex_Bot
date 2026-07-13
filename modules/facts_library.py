"""
Claw Bot — the facts reel library

Every facts reel ever finished, and a way to throw one away.

A reel is scattered across four places — the story JSON, the working stills, the
finished MP4s and their publish kit, and the pristine poster-less master. Deleting
one by hand means remembering all four, which is why nobody did, and why
`04_Outputs/final` had 212 facts files in it. This module is the one place that
knows the whole footprint of a reel.

    04_Outputs/facts/facts_{id}.json          the story (facts, captions, prompts)
    04_Outputs/storyboards/facts_{id}/        stills, per-beat wavs, part clips
    04_Outputs/final/facts_{id}_*.mp4         the reel, per aspect
    04_Outputs/final/facts_{id}_*_thumb_*.jpg the publish kit (+ title, description)
    04_Outputs/final/_posterless/facts_{id}_* the master before the poster frame

Deleting a reel does NOT forget its facts (see facts_memory): if a reel was bad you
want the next one to say something DIFFERENT, so what it said stays on the
do-not-repeat list.
"""

import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger("claw_bot.facts_library")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
OUTPUTS = PROJECT_ROOT / "04_Outputs"
FACTS_DIR = OUTPUTS / "facts"
STILLS_DIR = OUTPUTS / "storyboards"
FINAL_DIR = OUTPUTS / "final"
POSTERLESS_DIR = FINAL_DIR / "_posterless"

# A reel id is a timestamp: 20260713_152428. Nothing else is ever accepted — this
# string is used to glob for files to DELETE, and "*" would take the lot.
ID_RE = re.compile(r"^\d{8}_\d{6}$")


def valid_id(facts_id: str) -> bool:
    return bool(ID_RE.match((facts_id or "").strip()))


def _require_id(facts_id: str) -> str:
    fid = (facts_id or "").strip()
    if not valid_id(fid):
        raise ValueError(f"not a facts id: {facts_id!r}")
    return fid


def _when(facts_id: str) -> Optional[datetime]:
    try:
        return datetime.strptime(facts_id, "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def _story(facts_id: str) -> dict:
    p = FACTS_DIR / f"facts_{facts_id}.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning(f"facts_{facts_id}.json unreadable: {e}")
        return {}


def videos_for(facts_id: str) -> list:
    """The finished reels for this id, by aspect. Excludes the Discord preview
    re-encode (`_discord.mp4`) — that is a copy, not a deliverable."""
    fid = _require_id(facts_id)
    if not FINAL_DIR.exists():
        return []
    return sorted(p for p in FINAL_DIR.glob(f"facts_{fid}_*.mp4")
                  if not p.name.endswith("_discord.mp4"))


def describe(facts_id: str) -> Optional[dict]:
    """Everything the UI shows for one reel, or None when it isn't finished.

    A reel with no video is not in the library: the story JSON exists from the
    moment the LLM writes it, and a list built from those would be half full of
    runs that crashed at the render.
    """
    fid = (facts_id or "").strip()
    if not valid_id(fid):
        return None
    vids = videos_for(fid)
    if not vids:
        return None

    story = _story(fid)
    facts = [b for b in story.get("beats", []) if b.get("kind") == "fact"]
    main = next((v for v in vids if v.name.endswith("_9x16.mp4")), vids[0])
    stem = main.with_suffix("")
    thumb = Path(f"{stem}_thumb_9x16.jpg")
    desc = Path(f"{stem}_description.txt")
    title_f = Path(f"{stem}_title.txt")

    title = ""
    if title_f.exists():
        title = title_f.read_text(encoding="utf-8").strip()
    title = title or story.get("title") or f"facts {fid}"

    return {
        "id": fid,
        "title": title,
        "topic": story.get("topic", ""),
        "when": _when(fid),
        "video": main,
        "videos": vids,
        "thumb": thumb if thumb.exists() else None,
        "description": (desc.read_text(encoding="utf-8").strip()
                        if desc.exists() else ""),
        "facts": [b.get("narration", "") for b in facts],
        "n_facts": len(facts),
        "size_mb": round(sum(v.stat().st_size for v in vids) / 1e6, 1),
        "placeholder": bool(story.get("_placeholder")),
    }


def list_reels(limit: int = 0) -> list:
    """Every finished reel, newest first."""
    if not FINAL_DIR.exists():
        return []
    ids = set()
    for p in FINAL_DIR.glob("facts_*_9x16.mp4"):
        # facts_20260713_152428_9x16.mp4 -> 20260713_152428
        parts = p.stem.split("_")
        if len(parts) >= 3:
            fid = f"{parts[1]}_{parts[2]}"
            if valid_id(fid):
                ids.add(fid)
    out = [describe(i) for i in sorted(ids, reverse=True)]
    out = [r for r in out if r]
    return out[:limit] if limit else out


def _paths_of(facts_id: str) -> list:
    """Every file and folder this reel owns."""
    fid = _require_id(facts_id)
    paths = [FACTS_DIR / f"facts_{fid}.json", STILLS_DIR / f"facts_{fid}"]
    for d in (FINAL_DIR, POSTERLESS_DIR):
        if d.exists():
            # Catches the reel, the kit (title/description/publish/thumb/mascot art)
            # and the Discord preview — every name starts with the id.
            paths += sorted(d.glob(f"facts_{fid}*"))
    return paths


def footprint(facts_id: str) -> tuple:
    """(how many files, how many MB) a delete would remove."""
    n, size = 0, 0
    for p in _paths_of(facts_id):
        if p.is_dir():
            for f in p.rglob("*"):
                if f.is_file():
                    n += 1
                    size += f.stat().st_size
        elif p.exists():
            n += 1
            size += p.stat().st_size
    return n, round(size / 1e6, 1)


def delete_reel(facts_id: str, forget_facts: bool = False) -> dict:
    """Delete a reel and everything it owns. Irreversible — there is no recycle bin.

    The FACTS it told are kept on the do-not-repeat list by default: deleting a bad
    reel is a reason to want a DIFFERENT one next time, not the same one again.
    `forget_facts=True` releases them (for a reel written by the offline
    placeholder, or one you deleted because you want that exact ground re-covered).
    """
    fid = _require_id(facts_id)
    removed, failed = [], []
    for p in _paths_of(fid):
        try:
            if p.is_dir():
                shutil.rmtree(p)
                removed.append(p)
            elif p.exists():
                p.unlink()
                removed.append(p)
        except Exception as e:
            log.warning(f"could not delete {p}: {e}")
            failed.append(p)

    forgotten = 0
    if forget_facts:
        from modules import facts_memory as fm
        forgotten = fm.forget_reel(fid)

    log.info(f"facts reel {fid} deleted: {len(removed)} path(s) removed"
             + (f", {len(failed)} failed" if failed else "")
             + (f", {forgotten} fact(s) forgotten" if forgotten else ""))
    return {"id": fid, "removed": removed, "failed": failed,
            "forgotten": forgotten}
