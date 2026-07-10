"""Backfill title + thumbnail for videos rendered before the publish kit existed.

Usage (from 02_Agent):
    venv\\Scripts\\python tools\\backfill_publish_kits.py            # all real reels
    venv\\Scripts\\python tools\\backfill_publish_kits.py --dry-run

Skips PLACEHOLDER_* renders (facts written while the LLM was down) and anything
that already has a _title.txt.
"""

import argparse
import json
import sys
from pathlib import Path

_AGENT = Path(__file__).parent.parent.resolve()
if str(_AGENT) not in sys.path:
    sys.path.insert(0, str(_AGENT))

import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

from modules import publish_kit  # noqa: E402

PROJECT_ROOT = _AGENT.parent
FINAL_DIR = PROJECT_ROOT / "04_Outputs" / "final"
FACTS_DIR = PROJECT_ROOT / "04_Outputs" / "facts"
STORYBOARDS = PROJECT_ROOT / "04_Outputs" / "storyboards"


def _facts_jobs():
    for video in sorted(FINAL_DIR.glob("facts_*_9x16.mp4")):
        if video.name.startswith("PLACEHOLDER_") or video.stem.endswith("_discord"):
            continue
        fid = video.stem.replace("facts_", "").replace("_9x16", "")
        story_path = FACTS_DIR / f"facts_{fid}.json"
        if not story_path.exists():
            continue
        story = json.loads(story_path.read_text(encoding="utf-8"))
        if story.get("_placeholder"):
            continue                      # never dress up a placeholder reel
        sb = STORYBOARDS / f"facts_{fid}"
        stills = sorted(sb.glob("bg_*.png")) if sb.exists() else []
        still = stills[1] if len(stills) > 1 else (stills[0] if stills else None)
        context = "\n".join(b.get("narration", "") for b in story.get("beats", []))
        yield video, story.get("title", fid), context, \
            (story.get("description") or "").strip(), "facts short", still


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="rebuild existing kits")
    ap.add_argument("--only", help="one facts id, e.g. 20260709_233013")
    ap.add_argument("--limit", type=int, help="stop after N videos")
    ap.add_argument("--no-mascot", action="store_true",
                    help="skip USO mascot art (fast; uses the video's own still)")
    args = ap.parse_args()

    if args.no_mascot:
        import modules.runtime_settings as rs
        _orig = rs.get_mascot_thumbnails_enabled
        rs.get_mascot_thumbnails_enabled = lambda: False    # process-local only

    from modules import mascot
    ok, why = mascot.is_available()
    print(f"mascot: {'ON — ' + why if ok and not args.no_mascot else 'off (' + why + ')'}")
    if ok and not args.no_mascot:
        print("  note: one USO render per aspect per video (~2-3 min each)\n")

    jobs = list(_facts_jobs())
    if args.only:
        jobs = [j for j in jobs if args.only in j[0].name]
    if args.limit:
        jobs = jobs[: args.limit]

    done = skipped = 0
    for video, title, context, desc, mode, still in jobs:
        if not args.force and Path(f"{video.with_suffix('')}_title.txt").exists():
            skipped += 1
            continue
        print(f"→ {video.name}")
        if args.dry_run:
            print(f"   would use still: {still.name if still else '(video frame)'}")
            continue
        kit = publish_kit.attach(video, fallback_title=title, context=context,
                                 description=desc, mode=mode, source_image=still)
        print(f"   title : {kit.get('title')}")
        print(f"   source: {kit.get('thumb_source')}")
        if kit.get("mascot_scene"):
            print(f"   scene : {kit['mascot_scene']}")
        print(f"   thumb : {Path(kit['thumb_9x16']).name if kit.get('thumb_9x16') else 'FAILED'}")
        done += 1

    print(f"\n{done} kit(s) built, {skipped} already had one, {len(jobs)} candidate(s).")


if __name__ == "__main__":
    main()
