"""Validate the mascot reference image and render one preview thumbnail.

Run this the moment you drop a mascot in 02_Agent/assets/:

    venv\\Scripts\\python tools\\mascot_check.py                 # preview "goldfish"
    venv\\Scripts\\python tools\\mascot_check.py --topic bees
    venv\\Scripts\\python tools\\mascot_check.py --no-render     # checks only
"""

import argparse
import sys
from pathlib import Path

_AGENT = Path(__file__).parent.parent.resolve()
if str(_AGENT) not in sys.path:
    sys.path.insert(0, str(_AGENT))

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from PIL import Image  # noqa: E402

from modules import mascot as mas  # noqa: E402
from modules import publish_kit as pk  # noqa: E402


def check_image(p: Path) -> list[str]:
    """USO keys identity off this image. Flag what will hurt the transfer."""
    warn = []
    img = Image.open(p)
    w, h = img.size
    print(f"  size   : {w}x{h}")
    print(f"  mode   : {img.mode}")
    print(f"  bytes  : {p.stat().st_size / 1024:.0f} KB")
    if min(w, h) < 512:
        warn.append(f"small ({w}x{h}); 768px+ on the short side transfers better")
    if max(w, h) / min(w, h) > 2.2:
        warn.append("very elongated; a squarer crop of the character works better")
    if img.mode == "RGBA":
        warn.append("has alpha — USO sees it composited on black; "
                    "flatten onto a plain background for a cleaner reference")
    return warn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="goldfish")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--aspect", default="9x16", choices=["9x16", "16x9", "1x1"])
    args = ap.parse_args()

    print("=== mascot reference ===")
    p = mas.mascot_path()
    if p is None:
        print(f"  ❌ none found. Save yours as: {mas.ASSETS_DIR / 'mascot.png'}")
        print(f"     (accepted: {', '.join(mas.MASCOT_NAMES)})")
        return 1
    print(f"  ✅ {p}")
    for w in check_image(p):
        print(f"  ⚠️  {w}")

    print("\n=== renderer ===")
    ok, why = mas.backend_healthy()
    print(f"  {'✅' if ok else '❌'} {why}")
    if not ok:
        print("     Start ComfyUI, then re-run.")
        return 1

    if args.no_render:
        return 0

    print(f"\n=== preview: {args.topic} ===")
    title = f"{args.topic.title()} Facts"
    context = f"Surprising true facts about {args.topic}."
    scene = mas.scene_prompt(title, context, args.topic)
    print(f"  scene: {scene}")

    out_dir = _AGENT.parent / "04_Outputs"
    base = out_dir / f"mascot_preview_{args.topic}.png"
    png = mas.render_scene(scene, base, aspect=args.aspect)
    if not png:
        print("  ❌ render returned nothing — see the log above")
        return 1
    print(f"  ✅ art  : {png}")

    thumb = out_dir / f"mascot_preview_{args.topic}_thumb.jpg"
    size = pk.THUMB_9X16 if args.aspect == "9x16" else pk.THUMB_16X9
    if pk.render_thumbnail(png, title, thumb, size):
        print(f"  ✅ thumb: {thumb}")
    print("\nLooks right? Backfill everything with:")
    print("  venv\\Scripts\\python tools\\backfill_publish_kits.py --force")
    return 0


if __name__ == "__main__":
    sys.exit(main())
