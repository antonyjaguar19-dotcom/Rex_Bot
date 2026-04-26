"""
End-to-end test: generate a full 8-frame storyboard from an approved script.
Run from 02_Agent: python test_storyboard_generate.py <script_id>
                  (omit arg to auto-pick latest approved script)
"""
import sys
import logging
from pathlib import Path

_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)

from modules import storyboard_generator as sg
from modules.storyboard_generator import SCRIPTS_DIR


def _pick_latest_script() -> str:
    """Find the most recent VALID script JSON file's ID."""
    import json as _json
    files = sorted(SCRIPTS_DIR.glob("script_*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        print("No scripts found in 04_Outputs/scripts/ — run !generate_script first.")
        sys.exit(1)

    for f in files:
        if f.stat().st_size == 0:
            print(f"⚠ Skipping empty file: {f.name}")
            continue
        try:
            data = _json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"⚠ Skipping unparseable file {f.name}: {e}")
            continue
        if "shots" not in data or len(data.get("shots", [])) != 4:
            print(f"⚠ Skipping {f.name}: doesn't have 4 shots")
            continue
        # Found a valid script
        stem = f.stem
        return stem[len("script_"):]

    print("No VALID scripts found. Generate one first with !generate_script in Discord.")
    sys.exit(1)


def progress(status, current, total):
    print(f"  ▸ [{current}/{total}] {status}")


def main():
    if len(sys.argv) > 1:
        script_id = sys.argv[1]
    else:
        script_id = _pick_latest_script()
        print(f"No script_id given — using latest: {script_id}\n")

    print(f"Generating storyboard for: {script_id}")
    print(f"(This takes ~2 minutes for 4 frames.)\n")

    result = sg.generate_storyboard(
        script_id=script_id,
        aspect_ratio="16:9",
        progress_callback=progress,
    )

    print(f"\n{'=' * 60}")
    print(f"Storyboard generation: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"Backend: {result.backend_id}")
    print(f"Frames: {len(result.frames)}")
    for f in result.frames:
        print(f"  shot{f.shot_number}_{f.frame_type}: {f.image_path}")
    print(f"{'=' * 60}\n")

    if result.error:
        print(f"Error: {result.error}")
        sys.exit(1)


if __name__ == "__main__":
    main()