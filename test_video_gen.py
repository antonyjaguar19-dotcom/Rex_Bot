"""
Claw Bot — Standalone Video Generation Test

Loads the active video backend, picks a starting frame, generates one short
clip, prints the output path. No Discord bot, no LLM, no orchestration —
just proves the adapter wiring is correct.

Run from 02_Agent/ with venv active:
    python test_video_gen.py
"""

import logging
import sys
import time
from pathlib import Path

# Ensure modules/ resolves no matter where this is launched from
AGENT_DIR = Path(__file__).parent.resolve()
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from modules import video_backend as vb

PROJECT_ROOT = AGENT_DIR.parent.resolve()
STORYBOARD_DIR = PROJECT_ROOT / "04_Outputs" / "storyboards"


def find_test_image() -> Path:
    """Pick the first PNG/JPG from storyboards/ as the starting frame."""
    if not STORYBOARD_DIR.exists():
        raise SystemExit(f"Storyboard folder missing: {STORYBOARD_DIR}")
    candidates = sorted(
        list(STORYBOARD_DIR.glob("*.png")) + list(STORYBOARD_DIR.glob("*.jpg"))
    )
    if not candidates:
        raise SystemExit(
            f"No images found in {STORYBOARD_DIR}. "
            f"Generate a storyboard first or drop a test PNG here."
        )
    return candidates[0]


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 60)
    print("Claw Bot — Video Generation Standalone Test")
    print("=" * 60)

    # 1. Load the active video backend
    print("\n[1/4] Loading active video backend...")
    backend = vb.get_active_backend()
    print(f"      Backend: {backend.backend_id}")

    # 2. Health check ComfyUI
    print("\n[2/4] Health check...")
    ok, msg = backend.health_check()
    print(f"      {msg}")
    if not ok:
        raise SystemExit("ComfyUI not running. Start it before running this test.")

    # 3. Pick a starting frame
    print("\n[3/4] Picking starting frame...")
    image = find_test_image()
    print(f"      Using: {image.name}")

    # 4. Generate
    print("\n[4/4] Generating video (this will take a few minutes)...")
    test_prompt = (
        "The character moves gently and naturally within the scene. "
        "Soft ambient motion, subtle camera drift, warm storybook lighting. "
        "Smooth, cinematic, child-friendly animation."
    )
    start = time.time()
    try:
        out = backend.generate(
            prompt=test_prompt,
            input_image=image,
            frame_count=81,        # ~3.4s @ 24fps — fast sanity check
            fps=24,
            output_filename="test_video_gen_output.mp4",
        )
    except Exception as e:
        print(f"\n❌ FAILED: {type(e).__name__}: {e}")
        raise

    elapsed = int(time.time() - start)
    print(f"\n✅ SUCCESS in {elapsed}s")
    print(f"   Output: {out}")
    print(f"   Size:   {out.stat().st_size / (1024*1024):.1f} MB")


if __name__ == "__main__":
    main()
