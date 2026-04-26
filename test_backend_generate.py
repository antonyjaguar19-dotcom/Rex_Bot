"""
End-to-end test: generates one image using the active backend.
Run from 02_Agent: python test_backend_generate.py
"""
import sys
import logging
from pathlib import Path

_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from modules import image_backend as ib

backend = ib.get_active_backend()
print(f"\nBackend: {backend.backend_id}")

ok, msg = backend.health_check()
print(f"Health: {msg}")
if not ok:
    print("Backend unhealthy. Start ComfyUI first.")
    sys.exit(1)

test_prompt = (
    "A warm, hand-painted storybook illustration of a happy 5-year-old Indian boy "
    "hugging a bright red toy fire truck to his chest. He sits cross-legged on a "
    "woven jute rug, wearing a mustard yellow kurta, with short black hair. Soft "
    "afternoon sunlight streams through a window with light blue curtains behind "
    "him. Warm color palette, soft pastel tones, Pixar-inspired character design, "
    "gentle rim lighting, cozy children's picture book aesthetic."
)

print("\nGenerating (this takes ~30-40 seconds)...\n")
path = backend.generate(
    test_prompt,
    aspect_ratio="16:9",
    output_filename="test_zimage_1.png",
)
print(f"\n✓ Image saved: {path}\n")