"""
Character dataset builder — auto-generate a LoRA training set.

Phase A engine: the ACTIVE image backend (Z-Image Turbo). Generates a turnaround
of one character — varied angle / pose / expression, identical look token, plain
studio background — into 07_Training/datasets/<char>/.

This is the quick, dependency-free path (reuses the proven storyboard backend).
Phase B upgrade = Qwen-Image-Edit 2509 multi-angle (installed) for tighter
identity lock; swap _ANGLES driving with an edit-from-hero pass.

Curate after: delete frames where the face breaks or an accessory vanishes.
"""

import logging
import shutil
import sys
from pathlib import Path
from typing import Callable, Optional

_HERE = Path(__file__).parent.parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modules import image_backend as ib
from modules.lora import store

log = logging.getLogger("claw_bot.lora.dataset_builder")

# angle x pose matrix — keep accessories explicit so the trainer sees them often
_VIEWS = [
    ("front facing, standing straight, neutral expression", 0),
    ("front facing, smiling warmly", 11),
    ("three-quarter view from the left, standing", 23),
    ("three-quarter view from the right, standing", 37),
    ("left side profile, standing", 41),
    ("right side profile, standing", 53),
    ("back view, standing", 61),
    ("front facing, walking forward mid-stride", 73),
    ("front facing, arms crossed", 89),
    ("three-quarter left, looking up curiously", 97),
    ("three-quarter right, waving one hand", 103),
    ("front facing, sitting on the floor", 113),
    ("front facing, slight high angle looking down", 127),
    ("front facing, slight low angle looking up", 131),
    ("three-quarter left, running", 139),
    ("front facing, hands on hips confident", 149),
    ("close-up of face and shoulders, front facing", 151),
    ("close-up of face, three-quarter left", 163),
    ("front facing, crouching", 173),
    ("three-quarter right, jumping with joy", 181),
]


def build_dataset(
    name: str,
    look_token: str,
    style_suffix: str = "",
    *,
    n_images: int = 20,
    aspect_ratio: str = "1:1",
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> int:
    """Generate up to n_images turnaround frames into the char dataset dir.

    Returns count produced. Plain background + identical look token every frame.
    """
    dataset = store.dataset_dir(name)
    backend = ib.get_active_backend()
    look = look_token.strip().rstrip(".")
    style_pre = f"Art style: {style_suffix.strip().rstrip('.')}. " if style_suffix.strip() else ""

    views = _VIEWS[:max(1, n_images)]
    total = len(views)
    made = 0
    for i, (view, seed) in enumerate(views, 1):
        prompt = (
            f"{style_pre}Character reference sheet, single character alone. "
            f"{look}. {view}. "
            "Plain solid light-gray studio background, even soft lighting, sharp focus, "
            "the character's own accessories clearly visible, no other objects, no scenery, no text."
        )
        out_name = f"{store.slug(name)}_{i:02d}.png"
        target = dataset / out_name
        if progress_callback:
            progress_callback(f"dataset {name} {view[:24]}", i, total)
        try:
            saved = backend.generate(
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                output_filename=out_name,
                seed=seed,
            )
            saved = Path(saved)
            if saved.resolve() != target.resolve():
                shutil.move(str(saved), str(target))
            made += 1
        except Exception as e:
            log.warning(f"dataset frame {i}/{total} failed: {e}")
    log.info(f"dataset for '{name}': {made}/{total} frames -> {dataset}")
    return made


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    nm = sys.argv[1] if len(sys.argv) > 1 else "Pip"
    tok = sys.argv[2] if len(sys.argv) > 2 else (
        "7-year-old boy, messy orange hair, round brass goggles on forehead, "
        "blue scarf, brown leather satchel, green boots"
    )
    style = sys.argv[3] if len(sys.argv) > 3 else "soft 3D storybook render, Pixar-like, warm lighting"
    def _cb(m, c, t): print(f"  {m}: {c}/{t}")
    n = build_dataset(nm, tok, style, progress_callback=_cb)
    print("DATASET FRAMES:", n)
