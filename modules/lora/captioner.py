"""
Caption a character dataset for ai-toolkit.

ai-toolkit reads a sidecar <image>.txt caption next to each training image.
For a character LoRA the caption is intentionally simple and identical-ish:
    "<trigger>, <look token>"
e.g. "rexj_lily, 7-year-old girl, red wool coat, brass goggles on forehead"

Why so flat: we WANT the trigger token to absorb the character's full look
(face + outfit + accessories). Over-describing per-image teaches the model to
separate traits from the trigger — the opposite of what we want here.
"""

import logging
from pathlib import Path

log = logging.getLogger("claw_bot.lora.captioner")

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def caption_dataset(
    dataset_dir: Path,
    trigger: str,
    look_token: str,
    *,
    overwrite: bool = True,
) -> int:
    """Write a .txt caption beside every image. Returns number written."""
    dataset_dir = Path(dataset_dir)
    look = (look_token or "").strip().rstrip(".")
    caption = f"{trigger}, {look}" if look else trigger

    n = 0
    imgs = [p for p in sorted(dataset_dir.iterdir())
            if p.is_file() and p.suffix.lower() in IMG_EXTS]
    for img in imgs:
        txt = img.with_suffix(".txt")
        if txt.exists() and not overwrite:
            continue
        txt.write_text(caption, encoding="utf-8")
        n += 1
    log.info(f"captioned {n} image(s) in {dataset_dir} -> '{caption}'")
    return n


def count_images(dataset_dir: Path) -> int:
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        return 0
    return sum(1 for p in dataset_dir.iterdir()
               if p.is_file() and p.suffix.lower() in IMG_EXTS)
