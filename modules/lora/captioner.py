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
import re
from pathlib import Path

log = logging.getLogger("claw_bot.lora.captioner")

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


# Common class nouns — caption keeps ONLY the class, never the full look, so the
# trigger token (not the descriptive words) absorbs face + outfit + accessories.
_CLASS_WORDS = [
    "toddler", "baby", "boy", "girl", "child", "kid", "man", "woman", "teenager",
    "princess", "prince", "knight", "wizard", "witch", "fox", "dog", "cat",
    "rabbit", "bear", "robot", "dragon", "monster", "creature", "animal",
]


def class_from_look(look_token: str) -> str:
    """Pull a single class noun out of a look description ('...7yo boy...' -> 'boy')."""
    lt = (look_token or "").lower()
    for w in _CLASS_WORDS:
        if re.search(rf"\b{w}\b", lt):
            return w
    return "character"


def caption_dataset(
    dataset_dir: Path,
    trigger: str,
    look_token: str,
    *,
    overwrite: bool = True,
) -> int:
    """Write a .txt caption beside every image. Returns number written.

    Caption = trigger + class ONLY (e.g. 'rexj_pip, a boy'). The full look stays
    OUT so the trigger learns to carry the whole appearance; otherwise flux binds
    the look to the descriptive words and trigger-only recall is weak.
    """
    dataset_dir = Path(dataset_dir)
    cls = class_from_look(look_token)
    caption = f"{trigger}, a {cls}"

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
