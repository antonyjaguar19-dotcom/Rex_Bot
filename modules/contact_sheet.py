"""
Claw Bot — Contact-Sheet Builder

Composes one PNG containing every shot from a storyboard arranged in a grid
with shot-number labels. Posted to Discord above the approval prompt so the
user can review the whole storyboard at a glance instead of scrolling through
N individual frame embeds.
"""

import logging
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger("claw_bot.contact_sheet")

# Pick a grid layout for N images that yields close-to-square aspect.
def _grid_dims(n: int) -> tuple[int, int]:
    if n <= 0:
        return 1, 1
    if n == 1:
        return 1, 1
    if n == 2:
        return 2, 1
    if n <= 4:
        return 2, 2
    if n <= 6:
        return 3, 2
    if n <= 9:
        return 3, 3
    if n <= 12:
        return 4, 3
    if n <= 16:
        return 4, 4
    # Fallback: roughly square
    import math
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return cols, rows


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    # PIL default font is bitmap; try a TTF for crisp labels.
    for candidate in (
        "C:\\Windows\\Fonts\\segoeuib.ttf",   # Segoe UI Bold (Windows)
        "C:\\Windows\\Fonts\\arialbd.ttf",    # Arial Bold (Windows)
        "DejaVuSans-Bold.ttf",                # cross-platform fallback
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    return ImageFont.load_default()


def build_contact_sheet(
    image_paths: list[Path],
    output_path: Path,
    *,
    shot_labels: Optional[list[str]] = None,
    tile_width: int = 480,
    padding: int = 12,
    bg_color: tuple[int, int, int] = (24, 25, 28),
    label_color: tuple[int, int, int] = (240, 240, 240),
    label_height: int = 36,
) -> Path:
    """
    Build a contact-sheet PNG from a list of storyboard image paths.

    Args:
        image_paths: ordered list of frame PNGs (one per shot)
        output_path: where to save the composite PNG
        shot_labels: optional per-tile labels (default: "Shot 1", "Shot 2", ...)
        tile_width: width in pixels of each thumbnail (height auto from aspect)
        padding: pixel gap between tiles
        bg_color: RGB background fill
        label_color: RGB label text color
        label_height: pixel height reserved under each tile for the label

    Returns:
        Path to the saved PNG.
    """
    if not image_paths:
        raise ValueError("No images provided for contact sheet.")

    cols, rows = _grid_dims(len(image_paths))
    log.info(f"Contact sheet: {len(image_paths)} images → {cols}x{rows} grid")

    # Load + resize all tiles uniformly. Preserve aspect; pad to common height.
    tiles: list[Image.Image] = []
    max_tile_h = 0
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            log.warning(f"Could not open {p}: {e}; using placeholder")
            img = Image.new("RGB", (tile_width, int(tile_width * 9 / 16)), (40, 40, 45))
        # Scale width → tile_width
        w, h = img.size
        scale = tile_width / w
        new_h = max(1, int(h * scale))
        img = img.resize((tile_width, new_h), Image.LANCZOS)
        tiles.append(img)
        max_tile_h = max(max_tile_h, new_h)

    # Pad shorter tiles vertically so the grid lines up
    normalized: list[Image.Image] = []
    for tile in tiles:
        if tile.height == max_tile_h:
            normalized.append(tile)
            continue
        padded = Image.new("RGB", (tile_width, max_tile_h), bg_color)
        offset_y = (max_tile_h - tile.height) // 2
        padded.paste(tile, (0, offset_y))
        normalized.append(padded)
    tiles = normalized

    cell_w = tile_width
    cell_h = max_tile_h + label_height
    sheet_w = cols * cell_w + (cols + 1) * padding
    sheet_h = rows * cell_h + (rows + 1) * padding

    sheet = Image.new("RGB", (sheet_w, sheet_h), bg_color)
    draw = ImageDraw.Draw(sheet)
    font = _load_font(max(14, int(label_height * 0.55)))

    labels = shot_labels or [f"Shot {i + 1}" for i in range(len(image_paths))]

    for idx, tile in enumerate(tiles):
        col = idx % cols
        row = idx // cols
        x = padding + col * (cell_w + padding)
        y = padding + row * (cell_h + padding)
        sheet.paste(tile, (x, y))

        label = labels[idx] if idx < len(labels) else f"Shot {idx + 1}"
        # Centred label below tile
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            text_w, text_h = len(label) * 8, 14
        text_x = x + (cell_w - text_w) // 2
        text_y = y + max_tile_h + (label_height - text_h) // 2
        draw.text((text_x, text_y), label, font=font, fill=label_color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG", optimize=True)
    log.info(
        f"Contact sheet saved: {output_path} "
        f"({sheet_w}x{sheet_h}, {output_path.stat().st_size / 1024:.0f} KB)"
    )
    return output_path


if __name__ == "__main__":
    import sys, json
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python contact_sheet.py <script_id>")
        sys.exit(1)
    sid = sys.argv[1]
    project_root = Path(__file__).parent.parent.parent
    manifest = project_root / "04_Outputs" / "storyboards" / sid / "storyboard.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    imgs = [project_root / f["image_path"] for f in data["frames"]]
    out = project_root / "04_Outputs" / "storyboards" / sid / "_contact_sheet.png"
    build_contact_sheet(imgs, out)
    print(f"Saved: {out}")
