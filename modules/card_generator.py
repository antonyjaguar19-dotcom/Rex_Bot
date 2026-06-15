"""
Claw Bot — Title / End Card Generator (Phase 6C)

Builds the two bookend cards baked into the final video by assembly.py:
  - TITLE card  : blurred FIRST storyboard frame + the story title on top,
                  Kokoro speaks the title.
  - END card    : blurred LAST storyboard frame + the moral on top,
                  Kokoro speaks the moral ("for the parents" takeaway).

Each card is rendered PER ASPECT (text + blur sized to the exact canvas) so
it fills 9x16 / 16x9 / 1x1 with no black bars. The spoken audio is generated
ONCE and reused across aspects (same wav, different image).

Pure PIL + ffmpeg orchestration. No Discord, no LLM. Graceful: callers wrap in
try/except — a missing frame or TTS failure must NOT break assembly.
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont, ImageFilter

_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

log = logging.getLogger("claw_bot.card_generator")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
FFMPEG_EXE = PROJECT_ROOT / "00_Tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
FFPROBE_EXE = PROJECT_ROOT / "00_Tools" / "ffmpeg" / "bin" / "ffprobe.exe"
STORYBOARDS_DIR = PROJECT_ROOT / "04_Outputs" / "storyboards"
CARDS_DIR = PROJECT_ROOT / "04_Outputs" / "clips" / "_cards"

# Card video framerate. Assembly re-encodes everything to its own FPS anyway,
# but we author at a sane value so the card stands alone too.
FPS = 24
# Trailing freeze after speech ends so the last word breathes before the cut.
TAIL_PAD_SEC = 0.8
# Minimum card length even if speech is very short / absent.
MIN_CARD_SEC = 2.5
# Gaussian blur radius as a fraction of canvas width.
BLUR_FRAC = 0.018
# Dark scrim alpha over the blurred frame so text stays legible (0-255).
SCRIM_ALPHA = 110


# ==============================================================================
# IMAGE HELPERS
# ==============================================================================
def _load_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    faces = (
        ["C:\\Windows\\Fonts\\segoeuib.ttf", "C:\\Windows\\Fonts\\arialbd.ttf",
         "DejaVuSans-Bold.ttf"]
        if bold else
        ["C:\\Windows\\Fonts\\segoeui.ttf", "C:\\Windows\\Fonts\\arial.ttf",
         "DejaVuSans.ttf"]
    )
    for face in faces:
        try:
            return ImageFont.truetype(face, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _fit_cover(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Scale so the image COVERS the canvas (no bars), then centre-crop."""
    w, h = img.size
    scale = max(target_w / w, target_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


def _wrap_text(draw, text: str, font, max_w: int) -> list[str]:
    """Greedy word-wrap so each line fits within max_w pixels."""
    words = text.split()
    if not words:
        return [""]
    lines, cur = [], words[0]
    for word in words[1:]:
        trial = f"{cur} {word}"
        if draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    lines.append(cur)
    return lines


def _draw_block(
    draw, lines: list[str], font, canvas_w: int, top_y: int,
    fill=(255, 255, 255), line_gap_frac: float = 0.25,
) -> int:
    """Draw centred, shadowed text lines starting at top_y. Returns bottom y."""
    ascent, descent = font.getmetrics()
    line_h = ascent + descent
    gap = int(line_h * line_gap_frac)
    y = top_y
    shadow_off = max(2, line_h // 22)
    for line in lines:
        tw = draw.textlength(line, font=font)
        x = (canvas_w - tw) / 2
        draw.text((x + shadow_off, y + shadow_off), line, font=font,
                  fill=(0, 0, 0))
        draw.text((x, y), line, font=font, fill=fill)
        y += line_h + gap
    return y - gap


def _render_card_image(
    frame_path: Optional[Path], main_text: str, target_w: int, target_h: int,
    out_png: Path, *, label: Optional[str] = None,
) -> Path:
    """Blurred frame background + dark scrim + centred title/moral text."""
    # Background: blurred cover-fit frame, or a dark gradient-ish solid fallback.
    if frame_path and Path(frame_path).exists():
        try:
            bg = Image.open(frame_path).convert("RGB")
            bg = _fit_cover(bg, target_w, target_h)
            bg = bg.filter(ImageFilter.GaussianBlur(
                radius=max(8, int(target_w * BLUR_FRAC))))
        except Exception as e:
            log.warning(f"Card bg load failed ({frame_path}): {e}; using solid.")
            bg = Image.new("RGB", (target_w, target_h), (20, 22, 28))
    else:
        bg = Image.new("RGB", (target_w, target_h), (20, 22, 28))

    # Dark scrim for legibility.
    scrim = Image.new("RGBA", (target_w, target_h), (0, 0, 0, SCRIM_ALPHA))
    canvas = Image.alpha_composite(bg.convert("RGBA"), scrim)
    draw = ImageDraw.Draw(canvas)

    margin = int(target_w * 0.10)
    max_text_w = target_w - 2 * margin

    main_font = _load_font(max(20, int(target_w / 13)), bold=True)
    main_lines = _wrap_text(draw, main_text, main_font, max_text_w)

    label_font = None
    label_lines: list[str] = []
    if label:
        label_font = _load_font(max(14, int(target_w / 26)), bold=True)
        label_lines = [label.upper()]

    # Measure total block height to vertically centre it.
    def _block_h(lines, font):
        a, d = font.getmetrics()
        lh = a + d
        return len(lines) * lh + (len(lines) - 1) * int(lh * 0.25)

    total_h = _block_h(main_lines, main_font)
    label_gap = int(target_h * 0.03)
    if label_lines:
        total_h += _block_h(label_lines, label_font) + label_gap

    y = (target_h - total_h) // 2
    if label_lines:
        y = _draw_block(draw, label_lines, label_font, target_w, y,
                        fill=(255, 214, 120))
        y += label_gap
    _draw_block(draw, main_lines, main_font, target_w, y)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out_png, format="PNG")
    return out_png


# ==============================================================================
# AUDIO + VIDEO
# ==============================================================================
def synth_card_audio(text: str, out_wav: Path, tts=None) -> tuple[Optional[Path], float]:
    """Kokoro-speak `text`. Returns (wav_path, duration_sec) or (None, 0.0)."""
    if not text or not text.strip():
        return None, 0.0
    try:
        if tts is None:
            from modules.tts_engine import TTSEngine
            from modules import runtime_settings as rs
            tts = TTSEngine(voice=rs.get_effective_voice())
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        tts.synthesize(text, output_path=out_wav)
        dur = _probe_duration(out_wav)
        return out_wav, dur
    except Exception as e:
        log.warning(f"Card TTS failed for '{text[:40]}...': {e}")
        return None, 0.0


def _probe_duration(path: Path) -> float:
    cmd = [
        str(FFPROBE_EXE), "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
    return float(r.stdout.strip())


def _render_card_video(
    image_path: Path, audio_path: Optional[Path], duration: float, out_mp4: Path,
) -> Path:
    """Static image for `duration`s + optional spoken audio (silence-padded tail).

    NO -shortest: audio is mapped in full; the video is held to `duration`
    which is always >= audio length, so the last word is never cropped.
    """
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(FFMPEG_EXE), "-y", "-loglevel", "error",
           "-loop", "1", "-i", str(image_path)]
    if audio_path and Path(audio_path).exists():
        cmd += ["-i", str(audio_path)]
    else:
        cmd += ["-f", "lavfi", "-i",
                "anullsrc=channel_layout=stereo:sample_rate=48000"]
    cmd += [
        "-t", f"{duration:.3f}",
        "-r", str(FPS),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart",
        str(out_mp4),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg card render failed:\n{result.stderr.strip()}")
    return out_mp4


# ==============================================================================
# PUBLIC API
# ==============================================================================
def find_frame(script_id: str, which: str) -> Optional[Path]:
    """Return first or last storyboard frame path. which = 'first' | 'last'."""
    story_dir = STORYBOARDS_DIR / script_id
    frames = sorted(
        story_dir.glob("shot*_first.png"),
        key=lambda p: int(p.stem.split("shot")[1].split("_")[0]),
    )
    if not frames:
        return None
    return frames[0] if which == "first" else frames[-1]


def effective_card_duration(raw_audio_dur: float) -> float:
    """Final on-screen card length given the spoken-audio duration.

    Single source of truth so assembly's filter offsets/music length match the
    actual rendered card length exactly.
    """
    return max(MIN_CARD_SEC, raw_audio_dur + TAIL_PAD_SEC)


def build_card(
    *, script_id: str, kind: str, text: str, frame_path: Optional[Path],
    target_w: int, target_h: int, aspect_key: str,
    audio_path: Optional[Path], duration: float, label: Optional[str] = None,
) -> Path:
    """Render one card (image sized to canvas) + mux spoken audio → mp4.

    kind = 'title' | 'end'. Returns the card mp4 path.
    """
    dur = effective_card_duration(duration)
    png = CARDS_DIR / f"card_{script_id}_{kind}_{aspect_key}.png"
    mp4 = CARDS_DIR / f"card_{script_id}_{kind}_{aspect_key}.mp4"
    _render_card_image(frame_path, text, target_w, target_h, png, label=label)
    _render_card_video(png, audio_path, dur, mp4)
    return mp4
