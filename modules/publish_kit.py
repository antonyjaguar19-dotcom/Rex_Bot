"""
Claw Bot — publish kit (title + thumbnail + description)

Every finished video used to land in 04_Outputs/final as a bare MP4. Facts mode
wrote a description; nothing wrote a TITLE you could paste into YouTube, and
nothing produced a THUMBNAIL you could set. You did that by hand, per video.

`attach(video_path, ...)` now writes, next to the video:

    <stem>_title.txt          one line, ready to paste
    <stem>_description.txt    caption + hashtags (kept if the pipeline made one)
    <stem>_thumb_16x9.jpg     1280x720  — YouTube custom thumbnail
    <stem>_thumb_9x16.jpg     1080x1920 — Shorts / Reels / TikTok cover
    <stem>_publish.json       all of the above, machine-readable

Design rules:
- **Never fail the render.** A thumbnail is cosmetic; the video already cost an
  hour of GPU. Every step is wrapped, and a failure downgrades (LLM title ->
  the story's own title -> the filename) rather than raising.
- **Never invent facts.** The title falls back to text the pipeline already
  wrote. Unlike facts_writer's old `_fallback`, nothing here fabricates content
  that could be mistaken for real.
- The thumbnail frame is chosen by ffmpeg's `thumbnail` filter (most
  representative of a batch), not frame 0 — which is usually a fade-in.
"""

import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFilter, ImageStat

_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modules.card_generator import _load_font, _fit_cover, _wrap_text

log = logging.getLogger("claw_bot.publish_kit")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
WATERMARK_PNG = PROJECT_ROOT / "02_Agent" / "assets" / "watermark.png"

THUMB_16X9 = (1280, 720)
THUMB_9X16 = (1080, 1920)

# Bulk tools flip this so the mascot model stays resident across many videos.
# A pipeline rendering ONE video leaves it False: the next stage (Wan, Z-Image)
# needs the card back, and on 16 GB they cannot coexist.
KEEP_MODEL_WARM = False

# YouTube hard-caps titles at 100 chars; short ones read better on mobile.
TITLE_MAX = 80
# YouTube rejects custom thumbnails over 2 MB.
THUMB_MAX_BYTES = 2 * 1024 * 1024

_TITLE_SYS = (
    "You write titles for short-form video (YouTube Shorts, Reels, TikTok).\n"
    "Output ONLY valid JSON: {\"title\": \"...\"}\n"
    "Rules:\n"
    f"- At most {TITLE_MAX} characters. Shorter is better.\n"
    "- Must be TRUE to the content you are given. Never promise something the "
    "video does not deliver, never invent a fact or a number.\n"
    "- Curiosity, not clickbait. No ALL CAPS words, no 'You won't believe'.\n"
    "- At most one emoji, only if it earns its place.\n"
    "- No surrounding quotes, no hashtags, no channel name."
)


# ==============================================================================
# TITLE
# ==============================================================================

def _clean_title(text: str) -> str:
    """Strip the things models add anyway: quotes, hashtags, stray newlines."""
    t = (text or "").strip()
    t = t.split("\n")[0].strip()
    t = t.strip('"“”\'')
    t = re.sub(r"#\w+", "", t).strip()
    t = re.sub(r"\s+", " ", t)
    if len(t) > TITLE_MAX:
        # Cut on a word boundary rather than mid-word.
        t = t[:TITLE_MAX].rsplit(" ", 1)[0].rstrip(",.;:-") + "…"
    return t


# Hype and framing words a title may use even though the narration never does.
# Anything ELSE substantial has to be traceable to the video's own content.
_GENERIC_WORDS = {
    "facts", "fact", "truth", "truths", "secret", "secrets", "surprising",
    "amazing", "revealed", "reveal", "explained", "about", "these", "their",
    "there", "think", "know", "knew", "never", "really", "actually", "things",
    "thing", "every", "everything", "cosmic", "wild", "crazy", "weird",
    "strange", "hidden", "unbelievable", "incredible", "mysterious", "mystery",
    "science", "history", "world", "inside", "behind", "beyond", "before",
    "after", "still", "more", "most", "than", "that", "this", "with", "your",
    "you", "from", "into", "what", "when", "where", "which", "while", "would",
    "could", "should", "shorts", "short", "video",
    # short filler that carries no claim
    "have", "has", "had", "will", "wont", "cant", "does", "did", "are", "is",
    "was", "were", "been", "them", "they", "then", "here", "just", "only",
    "also", "much", "many", "some", "each", "over", "under", "like", "make",
    "made", "made", "best", "true", "real", "top", "and", "the", "for", "why",
    "how", "who", "its", "it", "of", "to", "in", "on", "at", "a", "an",
}

# 4+ letters: a made-up 4-letter noun ("eyes", "gold") must still be traceable.
_WORD_RE = re.compile(r"[a-z]{4,}")


def _title_is_grounded(title: str, context: str, fallback: str) -> bool:
    """Every substantial word in the title must come from the video's own text.

    A title that invents a claim is the same failure as narration that invents a
    fact. Note this checks FAITHFULNESS to the video, not truth: if the script
    says goldfish grow extra eyes, a title saying so is "grounded" — the error,
    if any, lives upstream in the writer.

    Deliberately conservative: an unrecognised word means we fall back to the
    pipeline's own title, which is always true to the content.
    """
    haystack = f"{context}\n{fallback}".lower()
    for word in _WORD_RE.findall(title.lower()):
        if word in _GENERIC_WORDS:
            continue
        # allow simple plural/verb inflections ("owners" <- "owner")
        stem = word.rstrip("s")
        if word in haystack or stem in haystack:
            continue
        log.warning(f"Title claim '{word}' is not in the video's content — rejecting "
                    f"title {title!r}")
        return False
    return True


def build_title(fallback: str, context: str = "", mode: str = "") -> str:
    """Ask the LLM for a title. Falls back to `fallback` on any failure —
    a cosmetic field must never take a finished render down with it, and a
    made-up title is worse than a plain one."""
    fallback = _clean_title(fallback) or "Untitled"
    if not context.strip():
        return fallback

    from_llm = None
    try:
        from modules.script_generator import _call_llm, _extract_json
        prompt = (
            f"Video type: {mode or 'short video'}\n"
            f"Working title: {fallback}\n\n"
            f"Content:\n{context[:1500]}\n\n"
            f"Write the best title for this video. Use only words and claims "
            f"that appear in the content above."
        )
        for attempt in (1, 2):
            raw = _call_llm(prompt, _TITLE_SYS, role="creative")
            candidate = _clean_title(_extract_json(raw).get("title", ""))
            if not candidate:
                continue
            if _title_is_grounded(candidate, context, fallback):
                from_llm = candidate
                break
            prompt += ("\n\nYour previous title invented something the video never "
                       "says. Use ONLY the content above.")
    except Exception as e:
        log.warning(f"Title LLM failed ({e}); using '{fallback}'")
        return fallback

    title = from_llm or fallback
    log.info(f"Title: {title}" + ("" if from_llm else " (LLM title rejected)"))
    return title


# ==============================================================================
# THUMBNAIL
# ==============================================================================

def _ffmpeg() -> Path:
    from modules.assembly import FFMPEG_EXE
    return FFMPEG_EXE


def grab_frame(video: Path, out_png: Path, skip_sec: float = 1.0) -> Optional[Path]:
    """Pull one representative frame.

    ffmpeg's `thumbnail` filter scores a batch of frames and returns the most
    representative — far better than frame 0, which is typically a fade-in or a
    title card. Skips the first second for the same reason.
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(_ffmpeg()), "-y", "-loglevel", "error",
        "-ss", str(skip_sec), "-i", str(video),
        "-vf", "thumbnail=100", "-frames:v", "1",
        str(out_png),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode == 0 and out_png.exists():
            return out_png
        log.warning(f"thumbnail frame grab failed: {r.stderr.strip()[:200]}")
    except Exception as e:
        log.warning(f"thumbnail frame grab errored: {e}")

    # Fallback: the very first frame is better than no thumbnail.
    try:
        r = subprocess.run(
            [str(_ffmpeg()), "-y", "-loglevel", "error", "-i", str(video),
             "-frames:v", "1", str(out_png)],
            capture_output=True, text=True, timeout=60)
        if r.returncode == 0 and out_png.exists():
            return out_png
    except Exception:
        pass
    return None


def _scrim(size: tuple[int, int], height_frac: float, opacity: int) -> Image.Image:
    """A bottom-up dark gradient so white text stays readable over any frame.

    Darkest at the very bottom, fading to nothing at the top of the band. Getting
    this backwards leaves a hard horizontal seam across the middle of the image.
    """
    w, h = size
    scrim = Image.new("L", (1, h), 0)
    band = max(1, int(h * height_frac))
    for i in range(band):                 # i = 0 at the bottom row
        y = h - 1 - i
        falloff = (1.0 - i / band) ** 1.6   # 1.0 at the bottom -> 0.0 at band top
        scrim.putpixel((0, y), int(opacity * falloff))
    # Blur along the seam; radius scales with the image so it never bands.
    return scrim.resize((w, h)).filter(ImageFilter.GaussianBlur(h * 0.01))


# Bold UI fonts (Segoe UI Bold, Arial Bold, DejaVu) carry no emoji glyphs, so a
# painted emoji comes out as a tofu box. The emoji stays in _title.txt, where it
# belongs — on the thumbnail it is dropped rather than drawn as ▯.
_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF←-⇿⌀-➿⬀-⯿︀-️‍]"
)


def strip_emoji(text: str) -> str:
    return re.sub(r"\s+", " ", _EMOJI_RE.sub("", text)).strip(" -–—·,")


def render_thumbnail(frame: Path, title: str, out_jpg: Path,
                     size: tuple[int, int] = THUMB_16X9) -> Optional[Path]:
    """Compose frame + darkening scrim + wrapped title -> JPEG."""
    try:
        title = strip_emoji(title) or title
        w, h = size
        img = _fit_cover(Image.open(frame).convert("RGB"), w, h)

        # Darken the lower third so the title reads. A fixed opacity is wrong:
        # over a white studio background the text still fought the plate, while
        # a dark frame went muddy. Scale the scrim to what is actually there.
        band = img.crop((0, int(h * 0.72), w, h)).convert("L")
        brightness = ImageStat.Stat(band).mean[0]
        opacity = int(min(245, max(170, 150 + brightness * 0.42)))
        dark = Image.new("RGB", (w, h), (0, 0, 0))
        img = Image.composite(dark, img, _scrim((w, h), 0.55, opacity))

        draw = ImageDraw.Draw(img)
        margin = int(w * 0.06)
        max_w = w - 2 * margin

        # Shrink until the title fits in at most 3 lines.
        size_px = int(h * (0.13 if w >= h else 0.075))
        for _ in range(14):
            font = _load_font(size_px, bold=True)
            lines = _wrap_text(draw, title, font, max_w)
            if len(lines) <= 3:
                break
            size_px = int(size_px * 0.88)
        else:
            lines = lines[:3]

        ascent, descent = font.getmetrics()
        line_h = ascent + descent
        gap = int(line_h * 0.16)
        total_h = len(lines) * line_h + (len(lines) - 1) * gap
        y = h - margin - total_h

        stroke = max(2, size_px // 14)
        for line in lines:
            tw = draw.textlength(line, font=font)
            x = (w - tw) / 2
            draw.text((x, y), line, font=font, fill=(255, 255, 255),
                      stroke_width=stroke, stroke_fill=(0, 0, 0))
            y += line_h + gap

        # No watermark is pasted here: the frame comes from a finished render,
        # which already has the Rexjaw logo burned in. Adding another gives you
        # two logos stacked in the corner.

        out_jpg.parent.mkdir(parents=True, exist_ok=True)
        quality = 92
        img.save(out_jpg, "JPEG", quality=quality, optimize=True)
        # YouTube rejects thumbnails over 2 MB.
        while out_jpg.stat().st_size > THUMB_MAX_BYTES and quality > 50:
            quality -= 8
            img.save(out_jpg, "JPEG", quality=quality, optimize=True)
        return out_jpg
    except Exception as e:
        log.warning(f"thumbnail render failed: {e}")
        return None


def _mascot_art(video: Path, title: str, context: str, mode: str,
                aspects: tuple, scene: str = "") -> dict:
    """Branded mascot artwork for this video, or {} when unavailable/disabled.

    Silent no-op until a mascot image exists, so this costs nothing before you
    add one. Never raises: a GPU hiccup here just means an ordinary thumbnail.
    """
    try:
        from modules import runtime_settings as rs
        if not rs.get_mascot_thumbnails_enabled():
            return {}
    except Exception:
        pass
    try:
        from modules import mascot
        wanted = tuple(a for a in aspects if a in mascot.NATIVE_ASPECTS)
        stem = video.with_suffix("").name

        # Reuse artwork already rendered for this video. Re-compositing a title
        # is milliseconds; a USO render is minutes, so a rerun (new title, new
        # font, tweaked scrim) must not go back to the GPU. Delete the
        # *_mascot_*.png files to force fresh art.
        cached = {a: video.parent / f"{stem}_mascot_{a}.png" for a in wanted}
        if cached and all(p.exists() for p in cached.values()):
            log.info(f"reusing cached mascot art for {stem}")
            return {a: p for a, p in cached.items()}

        # KEEP_MODEL_WARM is set by bulk tools that loop over many videos; a
        # cold reload of the 13.5 GB model costs ~4 min, a warm render 15 s.
        art = mascot.render_for_video(
            title=title, context=context, out_dir=video.parent,
            stem=stem, aspects=wanted,
            release_after=not KEEP_MODEL_WARM, scene=scene or None)
        return art or {}
    except Exception as e:
        log.warning(f"mascot art skipped ({e}); using a normal thumbnail")
        return {}


# ==============================================================================
# ONE CALL PER FINISHED VIDEO
# ==============================================================================

def attach(video: Path,
           fallback_title: str,
           context: str = "",
           description: str = "",
           mode: str = "",
           source_image: Optional[Path] = None,
           aspects: tuple = ("16x9", "9x16"),
           mascot_scene: str = "") -> dict:
    """Write title + thumbnails + description beside `video`. Never raises.

    `source_image` should be the CLEAN still the video was built from (a facts
    backdrop, shot 1's storyboard frame, a music scene). Prefer it: the rendered
    video has burned-in subtitles and read-along captions, so a frame grabbed
    from it puts the thumbnail title on top of the reel's own text.
    Falls back to a representative frame when no still is available.
    """
    kit: dict = {"video": str(video)}
    try:
        video = Path(video)
        if not video.exists():
            log.warning(f"publish kit: {video} does not exist")
            return kit
        stem = video.with_suffix("")

        title = build_title(fallback_title, context, mode)
        kit["title"] = title
        try:
            Path(f"{stem}_title.txt").write_text(title, encoding="utf-8")
            kit["title_file"] = f"{stem}_title.txt"
        except Exception as e:
            log.warning(f"could not write title file: {e}")

        if description:
            try:
                Path(f"{stem}_description.txt").write_text(description, encoding="utf-8")
                kit["description"] = description
                kit["description_file"] = f"{stem}_description.txt"
            except Exception as e:
                log.warning(f"could not write description file: {e}")

        # Preferred art: the mascot, rendered by USO into a scene about this
        # video. Falls back to the clean still, then to a frame of the render.
        mascot_art = _mascot_art(video, title, context, mode, aspects,
                                 scene=mascot_scene)
        if mascot_art:
            kit["thumb_source"] = "mascot"
            kit["mascot_scene"] = mascot_art.get("_scene", "")

        grabbed = None
        frame = None
        if not mascot_art:
            if source_image and Path(source_image).exists():
                frame = Path(source_image)
                kit["thumb_source"] = "still"
            else:
                grabbed = grab_frame(video, Path(f"{stem}_frame.png"))
                frame = grabbed
                kit["thumb_source"] = "video frame"

        for aspect in aspects:
            size = THUMB_16X9 if aspect == "16x9" else THUMB_9X16
            base = mascot_art.get(aspect) if mascot_art else frame
            if not base:
                continue
            out = Path(f"{stem}_thumb_{aspect}.jpg")
            if render_thumbnail(Path(base), title, out, size):
                kit[f"thumb_{aspect}"] = str(out)

        if grabbed:                          # only delete what we created
            try:
                grabbed.unlink(missing_ok=True)
            except Exception:
                pass

        try:
            Path(f"{stem}_publish.json").write_text(
                json.dumps(kit, indent=2), encoding="utf-8")
        except Exception as e:
            log.warning(f"could not write publish.json: {e}")

        n_thumbs = sum(1 for k in kit if k.startswith("thumb_") and k != "thumb_source")
        log.info(f"Publish kit ready for {video.name}: title + {n_thumbs} "
                 f"thumbnail(s) from the {kit.get('thumb_source', '?')}")
    except Exception as e:
        log.warning(f"publish kit failed for {video} (render is unaffected): {e}")
    return kit
