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

    # _WORD_RE only matches letters, so "Bees Have 9 Eyes" was 'grounded' while
    # the script said five. An invented number is the most visible lie we print.
    for num in re.findall(r"\d+", title):
        if num not in f"{context}\n{fallback}":
            log.warning(f"Title number '{num}' is not in the video's content — "
                        f"rejecting title {title!r}")
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



_HEADLINE_SYS = (
    "You write the few words printed ON a video thumbnail.\n"
    'Output ONLY valid JSON: {"headline": "...", "short": "..."}\n'
    "Rules:\n"
    "- headline: 2 to 4 words, at most 24 characters.\n"
    "- short: the SAME hook squeezed to at most 2 words and 14 characters, for\n"
    "  the tall phone thumbnail. It must still be a meaningful phrase on its own.\n"
    "  Never a truncation: 'Bees Have 5 Eyes' -> '5 Eyes', not 'Bees Have'.\n"
    "- Neither is the video title: they are the hook you read at a glance.\n"
    "- Both must read as natural English, not keywords jammed together:\n"
    "  'Bees 230 Flaps' is wrong, '230 Flaps A Second' is right.\n"
    "- Must be true to the content. Never invent a fact or a number.\n"
    "- No punctuation except a question mark. No emoji, no hashtags, no quotes.\n"
    'Examples: {"headline": "230 Flaps A Second", "short": "230 Flaps"}\n'
    '          {"headline": "Bees Have 5 Eyes", "short": "5 Eyes"}'
)

HEADLINE_MAX = 24
HEADLINE_SHORT_MAX = 14


def _clean_headline(text: str) -> str:
    t = strip_emoji((text or "").strip().strip('"').strip("'"))
    t = re.sub(r"[#*_`]", "", t)
    t = re.sub(r"\s+", " ", t).strip(" .,:;-")
    return t


CONTEXT_MAX = 2000


def _kit_context(kit: dict) -> str:
    """What the video actually says, for grounding a rerolled headline.

    attach() grounds against the narration but used to throw it away, so a reroll
    fell back to the description — a 222-character caption. The hook
    "230 Flaps A Second" was then rejected because the caption never says
    "second", only the narration does. Old kits have no stored context and still
    get the caption; new ones ground against the same text attach used.
    """
    if kit.get("context"):
        return kit["context"]
    parts = [kit.get("description", "")]
    df = kit.get("description_file")
    if not parts[0] and df and Path(df).exists():
        try:
            parts[0] = Path(df).read_text(encoding="utf-8")
        except Exception:
            pass
    parts.append(kit.get("title", ""))
    return "\n".join(p for p in parts if p).strip()


def _short_fallback(headline: str) -> str:
    """A short form for the portrait frame, when the LLM didn't give a usable one.

    Chopping trailing words is how you get "Bees Have" out of "Bees Have 5 Eyes",
    so we only accept a prefix that still says something: it has to end on a word
    that carries meaning, not on a verb or article. If nothing qualifies, return
    the full headline — a smaller point size beats a mutilated sentence.
    """
    words = headline.split()
    if len(headline) <= HEADLINE_SHORT_MAX:
        return headline

    # The payload sits at the END of "Bees Have [5 Eyes]" and at the START of
    # "[230 Flaps] A Second", so try both, tail first. A fragment that begins or
    # ends on a filler word ("A Second", "Bees Have") says nothing.
    candidates = [words[-2:], words[:2], words[-1:], words[:1]]
    for cand_words in candidates:
        cand = " ".join(cand_words)
        if not cand or len(cand) > HEADLINE_SHORT_MAX:
            continue
        if cand_words[0].lower() in _GENERIC_WORDS:
            continue
        if cand_words[-1].lower() in _GENERIC_WORDS:
            continue
        return cand
    return headline


def build_headline(title: str, context: str = "") -> tuple:
    """The words printed ON the thumbnail: (headline, short).

    Not the upload title. A YouTube title runs to 80 characters; the thumbnail
    wants a hook you can read at a glance, and a 9:16 frame holding a full-body
    mascot has room for about two words of giant type — given four, the image
    model silently drops the last two. So we ask for both forms at once and let
    each aspect use the one it can render whole.

    Grounded like the title: an invented word or number is rejected, and we fall
    back to the title's own opening words.
    """
    fallback = _clean_headline(" ".join(_clean_title(title).split()[:4]))[:HEADLINE_MAX]
    if not context.strip():
        return fallback, _short_fallback(fallback)
    try:
        from modules.script_generator import _call_llm, _extract_json
        prompt = (f"Video title: {title}\n\nContent:\n{context[:900]}\n\n"
                  f"Write the thumbnail headline.")
        for _ in (1, 2):
            raw = _call_llm(prompt, _HEADLINE_SYS, role="creative")
            data = _extract_json(raw)
            cand = _clean_headline(data.get("headline", ""))
            short = _clean_headline(data.get("short", ""))
            if not cand or len(cand) > HEADLINE_MAX:
                continue
            if not _title_is_grounded(cand, context, title):
                log.info(f"Headline rejected (ungrounded): {cand}")
                prompt += ("\n\nThat invented something the video never says. "
                           "Use the video's own words and numbers.")
                continue
            if (not short or len(short) > HEADLINE_SHORT_MAX
                    or not _title_is_grounded(short, context, title)):
                short = _short_fallback(cand)
            log.info(f"Headline: {cand}  (short: {short})")
            return cand, short
    except Exception as e:
        log.warning(f"Headline LLM failed ({e}); using '{fallback}'")
    return fallback, _short_fallback(fallback)


# ==============================================================================
# THUMBNAIL
# ==============================================================================

# Aspect suffix on every finished video: facts_20260710_184745_9x16.mp4
_ASPECT_RE = re.compile(r"_(9x16|16x9|1x1)(?:_|$)")

# The two thumbnails a platform will take. A 1x1 video still gets the portrait
# cover — nothing consumes a square thumbnail.
THUMB_ASPECTS = ("16x9", "9x16")


def _probe_orientation(video: Path) -> str:
    """9x16 or 16x9, straight from the stream dimensions."""
    from modules.assembly import FFPROBE_EXE
    try:
        out = subprocess.run(
            [str(FFPROBE_EXE), "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x",
             str(video)],
            capture_output=True, text=True, timeout=60).stdout.strip()
        w, h = (int(x) for x in out.split("x")[:2])
        return "9x16" if h > w else "16x9"
    except Exception as e:
        log.warning(f"could not probe {video.name} ({e}); assuming 9x16")
        return "9x16"


def video_aspects(video: Path) -> tuple:
    """Which thumbnails this render actually needs.

    Every caller used to take the default ("16x9", "9x16"), so a facts reel —
    which only ever exists as 9x16 — paid a second ~25 s Qwen render for a
    landscape thumbnail nothing could use, and horror (16x9 only) got a portrait
    one. A render's thumbnails should match the videos it produced, so we look
    for the siblings assembly wrote: <base>_9x16.mp4, _16x9.mp4, _1x1.mp4.
    """
    base = _ASPECT_RE.sub("_", video.stem).rstrip("_")
    found = set()
    for sibling in video.parent.glob(f"{base}_*.mp4"):
        m = _ASPECT_RE.search(sibling.stem + "_")
        if m:
            found.add(m.group(1))
    wanted = tuple(a for a in THUMB_ASPECTS if a in found)
    # No aspect suffix at all (manual mode names its own files): ask the stream.
    return wanted or (_probe_orientation(video),)


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


def save_thumbnail(frame: Path, out_jpg: Path,
                   size: tuple[int, int] = THUMB_16X9) -> Optional[Path]:
    """Cover-fit and save, painting nothing. Used when the image model already
    rendered the headline into the artwork — an overlay on top of baked type
    would double the title."""
    try:
        img = _fit_cover(Image.open(frame).convert("RGB"), *size)
        out_jpg.parent.mkdir(parents=True, exist_ok=True)
        quality = 92
        img.save(out_jpg, "JPEG", quality=quality, optimize=True)
        while out_jpg.stat().st_size > THUMB_MAX_BYTES and quality > 50:
            quality -= 8
            img.save(out_jpg, "JPEG", quality=quality, optimize=True)
        return out_jpg
    except Exception as e:
        log.warning(f"thumbnail save failed: {e}")
        return None


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
                aspects: tuple, scene: str = "", headline: str = "",
                headline_short: str = "") -> dict:
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
        flags_path = video.parent / f"{stem}_mascot.json"
        if cached and all(p.exists() for p in cached.values()):
            # The cache MUST carry `_baked_headline` with it. Returning bare paths
            # made every rerun think the art had no headline in it, so it painted
            # the title over type the model had already drawn — two headlines, one
            # thumbnail. Art without its flags is unusable, so re-render instead of
            # guessing (a warm render is ~15s; a doubled title ships forever).
            if flags_path.exists():
                try:
                    flags = json.loads(flags_path.read_text(encoding="utf-8"))
                    log.info(f"reusing cached mascot art for {stem} "
                             f"(headline {'baked' if flags.get('baked') else 'overlaid'})")
                    art = {a: p for a, p in cached.items()}
                    art["_baked_headline"] = bool(flags.get("baked"))
                    art["_scene"] = flags.get("scene", "")
                    art["_seed"] = flags.get("seed")
                    art["_backend"] = flags.get("backend", "")
                    return art
                except Exception as e:
                    log.warning(f"mascot art flags unreadable ({e}); re-rendering")
            else:
                log.info(f"cached mascot art for {stem} has no flags; re-rendering")

        # KEEP_MODEL_WARM is set by bulk tools that loop over many videos; a
        # cold reload of the 13.5 GB model costs ~4 min, a warm render 15 s.
        art = mascot.render_for_video(
            title=title, context=context, out_dir=video.parent,
            stem=stem, aspects=wanted,
            release_after=not KEEP_MODEL_WARM, scene=scene or None,
            headline=headline or None, headline_short=headline_short or None)
        return art or {}
    except mascot.MascotGpuFault as e:
        # A single video may still ship with a still-frame thumbnail, but the
        # caller must know the GPU is dead — a bulk run has to stop here.
        log.error(f"{e}")
        return {"_fatal": str(e)}
    except Exception as e:
        log.warning(f"mascot art skipped ({e}); using a normal thumbnail")
        return {}


# ==============================================================================
# EDIT + REGENERATE — when the bot's idea isn't the one you wanted
# ==============================================================================

def load_kit(video) -> dict:
    """The publish.json written beside a video, or {} if there isn't one."""
    p = Path(f"{Path(video).with_suffix('')}_publish.json")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning(f"unreadable publish kit {p.name}: {e}")
        return {}


def find_video(video_id: str):
    """Resolve a bare id ('20260709_201218') or a filename to a final MP4.

    Prefers the 9x16 master over its _discord re-encode, and never matches a
    PLACEHOLDER_* render (reels written while the LLM was down).
    """
    final_dir = PROJECT_ROOT / "04_Outputs" / "final"
    if not final_dir.exists():
        return None
    vid = str(video_id).strip()
    exact = final_dir / vid
    if exact.exists() and exact.suffix == ".mp4":
        return exact
    cands = [p for p in final_dir.glob(f"*{vid}*.mp4")
             if not p.name.startswith("PLACEHOLDER_")
             and not p.stem.endswith("_discord")]
    if not cands:
        return None
    nine = [p for p in cands if "9x16" in p.name]
    return (nine or cands)[0]


def latest_videos(limit: int = 10) -> list:
    """Most recent finals that have a publish kit, newest first."""
    final_dir = PROJECT_ROOT / "04_Outputs" / "final"
    if not final_dir.exists():
        return []
    vids = [p for p in final_dir.glob("*.mp4")
            if not p.name.startswith("PLACEHOLDER_")
            and not p.stem.endswith("_discord")
            and Path(f"{p.with_suffix('')}_publish.json").exists()]
    return sorted(vids, key=lambda p: p.stat().st_mtime, reverse=True)[:limit]


def regenerate_thumbnail(video, scene: str = "", title=None, seed=None,
                         aspects: Optional[tuple] = None,
                         headline: Optional[str] = None) -> dict:
    """Re-render one video's mascot artwork, optionally from YOUR scene.

    scene=""     -> ask the LLM for a fresh one (a re-roll, with a new seed)
    scene="..."  -> render exactly what you wrote
    title=...    -> repaint this title instead of the video's current one
    headline=... -> the words IN the art. Yours is used as written, no grounding
                    check: you are allowed to say what you mean.

    Raises ValueError on an unsafe scene and MascotGpuFault when the GPU is
    dead. Both are things the caller must SHOW you, not swallow — unlike
    attach(), which must never take a finished render down with it.
    """
    from modules import mascot

    video = Path(video)
    if not video.exists():
        raise FileNotFoundError(f"No such video: {video}")
    stem = video.with_suffix("")

    scene = (scene or "").strip()
    if scene:
        bad = mascot.scene_violation(scene)
        if bad:
            raise ValueError(f"That scene contains {bad!r}, which the mascot "
                             f"never appears with. Rewrite it without that.")
        if "mascot" not in scene.lower():
            scene = f"the mascot character {scene}"

    kit = load_kit(video)
    if title is None:
        title = kit.get("title", "")
    if not title:
        tf = Path(f"{stem}_title.txt")
        title = tf.read_text(encoding="utf-8").strip() if tf.exists() else video.stem

    # Re-render exactly the thumbnails this video already has, so a reroll never
    # invents a landscape cover for a portrait-only reel.
    if not aspects:
        have = tuple(a for a in THUMB_ASPECTS if kit.get(f"thumb_{a}"))
        aspects = have or video_aspects(video)

    # Cached art would be reused verbatim — the point of the cache, and exactly
    # wrong here.
    for a in aspects:
        Path(f"{stem}_mascot_{a}.png").unlink(missing_ok=True)

    # Reuse the stored hook when there is one: rerolling it would load Ollama
    # only to change the words under a thumbnail the user asked to keep.
    #
    # When rerolling, ground it against what the VIDEO says, not the mascot
    # scene. Passing the scene made grounding reject "230 Flaps A Second" — the
    # scene text never mentions 230, so a perfectly faithful hook looked invented.
    if headline:
        headline = _clean_headline(headline)
        headline_short = _short_fallback(headline)
    elif kit.get("headline_text"):
        headline = kit["headline_text"]
        headline_short = kit.get("headline_short") or _short_fallback(headline)
    else:
        headline, headline_short = build_headline(title, _kit_context(kit))
    kit["headline_text"] = headline
    kit["headline_short"] = headline_short

    art = mascot.render_for_video(
        title=title, context=kit.get("mascot_scene", "") or title,
        out_dir=video.parent, stem=stem.name, aspects=aspects,
        seed=seed, scene=scene or None, headline=headline,
        headline_short=headline_short)
    if not art:
        raise RuntimeError("Mascot render produced nothing — is ComfyUI up, "
                           "and is a mascot image installed?")

    baked = bool(art.get("_baked_headline"))
    for aspect in aspects:
        base = art.get(aspect)
        if not base:
            continue
        size = THUMB_16X9 if aspect == "16x9" else THUMB_9X16
        out = Path(f"{stem}_thumb_{aspect}.jpg")
        done = (save_thumbnail(Path(base), out, size) if baked
                else render_thumbnail(Path(base), title, out, size))
        if done:
            kit[f"thumb_{aspect}"] = str(out)

    kit.update({"video": str(video), "title": title, "thumb_source": "mascot",
                "mascot_scene": art.get("_scene", scene),
                "mascot_seed": art.get("_seed"),
                "headline": "baked into the art" if baked else "overlaid"})
    try:
        Path(f"{stem}_publish.json").write_text(json.dumps(kit, indent=2),
                                                encoding="utf-8")
        Path(f"{stem}_title.txt").write_text(title, encoding="utf-8")
    except Exception as e:
        log.warning(f"could not update publish kit files: {e}")
    log.info(f"Thumbnail regenerated for {video.name}: {kit['mascot_scene']}")
    return kit


# ==============================================================================
# ONE CALL PER FINISHED VIDEO
# ==============================================================================

def attach(video: Path,
           fallback_title: str,
           context: str = "",
           description: str = "",
           mode: str = "",
           source_image: Optional[Path] = None,
           aspects: Optional[tuple] = None,
           mascot_scene: str = "",
           thumbnail: bool = True) -> dict:
    """Write title + thumbnails + description beside `video`. Never raises.

    `thumbnail=False` writes the title and description but skips the thumbnail
    image entirely — no LLM headline, no Qwen render. Use it when a mode's
    thumbnail toggle is off; the reel still gets its paste-ready title.

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
        # A 9x16-only reel has no use for a landscape thumbnail, and rendering
        # one costs a second ~25 s Qwen pass.
        aspects = tuple(aspects) if aspects else video_aspects(video)
        kit["aspects"] = list(aspects)

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

        if not thumbnail:
            # Title + description only. Skips the LLM headline and the Qwen
            # render — the whole cost of a thumbnail — when the mode's toggle
            # is off.
            kit["thumb_source"] = "disabled"
            log.info(f"publish kit: thumbnail disabled for {video.name}")
        else:
            # The few words that go ON the thumbnail — not the upload title.
            # Written now, while Ollama is still resident: _mascot_art evicts it
            # to load Qwen.
            headline, headline_short = build_headline(title, context)
            kit["headline_text"] = headline
            kit["headline_short"] = headline_short
            # Keep what we grounded against, so a later reroll grounds the same way.
            if context.strip():
                kit["context"] = context[:CONTEXT_MAX]

            # Preferred art: the mascot, rendered by Qwen into a scene about this
            # video. Falls back to the clean still, then to a frame of the render.
            mascot_art = _mascot_art(video, title, context, mode, aspects,
                                     scene=mascot_scene, headline=headline,
                                     headline_short=headline_short)
            if mascot_art.get("_fatal"):
                kit["mascot_fatal"] = mascot_art["_fatal"]
                mascot_art = {}
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

            baked = bool(mascot_art.get("_baked_headline")) if mascot_art else False
            kit["headline"] = "baked into the art" if baked else "overlaid"
            for aspect in aspects:
                size = THUMB_16X9 if aspect == "16x9" else THUMB_9X16
                base = mascot_art.get(aspect) if mascot_art else frame
                if not base:
                    continue
                out = Path(f"{stem}_thumb_{aspect}.jpg")
                done = (save_thumbnail(Path(base), out, size) if baked
                        else render_thumbnail(Path(base), title, out, size))
                if done:
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
