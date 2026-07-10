"""
Claw Bot — Music Video Pipeline (orchestrator)

Drives the music-video render once a song JSON is approved:
  1. ACE-Step renders the song audio (audio_backend).
  2. Z-Image renders one mood-fit still per scene (image_backend) — NO character
     consistency, just the song's visual_style suffix.
  3. musicvideo_assembly stitches Ken Burns clips + the song into 3 aspect MP4s.

Core is front-end agnostic (returns paths); claw_bot / dashboard post the result.
"""

import logging
import random
import sys
from pathlib import Path
from typing import Callable, Optional

_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modules import audio_backend
from modules import image_backend
from modules import gpu_utils
from modules import runtime_settings as rs
from modules import song_generator as song_gen
from modules import musicvideo_assembly as mva
from modules.script_generator import get_style_description

log = logging.getLogger("claw_bot.musicvideo_pipeline")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
STILLS_DIR = PROJECT_ROOT / "04_Outputs" / "storyboards"   # reuse storyboard tree


import re as _re

# Vocal descriptors the LLM may have baked into ace_tags. We STRIP these before
# injecting the chosen vocal so the override actually wins (appending left the
# old "female vocal" in place → ACE-Step kept singing female regardless).
_VOCAL_DESC = _re.compile(
    # "female/male" + any vocal-ish role (vocal, voice, singer, rap, rapper,
    # rapping, sings, vocalist) — catches "female rap" too, not just "female vocal".
    r"\b(?:fe)?male\s+(?:vocals?|voice|voices|singer|singers|vocalist|"
    r"rap|rapper|rapping|sings|singing)\b"
    r"|\b(?:duet|choir)\b"                              # duet / choir (vocal descriptors)
    r"|\b(?:rap|solo)\s+vocals?\b"                      # rap/solo vocal (keeps bare genre 'rap')
    r"|\binstrumental\b|\bno\s+vocals?\b",              # instrumental / no vocals
    _re.I,
)


def _vocal_tags(song: dict) -> str:
    """Force the chosen vocal type into the ACE tag string: strip any vocal
    descriptor the LLM wrote, then put the chosen one FIRST (ACE-Step weights
    early tags higher)."""
    tags = (song.get("ace_tags") or "").strip()
    vocal = (song.get("vocal_type") or "auto").strip().lower()
    if vocal in ("auto", ""):
        return tags
    # remove existing vocal descriptors, tidy leftover commas
    tags = _VOCAL_DESC.sub("", tags)
    tags = _re.sub(r"\s*,\s*(?:,\s*)+", ", ", tags).strip(" ,")
    lead = "instrumental, no vocals" if vocal == "instrumental" else f"{vocal} vocal"
    return f"{lead}, {tags}".strip(", ") if tags else lead


def _effective_lyrics(song: dict) -> str:
    """For a true instrumental, ACE-Step wants the '[inst]' marker in the lyrics
    field — NOT a real lyric sheet (it would sing it) and NOT an empty string
    (that produced noisy/incoherent output). '[inst]' tells the model to compose
    music with no vocals."""
    if (song.get("vocal_type") or "").strip().lower() == "instrumental":
        return "[inst]"
    return song.get("lyrics", "")


def _scene_prompt(song: dict, scene: dict) -> str:
    """Build the final image prompt for one scene: scene + visual world + style."""
    style_id = song_gen.VISUAL_STYLE_TO_STYLE_ID.get(song.get("visual_style", "cartoon"), "cartoon")
    suffix = (get_style_description(style_id) or {}).get("prompt_suffix", "")
    parts = [scene.get("image_prompt", "").strip()]
    world = (song.get("visual_world") or "").strip()
    if world:
        parts.append(world)
    if suffix:
        parts.append(suffix)
    return ", ".join(p for p in parts if p)


def render_musicvideo(
    song: dict,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """Full render: audio -> stills -> assembled music videos. Returns the
    musicvideo_assembly outputs dict ({song_id, 9x16, 16x9, 1x1, ...})."""
    def _p(msg: str):
        log.info(msg)
        if progress_cb:
            try:
                progress_cb(msg)
            except Exception:
                pass

    song_id = song.get("song_id") or song.get("_id")
    scenes = song.get("scenes", [])
    if not scenes:
        raise ValueError("Song has no scenes to render.")

    aspect = rs.get_effective_aspect_ratio()

    # ---- 1. Song audio (ACE-Step) ----
    _p("🎵 rendering song (ACE-Step)...")
    gpu_utils.ensure_vram_free()
    ab = audio_backend.get_active_backend()
    song_audio = ab.generate(
        tags=_vocal_tags(song),
        lyrics=_effective_lyrics(song),
        duration_sec=float(song.get("duration_sec", 120)),
        bpm=(rs.get_tempo_override() or song.get("bpm")),
        keyscale=song.get("keyscale"),
        language=song.get("language", "en"),
        output_filename=f"song_{song_id}.mp3",
    )
    _p(f"🎵 song ready: {song_audio.name}")

    # ---- 2. Scene stills ----
    # USO mode (default): render scene 0 as the visual ANCHOR, then reference it
    # for every later scene so the protagonist/look stays consistent across the
    # whole music video. USO off -> previous behaviour (active backend, no
    # consistency). Falls back cleanly if USO is unhealthy.
    gpu_utils.free_comfyui_vram()
    use_uso = False
    if rs.get_music_image_backend() == "zturbo":
        # Fast + reliable stills, no cross-scene consistency (like facts mode).
        try:
            img = image_backend.get_named_backend("comfyui_zimage_turbo")
            ok, _msg = img.health_check()
            if ok:
                _p("music stills via Z-Image Turbo (fast, no anchor)")
            else:
                _p(f"Z-Image Turbo unhealthy ({_msg}); using active backend.")
                img = image_backend.get_active_backend()
        except Exception as e:
            _p(f"Z-Image Turbo unavailable ({e}); using active backend.")
            img = image_backend.get_active_backend()
    elif rs.get_uso_mode_enabled():
        use_uso = True
        try:
            img = image_backend.get_named_backend("comfyui_uso")
            ok, _msg = img.health_check()
            if not ok:
                _p(f"USO unhealthy ({_msg}); music stills use active backend.")
                img = image_backend.get_active_backend()
                use_uso = False
        except Exception as e:
            _p(f"USO unavailable ({e}); music stills use active backend.")
            img = image_backend.get_active_backend()
            use_uso = False
    else:
        img = image_backend.get_active_backend()

    out_dir = STILLS_DIR / f"song_{song_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_images: list[Path] = []
    anchor: Optional[Path] = None
    for i, sc in enumerate(scenes):
        _p(f"🖼️ scene {i+1}/{len(scenes)}...")
        prompt = _scene_prompt(song, sc)
        seed = random.randint(1, 2**31 - 1)
        gen_kwargs = dict(
            prompt=prompt,
            aspect_ratio=aspect,
            seed=seed,
            output_filename=str(out_dir / f"scene_{i:03d}.png"),
        )
        if use_uso and anchor is not None:
            path = img.generate(reference_image=anchor, **gen_kwargs)
        else:
            path = img.generate(**gen_kwargs)
        p = Path(path)
        scene_images.append(p)
        if use_uso and anchor is None:
            anchor = p  # first scene becomes the consistency anchor

    # ---- 3. Assemble Ken Burns + song ----
    _p("🎬 assembling music video...")
    gpu_utils.free_comfyui_vram()
    outputs = mva.assemble_musicvideo(song, song_audio, scene_images, progress_cb)
    outputs["song_audio"] = song_audio

    # Upload kit (title + thumbnail). Built from a scene still, not the render:
    # the finished video has lyric captions and the logo burned in.
    _p("🖼️ building title + thumbnail…")
    try:
        from modules import publish_kit
        video = outputs.get("9x16") or outputs.get("16x9")
        if video:
            kit = publish_kit.attach(
                Path(video),
                fallback_title=song.get("title", "Music Video"),
                context=f"Song style: {song.get('song_style', '')}\n"
                        f"Theme: {song.get('theme', '')}\n"
                        f"Lyrics:\n{(song.get('lyrics') or '')[:900]}",
                description=(song.get("description") or "").strip(),
                mode="music video",
                source_image=Path(scene_images[0]) if scene_images else None,
            )
            outputs["publish"] = kit
            outputs["title"] = kit.get("title")
            outputs["thumbnail"] = kit.get("thumb_9x16") or kit.get("thumb_16x9")
    except Exception as e:
        log.warning(f"publish kit skipped: {e}")

    _p("✅ music video complete")
    return outputs
