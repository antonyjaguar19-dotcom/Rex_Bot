"""
Claw Bot — mascot library

More than one mascot. A mascot used to be a single file, `assets/mascot.png`,
plus four optional angle views beside it — so the pipeline could only ever star
one character. This module turns that into a shelf you pick from:

    02_Agent/assets/mascots/
        jaguar-cub/
            mascot.png              <- primary reference (required)
            mascot_front.png        <- optional angles, same names as before
            mascot_threequarter.png
            mascot_side.png
            mascot_back.png
            voice.wav               <- optional: THIS mascot's cloned voice
            meta.json               <- {"name": "Jaguar Cub"} display name
        robot-owl/
            ...

The one that facts mode stars is `runtime_settings.active_mascot` (an id = the
folder name). Everything else — thumbnails, presenter stills, the cloned voice —
follows whatever is active, so switching mascot is one dropdown, not a file swap.

Back-compat: the old flat layout (`assets/mascot.png` + friends) still works and
is still what `mascot.py` falls back to when no library exists. `migrate()` copies
those files into `mascots/default/` once; it never deletes the originals, but once
`mascots/` exists the flat files are no longer consulted — so deleting every
mascot really does leave you with none, instead of resurrecting the old one.
"""

import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

log = logging.getLogger("claw_bot.mascot_library")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
FFPROBE_EXE = PROJECT_ROOT / "00_Tools" / "ffmpeg" / "bin" / "ffprobe.exe"

# The primary reference, in priority order — the first that exists wins.
PRIMARY_NAMES = ("mascot.png", "mascot.jpg", "mascot.jpeg", "mascot.webp")

# Optional extra views. Same filenames the flat layout used, so a folder is just
# the old assets dir with a name on it.
ANGLE_NAMES = (
    "mascot_front.png",
    "mascot_threequarter.png",
    "mascot_side.png",
    "mascot_back.png",
)

# The clip Chatterbox clones this mascot's voice from.
VOICE_NAMES = ("voice.wav", "voice.mp3", "voice.flac")

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
AUDIO_EXTS = (".wav", ".mp3", ".flac")

# Upload slots the UI offers. role -> filename it lands as.
ROLE_FILES = {
    "main": "mascot.png",
    "front": "mascot_front.png",
    "threequarter": "mascot_threequarter.png",
    "side": "mascot_side.png",
    "back": "mascot_back.png",
    "voice": "voice.wav",
}

# What a NEW mascot is asked for. The front view is the one the renderer actually
# conditions on (mascot_refs() hands over ONE reference by default — three of them
# made Qwen copy a reference's stance instead of acting out the scene). The other
# three are stored for the paths that ask for more, and they cost nothing.
INTAKE_VIEWS = ("front", "threequarter", "side", "back")
REQUIRED_VIEWS = ("front",)

# The voice is CLONED, not trained: Chatterbox reads this clip and speaks in its
# timbre — there is no training run and no model to fit. What it needs is a clean
# 10-second-ish sample: one speaker, no music, no room echo. Under ~5s it has too
# little to copy; past ~30s it is just a bigger file (and a slower load).
VOICE_IDEAL_SEC = 10.0
VOICE_MIN_SEC = 5.0
VOICE_MAX_SEC = 30.0

DEFAULT_ID = "default"


def _assets_dir() -> Path:
    """Late import on purpose: mascot.py imports us back, and the tests move its
    ASSETS_DIR to a tmp path — reading it at call time keeps them honest."""
    from modules import mascot
    return Path(mascot.ASSETS_DIR)


def mascots_dir() -> Path:
    return _assets_dir() / "mascots"


def library_exists() -> bool:
    """True once anything has been put on the shelf. While this is False,
    mascot.py keeps using the old flat `assets/mascot.png` layout."""
    return mascots_dir().is_dir()


# ==============================================================================
# IDs
# ==============================================================================

def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-")
    return s[:40] or "mascot"


def _unique_id(name: str) -> str:
    base = slugify(name)
    mid, n = base, 2
    while (mascots_dir() / mid).exists():
        mid = f"{base}-{n}"
        n += 1
    return mid


# ==============================================================================
# READ
# ==============================================================================

def _first(d: Path, names) -> Optional[Path]:
    for n in names:
        p = d / n
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def _display_name(d: Path) -> str:
    meta = d / "meta.json"
    if meta.exists():
        try:
            got = (json.loads(meta.read_text(encoding="utf-8")) or {}).get("name")
            if got:
                return str(got)
        except Exception as e:
            log.warning(f"unreadable meta.json for {d.name}: {e}")
    return d.name.replace("-", " ").title()


def describe(mid: str) -> Optional[dict]:
    """Everything the UI needs about one mascot, or None if it isn't there."""
    d = mascots_dir() / mid
    if not d.is_dir():
        return None
    primary = _first(d, PRIMARY_NAMES)
    angles = [d / n for n in ANGLE_NAMES
              if (d / n).exists() and (d / n).stat().st_size > 0]
    voice = _first(d, VOICE_NAMES)
    return {
        "id": mid,
        "name": _display_name(d),
        "dir": d,
        "image": primary or (angles[0] if angles else None),
        "angles": angles,
        "voice": voice,
        "ready": bool(primary or angles),
    }


def list_mascots() -> list:
    """Every mascot on the shelf, alphabetical. Empty list when there's none."""
    root = mascots_dir()
    if not root.is_dir():
        return []
    out = []
    for d in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not d.is_dir() or d.name.startswith(("_", ".")):
            continue
        got = describe(d.name)
        if got:
            out.append(got)
    return out


def get_active_id() -> Optional[str]:
    """The mascot facts mode stars right now.

    A stale id in the settings file (its folder was deleted) must not silently
    disable the mascot — fall through to the first one on the shelf instead.
    """
    from modules import runtime_settings as rs
    ids = [m["id"] for m in list_mascots()]
    if not ids:
        return None
    want = rs.get_active_mascot()
    if want in ids:
        return want
    return DEFAULT_ID if DEFAULT_ID in ids else ids[0]


def set_active_id(mid: str) -> None:
    from modules import runtime_settings as rs
    if mid and describe(mid) is None:
        raise ValueError(f"no mascot named {mid!r}")
    rs.set_active_mascot(mid)


def active() -> Optional[dict]:
    mid = get_active_id()
    return describe(mid) if mid else None


def mascot_dir(mid: Optional[str] = None) -> Optional[Path]:
    got = describe(mid) if mid else active()
    return got["dir"] if got else None


def primary_image(mid: Optional[str] = None) -> Optional[Path]:
    """The active mascot's main reference, or None when the library is unused
    or empty. mascot.mascot_path() falls back to the flat layout on None."""
    got = describe(mid) if mid else active()
    return got["image"] if got else None


def refs(mid: Optional[str] = None, max_refs: int = 1) -> list:
    """Reference images to condition on, best first — the angles if they exist,
    else the single primary. Same rule the flat layout used (see mascot.mascot_refs
    for why ONE front view beats three)."""
    got = describe(mid) if mid else active()
    if not got:
        return []
    if got["angles"]:
        return got["angles"][:max_refs]
    return [got["image"]] if got["image"] else []


def voice_ref(mid: Optional[str] = None) -> Optional[Path]:
    """The clip Chatterbox clones for this mascot.

    A mascot that carries its own voice.wav uses it — that is the whole point of
    a shelf: a different character should not speak in the last one's voice. With
    none, we fall back to the global setting (`assets/mascot_voice.wav` unless
    overridden), which is what every reel used before the library existed.
    """
    from modules import runtime_settings as rs
    got = describe(mid) if mid else active()
    if got and got["voice"]:
        return got["voice"]
    return rs.get_mascot_voice_ref()


# ==============================================================================
# WRITE
# ==============================================================================

def create(name: str, image: Optional[Path] = None,
           image_bytes: Optional[bytes] = None,
           filename: str = "mascot.png") -> str:
    """Put a new mascot on the shelf. Returns its id.

    An image is optional at creation — a mascot with no art is listed but not
    `ready`, and is_available() will say so rather than rendering something odd.
    """
    mid = _unique_id(name)
    d = mascots_dir() / mid
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(
        json.dumps({"name": (name or mid).strip()}), encoding="utf-8")
    if image or image_bytes:
        put_file(mid, "main", src=image, data=image_bytes, filename=filename)
    log.info(f"mascot created: {mid} ({name})")
    return mid


def audio_duration(path) -> Optional[float]:
    """Length of a clip in seconds, or None when ffprobe can't say.

    None means "unknown", never "bad" — a missing ffprobe must not block someone
    from adding a mascot.
    """
    try:
        out = subprocess.run(
            [str(FFPROBE_EXE), "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, check=True, timeout=60)
        return float(out.stdout.strip())
    except Exception as e:
        log.warning(f"could not probe {Path(path).name}: {e}")
        return None


def voice_warning(path) -> Optional[str]:
    """Why this clip will clone badly, or None if it's fine.

    A warning, not a veto: a 4-second clip still produces a voice, just a thinner
    one, and the owner is the one listening to it.
    """
    secs = audio_duration(path)
    if secs is None:
        return None
    if secs < VOICE_MIN_SEC:
        return (f"that clip is {secs:.0f}s — under {VOICE_MIN_SEC:.0f}s the clone "
                f"has little to copy. Aim for ~{VOICE_IDEAL_SEC:.0f}s.")
    if secs > VOICE_MAX_SEC:
        return (f"that clip is {secs:.0f}s — anything past {VOICE_MAX_SEC:.0f}s just "
                f"loads slower. ~{VOICE_IDEAL_SEC:.0f}s is plenty.")
    return None


def create_from_intake(name: str, views: dict, voice: Optional[Path] = None) -> str:
    """Create a mascot from the full intake: its views + its own voice clip.

    `views` maps a view name (INTAKE_VIEWS) to a file on disk. The FRONT view is
    required and is also installed as the primary reference — `mascot_path()` needs
    one, and a mascot whose primary is a side view renders a character seen from
    the side in every thumbnail.

    Nothing is half-created: if any file is rejected the folder is removed again,
    so a failed intake cannot leave a nameless, artless mascot on the shelf.
    """
    missing = [v for v in REQUIRED_VIEWS if not views.get(v)]
    if missing:
        raise ValueError(f"missing required view(s): {', '.join(missing)}")

    mid = create(name)
    try:
        put_file(mid, "main", src=Path(views["front"]))
        for view in INTAKE_VIEWS:
            src = views.get(view)
            if src:
                put_file(mid, view, src=Path(src))
        if voice:
            put_file(mid, "voice", src=Path(voice))
    except Exception:
        try:
            shutil.rmtree(mascots_dir() / mid)
        except Exception as e:
            log.warning(f"could not clean up half-built mascot {mid}: {e}")
        raise
    log.info(f"mascot {mid}: intake complete "
             f"({len([v for v in INTAKE_VIEWS if views.get(v)])} views, "
             f"voice={'yes' if voice else 'no'})")
    return mid


def put_file(mid: str, role: str, src: Optional[Path] = None,
             data: Optional[bytes] = None, filename: str = "") -> Path:
    """Install one file into a mascot's folder under the given role.

    `role` is one of ROLE_FILES — the image slots and `voice`. The extension
    follows the SOURCE file (a jpg stays a jpg), except for the angle slots, which
    are png-named by convention and are converted by nothing — so pass pngs for
    those, or accept the name.
    """
    role = (role or "main").strip().lower()
    if role not in ROLE_FILES:
        raise ValueError(f"role must be one of {tuple(ROLE_FILES)}")
    d = mascots_dir() / mid
    if not d.is_dir():
        raise ValueError(f"no mascot named {mid!r}")

    src_ext = Path(filename or (src.name if src else "")).suffix.lower()
    wanted = ROLE_FILES[role]
    if role == "voice":
        if src_ext not in AUDIO_EXTS:
            raise ValueError(f"voice clip must be one of {AUDIO_EXTS}")
        dest = d / f"voice{src_ext}"
        for other in VOICE_NAMES:            # only one voice per mascot
            if (d / other) != dest:
                (d / other).unlink(missing_ok=True)
    else:
        if src_ext and src_ext not in IMAGE_EXTS:
            raise ValueError(f"image must be one of {IMAGE_EXTS}")
        if role == "main":
            dest = d / f"mascot{src_ext or '.png'}"
            for other in PRIMARY_NAMES:      # a new main replaces the old one
                if (d / other) != dest:
                    (d / other).unlink(missing_ok=True)
        else:
            dest = d / wanted                # angles keep their fixed names

    if data is not None:
        dest.write_bytes(data)
    elif src is not None:
        shutil.copyfile(src, dest)
    else:
        raise ValueError("nothing to write: pass src= or data=")
    log.info(f"mascot {mid}: {role} <- {dest.name} ({dest.stat().st_size} bytes)")
    return dest


def rename(mid: str, name: str) -> None:
    d = mascots_dir() / mid
    if not d.is_dir():
        raise ValueError(f"no mascot named {mid!r}")
    (d / "meta.json").write_text(json.dumps({"name": name.strip()}),
                                 encoding="utf-8")


def remove(mid: str) -> None:
    """Delete a mascot, art and all. If it was the active one, the shelf's next
    mascot takes over — the active id must never point at nothing."""
    from modules import runtime_settings as rs
    d = mascots_dir() / mid
    if not d.is_dir():
        raise ValueError(f"no mascot named {mid!r}")
    shutil.rmtree(d)
    log.info(f"mascot removed: {mid}")
    if rs.get_active_mascot() == mid:
        left = [m["id"] for m in list_mascots()]
        rs.set_active_mascot(left[0] if left else "")


def migrate() -> Optional[str]:
    """Move the old single-mascot layout onto the shelf, once.

    COPIES `assets/mascot*.png` (+ the voice clip) into `mascots/default/`. The
    originals stay where they are — but `mascots/` now exists, and mascot.py only
    consults the flat layout while it does NOT, so the copies are what the
    pipeline reads from here on. Returns the new id, or None if there was nothing
    to migrate / the shelf already exists.
    """
    if library_exists():
        return None
    src = _assets_dir()
    flat_primary = _first(src, PRIMARY_NAMES)
    flat_angles = [src / n for n in ANGLE_NAMES
                   if (src / n).exists() and (src / n).stat().st_size > 0]
    if not (flat_primary or flat_angles):
        return None

    d = mascots_dir() / DEFAULT_ID
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps({"name": "Default"}), encoding="utf-8")
    if flat_primary:
        shutil.copyfile(flat_primary, d / flat_primary.name)
    for a in flat_angles:
        shutil.copyfile(a, d / a.name)
    legacy_voice = src / "mascot_voice.wav"
    if legacy_voice.exists() and legacy_voice.stat().st_size > 0:
        shutil.copyfile(legacy_voice, d / "voice.wav")

    from modules import runtime_settings as rs
    if not rs.get_active_mascot():
        rs.set_active_mascot(DEFAULT_ID)
    log.info(f"mascot library created; old mascot copied in as '{DEFAULT_ID}'")
    return DEFAULT_ID
