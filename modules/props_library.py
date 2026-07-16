"""
Claw Bot — props library (the "Props and other" shelf)

A reusable shelf of PROPS and scene elements, the same idea as the mascot shelf
(`mascot_library`) but for the things a video keeps coming back to — a doll, a puppy,
a tree, a chair, a rock — and for locations and secondary characters too. Drawn ONCE,
kept forever, reused across videos so a prop is never re-rendered.

    02_Agent/assets/props/
        rag-doll/
            prop.png                <- the reference image (the drawing of the prop)
            prop_front.png          <- optional extra views
            prop_side.png
            meta.json               <- {name, description, kind, rendered, source}
        labrador-puppy/
            ...

The deliberate DIFFERENCE from the mascot shelf: a mascot has ONE active star
(`runtime_settings.active_mascot`); props are MANY and are looked up BY NAME, not selected
one-at-a-time. So there is no `active` here — instead `find(name)` matches a script's word
("a doll") to a shelf entry, plural-folded, and that entry's picture is reused. That name
lookup IS the checklist: `list_props()` is every prop we have, and a script's props not on
it are the ones still to render.

`meta.json` fields:
    name        display name
    description the canonical, description-locked look (the words injected wherever the prop
                is named but has no reference slot — see lesson_objects.lock_descriptions)
    kind        "prop" | "location" | "character"
    rendered    True once its reference image has actually been drawn (vs a stub row)
    source      "auto" (extracted from a script and pinned) | "upload" (added by hand)
"""

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Optional

log = logging.getLogger("claw_bot.props_library")

PRIMARY_NAMES = ("prop.png", "prop.jpg", "prop.jpeg", "prop.webp")
ANGLE_NAMES = ("prop_front.png", "prop_side.png")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

# Upload slots the UI offers. role -> filename it lands as.
ROLE_FILES = {
    "main": "prop.png",
    "front": "prop_front.png",
    "side": "prop_side.png",
}

INTAKE_VIEWS = ("front", "side")
REQUIRED_VIEWS = ("front",)

KINDS = ("prop", "location", "character")
DEFAULT_KIND = "prop"


def _assets_dir() -> Path:
    """Late import (mascot.py owns ASSETS_DIR, and tests move it to a tmp path) — read it at
    call time so a redirected assets dir reaches the props shelf too."""
    from modules import mascot
    return Path(mascot.ASSETS_DIR)


def props_dir() -> Path:
    return _assets_dir() / "props"


def library_exists() -> bool:
    return props_dir().is_dir()


# ==============================================================================
# IDs + name matching
# ==============================================================================

def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-")
    return s[:40] or "prop"


def _unique_id(name: str) -> str:
    base = slugify(name)
    pid, n = base, 2
    while (props_dir() / pid).exists():
        pid = f"{base}-{n}"
        n += 1
    return pid


def _fold(word: str) -> str:
    """Plural-folded single word, so 'dolls' matches the 'doll' prop and 'puppies' the
    'puppy' — and 'octopus' survives. -ies is handled here (the shared singulariser only
    folds -s/-es); the rest defers to facts_memory._singular (keeps -us/-ss/-is/-as)."""
    from modules.facts_memory import _singular
    w = (word or "").strip().lower()
    if len(w) > 3 and w.endswith("ies"):
        w = w[:-3] + "y"                    # puppies -> puppy, before the generic fold
    return _singular(w)


# ==============================================================================
# READ
# ==============================================================================

def _first(d: Path, names) -> Optional[Path]:
    for n in names:
        p = d / n
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def _meta(d: Path) -> dict:
    m = d / "meta.json"
    if m.exists():
        try:
            return json.loads(m.read_text(encoding="utf-8")) or {}
        except Exception as e:
            log.warning(f"unreadable meta.json for {d.name}: {e}")
    return {}


def _write_meta(d: Path, meta: dict) -> None:
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def describe(pid: str) -> Optional[dict]:
    """Everything the UI/pipeline needs about one prop, or None if it isn't there."""
    d = props_dir() / pid
    if not d.is_dir():
        return None
    meta = _meta(d)
    primary = _first(d, PRIMARY_NAMES)
    angles = [d / n for n in ANGLE_NAMES
              if (d / n).exists() and (d / n).stat().st_size > 0]
    image = primary or (angles[0] if angles else None)
    return {
        "id": pid,
        "name": str(meta.get("name") or d.name.replace("-", " ").title()),
        "description": str(meta.get("description") or ""),
        "kind": str(meta.get("kind") or DEFAULT_KIND),
        "dir": d,
        "image": image,
        "angles": angles,
        # A prop is "rendered" once its reference picture exists; the meta flag is a hint but
        # the FILE is the truth (a meta that says rendered but has no image is not rendered).
        "rendered": bool(image) and bool(meta.get("rendered", bool(image))),
        "source": str(meta.get("source") or "upload"),
        "ready": bool(image),
    }


def list_props() -> list:
    """Every prop on the shelf, alphabetical. THIS is the checklist. Empty when none."""
    root = props_dir()
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


def get(pid: str) -> Optional[dict]:
    return describe(pid)


def find(name: str) -> Optional[dict]:
    """The prop a word/name refers to, matched by slug OR by any folded word in its name —
    so 'a doll' finds the 'rag-doll' prop and 'puppies' finds 'labrador-puppy'. None when the
    shelf has nothing for it: that miss is the signal to render + add it (the checklist)."""
    want_slug = slugify(name)
    want_words = {_fold(w) for w in re.findall(r"[a-z]+", (name or "").lower())}
    if not want_words and not want_slug:
        return None
    for p in list_props():
        if p["id"] == want_slug:
            return p
        name_words = {_fold(w) for w in re.findall(r"[a-z]+", p["name"].lower())}
        id_words = {_fold(w) for w in re.findall(r"[a-z]+", p["id"].lower())}
        if want_words & (name_words | id_words):
            return p
    return None


def primary_image(pid: str) -> Optional[Path]:
    got = describe(pid)
    return got["image"] if got else None


def file_for(pid: str, role: str) -> Optional[Path]:
    """The file a prop currently has in one slot, or None — what the UI SHOWS per slot."""
    got = describe(pid)
    if not got:
        return None
    role = (role or "main").strip().lower()
    if role == "main":
        return got["image"]
    name = ROLE_FILES.get(role)
    if not name:
        return None
    p = got["dir"] / name
    return p if p.exists() and p.stat().st_size > 0 else None


def refs(pid: str, max_refs: int = 1) -> list:
    """Reference image(s) to condition on — angles first, else the single primary. Same rule
    the mascot shelf uses (one front view beats three copied stances)."""
    got = describe(pid)
    if not got:
        return []
    if got["angles"]:
        return got["angles"][:max_refs]
    return [got["image"]] if got["image"] else []


# ==============================================================================
# WRITE
# ==============================================================================

def create(name: str, description: str = "", kind: str = DEFAULT_KIND,
           source: str = "upload", image: Optional[Path] = None,
           image_bytes: Optional[bytes] = None, filename: str = "prop.png") -> str:
    """Put a new prop on the shelf. Returns its id. Image optional (a stub row is listed but
    not `rendered` until its picture exists)."""
    kind = kind if kind in KINDS else DEFAULT_KIND
    pid = _unique_id(name)
    d = props_dir() / pid
    d.mkdir(parents=True, exist_ok=True)
    _write_meta(d, {"name": (name or pid).strip(), "description": (description or "").strip(),
                    "kind": kind, "rendered": False, "source": source})
    if image or image_bytes:
        put_file(pid, "main", src=image, data=image_bytes, filename=filename)
        set_rendered(pid, True)
    log.info(f"prop created: {pid} ({name}, kind={kind}, source={source})")
    return pid


def set_rendered(pid: str, rendered: bool = True) -> None:
    d = props_dir() / pid
    if not d.is_dir():
        raise ValueError(f"no prop named {pid!r}")
    meta = _meta(d)
    meta["rendered"] = bool(rendered)
    _write_meta(d, meta)


def set_description(pid: str, description: str) -> None:
    d = props_dir() / pid
    if not d.is_dir():
        raise ValueError(f"no prop named {pid!r}")
    meta = _meta(d)
    meta["description"] = (description or "").strip()
    _write_meta(d, meta)


def create_from_intake(name: str, description: str, kind: str, views: dict,
                       source: str = "upload") -> str:
    """Create a prop from the full intake: its description + views. Front is required and is
    installed as the primary. Nothing is half-created — a failed intake removes the folder."""
    missing = [v for v in REQUIRED_VIEWS if not views.get(v)]
    if missing:
        raise ValueError(f"missing required view(s): {', '.join(missing)}")
    pid = create(name, description=description, kind=kind, source=source)
    try:
        put_file(pid, "main", src=Path(views["front"]))
        for view in INTAKE_VIEWS:
            src = views.get(view)
            if src:
                put_file(pid, view, src=Path(src))
        set_rendered(pid, True)
    except Exception:
        try:
            shutil.rmtree(props_dir() / pid)
        except Exception as e:
            log.warning(f"could not clean up half-built prop {pid}: {e}")
        raise
    log.info(f"prop {pid}: intake complete "
             f"({len([v for v in INTAKE_VIEWS if views.get(v)])} views)")
    return pid


def put_file(pid: str, role: str, src: Optional[Path] = None,
             data: Optional[bytes] = None, filename: str = "") -> Path:
    """Install one image into a prop's folder under the given role. Stage-then-swap (a new
    main can BE built from the old one; delete after the copy is safely staged)."""
    role = (role or "main").strip().lower()
    if role not in ROLE_FILES:
        raise ValueError(f"role must be one of {tuple(ROLE_FILES)}")
    d = props_dir() / pid
    if not d.is_dir():
        raise ValueError(f"no prop named {pid!r}")

    src_ext = Path(filename or (src.name if src else "")).suffix.lower()
    if src_ext and src_ext not in IMAGE_EXTS:
        raise ValueError(f"image must be one of {IMAGE_EXTS}")
    if role == "main":
        dest = d / f"prop{src_ext or '.png'}"
        stale = [d / n for n in PRIMARY_NAMES]
    else:
        dest = d / ROLE_FILES[role]
        stale = []

    if data is None and src is None:
        raise ValueError("nothing to write: pass src= or data=")

    staged = d / f"_incoming{src_ext or '.bin'}"
    try:
        if data is not None:
            staged.write_bytes(data)
        else:
            shutil.copyfile(src, staged)
        for old in stale:
            if old != staged:
                old.unlink(missing_ok=True)
        shutil.copyfile(staged, dest)
    finally:
        staged.unlink(missing_ok=True)

    log.info(f"prop {pid}: {role} <- {dest.name} ({dest.stat().st_size} bytes)")
    return dest


def rename(pid: str, name: str) -> str:
    """Change a prop's DISPLAY name. The folder id never changes — the id is an address that
    a script's rendered artifacts and the checklist point at; the name is a label."""
    d = props_dir() / pid
    if not d.is_dir():
        raise ValueError(f"no prop named {pid!r}")
    clean = (name or "").strip()
    if not clean:
        raise ValueError("a prop needs a name")
    meta = _meta(d)
    meta["name"] = clean
    _write_meta(d, meta)
    log.info(f"prop {pid} renamed to '{clean}'")
    return clean


def remove(pid: str) -> None:
    """Delete a prop, image and all."""
    d = props_dir() / pid
    if not d.is_dir():
        raise ValueError(f"no prop named {pid!r}")
    shutil.rmtree(d)
    log.info(f"prop removed: {pid}")
