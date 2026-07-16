"""Claw Bot — the shared prop extractor + shelf bridge.

`lesson_objects` proved the pattern for ONE lesson: the writer PROPOSES recurring objects,
a CHECK keeps only those named in >=2 beats, and each is description-locked + pinned (drawn
once, cached under the lesson). This module generalises that so ANY pipeline can use it, and
adds the piece the user asked for: a prop is rendered ONCE, EVER — the pin is looked up on
the persistent props SHELF (`props_library`) and reused across videos, never re-rendered.

Two functions do the work, both mode-agnostic:

- `recurring(proposed, beat_texts)` — the CHECK. Same doctrine as lesson_writer: the model
  proposes, this keeps only nouns that really recur (>=2 beats), ranks by recurrence, caps.
- `ensure_ref(noun, desc, render_fn)` — the SHELF BRIDGE. `props_library.find(noun)` is the
  checklist: a hit that is already rendered returns the shelf image (NO re-render); a miss
  renders it once via `render_fn` and adds it to the shelf. `render_fn(desc, out_path)->bool`
  is supplied by the caller (the lesson wraps Qwen-Edit; a test passes a stub), so this module
  needs no GPU and is fully testable.
"""
import logging
import re
from pathlib import Path
from typing import Callable, Optional

from modules import props_library as pl

log = logging.getLogger("claw_bot.prop_extractor")

DEFAULT_MIN_BEATS = 2       # named in one beat only = a one-off, not a through-line
DEFAULT_CAP = 2             # Qwen-Edit affords 2 extra refs beside the person


def _head(noun: str) -> str:
    from modules.facts_memory import _singular
    return _singular(re.sub(r"[^a-z]", "", (noun or "").lower().split()[-1] if noun else ""))


def recurring(proposed: list, beat_texts: list,
              min_beats: int = DEFAULT_MIN_BEATS, cap: int = DEFAULT_CAP) -> list:
    """The recurring objects worth pinning, CHECKED against the script's own words.

    `proposed` = [{"noun","desc"}, ...] (from an LLM). `beat_texts` = the words of each beat/
    shot. Keep only a noun whose head word appears in >= `min_beats` beats (drops one-offs and
    anything the model invented), rank by how often it recurs, cap the list. Returns
    [{"noun","desc","head","count"}].
    """
    from modules.facts_memory import _singular
    beat_tokens = [{_singular(w) for w in re.findall(r"[a-z]+", (t or "").lower())}
                   for t in (beat_texts or [])]
    scored, seen = [], set()
    for it in proposed or []:
        if not isinstance(it, dict):
            continue
        noun = (it.get("noun") or "").strip().lower()
        desc = (it.get("desc") or "").strip()
        if not noun or not desc:
            continue
        head = _head(noun)
        if len(head) < 3 or head in seen:
            continue
        count = sum(1 for toks in beat_tokens if head in toks)      # THE CHECK
        if count < min_beats:
            continue
        seen.add(head)
        scored.append((count, {"noun": noun, "desc": desc, "head": head, "count": count}))
    scored.sort(key=lambda x: -x[0])
    return [o for _, o in scored[:cap]]


def ensure_ref(noun: str, desc: str,
               render_fn: Callable[[str, Path], bool],
               kind: str = "prop", source: str = "auto") -> tuple:
    """A reference image for this prop, drawn ONCE ever. Returns (path, status).

    status: "reused"   — already on the shelf and rendered; the SAME picture is returned, no
                         GPU touched (the whole point of the shelf).
            "rendered" — a shelf miss (or a stub row): drawn once via render_fn and added.
            "none"     — render_fn failed; the caller falls back to the description-lock.

    `render_fn(desc, out_path) -> bool` draws the prop ALONE into out_path; called ONLY on a
    miss. props_library.find() is the checklist — a hit here is the "we already have it".
    """
    hit = pl.find(noun)
    if hit and hit["rendered"] and hit["image"]:
        log.info(f"prop {noun!r} reused from the shelf ({hit['id']}) — not re-rendered")
        return hit["image"], "reused"

    if hit:                                     # a stub row with no picture yet
        pid = hit["id"]
    else:
        pid = pl.create(noun, description=desc, kind=kind, source=source)
    out = pl.props_dir() / pid / "prop.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    ok = False
    try:
        ok = bool(render_fn(desc, out))
    except Exception as e:                      # a failed pin is not a broken render
        log.warning(f"could not render prop {noun!r}: {e}")
    if ok and out.exists() and out.stat().st_size:
        pl.set_description(pid, desc)
        pl.set_rendered(pid, True)
        log.info(f"prop {noun!r} rendered once and added to the shelf ({pid})")
        return out, "rendered"
    log.warning(f"prop {noun!r} did not render; it will use the description-lock instead")
    return None, "none"


def checklist(nouns: list) -> list:
    """For a script's prop nouns: what the shelf already has vs what is still to render.

    Returns [{"noun","on_shelf","rendered","id"}] — the have/missing list surfaced in the
    dashboard and `!prop list`.
    """
    out = []
    for n in nouns or []:
        hit = pl.find(n)
        out.append({"noun": n,
                    "on_shelf": bool(hit),
                    "rendered": bool(hit and hit["rendered"]),
                    "id": hit["id"] if hit else None})
    return out
