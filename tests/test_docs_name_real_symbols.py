"""The docs may not name code that does not exist.

Three documents in this repo disagreed about the same dead module, and one of them told you
to go read a symbol that had never existed:

  - `IMAGE_FEEDBACK.md` and `CLAUDE.md` both said the lesson backdrop was
    `BACKDROPS[hash(lesson_id)]`. There is no `BACKDROPS` anywhere in the codebase. It is
    `lesson_pipeline.SETTINGS` + `setting_for(topic)` — and it keys on the TOPIC, not the
    lesson, so the doc also promised a per-lesson stability the code does not deliver.
  - `IMAGE_FEEDBACK.md` said visual QC was "proposed, not built" while it was built, tested
    and sitting in the tree.
  - `image_qc.py` argued the 32b was the answer; the image-supervisor skill said the 32b had
    been measured failing. (Both were wrong, for the same reason — see the ledger.)

Prose correction alone is what already failed, three times. This is the check.

It is DELIBERATELY narrow: only `module.SYMBOL` references where `module` is a real file in
`modules/`. It says nothing about whether the prose is true — only that the code it points at
is real. A doc that names a function that no longer exists is a doc that will send the next
person hunting for it.
"""
import re
from pathlib import Path

import pytest

AGENT_ROOT = Path(__file__).parent.parent.resolve()
MODULES = AGENT_ROOT / "modules"

DOCS = [
    AGENT_ROOT / "IMAGE_FEEDBACK.md",
    AGENT_ROOT / "CLAUDE.md",
    AGENT_ROOT / ".claude" / "skills" / "image-supervisor" / "SKILL.md",
]

# `module.SYMBOL` inside backticks. Backticks only: prose says "the lesson_writer.py file"
# and English sentences end in full stops, so an unquoted match is mostly punctuation.
_REF = re.compile(r"`([a-z_][a-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)")

# `mascot.py` is a FILENAME, not a symbol reference — the docs are full of them and they are
# not what this test is about (the file either exists or the path is obviously wrong).
_EXTENSIONS = {"py", "json", "md", "ps1", "bat", "txt", "log", "wav", "png", "mp4", "jpg"}

# Names that are deliberately illustrative, or belong to something other than modules/.
# An allow-list, so this test can never go red for a bad reason — but every entry needs a
# reason, or it is just a way to lose the check.
ALLOW = {
    # a stdlib/3rd-party module that shares a name with nothing here
    "json.dumps", "json.loads", "re.search", "re.sub", "os.environ",
    # historical names, quoted BECAUSE they are gone — the ledger's whole job is to record
    # what was deleted and why
    "image_qc.qc_still", "image_qc.checks_for_contract", "image_qc.DISABLED_CHECKS",
    "lesson_pipeline.run_visual_qc", "prop_extractor.checklist",
    "lesson_objects.detect",     # named in the ledger's self-sabotage note; real, see below
    # a models.json KEY that happens to share a name with modules/image_backend.py
    "image_backend.active",
    # quoted in CLAUDE.md §3L as THE EXAMPLE of what this test caught — it lives in
    # dashboard_nicegui, and the doc says so. A ledger has to be able to name the wrong name.
    "script_generator.update_shot_narration",
}


def _known_symbols(mod: str) -> set:
    """Every top-level name in modules/<mod>.py, by source text — no import, so a module with
    a heavy import graph cannot make this test slow or flaky."""
    p = MODULES / f"{mod}.py"
    if not p.is_file():
        return set()
    src = p.read_text(encoding="utf-8", errors="replace")
    out = set(re.findall(r"^(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", src, re.M))
    out |= set(re.findall(r"^([A-Za-z_][A-Za-z0-9_]*)\s*[:=]", src, re.M))
    return out


@pytest.mark.parametrize("doc", [d for d in DOCS], ids=lambda d: d.name)
def test_every_module_symbol_named_in_the_docs_exists(doc):
    if not doc.is_file():
        pytest.skip(f"{doc.name} is not in this checkout")
    text = doc.read_text(encoding="utf-8", errors="replace")

    bad = []
    for mod, sym in _REF.findall(text):
        if sym in _EXTENSIONS or f"{mod}.{sym}" in ALLOW:
            continue
        if not (MODULES / f"{mod}.py").is_file():
            continue                      # not one of ours — prose, or another package
        known = _known_symbols(mod)
        if sym not in known:
            bad.append(f"{mod}.{sym}")

    assert not bad, (
        f"{doc.name} names code that does not exist: {sorted(set(bad))}. "
        f"This is how `BACKDROPS[hash(lesson_id)]` survived in two documents describing a "
        f"symbol that was never written. Fix the doc, or add the name to ALLOW with a reason."
    )


def test_the_check_can_actually_fail():
    # A guard nobody has watched fail is a guard you are trusting on faith. This is the exact
    # shape of the bug it exists for.
    assert "SETTINGS" in _known_symbols("lesson_pipeline")
    assert "setting_for" in _known_symbols("lesson_pipeline")
    assert "BACKDROPS" not in _known_symbols("lesson_pipeline"), \
        "if BACKDROPS now exists, the ledger's row 17 needs revisiting"
    assert _known_symbols("a_module_that_does_not_exist") == set()
