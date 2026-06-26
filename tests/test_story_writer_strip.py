"""story_writer._strip_meta must scrub <think> blocks from thinking models."""
import sys
from pathlib import Path

_AGENT = Path(__file__).parent.parent.resolve()
if str(_AGENT) not in sys.path:
    sys.path.insert(0, str(_AGENT))

from modules import story_writer as sw


def test_strip_closed_think_block():
    raw = "<think>reasoning here, lots of it</think>The Real Title\n\nThe story body."
    out = sw._strip_meta(raw)
    assert "<think>" not in out and "reasoning" not in out
    assert "The Real Title" in out


def test_strip_keeps_answer_after_last_close():
    raw = "<think>step 1</think> The kitten purred softly."
    out = sw._strip_meta(raw)
    assert out.strip().startswith("The kitten purred")


def test_plain_text_unchanged():
    raw = "The Moonlit Kitten\n\nMittens crouched in the grass."
    out = sw._strip_meta(raw)
    assert "Mittens crouched in the grass." in out


def test_title_label_prefix_stripped():
    for raw in (
        "Title of the Story: Rusty Finds a Friend\n\nRusty beeped sadly.",
        "Title: The Last Crumb\n\nTwo mice scurried home.",
        "Story Title - Little Seeds\n\nSunny dug a hole.",
    ):
        title, body = sw._split_title_body(raw)
        assert not title.lower().startswith("title"), f"label leaked: {title!r}"
        assert body.strip()


def test_plain_title_not_mangled():
    title, body = sw._split_title_body("The Race\n\nPip hopped fast.")
    assert title == "The Race"


def test_strip_orphan_unclosed_think_tag():
    # No closing tag — the orphan <think> must not survive as a title line.
    raw = "<think>\nreasoning with no close\n\nThe Title\n\nStory body here."
    out = sw._strip_meta(raw)
    assert "<think>" not in out


def test_reasoning_leak_detected():
    assert sw._looks_like_reasoning_leak("Here's a thinking process: first I will...")
    assert sw._looks_like_reasoning_leak(" ".join(["word"] * 300))   # way over target
    assert not sw._looks_like_reasoning_leak(
        "Mittens crouched low. A spark floated by. She pounced gently into the night."
    )
