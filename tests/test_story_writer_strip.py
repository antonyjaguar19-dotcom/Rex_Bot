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
