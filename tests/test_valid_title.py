"""script_generator._valid_title rejects leaked-reasoning/tag titles."""
import sys
from pathlib import Path

_AGENT = Path(__file__).parent.parent.resolve()
if str(_AGENT) not in sys.path:
    sys.path.insert(0, str(_AGENT))

from modules import script_generator as sg


def test_rejects_think_and_tags():
    assert not sg._valid_title("<think>")
    assert not sg._valid_title("Here's a thinking process:")
    assert not sg._valid_title("")
    assert not sg._valid_title("   ")
    assert not sg._valid_title("a really long title that runs on and on for many words indeed")


def test_accepts_clean_titles():
    assert sg._valid_title("The Slow Race")
    assert sg._valid_title("Mittens and the Firefly")
    assert sg._valid_title("Fast Friends")
