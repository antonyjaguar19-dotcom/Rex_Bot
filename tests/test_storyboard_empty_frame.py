"""storyboard_generator: only WIDE establishing frames may be flagged empty,
and character-focused shots always get a character sheet injected even when the
prompt/narration refer to the cast by 'the twins'/'they' (no proper name)."""
import sys
from pathlib import Path

_AGENT = Path(__file__).parent.parent.resolve()
if str(_AGENT) not in sys.path:
    sys.path.insert(0, str(_AGENT))

from modules import storyboard_generator as sgg

_SCRIPT = {
    "style": "pixar",
    "setting": "a dusty attic",
    "characters": [
        {"name": "Graywhisker", "type": "animal", "gender": "male",
         "locked_visual_token": "a young male grey mouse with a blue scarf"},
        {"name": "Graytail", "type": "animal", "gender": "male",
         "locked_visual_token": "a young male grey mouse with a red scarf"},
    ],
}


def _shot(n, stype, narr, ffp):
    return {"shot_number": n, "shot_type": stype, "beat": "atmosphere",
            "speaker": "narrator", "narration": narr,
            "first_frame_prompt": ffp, "visual_description": narr}


def test_closeup_never_empty_and_gets_cast():
    # closeup that names no one ("the twins' heads") + subjectless narration
    sh = _shot(2, "closeup", "each with a patch of shiny gray fur.",
               "A closeup of the twins' heads, fur catching the light.")
    final = sgg._rewrite_as_paragraph(sh["first_frame_prompt"], _SCRIPT, sh)
    assert "Character sheet" in final            # full cast injected as fallback
    assert sgg._is_empty_frame(_SCRIPT, sh, final) is False


def test_wide_with_named_char_not_empty():
    sh = _shot(1, "wide", "Graywhisker and Graytail were twins,",
               "Wide attic, Graywhisker and Graytail sit on a crate.")
    final = sgg._rewrite_as_paragraph(sh["first_frame_prompt"], _SCRIPT, sh)
    assert sgg._is_empty_frame(_SCRIPT, sh, final) is False


def test_wide_pure_location_is_empty():
    sh = _shot(1, "wide", "The attic was silent at dawn.",
               "A wide empty attic full of dusty books at dawn.")
    final = sgg._rewrite_as_paragraph(sh["first_frame_prompt"], _SCRIPT, sh)
    assert sgg._is_empty_frame(_SCRIPT, sh, final) is True


def test_medium_no_name_still_not_empty():
    sh = _shot(5, "medium", "They argued back and forth.",
               "A medium shot as they face each other.")
    final = sgg._rewrite_as_paragraph(sh["first_frame_prompt"], _SCRIPT, sh)
    assert "Character sheet" in final
    assert sgg._is_empty_frame(_SCRIPT, sh, final) is False
