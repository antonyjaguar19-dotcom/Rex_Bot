"""Unparseable LLM output must survive long enough to be debugged.

_last_raw_llm_output.txt is rewritten by EVERY call, so the response that
actually failed was overwritten by the next success before anyone could look.
"""

import pytest

from modules import script_generator as sg


@pytest.fixture
def outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(sg, "OUTPUTS_DIR", tmp_path)
    return tmp_path


def test_bad_output_is_archived_with_its_own_name(outputs):
    with pytest.raises(ValueError, match="Could not extract JSON"):
        sg._extract_json("this is not json at all {{{")
    saved = list((outputs / "_bad_llm_output").glob("*.txt"))
    assert len(saved) == 1
    assert "not json" in saved[0].read_text(encoding="utf-8")


def test_error_message_points_at_the_archived_copy(outputs):
    with pytest.raises(ValueError) as e:
        sg._extract_json("garbage {")
    assert "_bad_llm_output" in str(e.value)


def test_a_later_success_does_not_clobber_the_failure(outputs):
    with pytest.raises(ValueError):
        sg._extract_json("garbage {")
    sg._extract_json('{"ok": true}')          # a later good call
    saved = list((outputs / "_bad_llm_output").glob("*.txt"))
    assert len(saved) == 1                    # the failure is still there
    assert "garbage" in saved[0].read_text(encoding="utf-8")


def test_archive_is_pruned(outputs, monkeypatch):
    monkeypatch.setattr(sg, "MAX_BAD_OUTPUTS", 3)
    for i in range(6):
        with pytest.raises(ValueError):
            sg._extract_json(f"bad {i} {{")
    saved = sorted((outputs / "_bad_llm_output").glob("*.txt"))
    assert len(saved) == 3                    # only the newest kept


def test_valid_json_still_parses(outputs):
    assert sg._extract_json('{"a": 1}') == {"a": 1}
    assert not (outputs / "_bad_llm_output").exists()
