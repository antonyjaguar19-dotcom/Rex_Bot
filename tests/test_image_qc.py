"""Visual QC: a closed yes/no checklist, biased toward NOT flagging (a false alarm = a real
re-render). The VLM call is mocked; these pin the parse + flag logic, not the model."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import image_qc as qc              # noqa: E402


def _png(p: Path) -> Path:
    p.write_bytes(b"\x89PNG\r\n\x1a\n fake")
    return p


def _mock_llm(monkeypatch, payload):
    import modules.script_generator as sg
    monkeypatch.setattr(sg, "_call_llm", lambda *a, **k: payload)


def test_a_clean_still_passes(tmp_path, monkeypatch):
    verdicts = {k: {"verdict": "pass", "note": ""} for k, _ in qc._CHECKS_TEACHING}
    _mock_llm(monkeypatch, json.dumps(verdicts))
    got = qc.qc_still(_png(tmp_path / "a.png"))
    assert got["ok"] is True
    assert got["fails"] == []


def test_a_confident_fail_is_flagged(tmp_path, monkeypatch):
    verdicts = {k: {"verdict": "pass", "note": ""} for k, _ in qc._CHECKS_TEACHING}
    verdicts["no_face_on_objects"] = {"verdict": "fail", "note": "smiling earth"}
    _mock_llm(monkeypatch, json.dumps(verdicts))
    got = qc.qc_still(_png(tmp_path / "b.png"))
    assert got["ok"] is False
    assert [f["key"] for f in got["fails"]] == ["no_face_on_objects"]
    assert "smiling earth" in got["fails"][0]["note"]


def test_an_unparseable_reply_does_not_block(tmp_path, monkeypatch):
    # QC must NEVER be the thing that loops a render: garbage in = pass, not fail.
    _mock_llm(monkeypatch, "the model rambled and returned no json")
    got = qc.qc_still(_png(tmp_path / "c.png"))
    assert got["ok"] is True


def test_a_model_error_does_not_block(tmp_path, monkeypatch):
    import modules.script_generator as sg
    def boom(*a, **k):
        raise RuntimeError("ollama down")
    monkeypatch.setattr(sg, "_call_llm", boom)
    got = qc.qc_still(_png(tmp_path / "d.png"))
    assert got["ok"] is True
    assert "ollama down" in got["error"]


def test_a_missing_image_is_not_a_failure(tmp_path, monkeypatch):
    got = qc.qc_still(tmp_path / "nope.png")
    assert got["ok"] is True


def test_facts_context_drops_the_teaching_only_checks():
    keys_teaching = {k for k, _ in qc.checks_for("teaching")}
    keys_facts = {k for k, _ in qc.checks_for("facts")}
    assert "no_face_on_objects" in keys_teaching
    assert "no_face_on_objects" not in keys_facts     # no props in a facts reel
    assert keys_facts < keys_teaching
