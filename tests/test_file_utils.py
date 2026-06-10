import json

from modules.file_utils import atomic_write_json, atomic_write_text


def test_roundtrip(tmp_path):
    p = tmp_path / "state.json"
    data = {"shots": [1, 2, 3], "title": "மழை நாள்", "emoji": "🦖"}
    atomic_write_json(p, data)
    assert json.loads(p.read_text(encoding="utf-8")) == data


def test_overwrite_keeps_valid_json(tmp_path):
    p = tmp_path / "state.json"
    atomic_write_json(p, {"v": 1})
    atomic_write_json(p, {"v": 2})
    assert json.loads(p.read_text(encoding="utf-8")) == {"v": 2}


def test_no_tmp_file_left_behind(tmp_path):
    p = tmp_path / "state.json"
    atomic_write_json(p, {"v": 1})
    assert not list(tmp_path.glob("*.tmp"))


def test_creates_parent_dirs(tmp_path):
    p = tmp_path / "a" / "b" / "state.json"
    atomic_write_text(p, "hello")
    assert p.read_text(encoding="utf-8") == "hello"


def test_default_serializer(tmp_path):
    p = tmp_path / "state.json"
    atomic_write_json(p, {"path": tmp_path}, default=str)
    assert json.loads(p.read_text(encoding="utf-8"))["path"] == str(tmp_path)
