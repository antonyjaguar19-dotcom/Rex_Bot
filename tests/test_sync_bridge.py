import pytest

from modules import sync_bridge as sb


@pytest.fixture(autouse=True)
def isolated_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(sb, "EVENTS_PATH", tmp_path / "sync_events.json")
    monkeypatch.setattr(sb, "CURSOR_PATH", tmp_path / "sync_cursor.json")


def test_emit_and_read_roundtrip():
    a = sb.emit("video_regen", script_id="s1", shot=3)
    b = sb.emit("final_done", script_id="s1")
    assert b == a + 1
    events = sb.read_new(0)
    assert [e["id"] for e in events] == [a, b]
    assert events[0]["data"] == {"script_id": "s1", "shot": 3}


def test_cursor_skips_handled_events():
    a = sb.emit("video_done", script_id="s1")
    sb.set_cursor(a)
    assert sb.read_new(sb.get_cursor()) == []
    b = sb.emit("video_done", script_id="s2")
    new = sb.read_new(sb.get_cursor())
    assert [e["id"] for e in new] == [b]


def test_latest_id_for_backlog_skip():
    assert sb.latest_id() == 0
    sb.emit("x")
    sb.emit("y")
    assert sb.latest_id() == 2


def test_event_log_capped():
    for i in range(sb._MAX_EVENTS + 50):
        sb.emit("tick", n=i)
    events = sb.read_new(0)
    assert len(events) == sb._MAX_EVENTS
    # ids keep increasing even after trimming
    assert events[-1]["id"] == sb._MAX_EVENTS + 50
