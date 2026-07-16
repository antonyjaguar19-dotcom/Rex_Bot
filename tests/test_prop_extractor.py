"""The shared prop extractor + shelf bridge: the CHECK, and render-once-ever reuse."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import prop_extractor as pe        # noqa: E402
from modules import props_library as pl          # noqa: E402
from modules import mascot                       # noqa: E402


@pytest.fixture()
def shelf(tmp_path, monkeypatch):
    monkeypatch.setattr(mascot, "ASSETS_DIR", tmp_path / "assets")
    return tmp_path


def test_the_check_keeps_only_objects_named_in_two_or_more_beats():
    proposed = [{"noun": "doll", "desc": "a block"},
                {"noun": "puppy", "desc": "a labrador"},
                {"noun": "spaceship", "desc": "invented"}]   # model made it up
    beats = ["the doll sits still", "your puppy runs", "the doll never eats", "a happy puppy"]
    got = pe.recurring(proposed, beats)
    heads = {o["head"] for o in got}
    assert heads == {"doll", "puppy"}, heads            # spaceship (0 beats) dropped
    # ranked by recurrence, capped at 2
    assert len(got) == 2
    assert got[0]["count"] >= got[1]["count"]


def test_the_cap_limits_the_list():
    proposed = [{"noun": n, "desc": "d"} for n in ("doll", "puppy", "rock")]
    beats = ["doll", "doll", "puppy", "puppy", "rock", "rock"]
    assert len(pe.recurring(proposed, beats, cap=2)) == 2


def _stub_render(ok=True):
    calls = {"n": 0}
    def render(desc, out_path):
        calls["n"] += 1
        if ok:
            Path(out_path).write_bytes(b"\x89PNG rendered")
        return ok
    render.calls = calls
    return render


def test_a_prop_is_rendered_once_then_reused_no_second_render(shelf):
    render = _stub_render()
    path1, s1 = pe.ensure_ref("doll", "a faceless block", render)
    assert s1 == "rendered" and path1.exists()
    assert render.calls["n"] == 1

    # a SECOND video names the same prop -> reused from the shelf, render NOT called again
    render2 = _stub_render()
    path2, s2 = pe.ensure_ref("a doll", "a faceless block", render2)   # plural/article-insensitive
    assert s2 == "reused"
    assert path2 == path1
    assert render2.calls["n"] == 0, "a prop already on the shelf must not re-render"


def test_a_failed_render_falls_back_to_the_description_lock(shelf):
    render = _stub_render(ok=False)
    path, status = pe.ensure_ref("rock", "a grey stone", render)
    assert status == "none" and path is None
    # it did not get marked rendered, so next time it will be tried again
    assert pl.find("rock")["rendered"] is False


def test_checklist_reports_have_and_missing(shelf):
    render = _stub_render()
    pe.ensure_ref("doll", "a block", render)              # now on the shelf, rendered
    got = {c["noun"]: c for c in pe.checklist(["doll", "dragon"])}
    assert got["doll"]["rendered"] is True
    assert got["dragon"]["on_shelf"] is False             # still to render
