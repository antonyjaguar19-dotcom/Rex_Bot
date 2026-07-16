"""The props shelf: reusable prop/scene references, looked up BY NAME (the checklist).

Mirrors the mascot library, minus voice, plus name lookup. The shelf's assets dir is
redirected to a tmp path (mascot.ASSETS_DIR), so nothing touches the real 02_Agent/assets.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import props_library as pl        # noqa: E402
from modules import mascot                      # noqa: E402


@pytest.fixture()
def shelf(tmp_path, monkeypatch):
    monkeypatch.setattr(mascot, "ASSETS_DIR", tmp_path / "assets")
    return tmp_path


def _img(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x89PNG\r\n\x1a\n fake image bytes")
    return p


def test_create_and_describe(shelf):
    pid = pl.create("Rag Doll", description="a faceless plastic block", kind="prop",
                    source="auto")
    got = pl.describe(pid)
    assert got["id"] == "rag-doll"
    assert got["name"] == "Rag Doll"
    assert got["description"] == "a faceless plastic block"
    assert got["kind"] == "prop"
    assert got["source"] == "auto"
    assert got["rendered"] is False          # no image yet — a stub row
    assert got["ready"] is False


def test_an_image_makes_it_rendered_and_ready(shelf, tmp_path):
    pid = pl.create("Tree")
    pl.put_file(pid, "main", src=_img(tmp_path / "src.png"))
    pl.set_rendered(pid, True)
    got = pl.describe(pid)
    assert got["ready"] and got["rendered"]
    assert got["image"].name.startswith("prop")


def test_find_matches_by_name_plural_folded(shelf, tmp_path):
    pl.create_from_intake("Rag Doll", "a block", "prop", {"front": _img(tmp_path / "d.png")})
    pl.create_from_intake("Labrador Puppy", "a golden lab", "character",
                          {"front": _img(tmp_path / "p.png")})
    # a script word finds its shelf entry, singular or plural
    assert pl.find("a doll")["id"] == "rag-doll"
    assert pl.find("dolls")["id"] == "rag-doll"
    assert pl.find("the puppy")["id"] == "labrador-puppy"
    assert pl.find("puppies")["id"] == "labrador-puppy"
    # a word we have nothing for is a MISS — the signal to render + add it
    assert pl.find("a spaceship") is None


def test_list_props_is_the_checklist(shelf, tmp_path):
    assert pl.list_props() == []
    pl.create("Rock")
    pl.create_from_intake("Chair", "a wooden chair", "prop", {"front": _img(tmp_path / "c.png")})
    shelf_now = {p["id"]: p for p in pl.list_props()}
    assert set(shelf_now) == {"rock", "chair"}
    assert shelf_now["rock"]["rendered"] is False     # still to render
    assert shelf_now["chair"]["rendered"] is True     # have it


def test_put_file_replaces_the_main_without_destroying_it(shelf, tmp_path):
    pid = pl.create("Ball")
    pl.put_file(pid, "main", src=_img(tmp_path / "a.png"))
    first = pl.primary_image(pid)
    assert first.exists()
    pl.put_file(pid, "main", src=_img(tmp_path / "b.jpg"))
    got = pl.describe(pid)
    assert got["image"].suffix == ".jpg"              # new main
    assert not (got["dir"] / "prop.png").exists()     # old one swapped out


def test_intake_rolls_back_on_a_bad_file(shelf):
    with pytest.raises(ValueError):
        pl.create_from_intake("Bad", "x", "prop", {"front": None})   # front required
    assert pl.find("Bad") is None
    assert not (pl.props_dir() / "bad").exists()


def test_rename_keeps_the_id(shelf):
    pid = pl.create("Old Name")
    pl.rename(pid, "New Name")
    assert pl.describe(pid)["name"] == "New Name"
    assert pl.describe(pid)["id"] == pid               # the id is an address


def test_remove(shelf):
    pid = pl.create("Gone")
    pl.remove(pid)
    assert pl.describe(pid) is None
