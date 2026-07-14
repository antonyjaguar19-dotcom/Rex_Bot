"""The dashboard page actually builds.

There was no test for this, and an HTTP 200 from the running server does not give
one: NiceGUI executes a @ui.page builder when a CLIENT CONNECTS, not on the HTTP
GET. So a typo in the page body — a bad widget argument, a NameError in a
renderer — served a perfectly healthy 200 and blew up only in the browser, which
is exactly the class of silent failure this project keeps paying for.

Building the page inside a manual Client context runs the real thing: every card,
every widget, and the first full_refresh() pass.
"""
import pytest

from nicegui import Client
from nicegui.page import page as PageCls

import modules.dashboard_nicegui as dash


@pytest.fixture
def page(caplog):
    client = Client(PageCls("/"), request=None)
    with client:
        dash.main_page()
    errors = [r for r in caplog.get_records("call") if r.levelname == "ERROR"]
    assert not errors, f"page build logged errors: {[r.message for r in errors]}"
    return client


def _labels(client) -> list:
    out = []
    for e in client.elements.values():
        for got in (getattr(e, "_text", None), e._props.get("label")):
            if got:
                out.append(str(got))
    return out


def test_the_page_builds(page):
    assert len(page.elements) > 100


def test_every_mode_tab_is_there(page):
    labels = _labels(page)
    for tab in ("Story", "Facts", "Music", "Lessons", "Manual", "Mascots",
                "Models", "Queue"):
        assert tab in labels, f"missing nav tab: {tab}"


def test_facts_mode_can_choose_its_mascot(page):
    """The dropdown that picks WHICH character presents the facts."""
    selects = [e for e in page.elements.values()
               if e._props.get("label") == "Mascot"]
    assert selects, "no mascot picker on the Facts card"


def test_the_facts_card_offers_nothing_but_the_mascot():
    """A facts reel is frozen (FACTS_MODE.md): animated, mascot-presented, with a
    music bed, a thumbnail and a 40s ceiling. Every one of those was a toggle once,
    and every toggle was a way to ship the wrong film by accident — a stale value
    with no widget on screen to explain it. WHO presents is the only choice left.
    """
    from pathlib import Path
    src = (Path(__file__).parent.parent / "modules" / "dashboard_nicegui.py") \
        .read_text(encoding="utf-8")
    facts_ui = src.split("FACTS SHORTS (own tab)")[1].split("STAGE 2")[0]

    widgets = [ln.strip() for ln in facts_ui.splitlines()
               if ("ui.switch(" in ln or "ui.number(" in ln or "ui.select(" in ln)
               and not ln.lstrip().startswith("#")]
    assert len(widgets) == 1 and "ui.select(" in widgets[0], \
        f"the Facts card should offer only the mascot picker, found: {widgets}"


def test_the_mascot_card_offers_a_family(page):
    # A lesson line that names someone — "when mummy hugs you" — can only SHOW them if
    # this mascot has their picture. Without one they come back as a TWIN of the mascot,
    # so they are left out of the shot instead.
    #
    # The Family button lives on a MASCOT CARD, and conftest's _no_gpu_mascot fixture
    # points the shelf at an empty tmp dir — so the built page has no cards to hang it on.
    # What the build proves is that the page still ASSEMBLES with the family code in it
    # (the `page` fixture asserts no ERROR was logged during the build). The button itself
    # is checked in the source, the same way the Facts card's widget count is.
    import inspect

    import modules.dashboard_nicegui as dash
    src = inspect.getsource(dash.main_page)
    assert "Family (" in src, "no Family control on the mascot card"
    assert "_open_family_dialog(" in src
    assert len(page.elements) > 100, "and the page still builds"


def test_the_family_dialog_uses_the_3x_upload_api():
    # e.content / e.name is the NiceGUI 2.x API and only fails in the browser. 3.x gives
    # e.file, and the save is AWAITED — so the handler has to be async.
    import inspect
    import modules.dashboard_nicegui as dash
    src = inspect.getsource(dash.main_page)
    start = src.index("def _open_family_dialog")
    end = src.index("def render_mascots")
    family = src[start:end]

    assert "async def _up" in family, "an upload handler must be async — e.file.save is awaited"
    assert "await e.file.save(" in family
    assert "e.content" not in family, "e.content is the NiceGUI 2.x upload API"

    # the GPU path never runs on the UI thread
    assert "_bg_gpu(" in family and "_try_begin(" in family

    # every per-row handler binds its row by DEFAULT ARGUMENT — closing over the loop
    # variable gives every row the LAST relation
    assert "_rel=rel" in family


def test_the_family_is_in_the_cards_signature():
    # The card shows "Family (2)". Fold the family into the signature or that count goes
    # stale the moment you add a third — a count that never changes is a card that never
    # redraws, which is exactly the bug _slot_sig was written to fix for the slots.
    import inspect
    import modules.dashboard_nicegui as dash
    src = inspect.getsource(dash.main_page)
    start = src.index("def _slot_sig")
    end = src.index("if not _changed(\"mascots\"", start)
    assert "_family_of(" in src[start:end]


def test_the_gate_has_a_confirm_a_redraw_and_reorder_arrows(page):
    # THE GATE. Almost every serious defect this pipeline has produced rendered cleanly,
    # logged nothing, and was wrong only in the file — and Wan then spent 8.5 minutes
    # animating it. Nothing may start that on a picture nobody has confirmed.
    import inspect

    import modules.dashboard_nicegui as dash
    src = inspect.getsource(dash.main_page)
    start = src.index("def render_lesson_script")
    end = src.index("def render_lesson_library", start)
    gate = src[start:end]

    assert "lesson_ok_" in gate, "no per-picture confirm checkbox"
    assert "set_beat_approved(" in gate
    assert "render_btn.disable()" in gate, "Render is not locked while a picture is unconfirmed"
    assert "not confirmed" in gate
    assert "redraw_still_action(" in gate, "no per-picture redraw"
    assert "lw.move_beat(" in gate, "no reorder"

    # THE TICKBOXES ARE NAMED. Two checkboxes that do completely different jobs sat side
    # by side, unlabelled, a few pixels apart: one says the picture is right, the other
    # spends 8.5 minutes of GPU.
    assert 'ui.checkbox("Looks right"' in gate
    assert 'ui.checkbox("Animate (~8 min)"' in gate

    # ↺ RESET, and only when there is something to reset. A button that does nothing is
    # worse than no button — this codebase has already shipped one ("Redo the pictures"
    # was a twenty-minute no-op).
    assert "lw.reset_order(" in gate
    assert "if lw.is_reordered(lesson):" in gate

    # The change signature has to carry three things or the row silently goes stale:
    #   approved — or ticking it would not redraw the row
    #   orig     — or "↺ Reset order" would not appear when you move a shot
    #   MTIME    — or a REDRAWN picture would keep showing the old one: the path never
    #              changes, only the bytes behind it
    sig_start = src.index('sig = ((lesson or {}).get("lesson_id")')
    sig = src[sig_start:src.index("if not _changed(\"lesson_script\"", sig_start)]
    assert 'b.get("approved")' in sig
    assert 'b.get("orig")' in sig
    assert "_mtime(" in sig

    assert len(page.elements) > 100


def test_the_disabled_button_is_a_courtesy_not_the_gate():
    # A stale tab, or a Discord command, must hit the same wall. The real gate is in the
    # pipeline.
    import inspect
    from modules import lesson_pipeline as lp
    src = inspect.getsource(lp.approve)
    assert "lw.unapproved(lesson)" in src
    assert "not confirmed" in src
