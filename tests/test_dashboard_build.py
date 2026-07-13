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
    for tab in ("Story", "Facts", "Music", "Manual", "Mascots", "Models", "Queue"):
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
