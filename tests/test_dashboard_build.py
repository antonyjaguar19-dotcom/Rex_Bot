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
    # Render is locked while a picture is unconfirmed. The disable() now lives in
    # _repaint_estimate, which owns both the label and the button off one read of the disk —
    # see test_the_render_button_cannot_disagree_with_the_gate.
    assert "btn.disable()" in src, "Render is not locked while a picture is unconfirmed"
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


def test_the_prompts_are_revealed(page):
    # None of it was visible ANYWHERE: what reaches Qwen is f"{scene}, {style}" — the
    # scene plus ~120 words of STYLE_TEACHING and a ~90-word negative, in no log, no JSON
    # and on no screen. The video prompt was worse: invented at render time inside a
    # throwaway dict and never stored.
    import inspect

    import modules.dashboard_nicegui as dash
    src = inspect.getsource(dash.main_page)
    start = src.index("def _render_prompt_panel")
    panel = src[start:src.index("def render_lesson_script", start)]

    assert "prompts_for(" in panel, "the panel must be built from the RENDERER's code"
    assert "WHAT QWEN ACTUALLY GETS" in panel
    assert "image_negative" in panel
    assert "lw.set_scene(" in panel, "an edited scene must go through the guards"
    assert "motion_prompt" in panel
    assert 'got.get("guarded")' in panel, "the UI must SAY when it rewrote your words"

    # READABLE. The sent prompts were tiny dimmed textareas (11px, opacity .7) using
    # Quasar's `autogrow`, which caps its own height — so a 200-word positive and a
    # 430-word negative came out clipped, greyed and unreadable. The whole point of the
    # panel is that you can READ what the model was told.
    assert "_readonly_block(" in panel
    assert "white-space: pre-wrap" in panel, "a long prompt must WRAP"
    assert "overflow-y: auto" in panel, "...and scroll rather than clip"
    assert "font-size: 11px" not in panel

    # the two editable prompts are in the row's change signature, or the panel would go on
    # showing the words you just replaced
    sig_start = src.index('sig = ((lesson or {}).get("lesson_id")')
    sig = src[sig_start:src.index("if not _changed(\"lesson_script\"", sig_start)]
    assert 'b.get("mascot_scene", "")' in sig
    assert 'b.get("motion_prompt", "")' in sig

    assert len(page.elements) > 100


def test_the_gate_shows_what_the_bot_noticed(page):
    # Every warning this pipeline made was a progress_cb string: the mother we have no photo
    # of and quietly cut, the prop that lost its reference slot to her and will drift (which
    # has NO other symptom anywhere), the crop that failed so a "closeup" is a full-body
    # shot. All true, all detected, and all gone the moment the feed scrolled — long before
    # this gate, which is the one place a person is looking.
    import inspect

    import modules.dashboard_nicegui as dash
    src = inspect.getsource(dash.main_page)
    start = src.index("def render_lesson_script")
    end = src.index("def render_lesson_library", start)
    gate = src[start:end]

    assert 'lesson.get("warnings")' in gate, "the gate does not read the warnings"
    assert "worth a look before you render" in gate
    # ...and it says out loud that they are not a wall, so a full list does not read as a
    # refusal and send Jeffy hunting for an override.
    assert "None of these stop the render" in gate

    # The change signature must carry them, or a warning raised by a redraw would not appear
    # until something else forced a repaint.
    sig_start = src.index('sig = ((lesson or {}).get("lesson_id")')
    sig = src[sig_start:src.index("if not _changed(\"lesson_script\"", sig_start)]
    assert '.get("warnings")' in sig


def test_a_warning_is_not_a_gate(tmp_path, monkeypatch):
    # THE RULE. Only a physical floor may refuse (facts _short_takes doctrine): the
    # unconfirmed picture and the Wan cap. A thing to look at must never become a wall — a
    # check that can sink a lesson is worse than the defect it was written for.
    from modules import lesson_pipeline as lp
    from modules import lesson_writer as lw
    monkeypatch.setattr(lw, "LESSONS_DIR", tmp_path)
    lid = "20260101_000000"
    (tmp_path / lid).mkdir()
    lw._save({"lesson_id": lid, "title": "t", "stage": "stills",
              "beats": [{"narration": "hi", "approved": True, "animate": False}]})
    for code in ("person_dropped", "object_desc_locked", "negative_not_sent",
                 "prop_conform_failed", "coverage_gap"):
        lw.add_warning(lid, code, f"{code} happened", line=1)
    assert len(lw.load_lesson(lid)["warnings"]) == 5
    got = lp.approve(lid)              # must not raise
    assert got["stage"] == "approved"


def test_the_render_button_cannot_disagree_with_the_gate(page):
    # The label recomputed `unapproved` on every tick; the button captured it ONCE at build
    # time. So ticking the last picture left "✅ every picture confirmed" beside a disabled
    # button reading "🔒 1 picture(s) not confirmed" — the gate lying in the one direction
    # that makes you doubt your own eyes. ONE function, ONE read of the disk, both.
    import inspect

    import modules.dashboard_nicegui as dash
    src = inspect.getsource(dash.main_page)
    start = src.index("def _repaint_estimate")
    body = src[start:src.index("# The lesson as a storyboard", start)]
    assert "lw.unapproved(" in body, "the repaint does not re-read the ticks"
    assert "btn.enable()" in body and "btn.disable()" in body, \
        "the repaint does not own the button's state"
    assert "not confirmed" in body

    gate = src[src.index("def render_lesson_script"):src.index("def render_lesson_library")]
    assert "_gate_btn[\"btn\"] = render_btn" in gate, "the button never reaches the repaint"


def test_the_gate_makes_you_open_the_picture_before_you_tick_it(page):
    # At 104px every defect this pipeline has produced is a few pixels wide — a face on a
    # ball, mouse ears, emoji eyes, a different child entirely. The tick claims "I looked at
    # this FULL SIZE" and nothing ever checked the claim was even possible.
    #
    # It must NOT disable the tick until seen: a wiring bug there would lock Jeffy out of his
    # own render, and a check that can sink a lesson is worse than the defect it was written
    # for. One extra click, once.
    import inspect

    import modules.dashboard_nicegui as dash
    src = inspect.getsource(dash.main_page)
    gate = src[src.index("def render_lesson_script"):src.index("def render_lesson_library")]

    assert "min(90vw, 1100px)" in gate, "there is no full-size view to open"
    assert "cursor: zoom-in" in gate
    assert '_big.on("show"' in gate, "opening the picture is not recorded"
    assert "_i not in _seen" in gate, "the tick does not consult whether you looked"
    assert "_d.open()" in gate, "the first tick does not open the picture"
    # ...and it never disables the tick.
    assert "ok.disable()" not in gate and "ok.set_enabled(False)" not in gate


def test_the_gate_shows_what_the_shot_promised(page):
    # The same contract the visual check is built from — so you and it judge the picture
    # against ONE list, not two. It is what turns "does this look OK?" (unanswerable at a
    # glance) into "is she holding the blue block and nothing else?" (anyone can).
    import inspect

    import modules.dashboard_nicegui as dash
    src = inspect.getsource(dash.main_page)
    gate = src[src.index("def render_lesson_script"):src.index("def render_lesson_library")]
    assert "_lesson_contract_line(" in gate

    got = dash._lesson_contract_line({
        "line": 1, "framing": "medium",
        "people": {"count": 1, "relation": None},
        "objects": [{"key": "doll", "noun": "doll", "desc": "a small faceless blue block"}],
        "hands": {"says": "clapping her hands together"},
        "banned": ["an apple or any fruit"]})
    assert "1 child, alone" in got
    assert "a small faceless blue block" in got
    assert "clapping her hands together" in got
    assert "an apple or any fruit" in got
    assert dash._lesson_contract_line({}) == ""
