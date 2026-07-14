"""The daily auto-reel: on, off, and at an hour you choose.

It used to be unconditional — hardcoded ON at 09:00 IST with no way to stop it — so
the GPU vanished for an hour every morning whether anyone wanted a reel that day or
not. And it rendered without taking the job lock, meaning the 9am reel could land on
top of a render you started by hand.
"""
import pytest

from modules import runtime_settings as rs


@pytest.fixture(autouse=True)
def settings(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "SETTINGS_PATH", tmp_path / "runtime_settings.json")
    return rs


def test_it_stays_on_by_default():
    """Default ON is what the bot has always done. Flipping the default to OFF would
    silently stop a machine that was relying on the morning reel."""
    assert rs.get_daily_facts_enabled() is True
    assert rs.get_daily_facts_hour() == 9


def test_off_and_back_on():
    rs.set_daily_facts_enabled(False)
    assert rs.get_daily_facts_enabled() is False
    rs.set_daily_facts_enabled(True)
    assert rs.get_daily_facts_enabled() is True


def test_the_hour_round_trips_and_rejects_nonsense():
    rs.set_daily_facts_hour(7)
    assert rs.get_daily_facts_hour() == 7
    for bad in (-1, 24, 99):
        with pytest.raises(ValueError):
            rs.set_daily_facts_hour(bad)
    assert rs.get_daily_facts_hour() == 7        # unchanged by the failed writes


def test_a_junk_hour_on_disk_falls_back_to_nine():
    rs._save({"daily_facts_hour": "breakfast"})
    assert rs.get_daily_facts_hour() == 9


def test_the_last_run_is_remembered():
    """The tick fires HOURLY, so this date is what stops a restart at 09:30 from
    rendering a second reel the same morning."""
    assert rs.get_daily_facts_last_run() == ""
    rs.set_daily_facts_last_run("2026-07-13")
    assert rs.get_daily_facts_last_run() == "2026-07-13"


# --- the tick's decision table (the logic, without importing discord) ---------

def _should_run(now_hour: int, today: str) -> bool:
    """Mirror of claw_bot.daily_tick()'s decision, so it can be tested without a
    Discord client. Kept honest by test_the_tick_matches_this_table below."""
    if not rs.get_daily_facts_enabled():
        return False
    if now_hour != rs.get_daily_facts_hour():
        return False
    return rs.get_daily_facts_last_run() != today


def test_the_tick_only_fires_at_the_chosen_hour():
    rs.set_daily_facts_hour(9)
    assert _should_run(9, "2026-07-13") is True
    assert _should_run(8, "2026-07-13") is False
    assert _should_run(10, "2026-07-13") is False


def test_the_tick_does_nothing_when_switched_off():
    rs.set_daily_facts_enabled(False)
    assert _should_run(9, "2026-07-13") is False


def test_the_tick_will_not_make_two_reels_in_one_day():
    rs.set_daily_facts_last_run("2026-07-13")
    assert _should_run(9, "2026-07-13") is False
    assert _should_run(9, "2026-07-14") is True


def test_the_tick_matches_this_table():
    """The table above is a copy, and a copy rots. Pin it to the real source."""
    import inspect
    from pathlib import Path
    src = Path(__file__).parent.parent / "claw_bot.py"
    body = src.read_text(encoding="utf-8").split("async def daily_tick")[1] \
              .split("async def daily_auto_generation")[0]
    assert "get_daily_facts_enabled()" in body
    assert "get_daily_facts_hour()" in body
    assert "get_daily_facts_last_run()" in body


def test_the_daily_reel_takes_the_gpu_lock():
    """It used to render straight onto the card. Two jobs on a 16 GB card is how a
    90-second clip becomes 16 minutes of thrashing."""
    from pathlib import Path
    src = (Path(__file__).parent.parent / "claw_bot.py").read_text(encoding="utf-8")
    body = src.split("async def daily_auto_generation")[1].split("\nasync def ")[0]
    assert "job_lock.acquire(" in body, "the daily reel must queue behind the GPU lock"
    assert "job_lock.release()" in body
