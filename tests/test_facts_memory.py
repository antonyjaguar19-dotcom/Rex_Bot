"""The bot must not tell you the same fact twice.

Ask Qwen for facts about octopuses three times and you get the three famous ones
three times — a model reaching for the most surprising fact reaches for the same
one every run. The second reel is then a re-upload with new pictures.

The load-bearing part is not the prompt ("don't repeat these") — it is the CHECK.
A model told not to repeat itself rewords the fact instead, and a reworded fact is
the same fact.
"""

import json

import pytest

from modules import facts_memory as fm
from modules import facts_writer as fw


@pytest.fixture(autouse=True)
def memory(tmp_path, monkeypatch):
    monkeypatch.setattr(fm, "MEMORY_PATH", tmp_path / "_memory.json")
    monkeypatch.setattr(fm, "PROJECT_ROOT", tmp_path)
    (tmp_path / "04_Outputs" / "facts").mkdir(parents=True)
    return tmp_path


# --- the check --------------------------------------------------------------

def test_a_reworded_fact_is_the_same_fact():
    """This is the whole feature. The prompt alone does not stop it."""
    prior = ["Octopuses have three hearts."]
    assert fm.is_repeat("An octopus has three hearts!", prior) == prior[0]
    assert fm.is_repeat("Octopuses, it turns out, have three hearts.", prior)


def test_a_genuinely_different_fact_gets_through():
    prior = ["Octopuses have three hearts."]
    assert fm.is_repeat("Two of an octopus's hearts stop beating when it swims.",
                        prior) is None
    assert fm.is_repeat("Octopus blood is blue because it carries copper.",
                        prior) is None


def test_the_number_is_part_of_the_fact():
    """'three hearts' and 'nine brains' must not merge just because both are
    'octopuses have N X'. Numbers are kept as tokens for exactly this."""
    prior = ["Octopuses have three hearts."]
    assert fm.is_repeat("Octopuses have nine brains.", prior) is None


def test_topics_are_normalised_but_not_invented():
    assert fm.topic_key("The Deep Ocean!") == fm.topic_key("deep ocean")
    assert fm.topic_key("Octopus") == fm.topic_key("octopuses")
    # Crude on purpose: a synonym is NOT the same topic, and pretending otherwise
    # would starve a legitimate new reel.
    assert fm.topic_key("octopus") != fm.topic_key("cephalopods")


# --- the memory -------------------------------------------------------------

def test_facts_are_remembered_per_topic():
    fm.record("bees", "20260713_120000", ["Bees have five eyes.",
                                          "A bee flaps its wings 230 times a second."])
    assert len(fm.seen_facts("Bees")) == 2
    assert fm.seen_facts("octopuses") == []
    assert fm.all_topics() == {"bee": 2}


def test_recording_the_same_reel_twice_does_not_double_it():
    facts = ["Bees have five eyes."]
    assert fm.record("bees", "20260713_120000", facts) == 1
    assert fm.record("bees", "20260713_120000", facts) == 0
    assert len(fm.seen_facts("bees")) == 1


def test_forget_topic_and_forget_reel():
    fm.record("bees", "20260713_120000", ["Bees have five eyes."])
    fm.record("bees", "20260713_130000", ["Bee brains map flowers."])
    assert fm.forget_reel("20260713_120000") == 1
    assert fm.seen_facts("bees") == ["Bee brains map flowers."]
    assert fm.forget_topic("bees") == 1
    assert fm.seen_facts("bees") == []


def test_the_avoid_clause_carries_the_old_facts_into_the_prompt():
    fm.record("bees", "20260713_120000", ["Bees have five eyes."])
    clause = fm.avoid_clause("bees")
    assert "Bees have five eyes." in clause and "NOT appear again" in clause
    assert fm.avoid_clause("volcanoes") == ""


def test_backfill_learns_from_the_reels_already_on_disk(memory):
    """The memory ships onto a machine with 31 reels in it. Starting empty means
    the first repeat topic repeats itself — which is the bug."""
    facts_dir = memory / "04_Outputs" / "facts"
    (facts_dir / "facts_20260709_072924.json").write_text(json.dumps({
        "facts_id": "20260709_072924", "topic": "octopuses",
        "beats": [{"kind": "hook", "narration": "Three things about octopuses."},
                  {"kind": "fact", "narration": "Octopuses have three hearts."},
                  {"kind": "fact", "narration": "Octopus blood is blue."},
                  {"kind": "outro", "narration": "Follow for more."}],
    }), encoding="utf-8")
    # A placeholder reel's narration is not a fact — remembering it would poison
    # the topic forever.
    (facts_dir / "facts_20260709_081210.json").write_text(json.dumps({
        "facts_id": "20260709_081210", "topic": "bees", "_placeholder": True,
        "beats": [{"kind": "fact",
                   "narration": "Here is an interesting thing about bees number 1."}],
    }), encoding="utf-8")

    assert fm.backfill_from_reels() == 2       # hook and outro are not facts
    assert fm.seen_facts("octopuses") == ["Octopuses have three hearts.",
                                          "Octopus blood is blue."]
    assert fm.seen_facts("bees") == []
    assert fm.backfill_from_reels() == 0       # runs once


# --- the writer -------------------------------------------------------------

def _reel(facts: list) -> str:
    return json.dumps({
        "title": "Octopus Facts", "hook": "Three things you didn't know.",
        "facts": [{"spoken": f, "caption": f[:20], "backdrop": "an octopus"}
                  for f in facts],
        "outro": "Follow for more.",
    })


def test_the_writer_drops_a_fact_it_has_already_told(monkeypatch):
    fm.record("octopuses", "20260713_120000", ["Octopuses have three hearts."])

    answers = [
        # First roll repeats the known fact, reworded, and pads with new ones.
        _reel(["An octopus has three hearts.",
               "Octopus blood is blue because of copper.",
               "They can taste with their arms.",
               "An octopus can squeeze through a coin-sized gap.",
               "They edit their own RNA."]),
    ]
    monkeypatch.setattr(fw, "_call_llm", lambda *a, **k: answers.pop(0))
    monkeypatch.setattr(fw, "OUTPUTS_DIR", fm.PROJECT_ROOT / "04_Outputs" / "facts")

    story = fw.generate_facts_short("octopuses", n_facts=5)
    spoken = [b["narration"] for b in story["beats"] if b["kind"] == "fact"]

    assert not any("three hearts" in s for s in spoken), "the old fact came back"
    assert len(spoken) == 4
    # And what it DID say is now on the list for next time.
    assert "They edit their own RNA." in fm.seen_facts("octopuses")


def test_a_reel_of_nothing_but_repeats_is_re_rolled_then_refused(monkeypatch):
    """An exhausted topic must fail loudly. Silently shipping four repeats is a
    re-upload, and this pipeline has paid for silent downgrades before."""
    old = ["Octopuses have three hearts.", "Octopus blood is blue.",
           "They taste with their arms.", "They edit their own RNA.",
           "They squeeze through coin-sized gaps."]
    fm.record("octopuses", "20260713_120000", old)

    calls = []

    def fake(prompt, *a, **k):
        calls.append(prompt)
        return _reel(old)                       # the same five, every time
    monkeypatch.setattr(fw, "_call_llm", fake)

    with pytest.raises(fw.FactsUnavailable) as e:
        fw.generate_facts_short("octopuses", n_facts=5)

    assert len(calls) == fw.LLM_ATTEMPTS, "a stale roll must be re-rolled"
    assert "already been used" in str(e.value)
    # The retry has to be TOLD what it repeated, or it just repeats it.
    assert "Those are burned" in calls[-1]


def test_a_thin_roll_is_not_blamed_on_the_memory(monkeypatch):
    """The model wrote one fact and nothing was repeated. That is a thin roll, not
    an exhausted topic — saying "everything was already used (0 facts on record)"
    sends you off clearing a memory that is empty."""
    monkeypatch.setattr(fw, "_call_llm", lambda *a, **k: _reel(["Only one."]))
    with pytest.raises(fw.FactsUnavailable, match="usable facts"):
        fw.generate_facts_short("volcanoes", n_facts=5)


def test_a_fresh_topic_is_written_normally(monkeypatch):
    monkeypatch.setattr(fw, "_call_llm", lambda *a, **k: _reel(
        ["Volcanoes can make lightning.", "Lava reaches 1200 degrees.",
         "There are volcanoes under the sea.", "Some erupt for decades.",
         "Ash can circle the globe."]))
    monkeypatch.setattr(fw, "OUTPUTS_DIR", fm.PROJECT_ROOT / "04_Outputs" / "facts")

    story = fw.generate_facts_short("volcanoes", n_facts=5)
    assert len([b for b in story["beats"] if b["kind"] == "fact"]) == 5
    assert len(fm.seen_facts("volcanoes")) == 5
