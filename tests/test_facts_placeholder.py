"""facts_writer must never silently ship placeholder narration.

With Ollama down, generate_facts_short() used to fall back to _fallback(), whose
lines read "Here is an interesting thing about honeybees number 1." That reel was
then voiced, subtitled, rendered and posted as if it were real content.
"""

import pytest

from modules import facts_writer as fw


@pytest.fixture(autouse=True)
def no_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(fw, "OUTPUTS_DIR", tmp_path)
    yield


def _break_llm(monkeypatch, exc=RuntimeError("ollama is down")):
    def boom(*a, **k):
        raise exc
    monkeypatch.setattr(fw, "_call_llm", boom)


def test_llm_down_raises_instead_of_placeholder(monkeypatch):
    _break_llm(monkeypatch)
    with pytest.raises(fw.FactsUnavailable) as e:
        fw.generate_facts_short("honeybees", 6)
    assert "Is Ollama running?" in str(e.value)
    assert "nothing was rendered" in str(e.value).lower()


def test_placeholder_text_never_reaches_a_default_call(monkeypatch):
    _break_llm(monkeypatch)
    try:
        story = fw.generate_facts_short("honeybees", 6)
    except fw.FactsUnavailable:
        return                      # the only acceptable outcome
    narration = " ".join(b.get("narration", "") for b in story["beats"]).lower()
    pytest.fail(f"placeholder shipped: {narration[:80]}")


def test_opt_in_placeholder_still_works_for_offline_tests(monkeypatch):
    _break_llm(monkeypatch)
    story = fw.generate_facts_short("honeybees", 6, allow_placeholder=True)
    assert story["_placeholder"] is True
    joined = " ".join(b.get("narration", "") for b in story["beats"]).lower()
    assert "interesting thing about honeybees" in joined   # the old stub text


def test_real_llm_output_is_not_marked_placeholder(monkeypatch):
    facts = [{"spoken": f"Bees do thing {i}.", "caption": f"Fact {i}",
              "backdrop": "a bee"} for i in range(6)]
    monkeypatch.setattr(fw, "_call_llm", lambda *a, **k: "{}")
    monkeypatch.setattr(fw, "_extract_json", lambda raw: {
        "title": "Bee Facts", "hook": "Six things.", "facts": facts,
        "outro": "Follow."})
    story = fw.generate_facts_short("bees", 6)
    assert story["_placeholder"] is False
    assert story["title"] == "Bee Facts"


def test_too_few_usable_facts_raises(monkeypatch):
    monkeypatch.setattr(fw, "_call_llm", lambda *a, **k: "{}")
    monkeypatch.setattr(fw, "_extract_json", lambda raw: {
        "title": "Thin", "hook": "h",
        "facts": [{"spoken": "only one.", "caption": "c", "backdrop": "b"}],
        "outro": "o"})
    with pytest.raises(fw.FactsUnavailable, match="usable facts"):
        fw.generate_facts_short("bees", 6)


def test_empty_facts_list_raises(monkeypatch):
    monkeypatch.setattr(fw, "_call_llm", lambda *a, **k: "{}")
    monkeypatch.setattr(fw, "_extract_json", lambda raw: {"title": "T", "facts": []})
    with pytest.raises(fw.FactsUnavailable):
        fw.generate_facts_short("bees", 6)
