"""
Claw Bot — Story Writer (Stage 1)

Pure-prose storytelling pass. Hands the LLM a chat-style brief and lets it
write a real kids' story the way a chatbot would — no JSON, no schemas, no
banned-word lists, no rule walls. The output is then parsed by the stage-2
structurer (script_generator._structure_story) into the production JSON.

Why two stages: the old single-pass system prompt was ~150 lines of rules.
Qwen 2.5 14B treated it like a checklist and emitted mechanical fill-in-the-
blank stories. Splitting story (creative) from structure (mechanical) lets
each stage do what it does best.
"""

import json
import logging
import re
import time as _t
from typing import Optional

import requests

from modules import model_registry

log = logging.getLogger("claw_bot.story_writer")


# ==============================================================================
# CHAT-STYLE SYSTEM PROMPT — minimal, warm, no schema
# ==============================================================================

_STORY_SYSTEM_PROMPT = """You are a beloved children's storyteller — the kind of writer who makes parents tear up and kids ask "again!"

Write a tiny illustrated story for kids ages 3 to 10. Around 30 seconds when read aloud — roughly 70 to 85 words of narration total.

# HONOR THE THEME EXACTLY (most important rule)
The theme names the subject. Use the EXACT characters, animals, and objects it names — never substitute a similar one. If the theme says "tortoise", the story has a tortoise, NOT a turtle, NOT a snail. If it says "rabbit and tortoise race", both a rabbit AND a tortoise appear and they race. If the theme is a known classic ("the tortoise and the hare", "the ant and the grasshopper"), keep its real characters and real ending. Do not swap species, rename the core animals, or change the famous outcome. Inventing a different animal than the one named is a failure.

Write like a real storyteller, not a worksheet. Use simple, warm language. Let the story breathe — small moments, a feeling, a tiny twist. Include 1 to 3 named characters with simple looks (a color, a piece of clothing, a defining feature). Give the story a small problem, a clear choice the character makes, and a payoff that lands.

# DON'T TELL THE MORAL, DON'T USE CLICHÉS
Never write the lesson in the narration. BANNED: "learned that…", "and so he learned", "the moral is", "you never know when…", "and that's how…". The ending shows what happened (an action, an image, a feeling) and lets the kid feel the lesson — it never states it. Also avoid the tired opener "Once upon a time"; start with the character or the world doing something.

Don't lecture. Don't label emotions ("she felt sad"). Let actions and pictures show feeling. Trust the kid to feel the moral.

# WRITE LOTS OF DIALOGUE — THE CHARACTERS TALK
This is a TALKING film. The characters speak out loud, and a narrator fills the gaps. Make it dialogue-heavy: most of the story should be spoken by the characters in their own voices and personalities.
- **EVERY spoken line MUST be in double quotes AND attributed to a character BY NAME** — `Bunny said, "Watch me — I'll win for sure!"` or `"Wait for me!" called Terry.` The attribution name must be one of YOUR named characters.
- **NEVER let a character's words appear as plain narration.** If someone asks, suggests, exclaims, or says anything — even one word like "Hmm…" or "Wait!" — it goes in quotes with a name. A spoken line written as bare narration (no quotes, no name) is a FAILURE.
- One speaker per quoted line. For a back-and-forth, alternate and attribute EACH line: `"But why you?" asked Terry. "Because I'm fastest!" crowed Bunny.`
- Keep spoken lines short and natural for a young child (6-14 words).
- The NARRATOR (unquoted prose) only DESCRIBES — what we see, the scene, an action, a passage of time. The narrator never speaks a character's thoughts or words. Use narrator lines to open, bridge, and close; let the CHARACTERS carry the middle.
- Still DON'T state the moral and DON'T label emotions — show feeling through what characters say and do.

# TINY EXAMPLE OF THE RIGHT SHAPE
Redcomb puffed out his bright red chest at dawn.
"I should wake the whole farm every morning!" he crowed.
"But why only you?" asked Featherlite, tilting her head.
"What if we both do it — your crow, then my song?" she offered.
Redcomb thought a moment. "Hmm… that does sound nice," he admitted.
And every morning after, his loud crow was followed by her sweet song.
(Notice: every spoken line is quoted AND named; only the scene-setting first and last lines are narration.)

Output PLAIN PROSE only. No headers, no JSON, no shot lists, no markdown. Open with the title on its own first line, then a blank line, then the story (mixing narrator prose and quoted character dialogue). That's it."""


# ==============================================================================
# OLLAMA CALL
# ==============================================================================

def _call_llm(user_prompt: str, system_prompt: str, *, temperature: float = 0.95) -> str:
    cfg = model_registry.get_for_role("creative") or model_registry.get_active("llm_backend")
    model_name = (
        cfg.get("model_id") or cfg.get("model_name") or cfg.get("model")
        or "qwen2.5:14b-instruct-q6_K"
    )
    url = cfg.get("server_url", "http://127.0.0.1:11434") + "/api/generate"

    combined = (
        f"SYSTEM INSTRUCTIONS (STRICTLY FOLLOW THESE):\n{system_prompt}\n\n"
        f"USER REQUEST:\n{user_prompt}"
    )
    payload = {
        "model": model_name,
        "prompt": combined,
        "stream": False,
        "think": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.95,
            "num_ctx": 8192,
            # Headroom: thinking models (qwen3) spend tokens on a <think> block
            # BEFORE the prose. 1024 could truncate mid-think on longer stories,
            # leaking an unclosed block. 2560 lets think close + prose follow.
            "num_predict": 2560,
        },
    }
    log.info(f"Stage 1 (story prose) → {model_name}, temp={temperature}")
    r = requests.post(url, json=payload, timeout=300)
    r.raise_for_status()
    return r.json().get("response", "").strip()


# ==============================================================================
# OUTPUT NORMALIZATION
# ==============================================================================

def _strip_meta(text: str) -> str:
    """Remove markdown fences, <think> blocks, leading labels the LLM may add.

    Thinking models (e.g. qwen3) prefix output with a <think>...</think> block
    that ollama's think=False does NOT reliably suppress. Strip the closed block;
    if the answer follows a CLOSED think, keep it. As a fallback, drop a leading
    block from the start up to the last </think> (handles nested/odd markers)."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Fallback: any stray closing tag → keep only what comes AFTER the last one
    # (the real answer), discarding leaked reasoning before it.
    if "</think>" in cleaned.lower():
        idx = cleaned.lower().rfind("</think>")
        cleaned = cleaned[idx + len("</think>"):]
    cleaned = re.sub(r"```[a-z]*", "", cleaned, flags=re.IGNORECASE).replace("```", "")
    # Drop leading "Title:", "Story:", etc. that some models add as headers.
    cleaned = re.sub(r"^(?:Title|Story|Narration)\s*[:\-]\s*", "", cleaned.strip(),
                     flags=re.IGNORECASE | re.MULTILINE)
    return cleaned.strip()


def _split_title_body(text: str) -> tuple[str, str]:
    """First non-empty line = title. Rest = story body."""
    lines = [ln.rstrip() for ln in text.splitlines()]
    title = ""
    body_start = 0
    for i, ln in enumerate(lines):
        if ln.strip():
            title = ln.strip().strip("#*_\"' ").strip()
            body_start = i + 1
            break
    body = "\n".join(lines[body_start:]).strip()
    # If everything ran together on one paragraph, the title may be the first
    # sentence — try splitting on ". " in that case
    if not body and title and "." in title:
        first, _, rest = title.partition(".")
        if 2 <= len(first.split()) <= 12 and rest.strip():
            title, body = first.strip(), rest.strip()
    return title or "Untitled Story", body


# ==============================================================================
# PUBLIC API
# ==============================================================================

def write_story(
    theme: str,
    style_hint: Optional[str] = None,
    culture_hint: Optional[str] = None,
    temperature: float = 0.95,
) -> dict:
    """
    Stage 1: hand the LLM a chat-style brief, get back flowing prose.

    Returns:
        {
          "title": str,
          "prose": str,           # the full narration body, written for ear
          "raw":   str,           # unparsed model output (for debugging)
          "duration_sec": float,
        }
    """
    _start = _t.time()
    user_prompt = f"Theme: {theme}"
    if style_hint:
        user_prompt += f"\nVisual style preference: {style_hint}"
    if culture_hint:
        user_prompt += f"\nCultural setting preference: {culture_hint}"
    user_prompt += (
        "\n\nWrite the story now. Title on line 1, blank line, then the story. "
        "Plain prose only."
    )

    raw = _call_llm(user_prompt, _STORY_SYSTEM_PROMPT, temperature=temperature)
    cleaned = _strip_meta(raw)
    title, body = _split_title_body(cleaned)

    word_count = len(body.split())
    log.info(
        f"Stage 1 done — title='{title}', body={word_count} words, "
        f"took {_t.time() - _start:.1f}s"
    )
    return {
        "title": title,
        "prose": body,
        "raw": raw,
        "duration_sec": round(_t.time() - _start, 1),
        "word_count": word_count,
    }


def revise_prose(
    previous_prose: str,
    previous_title: str,
    feedback: str,
    theme: str = "",
    temperature: float = 0.85,
    length_instruction: str = "",
) -> dict:
    """
    Stage 1 for revisions: rewrite the prose only, guided by feedback.
    Stays in chat-style register; the structurer re-shapes JSON downstream.

    length_instruction: optional override for the length target. When the
    feedback asks to expand ("more shots", "more detail", "longer"), the caller
    passes an explicit longer target here, which SUPERSEDES the 70-85 word cap
    baked into the system prompt. Empty = keep the default ~70-85 words.
    """
    _start = _t.time()
    length_line = length_instruction.strip() or "Keep the same target length (~70-85 words)."
    user_prompt = (
        f"Here is a previous version of a kids' story.\n\n"
        f"TITLE: {previous_title}\n\n"
        f"STORY:\n{previous_prose}\n\n"
        f"The reader gave this feedback:\n\"{feedback}\"\n\n"
        f"Rewrite the story applying the feedback. Keep the warm, simple voice. "
        f"{length_line} Title on line 1, blank line, then the "
        f"revised story. Plain prose only — no JSON, no labels."
    )
    if theme:
        user_prompt += f"\n\nOriginal theme: {theme}"

    raw = _call_llm(user_prompt, _STORY_SYSTEM_PROMPT, temperature=temperature)
    cleaned = _strip_meta(raw)
    title, body = _split_title_body(cleaned)
    word_count = len(body.split())
    log.info(
        f"Stage 1 revise done — title='{title}', body={word_count} words, "
        f"took {_t.time() - _start:.1f}s"
    )
    return {
        "title": title or previous_title,
        "prose": body or previous_prose,
        "raw": raw,
        "duration_sec": round(_t.time() - _start, 1),
        "word_count": word_count,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    import sys
    theme = " ".join(sys.argv[1:]) or "a shy little mouse learning to speak up"
    out = write_story(theme)
    print("=" * 60)
    print(f"TITLE: {out['title']}")
    print("=" * 60)
    print(out["prose"])
    print("=" * 60)
    print(f"Words: {out['word_count']}, took {out['duration_sec']}s")
