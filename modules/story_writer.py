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

Write a tiny illustrated story for kids ages 3 to 10. It reads aloud in about 30 SECONDS. AIM FOR ~75 WORDS TOTAL — a window of 70 to 85 words counting BOTH the narrator's prose AND every spoken line together. Hit that window: under ~65 words the story is too thin and rushed (add the missing beat or enrich a line); over ~90 words it runs too long (trim repeated lines). ~75 words is the sweet spot. Count the words before you finish.

# RIGHT LENGTH — FEW LINES, FULL ARC
A 30-second film holds about 6 to 8 little moments — enough for a complete arc, not a long conversation. Use AT MOST 5 short spoken lines of dialogue total, and keep each narrator sentence to one line. One quick exchange (a greeting, a question, an answer) is plenty; do NOT write a long back-and-forth.
Spend your ~75 words on a COMPLETE story, not a fragment. The three load-bearing beats must ALWAYS be present and clear: (1) the SPARK (the problem/surprise appears), (2) the TURNING POINT (the choice or action that decides the ending — e.g. the hare lying down to nap), and (3) the PAYOFF (the visible result / who wins). Trim repeated lines and chit-chat to make room — never trim one of those three.

# EVERY STORY NEEDS A COMPLETE ARC THAT ENDS ON THE PAYOFF
Walk the arc in order and STOP at the payoff: setup → spark → the character tries/struggles → the turning point that decides it → the payoff (the result we see). The LAST beat must be the payoff/result — do NOT keep going after the outcome is decided (no aimless "and then he thought about it" shots once the race is won). Each main character must behave IN CHARACTER for the tale: in the hare-and-tortoise, the RABBIT is over-confident and careless (boasts, then stops/naps), and the TORTOISE is slow but steady — never swap their attitudes or give the rabbit the tortoise's determined "one more step" lines.

# HONOR THE THEME EXACTLY (most important rule)
The theme names the subject. Use the EXACT characters, animals, and objects it names — never substitute a similar one. If the theme says "tortoise", the story has a tortoise, NOT a turtle, NOT a snail. If it says "rabbit and tortoise race", both a rabbit AND a tortoise appear and they race. If the theme is a known classic ("the tortoise and the hare", "the ant and the grasshopper"), keep its real characters and real ending. Do not swap species, rename the core animals, or change the famous outcome. Inventing a different animal than the one named is a failure.

Write like a real storyteller, not a worksheet. Use simple, warm language. Let the story breathe — small moments, a feeling, a tiny twist. Include 1 to 3 named characters with simple looks (a color, a piece of clothing, a defining feature). Give the story a small problem, a clear choice the character makes, and a payoff that lands.

# CAUSE → EFFECT THE PAYOFF (a child must understand WHY it ends this way)
The most important thing for a young child is that the ending is CAUSED by something they SAW happen earlier. Show the cause on screen, then show the result. Do not skip the turning point.
- If a character wins or loses, the reason must be a CHOICE or ACTION shown earlier in the story — not luck, not a time-skip. Show the moment that decides it.
- For "the tortoise and the hare": the hare loses because he is over-confident and STOPS — he naps, plays, or wanders off while ahead. SHOW that pause clearly (he lies down / closes his eyes / gets distracted). The tortoise keeps going and passes him. Without that nap beat the story makes no sense to a child — never omit it.
- Keep ONE clear winner when the tale has one. NEVER soften it into "they both won" / "everyone is a winner" / "they celebrated their own way." That erases the lesson. The one who earned it wins; the other learns something.
- Avoid abrupt time-skips ("hours passed", "the next day") that hide the cause. Keep the action continuous so the child sees the why.

**ONE name per character — use it every single time.** Give each character a single simple name and refer to them ONLY by that exact name throughout — in dialogue tags AND in narration. NEVER give a character a second name, nickname, title, or racing-name, and NEVER use their species as a stand-in name. If the rabbit is "Pip", he is "Pip" in every line — never also "Bunny", "the rabbit", or "Speedy". If the tortoise is "Tess", she is "Tess" everywhere — never also "Slow Steady" or "the tortoise". Two names for one character is a FAILURE.

# DON'T TELL THE MORAL, DON'T USE CLICHÉS
Never write the lesson in the narration. BANNED: "learned that…", "and so he learned", "the moral is", "you never know when…", "and that's how…". The ending shows what happened (an action, an image, a feeling) and lets the kid feel the lesson — it never states it. Also avoid the tired opener "Once upon a time"; start with the character or the world doing something.

Don't lecture. Don't label emotions ("she felt sad"). Let actions and pictures show feeling. Trust the kid to feel the moral.

# WRITE LOTS OF DIALOGUE — THE CHARACTERS TALK
This is a TALKING film. The characters speak out loud, and a narrator fills the gaps. Give the characters real voices — but keep it SHORT: at most 5 spoken lines in the whole 30-second story. A few well-chosen lines beat a long conversation. When a character speaks, it must be in quotes and named (below).
- **EVERY spoken line MUST be in double quotes AND attributed to a character BY NAME** — `Bunny said, "Watch me — I'll win for sure!"` or `"Wait for me!" called Terry.` The attribution name must be one of YOUR named characters.
- **The quotation marks wrap ONLY the spoken words. The name and the verb ("said", "called") go OUTSIDE the quotes — never inside.** WRONG: `"Pip said, "I'll win!""` (name and a second set of quotes inside). RIGHT: `Pip said, "I'll win!"`. Use the character's proper name in the tag, never the species word ("Bunny said"/"Tortoise replied" are wrong if the names are Pip and Tess).
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
    # Kill any orphan/unclosed think tags so a stray "<think>" never becomes the
    # title. (qwen3 sometimes dumps reasoning with no clean closing tag.)
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```[a-z]*", "", cleaned, flags=re.IGNORECASE).replace("```", "")
    # Drop leading "Title:", "Story:", etc. that some models add as headers.
    cleaned = re.sub(r"^(?:Title|Story|Narration)\s*[:\-]\s*", "", cleaned.strip(),
                     flags=re.IGNORECASE | re.MULTILINE)
    return cleaned.strip()


_REASONING_OPENERS = re.compile(
    r"^(here'?s|okay|ok\b|let me|let'?s think|i need to|i'?ll|i will|first,|"
    r"thinking|step \d|to write|the user|alright|so,|my task|i should)",
    re.IGNORECASE,
)


def _looks_like_reasoning_leak(body: str) -> bool:
    """A thinking model dumped its chain-of-thought instead of clean prose.
    Heuristic: far over the ~85-word story target, or opens with a reasoning
    marker. A normal kids' story runs ~70-180 words."""
    if not body:
        return True
    wc = len(body.split())
    if wc > 220:
        return True
    if _REASONING_OPENERS.match(body.strip()):
        return True
    return False


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

    # Think-leak guard: a thinking model (qwen3) occasionally dumps its reasoning
    # into the response instead of clean prose (hundreds of extra words, opening
    # with a reasoning marker). Retry ONCE — the model is usually clean on a
    # second pass. The downstream structurer can still salvage a leak, so we
    # accept the retry's output regardless.
    if _looks_like_reasoning_leak(body):
        log.warning(f"Stage-1 output looks like a reasoning leak "
                    f"({len(body.split())} words); retrying once.")
        raw2 = _call_llm(user_prompt, _STORY_SYSTEM_PROMPT, temperature=temperature)
        cleaned2 = _strip_meta(raw2)
        title2, body2 = _split_title_body(cleaned2)
        if not _looks_like_reasoning_leak(body2) or len(body2.split()) < len(body.split()):
            raw, title, body = raw2, title2, body2

    # Over-length guard: a 30-second film is ~70-85 words. The model often
    # overshoots (long dialogue scenes → 150-220 words → 12+ micro-shots
    # downstream). Retry ONCE with an explicit shorten brief, keep whichever
    # version is closer to the budget. ~110 words ≈ 44s is the trip-wire.
    if len(body.split()) > 110:
        over = len(body.split())
        log.warning(f"Stage-1 story is {over} words (>110, ~{over/2.5:.0f}s) — "
                    f"too long for a 30s film; retrying once for a tighter cut.")
        shorten_prompt = (
            user_prompt
            + f"\n\nYour previous draft was {over} words — FAR too long for a "
              f"30-second film. Tell the SAME story in 70-85 words TOTAL "
              f"(narrator prose + every spoken line combined). Use at most 5 "
              f"short spoken lines. Keep the title, characters, the one key "
              f"choice and the payoff; cut repeated beats and trim every "
              f"sentence. Count the words."
        )
        raw3 = _call_llm(shorten_prompt, _STORY_SYSTEM_PROMPT, temperature=0.7)
        cleaned3 = _strip_meta(raw3)
        title3, body3 = _split_title_body(cleaned3)
        # Accept the retry only if it is shorter and not a reasoning leak.
        if body3 and not _looks_like_reasoning_leak(body3) and \
                len(body3.split()) < len(body.split()):
            log.info(f"Shorten retry: {over} -> {len(body3.split())} words.")
            raw, title, body = raw3, (title3 or title), body3

    # Under-length guard: a 30-second film wants ~70-85 words. The model often
    # writes a tight 40-60 word arc that renders a thin ~20s video. Retry ONCE
    # asking to ENRICH to the target while keeping the SAME arc; keep whichever
    # version lands closer to ~75 words. ~62 words ≈ 25s is the trip-wire.
    elif len(body.split()) < 62:
        under = len(body.split())
        log.warning(f"Stage-1 story is {under} words (<62, ~{under/2.5:.0f}s) — "
                    f"too thin for a 30s film; retrying once to enrich.")
        enrich_prompt = (
            user_prompt
            + f"\n\nYour previous draft was only {under} words — too short for a "
              f"30-second film, which needs about 75 words (70-85). Tell the SAME "
              f"story with the SAME beats, characters, turning point and payoff, "
              f"but ENRICH it to ~75 words: add a one-line opening that sets the "
              f"scene, give the narrator sentences a little more vivid detail, and "
              f"add at most one more short spoken line. Do NOT add new plot or new "
              f"characters. Count the words — land between 70 and 85."
        )
        raw3 = _call_llm(enrich_prompt, _STORY_SYSTEM_PROMPT, temperature=0.85)
        cleaned3 = _strip_meta(raw3)
        title3, body3 = _split_title_body(cleaned3)
        # Accept only if it grew toward the target and isn't a reasoning leak/over-long.
        n3 = len(body3.split())
        if body3 and not _looks_like_reasoning_leak(body3) and under < n3 <= 100:
            log.info(f"Enrich retry: {under} -> {n3} words.")
            raw, title, body = raw3, (title3 or title), body3

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
