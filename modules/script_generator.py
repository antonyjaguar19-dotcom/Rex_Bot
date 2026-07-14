"""
Claw Bot — Script Generator (Session 1A)

Generates a 30-second kids' story script in JSON format.

Changes from previous version:
- LLM picks style ("pixar", "cartoon_saloon", "claymation", "stickman")
  based on story mood; user can override via revision feedback.
- LLM picks culture/setting (Indian, Western, Japanese, mixed, animal-kingdom,
  fantasy) based on theme — no longer hardcoded to Indian.
- LLM picks character types (human, animal, mixed) as fits the story.
- Safety filter still applies. Output schema extended with 'style' and 'culture'.
"""

import json
import logging
import re
import sys
import time as _t
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modules import model_registry
from modules import safety_filter as sf
from modules import runtime_settings as rs
from modules import generation_meta as gm
from modules import feedback_thinker as ft
from modules import story_writer as sw
from modules.file_utils import atomic_write_json

log = logging.getLogger("claw_bot.script_generator")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
OUTPUTS_DIR = PROJECT_ROOT / "04_Outputs" / "scripts"
STYLES_PATH = PROJECT_ROOT / "05_Config" / "styles.json"


# ==============================================================================
# STYLE LIBRARY ACCESSOR
# ==============================================================================

def _load_styles() -> dict:
    if not STYLES_PATH.exists():
        return {"default": "pixar", "available": {}}
    return json.loads(STYLES_PATH.read_text(encoding="utf-8"))


def get_available_style_ids() -> list[str]:
    styles = _load_styles()
    return list(styles.get("available", {}).keys())


def get_style_ids_for_mode(mode: str = "story") -> list[str]:
    """Style ids tagged for a pipeline mode (story|music_video|horror_story).
    Untagged styles (no 'modes' key) are treated as story styles."""
    out = []
    for sid, info in _load_styles().get("available", {}).items():
        modes = info.get("modes") or ["story"]
        if mode in modes:
            out.append(sid)
    return out


def get_style_description(style_id: str) -> dict:
    styles = _load_styles()
    return styles.get("available", {}).get(style_id, {})


def get_default_style() -> str:
    styles = _load_styles()
    return styles.get("default", "pixar")


# ==============================================================================
# SYSTEM PROMPT — the heart of the generator
# ==============================================================================

def _build_system_prompt() -> str:
    available_styles = get_style_ids_for_mode("story")
    styles_data = _load_styles().get("available", {})
    style_guide_lines = []
    for sid in available_styles:
        info = styles_data.get(sid, {})
        style_guide_lines.append(
            f'  - "{sid}": {info.get("description", "")} '
            f'Best for: {info.get("best_for", "")}'
        )
    style_guide = "\n".join(style_guide_lines)

    return f"""You output ONLY valid JSON. No prose, no preamble, no markdown fences. Start with {{ end with }}.

You are Claw Bot, a master children's storyteller for ages 3-10. You write tiny illustrated stories that feel like Pixar shorts or Bluey episodes — short, but with a real heart.

# WHAT MAKES A GOOD STORY (read this carefully)

A story is NOT a list of things that happen. A story has a SHAPE:

1. **HOOK** — show the character WANTING something or FEELING something specific. Not "a boy walks down a path" — instead "a boy is hurrying home before the rain". The audience must care in the first 3 seconds.

2. **SPARK** — something unexpected happens. A problem, a discovery, an obstacle, a stranger, a small disaster. This is what makes the story start. Without a spark, you just have someone existing.

3. **STRUGGLE** — the character tries something, or hesitates, or has to figure it out. This is where the moral lives — not in narration, but in WHAT THE CHARACTER CHOOSES TO DO.

4. **CHOICE / TURN** — the character makes a clear, visible decision. They could have walked away. They didn't. THIS is the heart of the story.

5. **CONSEQUENCE** — the world responds to the choice. Something visibly changes — a smile, a transformation, a
reward, a small wonder. The audience FEELS the payoff.

If your story is missing the SPARK, the CHOICE, or the CONSEQUENCE — start over. A flat sequence of "and then...
and then... and then..." is a failure.

# A WORKED EXAMPLE OF GOOD STORY SHAPE (fable density, scaled to 30 seconds)

Theme: "a proud rabbit and a steady tortoise race." A strong 7-shot version reads like this (narration only shown):
  1. (atmosphere) "Morning dew sparkled on the meadow as the woodland animals gathered to watch the great race."
  2. (hook) "Rex the rabbit grinned, sure his fast legs would beat the slow little tortoise easily."
  3. (spark) "The whistle blew, and Rex shot ahead in a blur while Terry plodded calmly behind."
  4. (struggle) "Far in front, Rex yawned and curled beneath a shady tree for a quick little nap."
  5. (observation) "Step after patient step, Terry kept moving, never once stopping to rest or look back."
  6. (choice/turn) "Terry passed the sleeping rabbit quietly and pressed on toward the distant finish line."
  7. (consequence) "Rex woke too late — Terry had won, and the meadow cheered for the steady little tortoise."
  8. (Moral) Slow and steady wins the race.
  
Notice: ~84 words total, full sentences, clear arc, a real turn, an earned ending. THIS is the density and shape you aim for. Match this richness — never produce thin one-line shots.

# BREATHING SHOTS (cinematography that lets emotion land)

Children's stories for ages 3-10 NEED pauses to let kids absorb what's happening. A pure-plot sequence feels rushed
and emotionally flat. Use these shot TYPES alongside the 5 beats above:

- **atmosphere** — wide establishing shot setting the mood (an empty playground at dawn, a quiet kitchen). No
character action; pure setting. Often opens the story.
- **reaction** — a close-up on a character's face right after something happened. No new action — just feeling. STILL needs a narration line — observational, describing what we see (e.g. "The ball flies past Buddy's nose." or "She pauses, eyes wide.") — NEVER leave narration empty.
- **moment-of-decision** — a held beat BEFORE the choice. The character looks down, hesitates, takes a breath.
Tension is the point.
- **observation** — the character watches something happen to others. Quiet, contemplative.

These are NOT padding. They are STORYTELLING beats that earn their place when the story needs to slow down for
emotional weight. A 30-second story can absolutely have an atmosphere shot AND a reaction shot.

You may use any of these as the FIRST shot (before the hook) if it earns the opening — like a wide atmospheric
shot establishing the world before the character appears.

Mark these in the `beat` field exactly as listed: `"atmosphere"`, `"reaction"`, `"moment-of-decision"`, `"observation"`.

# RULES

1. **Shot count & length: hit the 30-second buffer.** The finished video is ~30 seconds. Narration is read aloud at ~2.5 words/second, so the WHOLE story needs roughly **70-85 words of narration total** across all shots. Shot count is whatever it takes to reach that — usually **6-9 shots**. Do NOT stop at 5 thin shots; that leaves the video half-empty. Each shot's narration should be a full, vivid sentence (10-14 words), not a fragment. Count your words: if the total is under 70, the story is too short — add a beat or enrich the narration until it lands between 70 and 85 words.

2. **Each shot serves a purpose — but purposes include EMOTION, not just plot.** A shot is justified if it
advances plot OR lets the audience feel something. "He looks down at the broken cup, eyes wide" is a valid shot
even though no plot moved — it gives the choice that comes next its emotional weight. Don't delete reaction or
atmosphere shots just because plot didn't move; delete shots that do NEITHER (a character walking from A to B
when arrival is what matters).

3. **Show, don't tell.** Narration describes what the audience SEES, never what the character feels internally. NOT "she felt sad" — instead "she sat alone, head in her hands". NOT "he learned to share" — instead "he held out his cookie".

4. **Narration: 10-14 words per shot. NEVER empty. NEVER a fragment.** Write FULL sentences a picture-book narrator would read aloud — like a classic fable. Specific verbs, warm simple language, no emotion-labels. Breathing shots (atmosphere/reaction/observation/moment-of-decision) ALSO need full narration — observational instead of action-driven. 

GOOD (full, fable-like): "The morning sun spilled across the meadow as the animals gathered to watch." 
GOOD: "Rex laughed and bounded ahead, certain the slow tortoise could never catch him." 
BAD (fragment, too thin): "Rex runs fast." 
BAD: "Terry is slow." 

Each shot should advance the story AND add texture — a detail of the world, a flicker of feeling, a turn of events. Aim the whole story at 70-85 narration words total.

5. **Camera variety (CRITICAL CAMERA RULE):** For every `visual_description`, `first_frame_prompt`, and `last_frame_prompt`, you MUST specify a dynamic cinematic camera angle. Never use flat, side-by-side theatrical staging. Alternate between Extreme Close-Ups (ECU), Over-The-Shoulder (OTS), High-Angle Drone shots, and Low-Angle tracking shots. If two characters are interacting, use alternating close-ups or OTS shots instead of placing them side-by-side.

5b. **Shot type grammar (REQUIRED FIELD `shot_type`):** every shot declares its framing in a `shot_type` field. Pick exactly one:
- `"wide"` — establishing shot. The whole location is visible; any character is small in frame. REQUIRED for the first shot in every new location, and good for atmosphere beats.
- `"medium"` — character action, waist-up or full body. The default for plot beats (hook/spark/struggle/choice).
- `"closeup"` — the face fills the frame. Use for reaction and moment-of-decision beats so the emotion lands.
- `"insert"` — extreme close-up of HANDS + OBJECT (a paw, a beak, or fingers gripping something counts). REQUIRED whenever a character picks up, holds, drops, places, or manipulates an object — show the action detail, not the whole body. The face may be out of frame entirely.
Across the story: at least ONE wide establishing shot, at least ONE insert or closeup, and avoid two consecutive shots with the same shot_type — UNLESS it is a back-and-forth dialogue exchange, where alternating `closeup` (or `medium`) shots between the two speakers is correct and expected (shot/reverse-shot). Wide → medium → insert → closeup variety is what makes the film feel directed instead of staged.

6. **First & last frame coherence:** the locked_visual_token guarantees identical character appearance across ALL shots. Your first_frame_prompt and last_frame_prompt must NEVER re-describe the character's hair, clothes, glasses, age, species, or any appearance trait — refer to them by name only. Different pose/expression/position shows the change across the shot. If you describe the character's appearance inside a shot prompt, you are breaking the system.

7. **Motion prompt:** describes only physical motion + camera movement. Do NOT write dialogue or words spoken inside the motion_prompt. (Lip movement is generated automatically from the shot's audio — you only describe body + camera motion here.)

7b. **DIALOGUE (this is a TALKING film — characters speak):** Every shot has a `speaker` field — either `"narrator"` or the EXACT name of one character. Rules:
- **One speaker per shot.** Never two characters speaking in the same shot. A back-and-forth conversation MUST be split across consecutive shots, alternating the `speaker` (classic shot/reverse-shot).
- **Dialogue-heavy.** Most shots should be a CHARACTER speaking, in their own voice and personality. Use `"narrator"` shots mainly to open the story, bridge a time/place jump, or land the closing beat.
- When `speaker` is a character, the `narration` field holds their ACTUAL SPOKEN WORDS, first person, in character (e.g. `"I can do this if I just take one more step!"`) — NOT a description, NOT "he says".
- A speaking character MUST be present and visible in that shot, and for a `closeup`/`medium` speaking shot the speaker's FACE must be toward the camera (their mouth must be visible so it can be lip-synced).
- Keep each spoken line short and natural for a young child to follow (6-14 words).

8. **Style suffix added later** — do NOT include style tags ("anime style", "Pixar-style") in any prompt field.

9. **Characters need a clear feeling or want from shot 1.** Not "Rohan walks". Instead "Rohan, hurrying, almost steps on a tiny turtle". The want is built into the action.

10. **The moral is felt, not stated.** The "moral" field is a one-line summary FOR THE PARENT — but the narration must never say it directly.

11. **Pick a music mood that fits the story's emotional tone:**
- "cheerful" — upbeat happy stories, playful adventures
- "calm" — bedtime stories, quiet reflective moments
- "adventurous" — exploration, curiosity, journeys
- "tender" — emotional/heartfelt, family bonds, kindness
- "magical" — fantasy, fairy-tale, wonder, dreams
- "energetic" — fast-paced, action, excitement, silly fun

12. **VALID JSON ONLY:** Every field MUST be followed by a comma except the last one in its object. Every string MUST be in double quotes. NO comments. Test the JSON would parse before responding.

# OUTPUT FORMAT

{{
  "title": "<3-6 words>",
  "theme": "<the user's theme>",
  "culture": "indian | western | japanese | mixed | animal-kingdom | fantasy",
  "style": "pixar | cartoon_saloon | claymation | stickman",
  "duration_seconds": <total — narration is read aloud at ~2.5 words/sec>,
  "characters": [
    {{
      "name": "...",
      "type": "human | animal | creature",
      "appearance": "<one specific sentence describing ONLY visual traits, and it MUST state an explicit AGE first: for humans give a number ('a 7-year-old girl...'), for animals/creatures give a clear life stage ('a young sparrow...', 'an old grey-whiskered dog...'). Then species, color, body shape, clothing, distinctive features. NO poses, NO locations, NO actions. BAD: 'a black crow perched on a branch'. GOOD: 'a young sleek black crow with sharp amber eyes and glossy black feathers'.>",
      "locked_visual_token": "<the SAME visual description as 'appearance' but compressed to a tight 15-30 word phrase that can be pasted verbatim into every shot prompt. Must include: age + species/race + hair + clothing colors + ONE distinctive feature. NO actions, NO emotions, NO setting. Example: 'a 6-year-old boy with curly brown hair and round glasses, wearing a red t-shirt, blue shorts, and white sneakers'.>"
    }}
  ],
  "setting": "<one sentence>",
  "shots": [
    {{
      "shot_number": 1,
      "beat": "atmosphere | hook | spark | reaction | observation | struggle | moment-of-decision | choice | consequence",
      "shot_type": "wide | medium | closeup | insert",
      "speaker": "narrator | <exact character name from the characters list>",
      "narration": "<the spoken line for THIS shot. If speaker is a character: their actual spoken words, FIRST PERSON, in-character (no 'he said'), 8-15 words. If speaker is 'narrator': third-person narration, 10-14 words. NEVER empty.>",
      "visual_description": "<one sentence>",
      "first_frame_prompt": "<40-70 words: ONLY pose + setting + camera framing + lighting. DO NOT re-describe the character's hair, clothes, age, species, glasses, or any appearance trait — the locked_visual_token will be injected automatically. Refer to the character by NAME only. Example: 'Rohan stands at the edge of a dusty courtyard, head tilted down, hands clasped behind his back. Wide shot, late afternoon golden light, soft warm shadows.'>",
      "last_frame_prompt": "<40-70 words: SAME rules — pose + setting + framing + lighting only. Show a clearly different pose/expression/position from first_frame to convey the change across this shot. Refer to character by NAME only.>",
      "motion_prompt": "<30-60 words: only physical motion + camera, no speech>"
    }}
  ],
  "moral": "<one sentence summary for parents — never spoken in the story>",
  "music_mood": "cheerful | calm | adventurous | tender | magical | energetic"
}}

# VISUAL STYLES (pick one matching the story's mood):
{style_guide}

# SAFETY

Age-appropriate. No graphic violence, no fear-mongering, no heavy themes (death, illness, poverty) unless the user's theme explicitly asks. Conflict is good — danger and dread are not.

# OUTPUT

Output ONLY the JSON object. Start with {{ end with }}. Nothing else."""


# ==============================================================================
# OLLAMA CALL
# ==============================================================================

def _call_llm(prompt: str, system_prompt: str, role: str = "structurer", *,
              images: Optional[list] = None,
              options: Optional[dict] = None,
              cfg: Optional[dict] = None) -> str:
    """Call Ollama's generate endpoint and return raw text response.

    role: pipeline role for per-stage model routing (see model_registry.
    get_for_role). Defaults to 'structurer' (reliable JSON model) — the
    structurer + expand passes need schema discipline, not creativity.

    The extras are keyword-only, so every existing positional call site (and the
    test stubs that mimic this signature) is untouched:

    images:  base64 pictures for a VISION model. A text model does NOT error on
             these — it ignores them and answers from the prompt alone — so pass
             `cfg` too and be certain of what you are talking to.
    options: overrides the sampler. Transcription wants temperature ~0; the default
             0.75 is a licence to guess a word that is not on the page.
    cfg:     an explicit backend config, bypassing role routing. Vision uses this
             because get_for_role() falls back to the active TEXT model on a missing
             role, which would quietly send page images to a blind model.
    """
    cfg = (cfg or model_registry.get_for_role(role)
           or model_registry.get_active("llm_backend"))
    model_name = cfg.get("model_id") or cfg.get("model_name") or cfg.get("model") or "llama3.1:8b-instruct-q8_0"
    url = cfg.get("server_url", "http://127.0.0.1:11434") + "/api/generate"

    # Forcefully combine prompts to bypass missing Ollama Chat Templates
    combined_prompt = f"SYSTEM INSTRUCTIONS (STRICTLY FOLLOW THESE):\n{system_prompt}\n\nUSER REQUEST:\n{prompt}"

    payload = {
        "model": model_name,
        "prompt": combined_prompt,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.75,
            "top_p": 0.95,
            "num_ctx": 8192,
            "num_predict": 8192,
            **(options or {}),
        },
    }

    if images:
        payload["images"] = list(images)
        # Image tokens live in the PROMPT. Ollama truncates an over-long prompt from
        # the FRONT, so too small a context window drops the picture and leaves a
        # fluent model answering from the words alone — an invented page, no error.
        if payload["options"].get("num_ctx", 0) < 4096:
            raise ValueError("a vision call needs num_ctx >= 4096, or the image is "
                             "silently truncated out of the prompt")

    log.info(f"Calling LLM ({model_name}) with theme prompt: {prompt[:80]}..."
             + (f" [+{len(images)} image(s)]" if images else ""))
    log.info(f"System prompt starts with: {system_prompt[:50]}")
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()


def rewrite_narration(current: str, instruction: str = "", *,
                      shot: dict | None = None,
                      title: str = "", moral: str = "") -> str:
    """AI-rewrite ONE shot's voiceover narration line.

    `instruction` is optional guidance ("make it funnier", "shorter"). When
    empty, it rephrases freshly while keeping meaning + length. Returns a single
    clean line (falls back to `current` on failure). Used by both the dashboard
    and the Discord rewrite command.
    """
    shot = shot or {}
    beat = (shot.get("beat") or "").strip()
    visual = (shot.get("visual_description")
              or shot.get("action")
              or shot.get("description") or "").strip()
    system_prompt = (
        "You rewrite ONE line of voiceover narration for a 30-second animated "
        "kids' story. Output ONLY the rewritten narration line — no quotes, no "
        "labels, no commentary, no options. Keep it one or two short sentences "
        "(about 8-22 words), warm, simple, child-friendly, present tense, third "
        "person. Do NOT write character dialogue or speech tags."
    )
    parts = []
    if title:
        parts.append(f"Story title: {title}")
    if moral:
        parts.append(f"Moral: {moral}")
    if beat:
        parts.append(f"Story beat: {beat}")
    if visual:
        parts.append(f"What is on screen: {visual}")
    if current:
        parts.append(f"Current narration: {current}")
    parts.append(
        f"Rewrite instruction: {instruction}" if instruction.strip()
        else "Rewrite instruction: rephrase it freshly, keep the same meaning "
             "and roughly the same length."
    )
    user_prompt = "\n".join(parts) + "\n\nRewritten narration line:"
    try:
        out = _call_llm(user_prompt, system_prompt, role="creative")
    except Exception as e:
        log.warning(f"rewrite_narration LLM call failed: {e}")
        return current
    if not out:
        return current
    # Model may add a stray label/extra lines — take first non-empty line, strip quotes.
    line = next((ln.strip() for ln in out.splitlines() if ln.strip()), "")
    line = line.strip().strip('"').strip("'").strip()
    # Drop a leading "Narration:" style label if the model added one.
    for lbl in ("Rewritten narration line:", "Narration:", "Rewritten:"):
        if line.lower().startswith(lbl.lower()):
            line = line[len(lbl):].strip().strip('"').strip("'").strip()
    return line or current


# ==============================================================================
# JSON EXTRACTION + VALIDATION
# ==============================================================================

MAX_BAD_OUTPUTS = 20     # keep the last N unparseable LLM responses on disk


def _save_bad_output(raw: str) -> Path:
    """Archive an unparseable LLM response under its own timestamped name and
    prune old ones. Returns the path (used in the raised error message)."""
    try:
        bad_dir = OUTPUTS_DIR / "_bad_llm_output"
        bad_dir.mkdir(parents=True, exist_ok=True)
        path = bad_dir / f"{datetime.now():%Y%m%d_%H%M%S_%f}.txt"
        path.write_text(raw, encoding="utf-8")
        old = sorted(bad_dir.glob("*.txt"))[:-MAX_BAD_OUTPUTS]
        for f in old:
            f.unlink(missing_ok=True)
        return path
    except Exception as e:
        log.warning(f"Could not archive bad LLM output: {e}")
        return OUTPUTS_DIR / "_last_raw_llm_output.txt"


def _extract_json(raw: str) -> dict:
    """Pull JSON out of an LLM response, tolerantly. Saves raw output for debugging."""
    # Save raw output for debugging
    try:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUTS_DIR / "_last_raw_llm_output.txt").write_text(raw, encoding="utf-8")
    except Exception:
        pass

    # Strip <think>...</think> blocks (reasoning models)
    # Strip closed <think>...</think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    # Strip UNCLOSED <think>... blocks (model got cut off before closing tag)
    # Cut everything from <think> to the first { (the JSON start)
    if "<think>" in cleaned.lower():
        json_start = cleaned.find("{")
        if json_start > 0:
            cleaned = cleaned[json_start:]
    # Strip markdown fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())

    # Find the JSON block (greedy, outermost braces)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    candidate = match.group(0) if match else cleaned

    # Attempt 1: strict JSON
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        log.warning(f"Strict JSON parse failed: {e}. Trying repair...")

    # Attempt 2: common LLM JSON mistakes
    repaired = candidate
    # Fix MISSING commas between JSON fields: `"...value"\n    "next_key":` (Llama 3.1 often drops these)
    repaired = re.sub(r'("\s*)\n(\s*")', r'\1,\n\2', repaired)
    # Fix unescaped newlines inside strings (very common)
    # Replace literal newlines that are inside "..." strings with \n
    def _escape_newlines_in_strings(s: str) -> str:
        out = []
        in_string = False
        escape = False
        for ch in s:
            if escape:
                out.append(ch)
                escape = False
                continue
            if ch == "\\":
                out.append(ch)
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                out.append(ch)
                continue
            if in_string and ch == "\n":
                out.append("\\n")
                continue
            if in_string and ch == "\r":
                continue
            out.append(ch)
        return "".join(out)
    repaired = _escape_newlines_in_strings(repaired)

    try:
        return json.loads(repaired)
    except json.JSONDecodeError as e:
        log.error(f"Repair also failed: {e}")
        # `_last_raw_llm_output.txt` is rewritten by EVERY call, so by the time
        # anyone investigates, the offending output has been overwritten by a
        # later success. Keep a dated copy of the failures only.
        bad_path = _save_bad_output(raw)
        raise ValueError(
            f"Could not extract JSON from LLM response. "
            f"Raw output saved to: {bad_path}. "
            f"First 300 chars: {raw[:300]}"
        )


_SPEECH_VERBS = (
    r"(?:said|says|replied|reply|asked|answered|called|cried|shouted|whispered|"
    r"murmured|added|offered|admitted|agreed|suggested|announced|declared|"
    r"mumbled|sighed|laughed|grinned|crowed|clucked|chirped|barked|squeaked|"
    r"roared|exclaimed|continued|insisted|promised|warned|begged|wondered|"
    r"thought|spoke|yelled|giggled|hummed|snapped|gasped|moaned|boasted|"
    # action verbs the LLM uses as a dialogue tag ("Tortoise smiled, ...")
    r"smiled|nodded|beamed|shrugged|chuckled|grumbled|huffed|purred|smirked|"
    r"whimpered|sniffed|winked|waved|yawned|panted|breathed|groaned|whined|"
    r"cheered|pleaded|bragged|wailed|hollered|stammered|chimed|"
    # NOTE: only verbs that read as SPEECH/emotive attribution belong here.
    # Plain physical actions that commonly follow a pronoun in ordinary
    # narration (paused, pointed, gestured, blinked) are deliberately EXCLUDED —
    # the trailing scrub is greedy, so "he paused for a rest, ..." would lose the
    # rest of the sentence.
    r"frowned|scowled|mused|realized|realised|repeated|murmured|whispered|"
    r"exclaimed|grinned|called|cried|shouted|challenged|dared|urged|teased|"
    r"taunted|scoffed|retorted|countered|snickered|chirped)"
)
# Optional adverb(s) between the subject and the verb: "he softly replied".
_ADV = r"(?:\w+ly\s+)?"

# Verbs that ONLY ever introduce speech (never plain physical action). A line
# carrying "said Pip," / "Pip replied:" is dialogue even with no '?'/'!' and no
# first/second-person pronoun — unlike "smiled"/"nodded", which double as action.
_PURE_SPEECH_VERBS = (
    r"(?:said|says|asked|replied|reply|answered|called|cried|shouted|"
    r"whispered|exclaimed|added|murmured|announced|declared|yelled|"
    r"wondered|insisted|begged|warned|promised|crowed|chirped)"
)


def _strip_attribution(text: str, speaker: str, extra_names=None) -> str:
    """Remove `he said` / `, she replied` / `X asked,` attribution tags from a
    character's spoken line so TTS voices ONLY the dialogue. Deterministic —
    works even when the LLM used comma-attribution instead of quotes.

    `extra_names` = the full cast. A line is often mis-tagged `narrator` while
    still carrying `... asked Finn, surprised` — without the cast names as
    candidate subjects the tag survives and TTS speaks it. Passing every cast
    name lets the scrub catch the tag regardless of the shot's `speaker`."""
    if not text:
        return text
    t = text.strip()
    # Build a subject alternation: pronouns + speaker name tokens + every cast
    # name token (so a stray "asked Finn" is removed even on a narrator line).
    name_tokens = [tok for tok in speaker.split() if tok]
    for nm in (extra_names or []):
        name_tokens.extend(tok for tok in str(nm).split() if tok)
    name_tokens = [re.escape(tok) for tok in dict.fromkeys(name_tokens)]
    subj = "|".join(["he", "she", "they", "it"] + name_tokens) or "he|she|they|it"

    # Trailing: from the attribution tag to the END of the line — this also
    # drops any stage-action the author appended after it ("..., said Fox
    # nervously, tugging his tail." -> "..."). Safe to be greedy because `subj`
    # is the SPEAKER's own name + pronouns only (not every cast name), so a
    # subordinate clause about a DIFFERENT character ("..., while Terry continued
    # to crawl...") is never matched. Handles subject-verb ("he said") and
    # inverted ("said Fox") order.
    t = re.sub(rf"[,\s]+(?:{subj})\s+{_ADV}{_SPEECH_VERBS}\b.*$", "", t,
               flags=re.IGNORECASE)
    t = re.sub(rf"[,\s]+{_ADV}{_SPEECH_VERBS}\s+(?:{subj})\b.*$", "", t,
               flags=re.IGNORECASE)
    # Subject-less broken tag that starts its own sentence after the spoken line
    # ("See ya later! said to the sun as he sped past." -> "See ya later!"). A
    # sentence beginning with a bare speech verb is never real narration.
    t = re.sub(rf"(?<=[.!?])\s+{_ADV}{_SPEECH_VERBS}\b[^.!?]*[.!?]?\s*$", "", t,
               flags=re.IGNORECASE)
    # Leading: "Featherlite said, " / "He replied: " — the comma/colon is
    # REQUIRED so a real description ("He said nothing and left.") is not eaten.
    t = re.sub(rf"^(?:{subj})\s+{_ADV}{_SPEECH_VERBS}\b[,:]\s*", "", t,
               flags=re.IGNORECASE)
    t = t.strip().strip(",").strip()
    if not t:
        return text.strip()  # scrub ate everything — keep original
    # Re-capitalize + ensure terminal punctuation.
    t = t[0].upper() + t[1:]
    if t[-1] not in ".!?":
        t += "."
    return t


# Manner adverbs the LLM appends as a leaked stage direction AFTER the closing
# quote ('"Hello," quietly but clearly.' → narration "Hello, quietly but
# clearly."). These carry no speech VERB so _strip_attribution misses them. The
# scrub only fires when the ENTIRE trailing comma-clause is made of these words
# (+ and/but/yet), so a real spoken phrase like "Run, you can do it!" is safe.
_MANNER_ADVERBS = {
    "quietly", "softly", "loudly", "clearly", "nervously", "shyly", "boldly",
    "bravely", "calmly", "cheerfully", "sadly", "happily", "proudly", "gently",
    "firmly", "sweetly", "slowly", "eagerly", "hesitantly", "confidently",
    "warmly", "coldly", "sternly", "kindly", "politely", "excitedly",
    "anxiously", "timidly", "gleefully", "merrily", "brightly", "faintly",
    "weakly", "breathlessly", "solemnly", "earnestly", "quickly", "carefully",
    "curiously", "thoughtfully", "wearily", "grumpily", "crossly", "sheepishly",
}


# Gerund-led expressions/actions the LLM tacks onto a locked_visual_token
# ("...yellow boots, smiling", "...white fur, wagging its tail"). The token is
# injected verbatim into EVERY shot, so a baked-in action forces that pose into
# every frame (a worried "struggle" shot still shows the dog wagging). Strip
# trailing comma-clauses that are an action/expression, not a stable look.
_ACTION_GERUNDS = {
    "smiling", "grinning", "frowning", "laughing", "crying", "wagging",
    "beaming", "waving", "jumping", "running", "sitting", "standing", "looking",
    "holding", "walking", "hopping", "playing", "dancing", "singing", "sleeping",
    "winking", "pointing", "reaching", "leaning", "crouching", "kneeling",
    "stretching", "yawning", "nodding", "blinking", "gazing", "staring",
    "panting", "trotting", "flying", "swimming", "climbing", "skipping",
}


def _clean_locked_token(token: str) -> str:
    """Drop trailing comma-clauses that are an ACTION or EXPRESSION rather than a
    stable visual trait, so the injected token never forces a fixed pose into
    every shot. Only strips a trailing segment whose first word is a known
    action/expression gerund ('smiling', 'wagging its tail') — colors, clothes
    and features (which never start with such a gerund) are untouched."""
    if not token or "," not in token:
        return token.strip()
    segs = [s.strip() for s in token.split(",")]
    while len(segs) > 1:
        first = re.sub(r"[^A-Za-z]", "", segs[-1].split()[0]).lower() if segs[-1].split() else ""
        if first in _ACTION_GERUNDS:
            segs.pop()
        else:
            break
    return ", ".join(s for s in segs if s).strip().rstrip(".")


def _strip_trailing_stage_direction(text: str) -> str:
    """Drop a trailing ', <manner-adverb(s)>' clause leaked from prose like
    `said, "Hi," softly and shyly.` Only strips when every word in the trailing
    comma-clause is a manner adverb or a conjunction — never real spoken words."""
    if not text or "," not in text:
        return text
    head, _, tail = text.rstrip().rpartition(",")
    if not head.strip():
        return text
    tail_words = re.findall(r"[A-Za-z']+", tail)
    if not tail_words:
        return text
    conj = {"and", "but", "yet", "then"}
    if all(w.lower() in _MANNER_ADVERBS or w.lower() in conj for w in tail_words):
        out = head.strip().rstrip(",.! ?").strip()
        if out:
            if out[-1] not in ".!?":
                # keep the original sentence-final punctuation if it was ! or ?
                end = text.rstrip()[-1]
                out += end if end in "!?" else "."
            return out
    return text


_FIRST_SECOND_PERSON = re.compile(
    r"\b(I|I'm|I'll|I've|I'd|my|mine|me|we|we're|we'll|us|our|ours|"
    r"you|you're|you'll|your|yours|let's)\b",
    re.IGNORECASE,
)


def _looks_like_narration(text: str, speaker: str) -> bool:
    """True when a CHARACTER-tagged line is really third-person narration about
    that character (e.g. 'Little Fox watched...', 'He started hopping...').
    Conservative: only fires on clear third-person description so we can safely
    retag it to narrator (never the reverse — that would risk a wrong-face
    lip-sync). Questions/exclamations and any first/second-person line are kept
    as dialogue."""
    if not text:
        return False
    t = text.strip()
    if t[-1:] in "?!":
        return False
    if _FIRST_SECOND_PERSON.search(t):
        return False
    first = t.split()[0].strip(",.:;").lower() if t.split() else ""
    name_tokens = {tok.lower() for tok in speaker.split()}
    third_person_subjects = {"he", "she", "it", "they", "his", "her", "their", "the"}
    if first in name_tokens or first in third_person_subjects:
        return True
    return False


_ATTRIB_NAME = None  # lazy-compiled per cast (see _narrator_dialogue_owner)


def _narrator_dialogue_owner(text: str, cast_names: list) -> Optional[str]:
    """If a NARRATOR-tagged line is really a character's spoken line carrying an
    attribution tag (`... asked Finn, surprised`, `Finn wondered aloud, ...`),
    return that character's name so the caller can retag the speaker. Returns
    None for genuine narration. Conservative: only fires when the line ALSO
    carries a speech cue (a `?`/`!` or a first/second-person pronoun), so plain
    third-person action like `Finn smiled and ran.` is never mistaken for
    dialogue."""
    if not text or not cast_names:
        return None
    names = [(nm or "").strip() for nm in cast_names if (nm or "").strip()]

    # 1) Explicit PURE-speech attribution ("said Pip,", "Pip replied:") — a
    # certain dialogue signal, so no '?'/'!'/pronoun cue is required. Punctuation
    # after the tag is required so plain narration ("Pip said nothing and left")
    # is not mistaken for a tag.
    for nm in names:
        n = re.escape(nm)
        if re.search(rf"\b{_PURE_SPEECH_VERBS}\s+{n}\b\s*[,.!?]", text, re.IGNORECASE) or \
           re.search(rf"\b{n}\b\s+{_ADV}{_PURE_SPEECH_VERBS}\s*[,:]", text, re.IGNORECASE):
            return nm

    # The rest needs a spoken-line cue to avoid retagging plain narration.
    has_cue = ("?" in text or "!" in text or bool(_FIRST_SECOND_PERSON.search(text)))
    if not has_cue:
        return None

    # 2) Action-as-tag attribution ("Pip grinned,") — only with a cue present.
    for nm in names:
        n = re.escape(nm)
        if re.search(rf"\b{n}\b\s+{_ADV}{_SPEECH_VERBS}\b", text, re.IGNORECASE) or \
           re.search(rf"\b{_SPEECH_VERBS}\s+{n}\b", text, re.IGNORECASE):
            return nm

    # 3) Untagged dialogue that DIRECTLY ADDRESSES one character (vocative): in a
    # two-character scene the speaker is the OTHER character ("Tess, you ready?"
    # -> Pip speaks; "Pip! Wait!" -> Tess speaks). Restricted to vocative position
    # (name at start/end or comma-set-off) so a line where the name is the SUBJECT
    # ("Bunny took off, I'm ahead") is NOT mis-assigned to the other character.
    def _is_vocative(nm: str) -> bool:
        n = re.escape(nm)
        return bool(
            re.search(rf"(?:^|[,.!?]\s*){n}\s*[,!?]", text, re.IGNORECASE)
            or re.search(rf",\s*{n}\s*[.!?]*$", text, re.IGNORECASE)
        )
    addressed = [nm for nm in names if _is_vocative(nm)]
    subjects = [nm for nm in names
                if nm not in addressed and re.search(rf"\b{re.escape(nm)}\b", text, re.IGNORECASE)]
    others = [nm for nm in names if nm not in addressed]
    if len(addressed) == 1 and not subjects and len(others) == 1:
        return others[0]
    return None


def _validate_and_default(script: dict) -> dict:
    """Apply sensible defaults for missing fields. Does NOT cap shot count."""
    available_styles = get_available_style_ids()

    # Normalize characters: if LLM returned strings, wrap each in a dict
    raw_chars = script.get("characters", [])
    normalized = []
    for c in raw_chars:
        if isinstance(c, dict):
            normalized.append(c)
        elif isinstance(c, str):
            # Try to split on " - " or "—" to extract name+appearance
            parts = c.split(" - ", 1) if " - " in c else c.split(" — ", 1) if " — " in c else [c]
            if len(parts) == 2:
                normalized.append({"name": parts[0].strip(), "type": "character", "appearance": parts[1].strip()})
            else:
                normalized.append({"name": c.strip(), "type": "character", "appearance": ""})
    script["characters"] = normalized

    # Canonicalize names: "Greenfield the Rabbit" -> "Greenfield" (the species
    # belongs in `type`, not the name). Propagate the rename through every
    # `speaker` and `narration` so the spoken story matches the cast list.
    rename = {}
    for ch in script["characters"]:
        if not isinstance(ch, dict):
            continue
        nm = (ch.get("name") or "").strip()
        m = re.match(r"^(.+?)\s+the\s+\w+$", nm, re.IGNORECASE)
        if m and m.group(1).strip() and m.group(1).strip().lower() != nm.lower():
            short = m.group(1).strip()
            rename[nm] = short
            ch["name"] = short
            nm = short
        # Strip an article-led generic name ("the puppy", "a robot") to a proper
        # name ("Puppy", "Robot") — a leading article in a character name renders
        # as spoken text and reads as no-name. Title-case the remainder.
        am = re.match(r"^(?:the|a|an)\s+(\w[\w\s'-]*)$", nm, re.IGNORECASE)
        if am:
            short = am.group(1).strip().title()
            if short and short.lower() != nm.lower():
                rename[ch.get("name", "").strip()] = short
                ch["name"] = short
    if rename:
        for s in script.get("shots", []):
            if not isinstance(s, dict):
                continue
            spk = (s.get("speaker") or "").strip()
            if spk in rename:
                s["speaker"] = rename[spk]
            narr = s.get("narration")
            if isinstance(narr, str):
                for old, new in rename.items():
                    narr = re.sub(rf"\b{re.escape(old)}\b", new, narr)
                s["narration"] = narr
        log.info(f"Canonicalized character names: {rename}")

    # Ensure every character has a locked_visual_token. If the LLM didn't
    # provide one, derive it from appearance (truncated). This is the SINGLE
    # source of truth for character look across all shots.
    for ch in script["characters"]:
        if not isinstance(ch, dict):
            continue
        token = (ch.get("locked_visual_token") or "").strip()
        if not token:
            appearance = (ch.get("appearance") or "").strip()
            if appearance:
                # Compress to ~30 words max for prompt efficiency
                words = appearance.split()
                token = " ".join(words[:30])
            else:
                token = ch.get("name", "a character")
            log.warning(
                f"Character '{ch.get('name')}' missing locked_visual_token; "
                f"derived from appearance: '{token[:80]}...'"
            )
        # Strip the character's OWN name out of its token — the structurer
        # sometimes bakes it in ("a young grey mouse Wobbly with round ears"),
        # which then double-prints as "Wobbly is a ... mouse Wobbly ..." once the
        # injector prefixes the name. The token must be name-free appearance only.
        own_name = (ch.get("name") or "").strip()
        if own_name:
            token = re.sub(rf"\s*\b{re.escape(own_name)}\b\s*", " ", token).strip()
            token = re.sub(r"\s{2,}", " ", token).strip(" ,")
        # Strip any baked-in action/expression ("..., smiling", "..., wagging its
        # tail") so the injected token carries only the stable look.
        cleaned_token = _clean_locked_token(token)
        if cleaned_token and cleaned_token != token:
            log.info(f"Cleaned action/expression from token for "
                     f"'{ch.get('name')}': '{token}' -> '{cleaned_token}'")
            token = cleaned_token
        ch["locked_visual_token"] = token

    script.setdefault("title", "Untitled Story")
    script.setdefault("theme", "")
    script.setdefault("culture", "mixed")
    script.setdefault("style", get_default_style())
    script.setdefault("duration_seconds", 30)
    script.setdefault("characters", [])
    script.setdefault("setting", "")
    script.setdefault("shots", [])
    script.setdefault("moral", "")
    
    if script.get("music_mood") not in ("cheerful", "calm", "adventurous", "tender", "magical", "energetic"):
        script["music_mood"] = "cheerful"

    # Style must be one we know about; fall back to default
    if script["style"] not in available_styles:
        log.warning(
            f"LLM returned unknown style '{script['style']}'. Defaulting to '{get_default_style()}'."
        )
        script["style"] = get_default_style()

    # Minimum sanity floor only — let the model decide length
    shots = script.get("shots", [])
    if len(shots) < 3:
        raise ValueError(
            f"LLM produced only {len(shots)} shots — minimum is 3 "
            f"(need beginning/middle/end)"
        )
    if len(shots) > 20:
        log.warning(f"LLM produced {len(shots)} shots — capping at 20 for safety.")
        script["shots"] = shots[:20]

    # Detect duplicate narration (LLM padding)
    seen_narration = set()
    for s in script["shots"]:
        narr = s.get("narration", "").strip().lower()
        if narr and narr in seen_narration:
            log.warning(f"Duplicate narration detected: '{narr[:60]}...'")
        seen_narration.add(narr)

    # Beat-order guard: `consequence` is the payoff and must be the FINAL shot
    # only. The structurer occasionally puts it mid-story or swaps choice<->
    # consequence at the end. Demote any non-final `consequence` to observation,
    # then make the genuine last shot the consequence (kids 30s arcs end on the
    # payoff). Conservative: never touches atmosphere/hook openers.
    _shots = script.get("shots", [])
    if len(_shots) >= 4:
        last_i = len(_shots) - 1
        for i, s in enumerate(_shots):
            if (s.get("beat") or "").strip().lower() == "consequence" and i != last_i:
                log.info(f"Beat guard: demoting mid-story consequence at shot "
                         f"{i+1} -> observation.")
                s["beat"] = "observation"
        lb = (_shots[last_i].get("beat") or "").strip().lower()
        if lb in ("choice", "reaction", "observation"):
            log.info(f"Beat guard: final shot beat '{lb}' -> consequence (payoff).")
            _shots[last_i]["beat"] = "consequence"

    # Default shot_type when the LLM omitted it — derive from the beat
    _beat_to_shot_type = {
        "atmosphere": "wide",
        "reaction": "closeup",
        "moment-of-decision": "closeup",
    }
    for s in script["shots"]:
        st_val = (s.get("shot_type") or "").strip().lower()
        if st_val not in ("wide", "medium", "closeup", "insert"):
            beat = (s.get("beat") or "").strip().lower()
            s["shot_type"] = _beat_to_shot_type.get(beat, "medium")

    # Insert nudge: an `insert` (hands/paws + object close-up) is REQUIRED when a
    # character handles an object, but the structurer routinely skips it, leaving
    # a wall of mediums. If the story handles objects yet has ZERO inserts,
    # promote ONE medium whose text shows manipulation to `insert` for variety.
    _shots2 = script.get("shots", [])
    if _shots2 and not any((s.get("shot_type") or "").lower() == "insert" for s in _shots2):
        _manip = re.compile(
            r"\b(pick(?:s|ed)?|hold(?:s|ing)?|held|grab(?:s|bed)?|place(?:s|d)?|"
            r"put(?:s|ting)?|drop(?:s|ped)?|pour(?:s|ed)?|mix(?:es|ed|ing)?|"
            r"stir(?:s|red|ring)?|crack(?:s|ed|ing)?|cut(?:s|ting)?|lift(?:s|ed)?|"
            r"carry|carries|carried|hand(?:s|ed)?|plant(?:s|ed|ing)?|dig(?:s|ging)?|"
            r"paint(?:s|ed|ing)?|wrap(?:s|ped)?|press(?:es|ed)?|scoop(?:s|ed)?|"
            r"sprinkle(?:s|d)?|spread(?:s|ing)?)\b", re.IGNORECASE)
        for s in _shots2:
            if (s.get("shot_type") or "").lower() != "medium":
                continue
            text = " ".join(str(s.get(k, "")) for k in
                            ("narration", "first_frame_prompt", "visual_description"))
            if _manip.search(text):
                log.info(f"Insert nudge: shot {s.get('shot_number')} handles an "
                         f"object -> shot_type insert.")
                s["shot_type"] = "insert"
                break

    # Force sequential shot_number 1..N. The structurer sometimes repeats or
    # skips numbers (seen: two "shot 10"), which would make on-disk artifacts
    # (clip_{id}_shot{N}.mp4, approved_prompts keys, seeds) collide/overwrite.
    for idx, s in enumerate(script.get("shots", []), start=1):
        if isinstance(s, dict):
            s["shot_number"] = idx

    # Default + validate per-shot `speaker`. It selects the TTS voice and the
    # talking-vs-narrator render route (character → lip-synced backend later).
    # Missing, "narrator", or a name not in the cast → "narrator" so the
    # downstream voice lookup can never break. Known names keep canonical casing.
    canon = {}
    for c in script.get("characters", []):
        if isinstance(c, dict):
            nm = (c.get("name") or "").strip()
            if nm:
                canon[nm.lower()] = nm
    for s in script["shots"]:
        spk = (s.get("speaker") or "").strip()
        if not spk or spk.lower() == "narrator":
            s["speaker"] = "narrator"
        elif spk.lower() in canon:
            s["speaker"] = canon[spk.lower()]
        else:
            log.warning(
                f"Shot {s.get('shot_number')} speaker '{spk}' not in cast; "
                f"defaulting to narrator."
            )
            s["speaker"] = "narrator"

    cast_names = list(canon.values())

    # Forward retag: a NARRATOR line that is actually a character's spoken line
    # with an attribution tag ("... asked Finn, surprised") → retag speaker to
    # that character so it gets their voice (and the tag is scrubbed below).
    for s in script["shots"]:
        if (s.get("speaker") or "narrator") != "narrator":
            continue
        owner = _narrator_dialogue_owner(s.get("narration", ""), cast_names)
        if owner:
            log.info(f"Shot {s.get('shot_number')} narrator line is {owner}'s "
                     f"dialogue; retagging speaker -> {owner}.")
            s["speaker"] = owner

    # Inverse-leak guard: a character shot whose line is plainly third-person
    # narration about that character → retag to narrator (safe: VO, no lip-sync
    # to a wrong/absent face). Only the safe direction is ever auto-applied.
    for s in script["shots"]:
        spk = s.get("speaker")
        if spk and spk != "narrator" and _looks_like_narration(s.get("narration", ""), spk):
            log.info(f"Shot {s.get('shot_number')} looks like narration, not "
                     f"{spk}'s dialogue; retagging speaker -> narrator.")
            s["speaker"] = "narrator"

    # Deterministic narration scrub: strip quote marks so TTS never speaks them.
    # When `speaker` is a character the narration IS their dialogue, but the
    # quote characters themselves should not be voiced (apostrophes/contractions
    # are kept). Harmless for narrator lines too.
    for s in script["shots"]:
        narr = s.get("narration")
        if isinstance(narr, str) and ('"' in narr or "“" in narr or "”" in narr):
            cleaned = narr.replace('"', "").replace("“", "").replace("”", "")
            cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
            s["narration"] = cleaned

    # Strip dialogue attribution ("he said", ", she whispered to herself") from
    # EVERY shot so TTS voices only the words — never the stage tag. Runs on
    # narrator shots too (the LLM often leaves a spoken line tagged narrator with
    # its attribution tail attached); a true description has no such tag so it is
    # left untouched.
    for s in script["shots"]:
        narr = s.get("narration")
        if isinstance(narr, str):
            # Speaker-only subjects here: the narrator->character retag above
            # already fixed dialogue mislabeled as narrator, so feeding ALL cast
            # names into these destructive patterns only risks gutting genuine
            # narration that happens to start with "<Character> <action-verb>,".
            stripped = _strip_attribution(narr, s.get("speaker") or "")
            # Then drop any leaked trailing manner-adverb stage direction
            # ("Hello, quietly but clearly." -> "Hello.") on SPOKEN lines only —
            # narrator prose legitimately ends in adverbs ("she walked away
            # slowly").
            if (s.get("speaker") or "narrator") != "narrator":
                stripped = _strip_trailing_stage_direction(stripped)
            if stripped != narr:
                log.info(f"Shot {s.get('shot_number')} attribution scrubbed: "
                         f"'{narr}' -> '{stripped}'")
                s["narration"] = stripped

    # Pin ONE light direction for the whole film so every shot's lighting comes
    # from the same side (the assembler reads this; stops shot-to-shot flips).
    if not script.get("_light_direction"):
        script["_light_direction"] = "the upper right"

    # Cast a distinct speaking voice to each character (idempotent — never
    # overrides a voice the user already picked). Narrator uses the global voice.
    try:
        from modules import voice_casting as vc
        vc.assign_voices(script)
    except Exception as e:
        log.warning(f"Voice casting skipped: {e}")

    return script


# ==============================================================================
# DURATION / WORD-BUDGET ENFORCEMENT
# ==============================================================================

WORDS_PER_SECOND = 2.5
TARGET_SECONDS = 30
MIN_NARRATION_WORDS = 65    # ≈26s — below this we expand
MAX_NARRATION_WORDS = 95    # ≈38s — above this we don't push further
MAX_EXPAND_ATTEMPTS = 2


def _count_narration_words(script: dict) -> int:
    total = 0
    for shot in script.get("shots", []):
        narr = (shot.get("narration") or "").strip()
        if narr:
            total += len(narr.split())
    return total


def _expand_script_to_buffer(script: dict) -> dict:
    """If narration is too short to fill ~30s, ask the LLM to enrich it.
    Repeats up to MAX_EXPAND_ATTEMPTS. Returns the (possibly) expanded script."""
    attempts = 0
    while attempts < MAX_EXPAND_ATTEMPTS:
        words = _count_narration_words(script)
        est_sec = words / WORDS_PER_SECOND
        if words >= MIN_NARRATION_WORDS:
            log.info(f"Narration {words} words (~{est_sec:.0f}s) — fills buffer, no expansion.")
            return script

        log.info(
            f"Narration only {words} words (~{est_sec:.0f}s) — under {MIN_NARRATION_WORDS}. "
            f"Expanding (attempt {attempts + 1}/{MAX_EXPAND_ATTEMPTS})."
        )

        system_prompt = _build_system_prompt()
        user_prompt = (
            f"Here is a children's story JSON that is TOO SHORT — its narration totals only "
            f"{words} words (~{est_sec:.0f} seconds), but the finished video must be ~{TARGET_SECONDS} "
            f"seconds, which needs {MIN_NARRATION_WORDS}-{MAX_NARRATION_WORDS} narration words total.\n\n"
            f"```\n{json.dumps(script, indent=2, ensure_ascii=False)}\n```\n\n"
            f"Enrich this story so its narration totals {MIN_NARRATION_WORDS}-{MAX_NARRATION_WORDS} words. "
            f"You may: lengthen thin narration lines into full vivid sentences, and/or add 1-3 NEW shots "
            f"(atmosphere, reaction, or struggle beats) where the story can breathe. Keep the SAME characters, "
            f"the SAME locked_visual_token for each (do not change appearance), the same arc and moral. "
            f"PRESERVE each existing shot's `speaker` field exactly — keep character dialogue as dialogue "
            f"(do NOT convert a character's spoken line into narrator text, and do NOT invent dialogue for "
            f"narrator shots). New shots you add are usually `speaker: \"narrator\"` unless a character clearly speaks. "
            f"Renumber shots sequentially starting at 1. Output ONLY the JSON."
        )
        try:
            raw = _call_llm(user_prompt, system_prompt)
            expanded = _validate_and_default(_extract_json(raw))
            # Carry over locked tokens if the LLM dropped any
            orig_locks = {
                c.get("name", "").strip().lower(): c.get("locked_visual_token", "")
                for c in script.get("characters", []) if isinstance(c, dict)
            }
            for c in expanded.get("characters", []):
                if isinstance(c, dict):
                    nm = c.get("name", "").strip().lower()
                    if nm in orig_locks and orig_locks[nm] and not c.get("locked_visual_token"):
                        c["locked_visual_token"] = orig_locks[nm]
            new_words = _count_narration_words(expanded)
            # Only accept the expansion if it actually got longer
            if new_words > words:
                script = expanded
            else:
                log.warning("Expansion did not increase word count; keeping previous version.")
        except Exception as e:
            log.warning(f"Expansion attempt failed (keeping current script): {e}")
            return script
        attempts += 1

    final_words = _count_narration_words(script)
    log.info(f"Expansion done — final narration {final_words} words (~{final_words/WORDS_PER_SECOND:.0f}s).")
    return script


# ==============================================================================
# STAGE-2 STRUCTURER — turns stage-1 prose into the production JSON schema
# ==============================================================================

def _build_structurer_system_prompt(shot_min: int = 5, shot_max: int = 9) -> str:
    """Narrow, schema-only system prompt.

    Stage 1 (story_writer) already wrote the story in a natural voice. This
    stage's ONLY job is to split that prose into shots and emit valid JSON.
    No story rewriting. No tone changes. No banned-word policing. Narration
    in the JSON must be drawn from the prose verbatim (or near-verbatim — only
    splits and very light edits allowed).
    """
    available_styles = get_style_ids_for_mode("story")
    styles_data = _load_styles().get("available", {})
    style_guide_lines = []
    for sid in available_styles:
        info = styles_data.get(sid, {})
        style_guide_lines.append(
            f'  - "{sid}": {info.get("description", "")}'
        )
    style_guide = "\n".join(style_guide_lines)

    return f"""You output ONLY valid JSON. No prose, no preamble, no markdown fences. Start with {{ end with }}.

You are a story structurer. The story has already been WRITTEN by another author. Your only job is to break it into shots and emit the JSON schema below. You DO NOT rewrite the story. You DO NOT change the tone, vocabulary, or pacing.

# RULES

1. **Narration field = author's prose, split into shots.** Take the prose given to you, split it into {shot_min}-{shot_max} shots, and put each piece into the `narration` field of one shot. You may make tiny grammar fixes if a split leaves a half-sentence, but DO NOT rewrite the author's words. Keep the author's voice.

1a. **SPLIT ONLY AT SENTENCE ENDS (. ! ?).** Every shot's narration must be one or more COMPLETE sentences. NEVER split in the middle of a sentence — not at a comma, not at "and", not at "while". A fragment with no subject or no verb (e.g. "Once upon a time in a bright forest") is FORBIDDEN. If one sentence is long, keep it whole in one shot rather than breaking it. If two short sentences fit one beat, combine them into one shot.

1b. **PRESERVE PUNCTUATION VERBATIM.** Keep the author's commas, periods, and capitalization exactly. Do not drop the period after a name ("Mr. Snail", never "Mr Snail"). Each narration line ends with proper end punctuation.

1c. **DIALOGUE → speaker + spoken words (this is a TALKING film).** Every shot has a `speaker` field. Classify each sentence of the prose:
- **Is it something a character SAYS?** (in quotes, OR clearly spoken: a question, command, exclamation, greeting, or first-person "I/we/my" line addressed to another character). → It becomes its OWN shot. Set `speaker` to that character's EXACT name and put ONLY the spoken words in `narration` — strip the quotation marks AND any `X said`/`asked X` attribution. Example: prose `"Let's race!" said Mr. Snail.` → `speaker: "Mr. Snail"`, `narration: "Let's race!"`.
- **Is it the narrator describing the scene/action in third person?** → `speaker: "narrator"`, keep the prose verbatim.
Rules: ONE speaker per shot — never merge two characters' lines into one shot; a quoted exchange becomes consecutive shots that alternate `speaker`. **A character shot's narration is ONLY the spoken words — NEVER append a following description sentence to it.** If the prose is `"This is easy!" he yawned. Then he rested under a tree.`, that is TWO shots: shot A `speaker: "<character>"`, `narration: "This is easy!"`; shot B `speaker: "narrator"`, `narration: "Then he rested under a tree."`. Do not put the spoken line and the description in the same shot. **NEVER leave a spoken line tagged `narrator`.** A sentence is SPOKEN if it is a question, a command, an exclamation, a greeting, or uses first/second person ("I", "we", "you", "let's") — these are ALWAYS a character talking, never the narrator. If the prose did not mark who said it, INFER the speaker from the story: in a two-character scene the dialogue alternates between them, and lines often name the person being spoken TO (so the OTHER character is the speaker). Assign your best-guess character. Use `narrator` ONLY for third-person description of the scene or an action ("Twig took a deep breath", "The two friends reached the river").

Worked mapping (prose → shots):
- `"Shy! We need to cross today!" Twig called.` → speaker: Twig, narration: "Shy! We need to cross today!"
- `"But what if we can't?" whispered Shy.` → speaker: Shy, narration: "But what if we can't?"
- `Twig took a deep breath and hopped onto a flat rock.` → speaker: narrator (third-person action).
- `"Come on, it's safe!"` (unmarked, but Shy was just addressed, so Twig speaks) → speaker: Twig.

2. **Each shot's narration: roughly 8-15 words** of complete sentence(s), depending on where sentence breaks fall. Don't pad. Don't trim hard. Follow the prose.

3. **Pick a beat for each shot** from this fixed list:
   `atmosphere | hook | spark | reaction | observation | struggle | moment-of-decision | choice | consequence`
   The beats follow the story ARC in order: `atmosphere`/`hook` open it, then `spark` (the problem/surprise appears), then `struggle`/`observation`/`reaction` in the middle, then `moment-of-decision`, then `choice` (the character decides), and finally ONE `consequence` as the very LAST shot (the payoff).
   STRICT beat rules:
   - `spark`, `moment-of-decision`, `choice`, and `consequence` each appear AT MOST ONCE in the whole story.
   - `consequence` is ONLY the final shot — never put it in the middle, never use it twice. If several late shots all show the happy result, label the earlier ones `reaction`/`observation` and keep `consequence` for the last shot alone.
   - ORDER IS FIXED: `choice` always comes BEFORE `consequence`, never after. The VERY LAST shot is the `consequence` (the payoff/result we see). If the last shot shows the outcome or a final line landing the payoff, it is `consequence` — do NOT label it `choice`. A `choice` in the final slot or a `consequence` that is not last is WRONG.
   - NEVER repeat the same beat on two consecutive shots. For a back-and-forth dialogue in the middle, alternate `reaction`/`observation`/`struggle`, not the same beat over and over.
   - Match the beat to what the line actually DOES: a character merely answering a question or greeting is `reaction` or `observation`, NOT `choice`. `choice` is reserved for the one moment the character makes a real decision.

3b. **Pick a shot_type for each shot** (REQUIRED FIELD) — the camera framing:
   - `"wide"` — establishing shot, whole location visible, character small in frame. REQUIRED for the first shot in every new location; natural for atmosphere beats.
   - `"medium"` — character action, waist-up or full body. The default for plot beats.
   - `"closeup"` — the face fills the frame. Use for reaction and moment-of-decision beats.
   - `"insert"` — extreme close-up of HANDS + OBJECT (paw, beak, fingers count). REQUIRED whenever a character picks up, holds, drops, places, or manipulates an object.
   Across the story: at least ONE wide, at least ONE insert or closeup, and avoid two consecutive shots with the same shot_type — EXCEPT a dialogue exchange, where alternating `closeup`/`medium` shots between the two speakers is correct (shot/reverse-shot). For a `closeup`/`medium` shot where a character speaks, frame that speaker facing the camera (mouth visible for lip-sync).

4. **Extract characters from the prose.** For each named character the author mentioned, write a one-sentence `appearance` (visual traits only; NO poses, NO emotions, NO setting) that MUST state an explicit AGE first: for humans give a number ("a 7-year-old girl..."), for animals/creatures give a clear life stage ("a young sparrow...", "an old grey-whiskered dog..."). Then species, color, body shape, clothing, distinctive features. Then write a tight `locked_visual_token` (15-30 words) the renderer can paste verbatim into every shot prompt: `age + species/race + body-color + body-shape + clothing colors + ONE distinctive feature` — it must start with the age words.

4a. **The token must be CONCRETE and PURELY VISUAL — this is what keeps the character looking the same in every picture.** The image model draws exactly the nouns and colors it is given, so:
   - Lead with the SOLID FORM and its main COLOR right after the age: "a small round white robot...", "a young brown rabbit...", "a 6-year-old girl with black hair...". Name the body material/shape (round, boxy, fluffy, sleek) and the single dominant body color FIRST — never bury them.
   - BAN non-visual words from `appearance` AND `locked_visual_token`: no "energetic", "friendly", "cheerful", "happy", "curious", "shy", "brave", "cute", "magical", "lively". These describe personality, not pixels, and make the model draw a random creature. Replace them with a physical trait.
   - BAN vague glow-words as the lead descriptor ("glowing", "sparkly", "bright lights", "energetic form"). A robot is "a small round white robot with two round blue eyes", NOT "an energetic robot with yellow lights" (that renders a fuzzy yellow blob). If you want lights, attach them to a part: "a blue light strip on its chest".
   - For a NON-HUMAN character always name the species/form noun explicitly (robot, rabbit, mouse, turtle) so it can never drift into a different creature between shots.
   - For a HUMAN character, state the gender word AND the hair LENGTH/style right after the age ("a 7-year-old boy with short brown hair", "a 6-year-old girl with long curly black hair"). The renderer otherwise draws an ambiguous figure — a boy with only "brown hair" comes out looking like a girl. Always pin gender + hair length for humans.

4b. **ONE canonical name per character, used consistently.** Each character's `name` is a SHORT proper name only — NOT "Greenfield the Rabbit", just "Greenfield"; never put the species in the name. In EVERY `narration` line refer to that character by their exact canonical name. If the author's prose called the same character by a nickname or by their species as if it were a name ("Bunny", "the rabbit", "Tortoise"), REWRITE that mention to the canonical name so the spoken story and the pictures always match. Never introduce a second name for a character.

5. **If the prose only hints at a character's look ("a small girl", "a brown dog"), invent simple visual details that fit the story tone.** Pick concrete colors. Keep it kid-friendly. Be consistent across shots.

6. **Setting field**: one short sentence describing where the story takes place.

7. **Pick a music_mood** from: `cheerful | calm | adventurous | tender | magical | energetic` — match the emotional core of the story.

8. **Style field**: pick from `{', '.join(available_styles)}` — match the story's feel. Style guide:
{style_guide}

9. **Culture field**: pick from `indian | western | japanese | mixed | animal-kingdom | fantasy` — infer from prose, default `mixed`.

10. **Frame and motion prompts** (~20-40 words each) — these are the REAL image prompts, write them properly:
   - **ALWAYS name the character by their exact name** (e.g. "Beep", "Lila"). NEVER refer to a character by type or description ("the robot", "a small robot", "the boy", "a child", "a mouse"). The renderer injects each named character's locked look automatically — if you don't use the name, it draws a random stranger and breaks consistency.
   - `first_frame_prompt`: named subject + pose + camera framing + lighting. Do NOT re-describe hair/clothes/age/species (that is auto-injected). Example: "Beep rolls in through the classroom doorway, looking around shyly. Wide shot, warm morning light." For an establishing/atmosphere shot with no character on screen, describe the location only.
   - `last_frame_prompt`: same rules, a clearly different pose/expression to show change.
   - `motion_prompt`: physical motion + one camera move, no speech.

11. **Moral**: one-sentence summary of the lesson for the parent (NOT spoken in the story).

12. **JSON only. Every field comma-terminated except last. Test it would parse.**

# OUTPUT FORMAT

{{
  "title": "<from the author>",
  "theme": "<original theme>",
  "culture": "indian | western | japanese | mixed | animal-kingdom | fantasy",
  "style": "{' | '.join(available_styles)}",
  "duration_seconds": 30,
  "characters": [
    {{
      "name": "...",
      "type": "human | animal | creature",
      "appearance": "<visual traits sentence>",
      "locked_visual_token": "<tight 15-30 word phrase>"
    }}
  ],
  "setting": "<one sentence>",
  "shots": [
    {{
      "shot_number": 1,
      "beat": "atmosphere | hook | spark | reaction | observation | struggle | moment-of-decision | choice | consequence",
      "shot_type": "wide | medium | closeup | insert",
      "speaker": "narrator | <exact character name>",
      "narration": "<if speaker is a character: their SPOKEN words only (quotes + 'X said' stripped). If speaker is 'narrator': the author's prose verbatim. 6-15 words. NEVER empty.>",
      "visual_description": "<one short sentence>",
      "first_frame_prompt": "<~20-40 words: NAME the character (never 'the robot'/'the boy') + pose + camera framing + lighting; no appearance words>",
      "last_frame_prompt": "<~20-40 words: same rules, a different pose/expression to show change>",
      "motion_prompt": "<~20-40 words: physical motion + one camera move, no speech>"
    }}
  ],
  "moral": "<one sentence summary for parents>",
  "music_mood": "cheerful | calm | adventurous | tender | magical | energetic"
}}

Output ONLY the JSON. Start with {{ end with }}. Nothing else."""


def _valid_title(t: str) -> bool:
    """Reject leaked reasoning / tag garbage as a title (qwen3 sometimes dumps
    a <think> block, so the stage-1 'title' can be '<think>' or a reasoning
    fragment like 'Here's a thinking process:')."""
    t = (t or "").strip()
    if not t:
        return False
    low = t.lower()
    if "think" in low or "<" in t or ">" in t or t.endswith(":"):
        return False
    if not t[0].isalpha():
        return False
    return 1 <= len(t.split()) <= 8


def _structure_story(
    title: str,
    prose: str,
    theme: str,
    style_override: Optional[str] = None,
    culture_override: Optional[str] = None,
    shot_min: int = 5,
    shot_max: int = 9,
) -> dict:
    """
    Stage 2: feed the stage-1 prose to Qwen and ask it to emit the schema
    JSON. Calls _call_llm but with the structurer system prompt.

    shot_min/shot_max control how many shots the prose is split into. Defaults
    5-9; revisions that ask for "more shots" raise these so the structurer
    actually adds beats instead of clamping back to the original count.
    """
    system_prompt = _build_structurer_system_prompt(shot_min, shot_max)
    # Don't feed a leaked-reasoning title to the structurer — let it invent one.
    if not _valid_title(title):
        log.warning(f"Stage-1 title invalid ('{title[:40]}'); structurer will "
                    f"create its own.")
        title = ""
    user_prompt = (
        f"Original theme: {theme}\n\n"
        f"STORY TO STRUCTURE (author's prose — split this into shots; do NOT rewrite the words):\n\n"
        f"TITLE: {title}\n\n"
        f"{prose}\n\n"
        f"Now output the JSON. Use the title above. Use the prose above for the narration field "
        f"of each shot — split at natural sentence boundaries, do not paraphrase."
    )
    if style_override:
        user_prompt += f"\n\nUse style: {style_override}"
    if culture_override:
        user_prompt += f"\n\nUse culture: {culture_override}"

    raw = _call_llm(user_prompt, system_prompt)
    script = _extract_json(raw)
    script = _validate_and_default(script)
    # Use the author's title only when it's clean; otherwise keep the
    # structurer's own title, falling back to a safe default if that's bad too.
    if _valid_title(title):
        script["title"] = title
    elif not _valid_title(script.get("title", "")):
        script["title"] = "Untitled Story"
    return script


# ==============================================================================
# PUBLIC API — 2-stage flow
# ==============================================================================

def generate_script(theme: str, style_override: Optional[str] = None,
                    culture_override: Optional[str] = None) -> dict:
    """
    Generate a new script via 2-stage flow:
      Stage 1 (story_writer)  → organic prose, chat-style prompt, high temp
      Stage 2 (_structure_story) → JSON schema extraction, low temp
    The narration in each shot comes from the stage-1 prose, preserving voice.
    """
    _job_start = _t.time()

    # Apply persistent user override if caller didn't specify one
    effective_style = style_override or rs.get_style_override()

    # Detect explicit shot-count request in theme (e.g. "in 5 shots")
    shot_match = re.search(r"in (\d+)\s*shots?", theme, re.IGNORECASE)

    # === STAGE 1: write the story as prose ===
    log.info(f"Generating script for theme: '{theme[:80]}'")
    story = sw.write_story(
        theme=theme,
        style_hint=effective_style,
        culture_hint=culture_override,
    )
    log.info(
        f"Stage 1 prose: '{story['title']}' ({story['word_count']} words, "
        f"{story['duration_sec']}s)"
    )

    # === STAGE 2: structure prose into JSON schema ===
    log.info("Stage 2: structuring prose into JSON schema...")
    script = _structure_story(
        title=story["title"],
        prose=story["prose"],
        theme=theme,
        style_override=effective_style,
        culture_override=culture_override,
    )

    # Stash the original prose for later inspection / revisions
    script["_stage1_prose"] = story["prose"]
    script["_stage1_word_count"] = story["word_count"]

    # Ensure the story is long enough to fill the ~30s buffer.
    # Honor explicit shot-count request — skip expansion if user pinned count.
    if not shot_match:
        script = _expand_script_to_buffer(script)
        script = _validate_and_default(script)

    # Safety check on the generated script — filter returns (is_safe, blocked, warnings)
    is_safe, blocked_terms, warnings = sf.check_safety(script)
    if not is_safe:
        raise ValueError(
            f"Generated story contains hard-blocked terms: {blocked_terms}. "
            f"Try a different theme or word choice."
        )
    if warnings:
        log.info(f"Script generated with soft warnings: {warnings}")

    # Attach metadata
    now = datetime.now()
    script_id = now.strftime("%Y%m%d_%H%M%S")
    script["_id"] = script_id
    script["script_id"] = script_id  # legacy-compat
    script["_generated_at"] = now.isoformat()
    script["_revision_number"] = 1
    script["revision_number"] = 1  # legacy-compat

    # Apply style overrides post-hoc
    if effective_style:
        script["style"] = effective_style
    if culture_override:
        script["culture"] = culture_override

    # Save to disk
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS_DIR / f"script_{script_id}.json"
    atomic_write_json(output_path, script)
    log.info(f"Script saved: {output_path}")

    # Record to generation_meta (dashboard auto-refresh picks it up)
    try:
        llm_cfg = model_registry.get_active("llm_backend")
        llm_id = llm_cfg.get("_id", "unknown") if llm_cfg else "unknown"
    except Exception:
        llm_id = "unknown"
    gm.record({
        "kind": "script",
        "script_id": script_id,
        "duration_sec": round(_t.time() - _job_start, 1),
        "vram_peak_mb": 0,
        "settings": {
            "backend_id": llm_id,
            "style": script.get("style"),
            "culture": script.get("culture"),
            "shots": len(script.get("shots", [])),
            "title": script.get("title", "")[:60],
        },
        "success": True,
    })
    return script


# Words the LLM should add per NEW shot when expanding (≈ one narration line).
_WORDS_PER_NEW_SHOT = 12

_EXPAND_KEYWORDS = (
    "more shot", "add shot", "extra shot", "another shot", "more scene",
    "add scene", "more detail", "detailed", "longer", "expand", "elaborate",
    "flesh out", "more depth", "drag out", "slow it down", "slower pace",
    "break it up", "more beats", "add more", "make it bigger", "richer",
)
_SHRINK_KEYWORDS = (
    "fewer shot", "less shot", "shorter", "trim", "condense", "tighten",
    "cut it down", "too long", "remove a shot", "fewer scene",
)


def _resolve_revision_scale(feedback: str, cur_words: int, cur_shots: int):
    """Translate vague length feedback into concrete stage targets.

    Returns (length_instruction, shot_min, shot_max). Empty length_instruction
    + default 5-9 range when the feedback isn't about size (so normal surgical
    edits behave exactly as before).
    """
    fb = feedback.lower()

    # Explicit count wins: "add 3 shots", "make it 10 shots", "in 8 shots".
    target_shots = None
    m = re.search(r"(?:make it|in|to|now)\s+(\d{1,2})\s*shots?", fb)
    if m:
        target_shots = int(m.group(1))
    else:
        m = re.search(r"add\s+(\d{1,2})\s*(?:more\s+)?shots?", fb)
        if m:
            target_shots = cur_shots + int(m.group(1))

    wants_expand = target_shots is not None or any(k in fb for k in _EXPAND_KEYWORDS)
    wants_shrink = target_shots is None and any(k in fb for k in _SHRINK_KEYWORDS)

    if not wants_expand and not wants_shrink:
        return "", 5, 9  # not a size change — leave defaults

    if wants_shrink:
        target_shots = max(3, cur_shots - 2)
        new_words = max(MIN_NARRATION_WORDS, target_shots * _WORDS_PER_NEW_SHOT)
        instruction = (
            f"Make the story SHORTER and tighter — aim for about {new_words} words "
            f"total (the system's 70-85 word note still roughly applies). Cut the "
            f"weakest beats, keep the arc."
        )
        shot_min = max(3, target_shots - 1)
        shot_max = target_shots + 1
        return instruction, shot_min, shot_max

    # Expand path. No explicit number → grow by ~3 shots.
    if target_shots is None:
        target_shots = cur_shots + 3
    target_shots = max(cur_shots + 1, min(target_shots, 18))  # always more, cap 18

    new_words = max(target_shots * _WORDS_PER_NEW_SHOT, cur_words + 30)
    new_words = min(new_words, 260)  # sanity ceiling
    instruction = (
        f"IMPORTANT: ignore the 70-85 word length note in the system instructions "
        f"for THIS revision. Make the story LONGER and more detailed — aim for about "
        f"{new_words} words total so it can be split into roughly {target_shots} shots. "
        f"Add new beats (atmosphere, reactions, small struggles) and enrich existing "
        f"moments. Keep the same characters, arc, and moral."
    )
    shot_min = max(target_shots - 1, cur_shots + 1)
    shot_max = target_shots + 2
    return instruction, shot_min, shot_max


def revise_script(original_script: dict, feedback: str) -> dict:
    """
    Revise an existing script based on user feedback using the 2-stage flow:
      Stage 1 (story_writer.revise_prose) → rewrites the prose with feedback
      Stage 2 (_structure_story)          → re-structures into JSON schema
    """
    log.info(f"Revising via 2-stage — feedback: '{feedback[:80]}'")

    # --- Detect EXPAND intent up front -------------------------------------
    # "add more shots / make it more detailed / longer" must actually grow the
    # story. Without this, both stage prompts clamp back to ~70-85 words / 5-9
    # shots and the feedback feels ignored.
    cur_words = _count_narration_words(original_script)
    cur_shots = len(original_script.get("shots", []))
    length_instruction, shot_min, shot_max = _resolve_revision_scale(
        feedback, cur_words, cur_shots
    )
    if length_instruction:
        log.info(
            f"Expand intent detected — target ~{shot_min}-{shot_max} shots "
            f"(was {cur_shots}); growing narration from {cur_words} words."
        )

    # Best-effort thinker pass — still useful for surfacing extra hints to stage 1
    edit_plan = ft.think_about_feedback(original_script, feedback)
    plan_guidance = ft.plan_to_revision_prompt(edit_plan, feedback)

    original_title = original_script.get("title", "Untitled Story")
    theme = original_script.get("theme", "")
    # Prefer the stashed stage-1 prose; fall back to concatenated narration
    original_prose = (
        original_script.get("_stage1_prose")
        or " ".join(
            (s.get("narration") or "").strip()
            for s in original_script.get("shots", [])
            if (s.get("narration") or "").strip()
        ).strip()
    )

    feedback_for_stage1 = feedback
    if plan_guidance:
        feedback_for_stage1 = f"{feedback}\n\nEditor notes:\n{plan_guidance}"

    # === STAGE 1: revise the prose ===
    revised_story = sw.revise_prose(
        previous_prose=original_prose,
        previous_title=original_title,
        feedback=feedback_for_stage1,
        theme=theme,
        length_instruction=length_instruction,
    )

    # === STAGE 2: structure revised prose ===
    revised = _structure_story(
        title=revised_story["title"] or original_title,
        prose=revised_story["prose"] or original_prose,
        theme=theme,
        style_override=original_script.get("style"),
        culture_override=original_script.get("culture"),
        shot_min=shot_min,
        shot_max=shot_max,
    )
    revised["theme"] = theme  # keep original theme through revisions
    revised["_stage1_prose"] = revised_story["prose"]
    revised["_stage1_word_count"] = revised_story["word_count"]

    # Safety net: if the LLM dropped or mangled any locked_visual_token,
    # restore it from the original. The lock only changes when the user
    # explicitly asks to change appearance.
    feedback_lower = feedback.lower()
    appearance_keywords = (
        "appearance", "looks like", "look like", "wearing", "outfit",
        "clothes", "hair", "glasses", "age", "older", "younger",
        "redesign", "redraw character",
    )
    user_wants_appearance_change = any(k in feedback_lower for k in appearance_keywords)

    if not user_wants_appearance_change:
        original_locks = {
            c.get("name", "").strip().lower(): c.get("locked_visual_token", "")
            for c in original_script.get("characters", [])
            if isinstance(c, dict)
        }
        for c in revised.get("characters", []):
            if not isinstance(c, dict):
                continue
            name = c.get("name", "").strip().lower()
            if name in original_locks and original_locks[name]:
                if c.get("locked_visual_token", "") != original_locks[name]:
                    log.info(
                        f"Restoring locked_visual_token for '{c.get('name')}' "
                        f"(revision changed it, but user didn't ask for appearance change)"
                    )
                    c["locked_visual_token"] = original_locks[name]

    # Preserve identity + bump revision counter
    rev_num = int(original_script.get("_revision_number", 1)) + 1
    original_id = original_script.get("_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
    # Revised scripts get _vN suffix if the original id already exists
    new_id = f"{original_id}_v{rev_num}"

    revised["_id"] = new_id
    revised["script_id"] = new_id  # legacy-compat
    revised["_generated_at"] = datetime.now().isoformat()
    revised["_revision_number"] = rev_num
    revised["revision_number"] = rev_num  # legacy-compat
    revised.setdefault("_parent_id", original_id)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS_DIR / f"script_{new_id}.json"
    atomic_write_json(output_path, revised)
    log.info(f"Revised script saved: {output_path}")
    return revised


def format_for_discord(script: dict) -> str:
    """Format a script into a readable Discord message."""
    title = script.get("title", "Untitled")
    theme = script.get("theme", "")
    style = script.get("style", "")
    culture = script.get("culture", "")
    moral = script.get("moral", "")
    rev = script.get("_revision_number", 1)
    rev_suffix = f" (revision {rev})" if rev > 1 else ""
    script_id = script.get("_id", "")

    style_info = get_style_description(style)
    style_label = style_info.get("display_name", style) if style_info else style

    audio_first = bool(script.get("_audio_first"))
    if audio_first:
        # Real voiced length from the rendered master track (sum of shot windows).
        voiced = sum(float(s.get("win_dur") or 0) for s in script.get("shots", []))
        dur_line = (f"🎙️ **Audio-first** · voiceover **rendered** "
                    f"({voiced:.0f}s) — pauses set the cuts")
    else:
        dur_line = (f"⏱️ Est. narration: "
                    f"**~{_count_narration_words(script) / WORDS_PER_SECOND:.0f}s** "
                    f"({_count_narration_words(script)} words)")

    lines = [
        f"**📖 {title}**{rev_suffix}",
        f"_Theme: {theme}_",
        f"🎨 Style: **{style_label}**  ·  🌍 Culture: **{culture}**  ·  🎬 Shots: **{len(script.get('shots', []))}**",
        dur_line,
        f"ID: `{script_id}`",
        "",
        "**Characters:**",
    ]
    for c in script.get("characters", []):
        if isinstance(c, dict):
            name = c.get('name', 'Someone')
            ctype = c.get('type', 'character')
            appearance = c.get('appearance', '')
            lock = c.get('locked_visual_token', '')
            lines.append(f"• **{name}** _({ctype})_ — {appearance}")
            if lock and lock != appearance:
                lines.append(f"   🔒 _Locked: {lock[:200]}_")
        elif isinstance(c, str):
            lines.append(f"• {c}")
    lines.append("")
    lines.append(f"**Setting:** {script.get('setting', '')}")
    lines.append("")
    if audio_first:
        lines.append("**Shots:** _(narration is voiced & locked — pauses set the cuts)_")
    else:
        lines.append("**Shots:**")
    for shot in script.get("shots", []):
        n = shot.get("shot_number", "?")
        narration = shot.get("narration", "")
        vd = shot.get("visual_description", "")
        wd = shot.get("win_dur")
        tag = f" `⏱{float(wd):.1f}s`" if audio_first and wd else ""
        lines.append(f"**{n}.**{tag} 🎙️ *{narration}*")
        lines.append(f"       🖼️ {vd}")
    lines.append("")
    lines.append(f"✨ **Moral:** {moral}")
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    test_theme = "a shy little mouse learning to speak up"
    result = generate_script(test_theme)
    print(json.dumps(result, indent=2))