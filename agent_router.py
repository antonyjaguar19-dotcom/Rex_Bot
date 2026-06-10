"""
Claw Bot — Agent Router (Step 2)

The brain. Reads chat history, calls Qwen, returns:
  { reply: "...", tool_call: {name, args} | None }

Step 2 only wires ONE tool: generate_script. More added in Step 4.
"""

import json
import logging
import re
from typing import Optional

import sys
from pathlib import Path

import requests

# Make `modules` importable when running this file directly
_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modules import agent
from modules import model_registry

log = logging.getLogger("claw_bot.agent_router")

OLLAMA_URL = "http://127.0.0.1:11434"
MAX_HISTORY_TURNS = 20  # how many past turns to send to Qwen each call


# ==============================================================================
# TOOL DEFINITIONS — what the agent can do
# ==============================================================================
# Each tool: name, when to use, args schema. Qwen sees this in its system prompt.

TOOLS = [
    {
        "name": "generate_script",
        "description": (
            "Create a brand-new 30-second children's story from a theme. "
            "Use when the user asks for a NEW story, gives a theme, or says "
            "'tell me a story about X', 'make a new one', 'try again with a different idea'."
        ),
        "args": {
            "theme": "string — the story theme (required)",
            "style": "string OR null — storybook|cartoon|anime|watercolor|pixelart (only if user asked)",
            "culture": "string OR null — indian|western|japanese|mixed|animal-kingdom|fantasy (only if user asked)",
        },
    },
    {
        "name": "revise_script",
        "description": (
            "Modify the current/most-recent script based on user feedback. "
            "Use when user says 'make X scared', 'shorten shot 3', 'change the ending', "
            "'rewrite with a different setting', etc. Operates on a script that already exists."
        ),
        "args": {
            "feedback": "string — what to change, in the user's words (required)",
            "script_id": "string OR null — only if user gave an explicit ID; else uses current",
        },
    },
    {
        "name": "start_storyboard",
        "description": (
            "Render the storyboard images for an APPROVED script. Use when user says "
            "'make the storyboard', 'render it', 'generate the images', 'show me what it looks like'. "
            "Only valid AFTER a script exists and has been approved."
        ),
        "args": {
            "script_id": "string OR null — only if user gave an explicit ID; else uses current",
        },
    },
    {
        "name": "regenerate_shot",
        "description": (
            "Re-render a specific storyboard shot's image. Use when user says "
            "'redo shot 2', 'shot 3 looks wrong', 'regenerate the close-up'."
        ),
        "args": {
            "shot_numbers": "array of integers — shot numbers to regenerate, e.g. [2] or [3, 4] (required)",
            "script_id": "string OR null — only if user gave an explicit ID; else uses current",
        },
    },
    {
        "name": "start_video",
        "description": (
            "Generate animated video clips from an APPROVED storyboard. Use when user says "
            "'make the videos', 'animate it', 'render the motion'. Only valid AFTER storyboard exists."
        ),
        "args": {
            "script_id": "string OR null — only if user gave an explicit ID; else uses current",
        },
    },
    {
        "name": "list_recent_scripts",
        "description": (
            "Show the user a list of their recent stories. Use when they ask "
            "'what stories have I made', 'show my recent ones', 'list scripts'."
        ),
        "args": {},
    },
    {
        "name": "no_tool",
        "description": (
            "No action needed — pure conversation. "
            "Greetings, thanks, small talk, clarifying questions, status questions."
        ),
        "args": {},
    },
]


def _format_tools_for_prompt() -> str:
    lines = []
    for t in TOOLS:
        lines.append(f'## Tool: `{t["name"]}`')
        lines.append(f'**When to use:** {t["description"]}')
        if t["args"]:
            lines.append("**Arguments:**")
            for arg, desc in t["args"].items():
                lines.append(f'  - `{arg}`: {desc}')
        else:
            lines.append("**Arguments:** none")
        lines.append("")
    return "\n".join(lines)


# ==============================================================================
# SYSTEM PROMPT
# ==============================================================================

def _build_system_prompt() -> str:
    tools_block = _format_tools_for_prompt()
    return f"""/no_think

You output ONLY valid JSON. No prose, no preamble, no markdown fences. Start with {{ end with }}.

You are Claw Bot — a warm, conversational creative producer who helps a VFX artist make 30-second illustrated stories for kids. You are NOT a robotic command dispatcher. You speak like a friendly collaborator.

# YOUR JOB

For each user message, decide:
1. **What does the user actually want?** Read the conversation history. Use context.
2. **Do I need to call a tool, or just chat?**
3. **What should I say back?** Warm, brief, human. Never robotic.

# AVAILABLE TOOLS

{tools_block}

# OUTPUT FORMAT (always JSON)

{{
  "reasoning": "<one short sentence — what the user wants and what you'll do>",
  "reply": "<your message to the user — natural, conversational, brief>",
  "tool_call": {{
    "name": "<tool name from the list above>",
    "args": {{ ... }}
  }}
}}

If no tool is needed, set `"tool_call": {{"name": "no_tool", "args": {{}}}}`.

# REPLY STYLE RULES

- Sound like a person, not a help desk. "Sure, working on it!" not "Acknowledged. Generating script."
- Brief. 1-3 sentences usually. No lists unless the user asked for one.
- If you're calling a tool, your reply should ACKNOWLEDGE what you're doing — "Cooking up a turtle story for you, one sec..." — not "I will now invoke generate_script."
- Don't mention tool names, JSON, or any internals.
- If the user is unclear, ASK A QUESTION. Better to clarify than guess.

# CRITICAL — WHEN NOT TO CALL A TOOL

A tool is for taking ACTION on something the user just asked you to do. You must NOT call a tool if:

- The user said "thanks", "ok", "cool", "great", "hi", "yo", "okay", "sure", or any acknowledgement
- The user is just chatting, asking questions, or making small talk
- A `[tool: ... dispatched successfully ...]` message appears in recent history — that means the tool ALREADY RAN and the result was already shown to the user. Do NOT re-run it. Acknowledge it casually if relevant ("Hope you like it!" / "Let me know what you think!") or just chat.
- The user's message has nothing to do with creating, revising, or rendering a story

Re-running tools the user did not explicitly ask for again is BAD and creates duplicate work.

# READING TOOL HISTORY

Lines that look like `[TOOL] [generate_script dispatched successfully — output already shown to user above]` mean a previous tool call has COMPLETED. The user already saw the output. Do NOT pretend the tool is still running. Do NOT say "almost ready" or "one moment" — that work is finished.

# WHEN TO CALL A TOOL

Only call a tool when the user is making a NEW request, e.g.:
- "tell me a story about X" → generate_script (new request, no recent generate_script in history for this theme)
- "make a different one" / "try again" → generate_script (NEW request)
- After "thanks!" / "looks good!" / "okay" / "hi" → no_tool (just chat back)

# CRITICAL — DON'T HALLUCINATE THEMES

If the user just said "hi", "hello", "yo" or any greeting with NO topic — call no_tool. Do NOT invent a theme from previous channel memory. A new "hi" is a fresh greeting, not a continuation.

# WHEN TO CALL A TOOL

Only call a tool when the user is making a NEW request, e.g.:
- "tell me a story about X" → generate_script (NEW request)
- "make a different one" → generate_script (NEW request)
- After "thanks!" or "looks good!" → no_tool (just chat back)
- After "ok cool" → no_tool (acknowledgement, not a request)

# PIPELINE STAGES & DISAMBIGUATION

The user's project goes through stages, shown as `PIPELINE STAGE: <name>` above:
- `idle` → no work yet
- `script_generated` → a script exists, awaiting user approval
- `script_approved` → script approved, storyboard images being made
- `storyboard_generated` → storyboard images posted, awaiting approval
- `storyboard_approved` → storyboard approved, videos being made
- `video_generated` → videos posted

When the user says vague things like "redo shot 2", "fix it", "the butterfly is wrong", consider the stage:

- At `script_generated` → "fix it" probably means revise_script
- At `storyboard_generated` → "redo shot 2" probably means regenerate_shot (the image)
- At `video_generated` → "redo shot 2" probably means regenerate the video clip

# WHEN TO ASK FOR CLARIFICATION

If the user's intent is genuinely ambiguous, DO NOT GUESS. Set tool_call to `no_tool` and use your reply to ask the user a clarifying question. Offer 2-3 specific options.

Example: User says "the butterfly is missing" at stage `storyboard_generated`.
- Reply: "Should I (1) regenerate the storyboard image for that shot with the butterfly added, or (2) rewrite the script prompts for that shot first? Which one?"

Better to ask than to do wasted work. Asking is cheap; bad regeneration is expensive.

# OUTPUT

Output ONLY the JSON object. Start with {{ end with }}. Nothing else."""


# ==============================================================================
# OLLAMA CALL
# ==============================================================================

def _call_router_llm(user_text: str, history: list[dict],
                     summary: str = "", stage: str = "idle",
                     current_script_id: str = "") -> str:
    """Call the active LLM with system prompt + summary + history + new message."""
    cfg = model_registry.get_active("llm_backend")
    model_name = (
        cfg.get("model_id") or cfg.get("model_name")
        or cfg.get("model") or "qwen3story"
    )
    url = cfg.get("server_url", OLLAMA_URL) + "/api/generate"

    transcript_lines = []
    if summary:
        transcript_lines.append(f"PRIOR CONVERSATION SUMMARY:\n{summary}\n")

    transcript_lines.append(f"PIPELINE STAGE: {stage}")
    if current_script_id:
        transcript_lines.append(f"CURRENT SCRIPT ID: {current_script_id}")
    transcript_lines.append("")
    transcript_lines.append("RECENT CONVERSATION:")
    for t in history[-MAX_HISTORY_TURNS:]:
        role = t.get("role", "user").upper()
        text = t.get("text", "")
        if role == "ASSISTANT":
            tool = (t.get("meta") or {}).get("tool_called")
            if tool and tool != "no_tool":
                text = f"{text}  [→ called tool: {tool}]"
        transcript_lines.append(f"[{role}] {text}")
    transcript_lines.append(f"[USER] {user_text}")
    transcript = "\n".join(transcript_lines)

    system_prompt = _build_system_prompt()
    combined = (
        f"SYSTEM INSTRUCTIONS (STRICTLY FOLLOW):\n{system_prompt}\n\n"
        f"{transcript}\n\n"
        f"Now output your JSON response."
    )

    payload = {
        "model": model_name,
        "prompt": combined,
        "stream": False,
        "options": {
            "temperature": 0.5,
            "top_p": 0.9,
            "num_ctx": 16384,
            "num_predict": 2048,
        },
    }

    log.info(f"Router calling {model_name} | stage={stage} | user: {user_text[:60]}...")
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    return r.json().get("response", "").strip()


# ==============================================================================
# JSON EXTRACTION
# ==============================================================================

def _extract_json(raw: str) -> dict:
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    # Handle unclosed <think> blocks
    if "<think>" in cleaned.lower():
        json_start = cleaned.find("{")
        if json_start > 0:
            cleaned = cleaned[json_start:]
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    candidate = match.group(0) if match else cleaned

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        repaired = re.sub(r",(\s*[}\]])", r"\1", candidate)
        return json.loads(repaired)


# ==============================================================================
# PUBLIC API
# ==============================================================================

def route(channel_id: int | str, user_text: str) -> dict:
    """
    Take a user message, return router decision:
      { reasoning, reply, tool_call: {name, args} }

    Does NOT execute the tool — caller does that.
    Does NOT save memory — caller does that (so user/assistant turns stay in order).
    """
    memory = agent.load_memory(channel_id)
    history = memory.get("recent_turns", [])
    summary = memory.get("summary", "")
    stage = memory.get("stage", "idle")
    current_script_id = memory.get("current_script_id", "")

    raw = _call_router_llm(user_text, history, summary, stage, current_script_id)

    try:
        decision = _extract_json(raw)
    except Exception as e:
        log.error(f"Router JSON parse failed: {e} | raw start: {raw[:200]}")
        # Graceful fallback: just chat back, no tool
        return {
            "reasoning": "router parse error, falling back to chat",
            "reply": "Hmm, my brain hiccuped — could you say that again?",
            "tool_call": {"name": "no_tool", "args": {}},
        }

    # Defensive defaults
    decision.setdefault("reasoning", "")
    decision.setdefault("reply", "")
    decision.setdefault("tool_call", {"name": "no_tool", "args": {}})
    if not isinstance(decision["tool_call"], dict):
        decision["tool_call"] = {"name": "no_tool", "args": {}}
    decision["tool_call"].setdefault("name", "no_tool")
    decision["tool_call"].setdefault("args", {})

    log.info(
        f"Router decision: tool={decision['tool_call']['name']} | "
        f"reasoning: {decision['reasoning'][:80]}"
    )
    return decision


# ==============================================================================
# SELF-TEST — runs against your live Ollama server
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    test_channel = "_router_test"
    agent.clear_memory(test_channel)

    test_cases = [
        "hi there",
        "tell me a story about a brave little tomato",
        "thanks!",
        "make it in anime style",
    ]

    for user_msg in test_cases:
        print("\n" + "=" * 60)
        print(f"USER: {user_msg}")
        print("=" * 60)

        decision = route(test_channel, user_msg)
        print(f"💭 Reasoning: {decision['reasoning']}")
        print(f"💬 Reply: {decision['reply']}")
        print(f"🔧 Tool: {decision['tool_call']['name']}")
        if decision['tool_call']['args']:
            print(f"   Args: {json.dumps(decision['tool_call']['args'], indent=2)}")

        # Persist user + assistant turns so context builds across the test
        agent.append_turn(test_channel, "user", user_msg)
        agent.append_turn(
            test_channel, "assistant", decision["reply"],
            meta={"tool_called": decision["tool_call"]["name"]}
        )
