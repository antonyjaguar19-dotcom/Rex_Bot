"""
Claw Bot — Agent Module (Step 1: Memory Scaffolding)

This is the brain that will eventually replace command-based interaction.
Right now: just memory load/save. No LLM calls yet.

Memory model:
- One JSON file per Discord channel: 06_Memory/<channel_id>.json
- Structure: { "summary": "...", "recent_turns": [...] }
- Each turn: { "role": "user|assistant|tool", "text": "...", "ts": "...", "meta": {...} }
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from modules.file_utils import atomic_write_json

log = logging.getLogger("claw_bot.agent")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
MEMORY_DIR = PROJECT_ROOT / "06_Memory"


# ==============================================================================
# MEMORY LOAD / SAVE
# ==============================================================================

def _memory_path(channel_id: int | str) -> Path:
    return MEMORY_DIR / f"{channel_id}.json"


def load_memory(channel_id: int | str) -> dict:
    """Load a channel's chat memory. Returns empty memory if file doesn't exist."""
    path = _memory_path(channel_id)
    if not path.exists():
        return {"summary": "", "recent_turns": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning(f"Memory file corrupt for channel {channel_id}: {e}. Starting fresh.")
        return {"summary": "", "recent_turns": []}


def save_memory(channel_id: int | str, memory: dict) -> None:
    """Persist memory to disk."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    path = _memory_path(channel_id)
    atomic_write_json(path, memory)


# ==============================================================================
# TURN MANAGEMENT
# ==============================================================================

def append_turn(
    channel_id: int | str,
    role: str,
    text: str,
    meta: Optional[dict] = None,
) -> dict:
    """
    Add a turn to the channel's memory and persist.
    role: 'user' | 'assistant' | 'tool'
    Returns the updated memory dict.
    """
    if role not in ("user", "assistant", "tool", "system"):
        raise ValueError(f"Invalid role: {role}")

    memory = load_memory(channel_id)
    memory.setdefault("summary", "")
    memory.setdefault("recent_turns", [])

    turn = {
        "role": role,
        "text": text,
        "ts": datetime.now().isoformat(),
    }
    if meta:
        turn["meta"] = meta

    memory["recent_turns"].append(turn)
    save_memory(channel_id, memory)
    return memory


def get_recent_turns(channel_id: int | str, n: int = 20) -> list[dict]:
    """Get the last N turns from memory. Useful for previewing/debugging."""
    memory = load_memory(channel_id)
    return memory.get("recent_turns", [])[-n:]


def get_summary(channel_id: int | str) -> str:
    """Get the rolling summary for a channel."""
    return load_memory(channel_id).get("summary", "")


def clear_memory(channel_id: int | str) -> None:
    """Wipe a channel's memory entirely. (For !forget command later.)"""
    path = _memory_path(channel_id)
    if path.exists():
        path.unlink()
        log.info(f"Cleared memory for channel {channel_id}")


# ==============================================================================
# DEBUG / INTROSPECTION
# ==============================================================================

def memory_stats(channel_id: int | str) -> dict:
    """Quick stats for debugging."""
    memory = load_memory(channel_id)
    turns = memory.get("recent_turns", [])
    summary = memory.get("summary", "")
    total_text_chars = sum(len(t.get("text", "")) for t in turns) + len(summary)
    return {
        "channel_id": str(channel_id),
        "turn_count": len(turns),
        "summary_length_chars": len(summary),
        "total_text_chars": total_text_chars,
        "approx_tokens": total_text_chars // 4,  # rough estimate
        "first_turn_ts": turns[0]["ts"] if turns else None,
        "last_turn_ts": turns[-1]["ts"] if turns else None,
    }

# ==============================================================================
# CONTEXT TRACKING — "current script" per channel
# ==============================================================================

def set_current_script(channel_id: int | str, script_id: str) -> None:
    """Remember the most recent script for a channel."""
    memory = load_memory(channel_id)
    memory["current_script_id"] = script_id
    save_memory(channel_id, memory)


def get_current_script(channel_id: int | str) -> Optional[str]:
    """Get the most recent script ID for this channel, or None."""
    return load_memory(channel_id).get("current_script_id")

# ==============================================================================
# STAGE TRACKING — what stage of pipeline is the channel currently on?
# ==============================================================================

# Stages: "idle" | "script_generated" | "script_approved" | "storyboard_generated"
#         | "storyboard_approved" | "video_generated"

def set_stage(channel_id: int | str, stage: str) -> None:
    memory = load_memory(channel_id)
    memory["stage"] = stage
    save_memory(channel_id, memory)


def get_stage(channel_id: int | str) -> str:
    return load_memory(channel_id).get("stage", "idle")

# ==============================================================================
# SELF-TEST (run this file directly to verify memory works)
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    test_channel = "_test_channel"
    print("=" * 60)
    print("STEP 1 SELF-TEST — memory scaffolding")
    print("=" * 60)

    # Clear any previous test data
    clear_memory(test_channel)

    # Add a few turns
    append_turn(test_channel, "user", "Tell me a story about a turtle")
    append_turn(test_channel, "assistant", "Sure! Here's a draft...", meta={"tool_called": "generate_script", "script_id": "test_001"})
    append_turn(test_channel, "user", "Make the turtle scared at the start")
    append_turn(test_channel, "assistant", "Updated! Here's the revision...", meta={"tool_called": "revise_script"})

    # Inspect
    stats = memory_stats(test_channel)
    print(f"\n📊 Memory stats:\n{json.dumps(stats, indent=2)}")

    print(f"\n📝 Recent turns:")
    for t in get_recent_turns(test_channel):
        meta = f" [{t.get('meta', {}).get('tool_called', '')}]" if t.get("meta") else ""
        print(f"  {t['role']:10s} {t['text'][:60]}{meta}")

    print(f"\n💾 Memory file location:\n  {_memory_path(test_channel)}")
    print("\n✅ Self-test passed if you see 4 turns above.")
    print("   Run again to verify clear_memory wipes it (next run starts fresh).")
