import sys
from pathlib import Path

# Make `modules` importable exactly the way claw_bot.py does it
# (sys.path.insert of 02_Agent — never `python -m modules.X`).
AGENT_ROOT = Path(__file__).parent.parent.resolve()
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))
