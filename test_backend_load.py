"""
Quick test: loads the active image backend and runs a health check.
Run from 02_Agent: python test_backend_load.py
"""
import sys
import logging
from pathlib import Path

# Make sure 02_Agent is on the path
_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from modules import image_backend as ib

backend = ib.get_active_backend()
print(f"\nLoaded backend: {backend.backend_id}")
print(f"Config keys: {list(backend.config.keys())}")

ok, msg = backend.health_check()
print(f"Health: {'OK ✓' if ok else 'FAIL ✗'} — {msg}\n")

if not ok:
    sys.exit(1)