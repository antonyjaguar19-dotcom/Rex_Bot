"""Quick inspector: dump every node's class_type, title, and input keys."""
import json
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
default_wf = "ltx2_video_only_v1_api.json"
filename = sys.argv[1] if len(sys.argv) > 1 else default_wf
WF = PROJECT_ROOT / "05_Config" / "workflows" / filename

wf = json.loads(WF.read_text(encoding="utf-8"))
print(f"Workflow: {WF.name}")
print(f"Total nodes: {len(wf)}\n")

for nid in sorted(wf.keys(), key=lambda x: int(x) if str(x).isdigit() else 99999):
    node = wf[nid]
    if not isinstance(node, dict):
        continue
    ctype = node.get("class_type", "?")
    title = node.get("_meta", {}).get("title", "")
    inputs = node.get("inputs", {})

    print(f"Node {nid}: {ctype}    [title: {title!r}]")
    for k, v in inputs.items():
        # Show literal values inline; show wires as references
        if isinstance(v, list) and len(v) == 2:
            print(f"    {k}: <wire from node {v[0]}, output {v[1]}>")
        elif isinstance(v, str) and len(v) > 100:
            print(f"    {k}: {v[:100]!r}... (truncated)")
        else:
            print(f"    {k}: {v!r}")
    print()
