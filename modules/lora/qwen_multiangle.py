"""
Qwen-Image-Edit multi-angle dataset generator.

Feeds ONE hero image into the installed Qwen-Image-Edit 2509 multi-angle
workflow (05_Config/workflows/multiple_character_angles-v1.0.json) and pulls 8
identity-locked angle views (close-up, wide, rotate +/-45/+/-90, aerial,
low-angle). Far tighter identity + accessory retention than Z-Turbo turnaround,
because every view is an EDIT of the same hero, not a fresh generation.

Used to build a strong character-LoRA training set: 1-3 heroes -> 8-24 frames.
"""

import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import requests

_AGENT = Path(__file__).parent.parent.parent.resolve()
if str(_AGENT) not in sys.path:
    sys.path.insert(0, str(_AGENT))
from modules.lora import store

log = logging.getLogger("claw_bot.lora.qwen_multiangle")

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
WORKFLOW = PROJECT_ROOT / "05_Config" / "workflows" / "multiple_character_angles-v1.0.json"
SERVER = "http://127.0.0.1:8188"


def _upload(server: str, img: Path) -> str:
    with open(img, "rb") as f:
        r = requests.post(f"{server}/upload/image",
                          files={"image": (img.name, f, "image/png")},
                          data={"overwrite": "true"}, timeout=60)
    r.raise_for_status()
    return r.json().get("name", img.name)


def _find_load_image_node(wf: dict) -> str:
    for nid, n in wf.items():
        if isinstance(n, dict) and n.get("class_type") == "LoadImage":
            return nid
    raise RuntimeError("no LoadImage node in multiangle workflow")


def generate_angles(hero: Path, out_dir: Path, prefix: str,
                    server: str = SERVER, timeout: int = 1200) -> list[Path]:
    """Run the multi-angle workflow on `hero`, save all outputs into out_dir.

    Returns list of saved frame paths (typically 8).
    """
    hero = Path(hero)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    wf = json.loads(WORKFLOW.read_text(encoding="utf-8-sig"))

    name = _upload(server, hero)
    wf[_find_load_image_node(wf)]["inputs"]["image"] = name
    log.info(f"multiangle hero={hero.name} uploaded as {name}")

    cid = str(uuid.uuid4())
    r = requests.post(f"{server}/prompt", json={"prompt": wf, "client_id": cid}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"ComfyUI rejected multiangle (HTTP {r.status_code}): {r.text[:400]}")
    pid = r.json()["prompt_id"]

    start = time.time()
    entry = None
    while time.time() - start < timeout:
        h = requests.get(f"{server}/history/{pid}", timeout=10)
        if h.status_code == 200 and pid in h.json():
            e = h.json()[pid]
            st = e.get("status", {})
            if st.get("completed") is True:
                entry = e; break
            if st.get("status_str") == "error":
                raise RuntimeError(f"multiangle error: {st.get('messages')}")
        time.sleep(2)
    if entry is None:
        raise TimeoutError(f"multiangle timed out after {timeout}s")

    saved = []
    i = 0
    for _, out in entry.get("outputs", {}).items():
        for img in out.get("images", []):
            if img.get("type") != "output":
                continue
            url = (f"{server}/view?filename={img['filename']}"
                   f"&subfolder={img.get('subfolder','')}&type=output")
            d = requests.get(url, timeout=120)
            if d.status_code == 200:
                i += 1
                p = out_dir / f"{prefix}_{i:02d}.png"
                p.write_bytes(d.content)
                saved.append(p)
    log.info(f"multiangle produced {len(saved)} frames -> {out_dir}")
    return saved


def build_dataset(name: str, heroes: list[Path], server: str = SERVER) -> int:
    """Run multiangle on each hero, collect into the character dataset dir."""
    ds = store.dataset_dir(name)
    total = 0
    for hi, hero in enumerate(heroes, 1):
        frames = generate_angles(hero, ds, prefix=f"{store.slug(name)}_h{hi}", server=server)
        total += len(frames)
    log.info(f"qwen-multiangle dataset for '{name}': {total} frames")
    return total


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    hero = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        PROJECT_ROOT / "07_Training" / "datasets" / "pip" / "pip_01.png"
    out = PROJECT_ROOT / "04_Outputs" / "lora_test" / "multiangle_probe"
    frames = generate_angles(hero, out, "probe")
    print("FRAMES:", len(frames), "->", out)
