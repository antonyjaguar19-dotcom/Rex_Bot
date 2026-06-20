"""
Phase A end-to-end driver: train ONE character LoRA, then prove it holds
across varied shots on the Flux.1-dev + LoRA backend.

Run (bot venv):
    python test_lora_e2e_run.py            # full: dataset(if missing) -> train -> render
    python test_lora_e2e_run.py --steps 800
    python test_lora_e2e_run.py --skip-train   # just render with existing LoRA

Outputs comparison frames to 04_Outputs/lora_test/ and posts progress to #status.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

from modules.lora import store, captioner, dataset_builder, trainer
from modules import model_registry as mr
from modules.image_backends import comfyui_flux_lora as fl

CHAR = "Pip"
TOKEN = ("7-year-old boy, messy orange hair, round brass goggles resting on forehead, "
         "blue scarf, brown leather satchel across chest, green rubber boots")
STYLE = "soft 3D storybook render, Pixar-like, warm cinematic lighting"
OUT = Path(__file__).parent.parent / "04_Outputs" / "lora_test"

# varied shots — same character, different framing/action/scene.
# LoRA proof = face + goggles + scarf + satchel identical across all.
SHOTS = [
    ("front_park", f"{CHAR} standing in a sunny park waving hello, full body, front view"),
    ("side_run",   f"{CHAR} running across a wooden bridge, side view, dynamic"),
    ("closeup",    f"{CHAR} smiling close-up portrait, looking at camera"),
    ("sit_read",   f"{CHAR} sitting on grass reading a book, three-quarter view"),
]


def _post(msg):
    try:
        import tools.status_post as sp
        sp.post(msg)
    except Exception as e:
        print("status post failed:", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--min-imgs", type=int, default=12)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. dataset
    have = captioner.count_images(store.dataset_dir(CHAR))
    print(f"[dataset] {have} images for {CHAR}")
    if have < args.min_imgs and not args.skip_train:
        _post(f"🎨 dataset short ({have}); generating more for {CHAR}…")
        dataset_builder.build_dataset(CHAR, TOKEN, STYLE, n_images=20)
        have = captioner.count_images(store.dataset_dir(CHAR))
    print(f"[dataset] final {have} images")

    # 2. train
    if not args.skip_train:
        # free ComfyUI VRAM so the 12B Flux train fits the 16GB card
        try:
            from modules import gpu_utils as _g
            _g.free_comfyui_vram()
            try:
                _g.free_ollama_vram()
            except Exception:
                pass
            print("[vram] freed ComfyUI/Ollama before training")
        except Exception as e:
            print("[vram] free failed (continuing):", e)
        _post(f"🏋️ training LoRA '{CHAR}' — {have} imgs, {args.steps} steps. ~30-60min.")
        t0 = time.time()
        def cb(m, c, t):
            if c % 100 == 0:
                _post(f"… {m} {c}/{t}")
        entry = trainer.train_character(
            CHAR, TOKEN, steps=args.steps, rank=args.rank,
            progress_callback=cb, register=True,
        )
        print("[train] registered:", json.dumps(entry, indent=2))
        _post(f"✅ trained '{CHAR}' v{entry['version']} in {int(time.time()-t0)}s → {entry['lora_file']}")
    else:
        entry = store.get(CHAR)
        if not entry:
            print("no existing LoRA; drop --skip-train"); sys.exit(1)

    # 3. render proof shots on flux_lora backend
    cfg = dict(mr.get_available("image_backend", "comfyui_flux_lora")); cfg["_id"] = "comfyui_flux_lora"
    backend = fl.Backend(cfg)
    ok, msg = backend.health_check()
    print("[render] backend health:", ok, msg)
    if not ok:
        _post(f"❌ ComfyUI down, cannot render: {msg}"); sys.exit(1)

    _post(f"🖼️ rendering {len(SHOTS)} proof shots with '{CHAR}' LoRA…")
    made = []
    for name, prompt in SHOTS:
        full = f"Art style: {STYLE}. {prompt}."
        try:
            p = backend.generate(prompt=full, aspect_ratio="1:1",
                                 output_filename=f"../lora_test/{CHAR.lower()}_{name}.png")
            made.append(str(p))
            print("[render]", name, "->", p)
        except Exception as e:
            print("[render] FAIL", name, e)
            _post(f"⚠️ shot {name} failed: {e}")

    _post(f"🎬 e2e done — {len(made)}/{len(SHOTS)} shots in 04_Outputs/lora_test/. "
          f"Eyeball: goggles+scarf+satchel consistent across all?")
    print("DONE. frames:", made)


if __name__ == "__main__":
    main()
