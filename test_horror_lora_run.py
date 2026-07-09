"""One-off: render 5 horror stills via flux_lora backend + Horrorstyle style LoRA
(no character LoRA). Verifies the style LoRA actually loads and renders."""
import sys, time, importlib
from pathlib import Path

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from modules import model_registry, gpu_utils
from modules import runtime_settings as rs

OUT = HERE.parent / "04_Outputs" / "storyboards" / "_lora_test"
OUT.mkdir(parents=True, exist_ok=True)

PROMPTS = [
    "an abandoned hospital corridor at night, flickering fluorescent light, peeling walls, wet floor reflections",
    "a pale 40-year-old man in a torn coat standing alone in a dark forest, fog, moonlight through bare trees",
    "a child's empty bedroom, toys scattered, a shadow stretching across the wall, dim lamp",
    "a decrepit farmhouse kitchen, rusted knives, blood-dark stains, single candle, deep shadows",
    "a foggy graveyard at midnight, leaning headstones, a distant figure watching, cold blue light",
]

style = rs.get_horror_style_lora()
weight = rs.get_horror_style_lora_weight()
print(f"style LoRA = {style} @ {weight}")

cfg = model_registry.get_available("image_backend", "comfyui_flux_lora")
mod = importlib.import_module(cfg["module_path"])
backend = mod.Backend(cfg)
ok, msg = backend.health_check()
print("health:", ok, msg)

gpu_utils.free_comfyui_vram()
style_kw = {"use_char_lora": False,
            "extra_loras": [{"lora_file": style, "weight": weight}]}

for i, p in enumerate(PROMPTS):
    t0 = time.time()
    print(f"\n[{i+1}/5] {p[:60]}...")
    out = backend.generate(
        prompt=p + ", photorealistic, cinematic horror, film grain, dark, eerie",
        aspect_ratio="16:9",
        seed=1000 + i,
        output_filename=str(OUT / f"lora_test_{i:02d}.png"),
        **style_kw,
    )
    print(f"  saved {out}  ({time.time()-t0:.0f}s)")

print(f"\nDONE -> {OUT}")
