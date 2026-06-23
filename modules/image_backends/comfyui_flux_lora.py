"""
Claw Bot — Flux.1-dev + Character-LoRA image backend (ComfyUI API).

This is the production CHARACTER-SHOT backend. It runs Flux.1-dev (same base the
character LoRAs were trained on) and, per shot, auto-stacks the LoRA(s) of any
registered character whose name appears in the prompt — baking in that
character's exact face + outfit + accessories.

Resolution flow:
    prompt text  --store.resolve_for_prompt-->  [Pip entry, ...]
        -> insert LoraLoaderModelOnly node per character (chained on model)
        -> prepend each trigger word to the prompt

Non-character / establishing shots resolve to zero LoRAs and render as plain
Flux.1-dev. Z-Image Turbo stays the speed backend for those if preferred.
"""

import json
import logging
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import requests

_AGENT_DIR = Path(__file__).parent.parent.parent.resolve()
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

from modules.image_backend import ImageBackend
from modules import model_registry
from modules import runtime_settings as rs
from modules.lora import store as lora_store

log = logging.getLogger("claw_bot.image_backend.flux_lora")

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
WORKFLOW_PATH = PROJECT_ROOT / "05_Config" / "workflows" / "flux_dev_lora_api.json"
OUTPUT_DIR = PROJECT_ROOT / "04_Outputs" / "storyboards"


class Backend(ImageBackend):
    """Flux.1-dev + dynamic character-LoRA stacking via ComfyUI."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.server_url = config.get("server_url", "http://127.0.0.1:8188").rstrip("/")
        # all-in-one flux1-dev fp8 checkpoint (model+clip+vae), loaded via CheckpointLoaderSimple
        self.ckpt_name = config.get("ckpt_name", "flux1-dev-fp8.safetensors")
        self.steps = int(config.get("steps", 20))
        self.guidance = float(config.get("guidance", 3.5))
        self.cfg = float(config.get("cfg", 1.0))
        self.sampler = config.get("sampler", "euler")
        self.scheduler = config.get("scheduler", "simple")
        self._last_seed = -1

        if not WORKFLOW_PATH.exists():
            raise FileNotFoundError(f"Workflow not found: {WORKFLOW_PATH}")
        self._template = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8-sig"))

    # ------------------------------------------------------------------ contract
    def health_check(self) -> tuple[bool, str]:
        try:
            r = requests.get(f"{self.server_url}/system_stats", timeout=5)
            return (r.status_code == 200,
                    f"ComfyUI alive at {self.server_url}" if r.status_code == 200
                    else f"ComfyUI HTTP {r.status_code}")
        except Exception as e:
            return False, f"Cannot reach ComfyUI at {self.server_url}: {e}"

    def format_prompt_for_backend(self, raw_prompt: str) -> str:
        return raw_prompt.strip()

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        aspect_ratio: str = "16:9",
        output_filename: Optional[str] = None,
        seed: Optional[int] = None,
        cfg_override: Optional[float] = None,
        **kwargs,
    ) -> Path:
        if seed is None:
            seed = random.randint(1, 2**31 - 1)
        self._last_seed = seed

        width, height = model_registry.get_resolution(aspect_ratio)

        # which character LoRAs does this prompt need?
        loras = lora_store.resolve_for_prompt(prompt)
        triggers = [e["trigger"] for e in loras]
        final_prompt = prompt.strip()
        if triggers:
            final_prompt = ", ".join(triggers) + ", " + final_prompt
            log.info(f"char LoRAs: {[e['lora_file'] for e in loras]}")

        rs_steps = rs.get_steps_override()
        steps = rs_steps if rs_steps is not None else self.steps

        wf = self._build_workflow(final_prompt, width, height, seed, steps, loras)

        client_id = str(uuid.uuid4())
        prompt_id = self._submit(wf, client_id)
        history = self._wait(prompt_id, timeout=900)
        return self._download(history, prompt_id, output_filename)

    # ------------------------------------------------------------------ internals
    def _build_workflow(self, prompt, width, height, seed, steps, loras) -> dict:
        wf = json.loads(json.dumps(self._template))

        wf["10"]["inputs"]["ckpt_name"] = self.ckpt_name
        wf["20"]["inputs"]["text"] = prompt
        wf["22"]["inputs"]["guidance"] = self.guidance
        wf["30"]["inputs"]["width"] = int(width)
        wf["30"]["inputs"]["height"] = int(height)
        wf["40"]["inputs"].update(
            seed=int(seed), steps=int(steps), cfg=self.cfg,
            sampler_name=self.sampler, scheduler=self.scheduler,
        )

        # chain LoRA loaders on the model path: UNET(10) -> lora -> ... -> KSampler(40)
        prev = ["10", 0]
        for i, e in enumerate(loras, 1):
            nid = f"70{i}"
            wf[nid] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "model": prev,
                    "lora_name": e["lora_file"],
                    "strength_model": float(e.get("weight", 0.9)),
                },
                "_meta": {"title": f"lora {e['trigger']}"},
            }
            prev = [nid, 0]
        wf["40"]["inputs"]["model"] = prev
        return wf

    def _submit(self, wf, client_id) -> str:
        r = requests.post(f"{self.server_url}/prompt",
                          json={"prompt": wf, "client_id": client_id}, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"ComfyUI rejected prompt (HTTP {r.status_code}): {r.text[:400]}")
        data = r.json()
        if "prompt_id" not in data:
            raise RuntimeError(f"No prompt_id in response: {data}")
        return data["prompt_id"]

    def _wait(self, prompt_id, timeout=1800) -> dict:
        # Tolerant polling: the FIRST flux render cold-loads the ~16GB fp8
        # checkpoint, during which ComfyUI is unresponsive >10s. Use a generous
        # per-request timeout + retry on connection/read timeouts (don't crash).
        start = time.time()
        fails = 0
        while time.time() - start < timeout:
            try:
                r = requests.get(f"{self.server_url}/history/{prompt_id}", timeout=60)
                fails = 0
                if r.status_code == 200:
                    data = r.json()
                    if prompt_id in data:
                        entry = data[prompt_id]
                        st = entry.get("status", {})
                        if st.get("completed") is True:
                            return entry
                        if st.get("status_str") == "error":
                            raise RuntimeError(f"ComfyUI error: {st.get('messages')}")
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                fails += 1
                if fails >= 8:
                    raise RuntimeError(f"Lost ComfyUI for {prompt_id} after 8 poll timeouts")
                time.sleep(5)
                continue
            time.sleep(2)
        raise TimeoutError(f"Timed out after {timeout}s for {prompt_id}")

    def _download(self, history_entry, prompt_id, output_filename) -> Path:
        outputs = history_entry.get("outputs", {})
        info = None
        for _, out in outputs.items():
            if out.get("images"):
                info = out["images"][0]
                break
        if info is None:
            raise RuntimeError("ComfyUI produced no image.")
        url = (f"{self.server_url}/view?filename={info['filename']}"
               f"&subfolder={info.get('subfolder','')}&type={info.get('type','output')}")
        r = requests.get(url, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"Download failed HTTP {r.status_code}")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if output_filename:
            target = OUTPUT_DIR / output_filename
            target.parent.mkdir(parents=True, exist_ok=True)
        else:
            target = OUTPUT_DIR / f"img_fluxlora_{prompt_id[:12]}.png"
        target.write_bytes(r.content)
        log.info(f"saved {target}")
        return target


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from modules import image_backend as ib
    b = ib.get_active_backend()
    print("health:", b.health_check())
