"""
Character LoRA trainer — ai-toolkit wrapper.

Builds an ai-toolkit Flux.1-dev LoRA config and runs it in the dedicated
07_Training venv (separate from the bot venv). Base model is loaded QUANTIZED
so a 12B Flux fits a 16GB RTX 5080.

Flow:
    dataset (manual imgs + .txt captions)  ->  config.yaml  ->  ai-toolkit run.py
        ->  07_Training/output/<trigger>/<trigger>.safetensors

Caller (claw_bot / cli) then registers the result via lora.store.register().

NOTE: ai-toolkit + flux1-dev weights must be installed first
(see 07_Training/setup_aitoolkit.sh). train_character() raises a clear error
if the trainer or run.py is missing.
"""

import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

_HERE = Path(__file__).parent.parent.parent.resolve()  # 02_Agent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from dotenv import load_dotenv
from modules.lora import store, captioner

log = logging.getLogger("claw_bot.lora.trainer")

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
TRAIN_ROOT = PROJECT_ROOT / "07_Training"
AITK_DIR = TRAIN_ROOT / "ai-toolkit"
AITK_PY = AITK_DIR / "venv" / "Scripts" / "python.exe"
AITK_RUN = AITK_DIR / "run.py"
OUTPUT_DIR = TRAIN_ROOT / "output"
CONFIG_DIR = TRAIN_ROOT / "configs"
SECRETS = PROJECT_ROOT / "05_Config" / "secrets.env"

BASE_MODEL = "black-forest-labs/FLUX.1-dev"

# ai-toolkit Flux LoRA config. Quantized base + gradient checkpointing => fits 16GB.
# Sampling disabled (sample_every huge) to save VRAM/time during a headless run.
_CONFIG_TMPL = """\
job: extension
config:
  name: "{name}"
  process:
    - type: sd_trainer
      training_folder: "{training_folder}"
      device: cuda:0
      trigger_word: "{trigger}"
      network:
        type: lora
        linear: {rank}
        linear_alpha: {rank}
      save:
        dtype: float16
        save_every: {save_every}
        max_step_saves_to_keep: 2
      datasets:
        - folder_path: "{dataset}"
          caption_ext: txt
          caption_dropout_rate: 0.05
          shuffle_tokens: false
          cache_latents_to_disk: true
          resolution: [768, 1024]
      train:
        batch_size: 1
        steps: {steps}
        gradient_accumulation_steps: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: flowmatch
        optimizer: adamw8bit
        lr: 1e-4
        dtype: bf16
        disable_sampling: true
      model:
        name_or_path: "{base_model}"
        is_flux: true
        quantize: true
        low_vram: true
      sample:
        sampler: flowmatch
        sample_every: 1000000
        skip_first_sample: true
        prompts:
          - "{trigger}"
meta:
  name: "{name}"
  version: '1.0'
"""


def _check_installed():
    if not AITK_RUN.exists() or not AITK_PY.exists():
        raise RuntimeError(
            "ai-toolkit not installed. Run 07_Training/setup_aitoolkit.sh first "
            f"(expected {AITK_RUN} and {AITK_PY})."
        )


def build_config(name_slug: str, trigger: str, dataset: Path,
                 steps: int, rank: int) -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_every = max(250, steps // 4)
    text = _CONFIG_TMPL.format(
        name=trigger,
        trigger=trigger,
        training_folder=str(OUTPUT_DIR).replace("\\", "/"),
        dataset=str(dataset).replace("\\", "/"),
        steps=steps,
        rank=rank,
        save_every=save_every,
        base_model=BASE_MODEL,
    )
    cfg_path = CONFIG_DIR / f"{trigger}.yaml"
    cfg_path.write_text(text, encoding="utf-8")
    log.info(f"wrote training config: {cfg_path}")
    return cfg_path


def _env() -> dict:
    env = dict(os.environ)
    load_dotenv(SECRETS)
    tok = os.environ.get("HF_TOKEN")
    if tok:
        env["HF_TOKEN"] = tok
        env["HUGGING_FACE_HUB_TOKEN"] = tok
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


_STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


def train_character(
    name: str,
    look_token: str,
    *,
    steps: int = 1500,
    rank: int = 16,
    source_script: str = "",
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    register: bool = True,
) -> dict:
    """Caption dataset -> run ai-toolkit -> (optionally) register in store.

    Returns the store manifest entry (if register) or {'lora_path': ...}.
    Raises if dataset empty, trainer missing, or no output produced.
    """
    _check_installed()
    trigger = store.trigger_for(name)
    dataset = store.dataset_dir(name)

    n_imgs = captioner.count_images(dataset)
    if n_imgs == 0:
        raise RuntimeError(
            f"No training images in {dataset}. Drop 15-30 images of '{name}' there first."
        )
    if n_imgs < 10:
        log.warning(f"only {n_imgs} images for '{name}' — LoRA may undertrain (want 15-30).")
    captioner.caption_dataset(dataset, trigger, look_token, overwrite=True)

    cfg = build_config(store.slug(name), trigger, dataset, steps, rank)

    cmd = [str(AITK_PY), str(AITK_RUN), str(cfg)]
    log.info(f"launch train: {' '.join(cmd)} (imgs={n_imgs}, steps={steps}, rank={rank})")
    proc = subprocess.Popen(
        cmd, cwd=str(AITK_DIR), env=_env(),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    last_line = ""
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        last_line = line
        m = _STEP_RE.search(line)
        if m and progress_callback:
            try:
                cur, tot = int(m.group(1)), int(m.group(2))
                if tot >= steps * 0.5:  # ignore unrelated x/y counters
                    progress_callback(f"training {name}", cur, tot)
            except Exception:
                pass
        log.info(f"[aitk] {line}")
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"ai-toolkit exited {ret}. Last: {last_line}")

    # locate final safetensors
    out = OUTPUT_DIR / trigger
    cands = sorted(out.glob(f"{trigger}.safetensors")) or sorted(out.glob("*.safetensors"))
    if not cands:
        raise RuntimeError(f"training finished but no .safetensors under {out}")
    lora_path = cands[-1]
    log.info(f"trained LoRA: {lora_path}")

    if register:
        return store.register(
            name, lora_path,
            base_model=BASE_MODEL, rank=rank, steps=steps,
            source_script=source_script,
        )
    return {"lora_path": str(lora_path), "trigger": trigger}


# --------------------------------------------------------------------------
# CLI:  python modules/lora/trainer.py "Lily" "7yo girl, red coat, brass goggles" [steps]
# (run via the BOT venv; it shells out to the ai-toolkit venv)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if len(sys.argv) < 3:
        print('usage: trainer.py "<Name>" "<look token>" [steps] [rank]')
        sys.exit(2)
    nm, tok = sys.argv[1], sys.argv[2]
    st = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
    rk = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    def _cb(msg, c, t):
        print(f"  {msg}: {c}/{t}")
    entry = train_character(nm, tok, steps=st, rank=rk, progress_callback=_cb)
    print("DONE:", entry)
