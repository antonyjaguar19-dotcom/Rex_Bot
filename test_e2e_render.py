"""Full render from an existing audio-first script with approved prompts:
storyboard -> silent clips -> assemble. Verifies the back half of the pipeline.

Run: venv\Scripts\python test_e2e_render.py <script_id>
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from modules import storyboard_generator as sbg
from modules import assembly as asm
from modules import clip_generator as cg
from modules import runtime_settings as rs
from modules import prompt_approval as pap
from modules import gpu_utils

ROOT = Path(__file__).parent.parent
SCRIPTS = ROOT / "04_Outputs" / "scripts"
STORYBOARDS = ROOT / "04_Outputs" / "storyboards"


def log(m): print(f"[render {time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    sid = sys.argv[1]
    script = json.loads((SCRIPTS / f"script_{sid}.json").read_text(encoding="utf-8"))
    shots = script["shots"]
    log(f"{sid}: '{script['title']}' {len(shots)} shots, engine={script.get('_tts_engine')}")

    try: gpu_utils.free_ollama_vram()
    except Exception: pass

    log("STEP A storyboard...")
    sbg.generate_storyboard(sid, aspect_ratio="16:9",
                            progress_callback=lambda m, i, n: log(f"  sb {i}/{n}"))
    manifest = json.loads((STORYBOARDS / sid / "storyboard.json").read_text("utf-8"))
    img = {f["shot_number"]: ROOT / f["image_path"] for f in manifest["frames"]
           if f.get("shot_number") and f.get("image_path")}
    log(f"  images: {sorted(img)}")

    log("STEP B silent clips...")
    clip_gen = cg.ClipGenerator(sync_mode=rs.get_effective_sync_mode())
    fail = []
    for sh in shots:
        n = sh["shot_number"]
        if n not in img: fail.append(n); continue
        ap = pap.get_shot_prompts(sid, n) or {}
        motion = (ap.get("motion_prompt") or sh.get("motion_prompt") or sh.get("visual_description") or "").strip()
        motion += " No speech. No mouth movement."
        seed = ap.get("motion_seed", -1); seed = seed if isinstance(seed, int) and seed > 0 else None
        wd = float(sh.get("win_dur") or 0)
        t0 = time.time()
        try:
            clip_gen.generate_silent_clip(shot_id=f"{sid}_shot{n}", motion_prompt=motion,
                storyboard_image=img[n], target_dur=wd,
                output_filename=f"clip_{sid}_shot{n}.mp4", seed=seed, beat=(sh.get("beat") or "").lower())
            log(f"  shot {n} ok ({wd:.1f}s) {time.time()-t0:.0f}s")
        except Exception as e:
            fail.append(n); log(f"  shot {n} FAIL {type(e).__name__}: {str(e)[:120]}")
    if fail: log(f"ABORT — failed shots {fail}"); return

    log("STEP C assemble...")
    res = asm.assemble_final(sid, progress_cb=lambda m: log(f"  asm {m}"))
    log(f"DONE {res}")


if __name__ == "__main__":
    main()
