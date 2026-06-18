"""Resume audio-first e2e from STEP 3: silent clips -> assemble. Reuses the
already-rendered storyboards + approved prompts (no re-render).

Run: venv\Scripts\python test_e2e_clips.py [script_id]
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from modules import prompt_approval as pap
from modules import assembly as asm
from modules import clip_generator as cg
from modules import runtime_settings as rs
from modules import gpu_utils

ROOT = Path(__file__).parent.parent
SCRIPTS = ROOT / "04_Outputs" / "scripts"
STORYBOARDS = ROOT / "04_Outputs" / "storyboards"


def log(m):
    print(f"[clips {time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    sid = sys.argv[1] if len(sys.argv) > 1 else "20260618_084036"
    script = json.loads((SCRIPTS / f"script_{sid}.json").read_text(encoding="utf-8"))
    shots = script.get("shots", [])
    log(f"script {sid}: '{script['title']}' — {len(shots)} shots")

    manifest = json.loads((STORYBOARDS / sid / "storyboard.json").read_text("utf-8"))
    image_by_shot = {
        f.get("shot_number"): ROOT / f.get("image_path")
        for f in manifest.get("frames", [])
        if f.get("shot_number") and f.get("image_path")
    }
    log(f"images on disk: {sorted(image_by_shot.keys())}")

    try:
        gpu_utils.free_ollama_vram()
    except Exception:
        pass

    log("STEP 3 — silent clips (Wan, sized to win_dur)...")
    clip_gen = cg.ClipGenerator(sync_mode=rs.get_effective_sync_mode())
    done, failed = [], []
    for shot in shots:
        num = shot.get("shot_number")
        if num not in image_by_shot:
            log(f"  shot {num}: NO IMAGE"); failed.append(num); continue
        approved = pap.get_shot_prompts(sid, num) or {}
        motion = (approved.get("motion_prompt", "").strip()
                  or shot.get("motion_prompt", "").strip()
                  or shot.get("visual_description", "").strip())
        motion += " No speech. No mouth movement."
        seed = approved.get("motion_seed", -1)
        seed_arg = seed if isinstance(seed, int) and seed > 0 else None
        wd = float(shot.get("win_dur") or 0)
        t0 = time.time()
        try:
            out = clip_gen.generate_silent_clip(
                shot_id=f"{sid}_shot{num}", motion_prompt=motion,
                storyboard_image=image_by_shot[num], target_dur=wd,
                output_filename=f"clip_{sid}_shot{num}.mp4",
                seed=seed_arg, beat=(shot.get("beat") or "").lower(),
            )
            done.append(num)
            log(f"  shot {num}: {out.name} (target {wd:.2f}s) in {time.time()-t0:.0f}s")
        except Exception as e:
            failed.append(num)
            log(f"  shot {num}: FAILED {type(e).__name__}: {str(e)[:160]}")

    log(f"clips done={done} failed={failed}")
    if failed:
        log("ABORT assemble — not all clips rendered."); return

    log("STEP 4 — assemble (self-dispatch -> audio-first)...")
    result = asm.assemble_final(sid, progress_cb=lambda m: log(f"  asm: {m}"))
    log(f"DONE — {result}")


if __name__ == "__main__":
    main()
