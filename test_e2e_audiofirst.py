"""Headless audio-first end-to-end: prompts -> storyboard -> silent clips ->
assemble. Drives the SAME wired core functions the bot/dashboard call, without
Discord. Reuses an already-voiced audio-first script.

Run: venv\Scripts\python test_e2e_audiofirst.py [script_id]
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from modules import prompt_approval as pap
from modules import storyboard_generator as sbg
from modules import assembly as asm
from modules import clip_generator as cg
from modules import runtime_settings as rs

ROOT = Path(__file__).parent.parent
SCRIPTS = ROOT / "04_Outputs" / "scripts"
STORYBOARDS = ROOT / "04_Outputs" / "storyboards"


def log(m):
    print(f"[e2e {time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    sid = sys.argv[1] if len(sys.argv) > 1 else "20260618_084036"
    spath = SCRIPTS / f"script_{sid}.json"
    script = json.loads(spath.read_text(encoding="utf-8"))
    shots = script.get("shots", [])
    assert script.get("_audio_first"), "script is not audio-first!"
    log(f"script {sid}: '{script['title']}' — {len(shots)} shots, "
        f"audio_first={script.get('_audio_first')}, "
        f"master={Path(script.get('_master_audio','')).name}")

    # 1) prompts (image + motion per shot), auto-approve all
    log("STEP 1 — building prompts (LLM)...")
    state = pap.generate_all_prompts(
        script, progress=lambda m, i, n: log(f"  prompt {i}/{n}: {m}"))
    for k in state["prompts"]:
        state["prompts"][k]["approved"] = True
    pap._save(state)
    log(f"  prompts saved + approved ({len(state['prompts'])} shots)")

    # 2) storyboard images
    log("STEP 2 — rendering storyboard images (ComfyUI)...")
    res = sbg.generate_storyboard(
        sid, aspect_ratio="16:9",
        progress_callback=lambda m, i, n: log(f"  sb {i}/{n}: {m}"))
    log(f"  storyboard done: {res}")

    manifest = json.loads((STORYBOARDS / sid / "storyboard.json").read_text("utf-8"))
    image_by_shot = {
        f.get("shot_number"): ROOT / f.get("image_path")
        for f in manifest.get("frames", [])
        if f.get("shot_number") and f.get("image_path")
    }
    log(f"  images: {sorted(image_by_shot.keys())}")

    # 3) silent clips sized to each window
    log("STEP 3 — rendering silent clips (Wan, sized to win_dur)...")
    clip_gen = cg.ClipGenerator(sync_mode=rs.get_effective_sync_mode())
    for shot in shots:
        num = shot.get("shot_number")
        if num not in image_by_shot:
            log(f"  shot {num}: NO IMAGE, skipping"); continue
        approved = pap.get_shot_prompts(sid, num) or {}
        motion = (approved.get("motion_prompt", "").strip()
                  or shot.get("motion_prompt", "").strip()
                  or shot.get("visual_description", "").strip())
        motion += " No speech. No mouth movement."
        seed = approved.get("motion_seed", -1)
        seed_arg = seed if isinstance(seed, int) and seed > 0 else None
        wd = float(shot.get("win_dur") or 0)
        t0 = time.time()
        out = clip_gen.generate_silent_clip(
            shot_id=f"{sid}_shot{num}",
            motion_prompt=motion,
            storyboard_image=image_by_shot[num],
            target_dur=wd,
            output_filename=f"clip_{sid}_shot{num}.mp4",
            seed=seed_arg, beat=(shot.get("beat") or "").lower(),
        )
        log(f"  shot {num}: clip {out.name} (target {wd:.2f}s) in {time.time()-t0:.0f}s")

    # 4) assemble (assemble_final self-dispatches to audio-first)
    log("STEP 4 — assembling finals (self-dispatch -> audio-first)...")
    result = asm.assemble_final(sid, progress_cb=lambda m: log(f"  asm: {m}"))
    log(f"DONE — {result}")


if __name__ == "__main__":
    main()
