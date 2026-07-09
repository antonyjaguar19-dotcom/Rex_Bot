"""Full kids (audio-first) e2e from a THEME: write+voice story -> prompts ->
storyboard -> silent clips (Wan, sized to voiced windows) -> assemble finals.

Drives the SAME wired core functions the bot/dashboard call, no Discord.

Run: venv\Scripts\python run_kids_e2e.py "theme" [style]
     venv\Scripts\python run_kids_e2e.py --reuse <script_id>   # skip writing
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from modules import audio_first_pipeline as afp
from modules import prompt_approval as pap
from modules import storyboard_generator as sbg
from modules import assembly as asm
from modules import clip_generator as cg
from modules import runtime_settings as rs

ROOT = Path(__file__).parent.parent
SCRIPTS = ROOT / "04_Outputs" / "scripts"
STORYBOARDS = ROOT / "04_Outputs" / "storyboards"


def log(m):
    print(f"[kids-e2e {time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    args = sys.argv[1:]
    style = None
    if args and args[0] == "--reuse":
        sid = args[1]
        script = json.loads((SCRIPTS / f"script_{sid}.json").read_text("utf-8"))
        log(f"reusing script {sid}: '{script['title']}'")
    else:
        theme = args[0] if args else "a shy firefly who is afraid of the dark"
        style = args[1] if len(args) > 1 else None
        log(f"STEP 0 — writing + voicing story (theme: {theme!r}, style={style})...")
        t0 = time.time()
        script = afp.generate_script_audio_first(
            theme, style_override=style,
            progress_cb=lambda m: log(f"  gen: {m}"))
        sid = script["_id"]
        log(f"  story '{script['title']}' — {len(script['shots'])} shots, "
            f"engine={script.get('_tts_engine')} in {time.time()-t0:.0f}s")

    shots = script.get("shots", [])
    assert script.get("_audio_first"), "script is not audio-first!"

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

    # 3) silent clips sized to each voiced window
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
    log(f"DONE — script {sid} — {result}")


if __name__ == "__main__":
    main()
