"""Full audio-first e2e from a theme: script (VoxCPM2 designed voices) ->
prompts -> storyboard -> silent clips -> assemble (with title/moral cards)."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from modules.audio_first_pipeline import generate_script_audio_first
from modules import prompt_approval as pap
from modules import storyboard_generator as sbg
from modules import assembly as asm
from modules import clip_generator as cg
from modules import runtime_settings as rs
from modules import gpu_utils

ROOT = Path(__file__).parent.parent
STORYBOARDS = ROOT / "04_Outputs" / "storyboards"


def log(m): print(f"[full {time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    theme = " ".join(sys.argv[1:]) or "a curious kitten and a wise old dog explore a garden"
    log("STEP 1 — script + VoxCPM2 designed voices...")
    s = generate_script_audio_first(theme, progress_cb=lambda m: log(f"  {m}"))
    sid = s["_id"]
    log(f"  script {sid}: '{s['title']}' engine={s.get('_tts_engine')} shots={len(s['shots'])}")

    log("STEP 2 — prompts...")
    st = pap.generate_all_prompts(s, progress=lambda m, i, n: None)
    for k in st["prompts"]:
        st["prompts"][k]["approved"] = True
    pap._save(st)

    try: gpu_utils.free_ollama_vram()
    except Exception: pass

    log("STEP 3 — storyboard...")
    sbg.generate_storyboard(sid, aspect_ratio="16:9",
                            progress_callback=lambda m, i, n: log(f"  sb {i}/{n}"))
    manifest = json.loads((STORYBOARDS / sid / "storyboard.json").read_text("utf-8"))
    img = {f["shot_number"]: ROOT / f["image_path"] for f in manifest["frames"]
           if f.get("shot_number") and f.get("image_path")}

    log("STEP 4 — silent clips...")
    clip_gen = cg.ClipGenerator(sync_mode=rs.get_effective_sync_mode())
    fail = []
    for sh in s["shots"]:
        n = sh["shot_number"]
        if n not in img: fail.append(n); continue
        ap = pap.get_shot_prompts(sid, n) or {}
        motion = (ap.get("motion_prompt") or sh.get("motion_prompt") or sh.get("visual_description") or "").strip()
        motion += " No speech. No mouth movement."
        seed = ap.get("motion_seed", -1); seed = seed if isinstance(seed, int) and seed > 0 else None
        t0 = time.time()
        try:
            clip_gen.generate_silent_clip(shot_id=f"{sid}_shot{n}", motion_prompt=motion,
                storyboard_image=img[n], target_dur=float(sh.get("win_dur") or 0),
                output_filename=f"clip_{sid}_shot{n}.mp4", seed=seed, beat=(sh.get("beat") or "").lower())
            log(f"  shot {n} ok {time.time()-t0:.0f}s")
        except Exception as e:
            fail.append(n); log(f"  shot {n} FAIL {type(e).__name__}: {str(e)[:120]}")
    if fail: log(f"ABORT failed shots {fail}"); return

    log("STEP 5 — assemble (+ title/moral cards)...")
    r = asm.assemble_final(sid, progress_cb=lambda m: log(f"  asm {m}"))
    log(f"DONE sid={sid} {r}")


if __name__ == "__main__":
    main()
