"""QA driver: per kids style -> generate story (forced style) -> render first-frame
storyboard stills. Caps each story to first N shots for fast image review. Writes a
JSON summary + lists image paths for inspection. Story text kept full for eval."""
import sys, json, time, traceback
from pathlib import Path

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from modules import script_generator as sg
from modules import storyboard_generator as sb
from modules import runtime_settings as rs

SCRATCH = Path(r"C:/Users/anton/AppData/Local/Temp/claude/E--Rexjaw-VFX-02-Agent/622b374f-d45f-4c9c-b4c3-af3d8b64cac2/scratchpad")
SCRATCH.mkdir(parents=True, exist_ok=True)
SCRIPTS_DIR = HERE.parent / "04_Outputs" / "scripts"
SB_DIR = HERE.parent / "04_Outputs" / "storyboards"
CAP_SHOTS = 3   # render only first N shots per style (speed)

import os
_ALL_JOBS = [
    ("pixar",          "a shy little robot learns to make friends on its first day at school"),
    ("cartoon_saloon", "a young girl and a magical sea turtle protect their island's old lighthouse"),
    ("claymation",     "two clumsy mice secretly bake a birthday cake for their grandmother"),
    ("stickman",       "a stick-figure boy discovers why taking turns makes the playground more fun"),
]
_only = set((os.environ.get("QA_ONLY") or "").split(",")) - {""}
JOBS = [j for j in _ALL_JOBS if not _only or j[0] in _only]


def run():
    rs.set_reference_mode_enabled(False)  # no cast/env sheets — judge raw first frames
    summary = []
    for style, theme in JOBS:
        rec = {"style": style, "theme": theme}
        try:
            print(f"\n===== {style.upper()} =====", flush=True)
            print(f"[{style}] writing story...", flush=True)
            t0 = time.time()
            script = sg.generate_script(theme, style_override=style)
            sid = script["_id"]
            rec["script_id"] = sid
            rec["title"] = script.get("title")
            rec["style_field"] = script.get("style")
            rec["moral"] = script.get("moral")
            rec["characters"] = [
                {"name": c.get("name"), "appearance": c.get("appearance"),
                 "locked_visual_token": c.get("locked_visual_token")}
                for c in script.get("characters", [])
            ]
            shots = script.get("shots", [])
            rec["n_shots_full"] = len(shots)
            rec["shots"] = [
                {"n": s.get("shot_number"), "beat": s.get("beat"),
                 "shot_type": s.get("shot_type"), "speaker": s.get("speaker"),
                 "narration": s.get("narration"),
                 "first_frame_prompt": s.get("first_frame_prompt")}
                for s in shots
            ]
            print(f"[{style}] story '{script.get('title')}' "
                  f"{script.get('_stage1_word_count')}w, {len(shots)} shots "
                  f"({time.time()-t0:.0f}s)", flush=True)

            # Cap shots on disk for fast render
            capped = dict(script)
            capped["shots"] = shots[:CAP_SHOTS]
            (SCRIPTS_DIR / f"script_{sid}.json").write_text(
                json.dumps(capped, ensure_ascii=False, indent=2), encoding="utf-8")

            print(f"[{style}] rendering {len(capped['shots'])} first-frame stills...", flush=True)
            t1 = time.time()
            res = sb.generate_storyboard(sid, aspect_ratio="16:9")
            imgs = []
            for fr in res.frames:
                p = getattr(fr, "image_path", None) or getattr(fr, "path", None)
                if p:
                    imgs.append(str(p))
            rec["images"] = imgs
            rec["render_sec"] = round(time.time() - t1, 1)
            print(f"[{style}] rendered {len(imgs)} stills ({rec['render_sec']}s)", flush=True)
        except Exception as e:
            rec["error"] = f"{e}\n{traceback.format_exc()}"
            print(f"[{style}] ERROR: {e}", flush=True)
        summary.append(rec)
        (SCRATCH / "qa_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    rs.set_reference_mode_enabled(True)
    print("\n===== DONE =====", flush=True)
    print("summary:", SCRATCH / "qa_summary.json", flush=True)


if __name__ == "__main__":
    run()
