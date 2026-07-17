"""USO vs Qwen-Edit on the SAME shots: does the negative prompt actually buy anything?

The question this answers, and nothing else: the default lesson backend (USO) wires
ConditioningZeroOut into its negative input and its generate() takes no negative at all, so
~430 words of NEGATIVE_TEACHING are discarded — including the conditional phantom-apple ban.
Qwen-Edit honours the negative (at full_quality: 20 steps, cfg 2.5). Measured consequence on
the real book: 3 of the 9 shots in lesson 20260716_000517 have an apple in her hand that
their scene never asked for.

So: re-render those exact beats on BOTH backends, same scene, same seed, same references, and
LOOK at them. Nothing is judged here by a model — the pictures are written side by side and a
person reads them. That is the whole doctrine of this pipeline.

    venv\\Scripts\\python run_backend_ab.py
    venv\\Scripts\\python run_backend_ab.py --lesson 20260716_000517 --beats 1,4,6

Writes to 04_Outputs/lessons/<id>/_ab/<backend>_still_NN.png. Touches NOTHING else: the
lesson's own stills, its lesson.json and the live runtime setting are all restored/untouched.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent.resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from modules import cast
from modules import gpu_memory
from modules import lesson_pipeline as lp
from modules import lesson_writer as lw
from modules import mascot
from modules import mascot_library as ml
from modules import runtime_settings as rs

DEFAULT_LESSON = "20260716_000517"
# The three phantom-apple shots. Their scenes name no fruit; the render put an apple in her
# hand anyway. If the negative is worth anything, this is where it shows.
DEFAULT_BEATS = "1,4,6"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lesson", default=DEFAULT_LESSON)
    ap.add_argument("--beats", default=DEFAULT_BEATS, help="0-based beat indexes")
    # This lesson was drawn with nakshu. Re-rendering it under whoever happens to be active
    # would compare two backends AND two children at once, and prove nothing about either.
    ap.add_argument("--mascot", default="nakshu")
    args = ap.parse_args()

    lesson = lw.load_lesson(args.lesson)
    if not lesson:
        print(f"no lesson {args.lesson!r}", flush=True)
        return 1
    idxs = [int(x) for x in args.beats.split(",") if x.strip() != ""]

    out_dir = lp.LESSONS_DIR / args.lesson / "_ab"
    out_dir.mkdir(parents=True, exist_ok=True)

    was = rs.get_lesson_image_backend()
    was_mascot = ml.get_active_id()
    print(f"=== A/B on {args.lesson}, beats {idxs} (0-based) ===", flush=True)
    print(f"live settings: backend={was!r}, mascot={was_mascot!r} — both restored at the end\n",
          flush=True)
    if args.mascot and args.mascot != was_mascot:
        rs.set_active_mascot(args.mascot)

    cast.migrate()
    mid = ml.get_active_id()
    backdrop = lp.setting_for(lesson.get("topic", "") or lesson.get("title", ""))
    print(f"mascot: {mascot.active_mascot_name()}  ·  setting: {backdrop}\n", flush=True)

    for i in idxs:
        b = lesson["beats"][i]
        print(f"--- beat {i}  [{b.get('framing', '?')}]", flush=True)
        print(f"    line : {b.get('narration', '')[:100]}", flush=True)
        print(f"    scene: {b.get('mascot_scene', '')[:160]}", flush=True)
        # Say out loud whether this scene even mentions fruit — the apple ban is CONDITIONAL
        # (only added when the scene names none), so a shot that legitimately holds an apple
        # would prove nothing.
        import re as _re
        names_fruit = bool(_re.search(r"apple|fruit|banana|orange|berry|berries|grape",
                                      b.get("mascot_scene", ""), _re.I))
        print(f"    scene names fruit: {names_fruit}"
              + ("   <- an apple here is NOT a phantom" if names_fruit else
                 "   <- any apple in this picture is a hallucination"), flush=True)

    try:
        for backend in ("uso", "qwen"):
            rs.set_lesson_image_backend(backend)
            label = gpu_memory.FLUX_USO if backend == "uso" else gpu_memory.QWEN_EDIT
            neg = lp.negative_is_sent()
            print(f"\n=== {backend.upper()}  (negative actually sent: {neg}) ===", flush=True)
            gpu_memory.acquire(label)
            try:
                for i in idxs:
                    sp = out_dir / f"{backend}_still_{i:02d}.png"
                    t0 = time.time()
                    # _draw_one is THE draw path — the same one prepare and redraw use, so
                    # this cannot drift from what a real render would do. It writes the
                    # contract/delivered record onto the in-memory lesson; we pass a COPY so
                    # the real lesson.json is never touched by an experiment.
                    got = lp._draw_one(dict(lesson, lesson_id=args.lesson),
                                       i, backdrop, mid, sp,
                                       lambda m: print(f"      {m}", flush=True))
                    ok = "ok" if got else "FAILED"
                    print(f"    beat {i}: {ok} in {time.time()-t0:.0f}s -> {sp.name}",
                          flush=True)
            finally:
                gpu_memory.release(label)
    finally:
        rs.set_lesson_image_backend(was)
        rs.set_active_mascot(was_mascot)
        print(f"\nrestored: backend={was!r}, mascot={was_mascot!r}", flush=True)

    print(f"\nLOOK at them: {out_dir}", flush=True)
    for i in idxs:
        print(f"  uso_still_{i:02d}.png   vs   qwen_still_{i:02d}.png", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
