"""End-to-end lesson render harness (dashboard-less), for the FRAMING feature.

Drives the real pipeline on the real book + the active mascot, through to the STILLS
gate — write the lesson, write the scenes (Ollama), voice it (Chatterbox clone), draw a
picture per beat (Qwen-Edit). It stops there, exactly where the dashboard does: the video
stage is human-gated and each animated Wan shot is ~8 min, so an automated run confirms
the shot list, the framing and the pictures, not the final MP4.

What it proves:
  - every beat gets a framing, and neighbours differ (the shot list)
  - the framing REACHES the prompt (the full-body clause is swapped for this shot's)
  - the sequences group the shots like a storyboard
  - nakshu renders at every framing, identity intact (LOOK at the stills it prints)

Run:  venv\\Scripts\\python run_lesson_e2e.py
Env:  LESSON_BOOK (default the Class-1 EVS book), LESSON_TOPIC (default t01 Living/Non-living)
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent.resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Windows console/redirect defaults to cp1252, which cannot encode the ✅/emoji the
# pipeline logs. Force UTF-8 so a progress line never crashes the run.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from modules import lesson_writer as lw
from modules import lesson_pipeline as lp
from modules import mascot
from modules import runtime_settings as rs

BOOK = os.environ.get("LESSON_BOOK", "20260714_091120")
TOPIC = os.environ.get("LESSON_TOPIC", "t01")
MASCOT = os.environ.get("LESSON_MASCOT", "nakshu")


def _p(msg: str) -> None:
    print(f"  {msg}", flush=True)


def main() -> int:
    t0 = time.time()
    print(f"=== LESSON E2E — book {BOOK} / topic {TOPIC} / mascot {MASCOT} ===",
          flush=True)

    # WHO presents. rs.active_mascot resolves every presenter still + the cloned voice.
    rs.set_active_mascot(MASCOT)
    print(f"active mascot: {mascot.active_mascot_name()}", flush=True)
    ok, why = mascot.is_available()
    print(f"mascot available: {ok} — {why}", flush=True)
    if not ok:
        print("ABORT: mascot/backend not ready", flush=True)
        return 1

    # 1) WRITE — the shot list. Fast (Ollama only); the framing is stamped here.
    print("\n--- write_lesson ---", flush=True)
    lesson = lw.write_lesson(BOOK, TOPIC, progress_cb=_p)
    lid = lesson["lesson_id"]
    beats = lesson["beats"]

    print(f"\nlesson {lid}: {len(beats)} beats", flush=True)
    print(f"{'#':>2}  {'kind':<9} {'framing':<12} narration", flush=True)
    for i, b in enumerate(beats):
        print(f"{i+1:>2}  {b['kind']:<9} {b.get('framing','?'):<12} "
              f"{b['narration'][:54]}", flush=True)

    # The storyboard grouping.
    print("\nsequences:", flush=True)
    for n, grp in enumerate(lw.sequences(lesson), start=1):
        shots = ", ".join(str(s + 1) for s in grp["shots"])
        print(f"  Sequence {n}: {grp['title']}  (shots {shots})", flush=True)

    # The recurring objects pinned for consistency (the doll, the puppy).
    objs = lesson.get("objects", [])
    print(f"\nrecurring objects pinned ({len(objs)}):", flush=True)
    for o in objs:
        print(f"  • {o['noun']}: {o['desc'][:70]}", flush=True)

    # Assert the framing invariants before paying for a render.
    fr = [b.get("framing") for b in beats]
    assert all(f in mascot.LESSON_FRAMINGS for f in fr), f"bad framing: {fr}"
    assert all(not (fr[i] == fr[i + 1] and fr[i] in ("medium", "wide"))
               for i in range(len(fr) - 1)), f"workhorse neighbours clash: {fr}"
    print("✅ framings valid + no workhorse neighbours clash", flush=True)

    # 2) PREPARE — scenes (Ollama) + voice (clone) + a still per beat (Qwen-Edit).
    #    The slow part: ~100s a still. gpu_memory owns the residency dance.
    print("\n--- prepare_lesson (scenes + voice + stills) ---", flush=True)
    lesson = lp.prepare_lesson(lesson, progress_cb=_p)

    # 3) VERIFY the framing reached the PROMPT for a sample of the framings.
    print("\n--- prompt check: framing swapped the full-body clause ---", flush=True)
    seen = {}
    for i, b in enumerate(lesson["beats"]):
        f = b.get("framing")
        if f in seen:
            continue
        seen[f] = i
        pr = lp.prompts_for(lid, i)
        pos = pr["image_positive"]
        clause = mascot.FRAMING_CLAUSE.get(f, "")
        has_clause = clause and clause in pos
        # medium/wide legitimately keep "full body visible" replaced too; only report.
        full_body_gone = mascot._FULL_BODY_CLAUSE not in pos
        print(f"  line {i+1:>2} [{f:<12}] clause_in_prompt={bool(has_clause)} "
              f"full_body_clause_gone={full_body_gone}", flush=True)

    # Report.
    d = lp.LESSONS_DIR / lid
    print(f"\n=== DONE in {(time.time()-t0)/60:.1f} min ===", flush=True)
    print(f"lesson dir : {d}", flush=True)
    print(f"stills     : {d / 'stills'}", flush=True)
    print(f"stage      : {lesson.get('stage')}  "
          f"({lesson.get('estimated_seconds')}s of narration)", flush=True)
    print("\nLOOK at each still — identity intact across framings is the real test.",
          flush=True)
    for i, b in enumerate(lesson["beats"]):
        print(f"  still_{i:02d}.png  [{b.get('framing'):<12}] {b.get('mascot_scene','')[:70]}",
              flush=True)

    # A machine-readable summary next to the lesson.
    summary = {
        "lesson_id": lid, "book": BOOK, "topic": TOPIC, "mascot": MASCOT,
        "beats": [{"i": i + 1, "kind": b["kind"], "framing": b.get("framing"),
                   "still": b.get("still"), "scene": b.get("mascot_scene", "")}
                  for i, b in enumerate(lesson["beats"])],
        "sequences": lw.sequences(lesson),
    }
    (d / "e2e_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nsummary: {d / 'e2e_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
