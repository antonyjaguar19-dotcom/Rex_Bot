"""Full facts-shorts e2e from a TOPIC: write facts -> kokoro narration ->
mood backdrops -> 9x16 Ken Burns + big centered text -> final mp4.

Run: venv\Scripts\python run_facts_e2e.py "topic" [n_facts]
     venv\Scripts\python run_facts_e2e.py --reuse <facts_id>
"""
import sys
import time
import traceback
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def log(m):
    print(f"[facts-e2e {time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    from modules import facts_writer as fw
    from modules import facts_pipeline as fp

    args = sys.argv[1:]
    animate = "--animate" in args
    args = [a for a in args if a != "--animate"]
    if args and args[0] == "--reuse":
        story = fw.load_facts(args[1])
        if not story:
            log(f"facts_id {args[1]} not found"); sys.exit(1)
        log(f"reusing '{story['title']}' ({args[1]})")
    else:
        topic = args[0] if args else "the deep ocean"
        n = int(args[1]) if len(args) > 1 else 6
        t0 = time.time()
        story = fw.generate_facts_short(topic, n_facts=n, progress_cb=lambda m: log(f"gen: {m}"))
        log(f"story '{story['title']}' — {len(story['beats'])} beats in {time.time()-t0:.0f}s")

    t0 = time.time()
    log(f"render mode: {'WAN animate' if animate else 'Ken Burns'}")
    out = fp.render_facts(story, progress_cb=lambda m: log(m), animate=animate)
    vid = Path(out.get("9x16") or "")
    size = vid.stat().st_size / 1e6 if vid.exists() else 0
    log(f"DONE in {(time.time()-t0)/60:.1f} min — {vid} ({size:.0f} MB, {out.get('duration',0):.0f}s)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        log(f"FAILED: {e}")
        sys.exit(1)
