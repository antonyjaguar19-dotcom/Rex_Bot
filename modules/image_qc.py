"""Claw Bot — visual QC: a vision model LOOKS at a rendered still and checks it against the
known defect ledger (02_Agent/IMAGE_FEEDBACK.md), as a CLOSED yes/no checklist.

An earlier Qwen2.5-VL-7B QC pass was removed for being net-negative: asked open-endedly to
"find problems" it invented them, and a false "bad" costs a real re-render. This is a
different task on a stronger model (qwen2.5vl:32b): a FIXED list of yes/no questions, each
tied to a real shipped defect, answered against the actual pixels. The bias is deliberately
toward NOT flagging — a check only fails on an explicit, confident "yes, this is wrong";
anything unsure, unparseable, or errored counts as a pass, so QC can never be the thing that
loops the render.

Not wired into the pipeline yet — this is the engine. `qc_still(png)` returns the verdict;
the caller decides whether to re-roll or just surface it.
"""
import base64
import json
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger("claw_bot.image_qc")

# 7b -> 32b (jeffy, 2026-07-16, round two): the 7b passed reels with a FACE ON A BALL and a
# hallucinated doll-with-a-face — the two worst defects in a lesson — because a weak VLM
# answers "pass" to a subtle yes/no it cannot actually see. The whole point of QC here is to
# CATCH those and auto-re-roll, so it needs the stronger model. (The 32b "earned nothing"
# earlier only because that render happened to be clean.)
QC_MODEL = "qwen2.5vl:32b"
QC_SERVER = "http://127.0.0.1:11434"

# Each check = (key, the yes/no question). Written so that "yes" means BROKEN, and the model
# is told to answer PASS/FAIL — pass = the image is fine on this point. Tied 1:1 to rows in
# IMAGE_FEEDBACK.md. `teaching` context = a lesson still (one mascot child, props, no faces
# on inanimate things); `facts` context drops the teaching-only prop checks.
_CHECKS_TEACHING = [
    ("one_child_human",
     "Is the main character a single HUMAN child, with NO animal ears, paws, whiskers, "
     "tail or fur, and NO second identical twin of her in the frame?"),
    ("warm_face",
     "Is her face visible with a warm, friendly, neutral-or-happy expression — NOT angry, "
     "snarling, crying, blank/featureless, and with normal eyes (no hearts or stars for "
     "eyes)?"),
    ("no_face_on_objects",
     "Look at EVERY non-living object (toy, block, ball, rock, plate, the sun, a cloud). "
     "Does every one of them have a BLANK surface with NO eyes, NO mouth, NO smile and NO "
     "cartoon face of any kind? Answer fail if ANY object has a face or googly eyes on it."),
    ("no_doll_with_a_face",
     "Is the child NOT holding a doll, figure or plush toy that has a face, eyes or hair? "
     "(A plain faceless building block is fine; a doll with a face is a fail.)"),
    ("no_hallucinated_prop",
     "Is she holding ONLY things a teaching scene would sensibly show (a block, a plant, an "
     "apple, or nothing), with NO random extra toy, doll or object that does not belong?"),
    ("hands_busy",
     "Are her arms and hands doing something natural (holding, pointing, gesturing) rather "
     "than spread straight out sideways in a stiff T-pose?"),
    ("fully_dressed",
     "Is the child fully and normally dressed, with NO bare midriff, crop top or exposed "
     "belly?"),
]
_CHECKS_FACTS = [c for c in _CHECKS_TEACHING if c[0] in
                 ("one_child_human", "warm_face", "hands_busy", "fully_dressed")]

_SYS = (
    "You are a strict but fair quality checker for a children's cartoon still. You are given "
    "ONE image and a numbered checklist. Answer ONLY from what you can actually see in the "
    "image. For each item reply \"pass\" if the image is fine on that point and \"fail\" ONLY "
    "if you are confident the image is clearly wrong on that point. If you are unsure, reply "
    "\"pass\" — a false alarm wastes a re-render. Reply as compact JSON: "
    "{\"<key>\": {\"verdict\": \"pass\"|\"fail\", \"note\": \"<=8 words\"}, ...} and nothing "
    "else."
)


def _img_b64(png: Path) -> str:
    return base64.b64encode(Path(png).read_bytes()).decode("ascii")


def _cfg() -> dict:
    return {"model_id": QC_MODEL, "server_url": QC_SERVER, "vision": True, "_id": "vision_qc"}


def checks_for(context: str) -> list:
    return _CHECKS_FACTS if (context or "").strip().lower() == "facts" else _CHECKS_TEACHING


def qc_still(png: Path, context: str = "teaching") -> dict:
    """LOOK at one still and check it against the ledger. Returns
    {ok, fails: [{key, note}], checks: {key: verdict}, raw}. `ok` is False only when a check
    confidently FAILED; a model/parse error returns ok=True (QC never blocks on itself)."""
    png = Path(png)
    if not png.exists() or not png.stat().st_size:
        return {"ok": True, "fails": [], "checks": {}, "raw": "", "error": "no image"}

    checklist = checks_for(context)
    numbered = "\n".join(f"{i+1}. [{key}] {q}" for i, (key, q) in enumerate(checklist))
    from modules.script_generator import _call_llm, _extract_json
    try:
        raw = _call_llm(
            f"Check this cartoon still against the checklist:\n{numbered}",
            _SYS, cfg=_cfg(), images=[_img_b64(png)],
            options={"temperature": 0.1, "top_p": 0.9, "num_predict": 500},
        )
    except Exception as e:
        log.warning(f"visual QC could not run on {png.name}: {e}")
        return {"ok": True, "fails": [], "checks": {}, "raw": "", "error": str(e)}

    try:
        got = _extract_json(raw) or {}
    except Exception as e:              # unparseable reply must not loop a render
        log.warning(f"visual QC reply unparseable on {png.name}: {e}")
        return {"ok": True, "fails": [], "checks": {}, "raw": raw, "error": "unparseable"}
    checks, fails = {}, []
    for key, _q in checklist:
        entry = got.get(key) or {}
        verdict = str(entry.get("verdict", "pass")).strip().lower()
        checks[key] = verdict
        if verdict == "fail":           # ONLY an explicit fail counts
            fails.append({"key": key, "note": str(entry.get("note", ""))[:60]})
    ok = not fails
    (log.info if ok else log.warning)(
        f"visual QC {png.name}: {'clean' if ok else 'FLAGGED ' + ', '.join(f['key'] for f in fails)}")
    return {"ok": ok, "fails": fails, "checks": checks, "raw": raw}


def available() -> tuple:
    """Is the QC vision model pulled and Ollama answering? (ok, why)."""
    try:
        import requests
        r = requests.get(QC_SERVER + "/api/tags", timeout=10)
        r.raise_for_status()
        names = [m.get("name", "") for m in r.json().get("models", [])]
    except Exception as e:
        return False, f"Ollama is not answering ({e})"
    if not any(n == QC_MODEL or n.startswith(QC_MODEL) for n in names):
        return False, f"{QC_MODEL} not pulled — run `ollama pull {QC_MODEL}`"
    return True, "ready"
