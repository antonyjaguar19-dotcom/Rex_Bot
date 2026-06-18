"""Generate a fresh audio-first script + prompts and audit character
consistency: does every shot a character appears in carry the SAME locked
visual token, and does gender stay consistent with the voice?"""
import json, sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from modules.audio_first_pipeline import generate_script_audio_first
from modules import prompt_approval as pap
from modules import voice_casting as vc


def log(m): print(m, flush=True)


def main():
    theme = " ".join(sys.argv[1:]) or "a brave little fox and a wise old owl cross a dark forest"
    log(f"== GENERATING SCRIPT (audio-first, VoxCPM clone) ==")
    s = generate_script_audio_first(theme, progress_cb=lambda m: log(f"  · {m}"))
    sid = s["_id"]
    log(f"\nTITLE: {s['title']}  id={sid}  engine={s.get('_tts_engine')}  shots={len(s['shots'])}")

    log("\n== CAST ==")
    cast = {}
    for c in s.get("characters", []):
        nm = c.get("name"); tok = c.get("locked_visual_token", "")
        cast[nm] = tok
        v = vc.resolve_voice(s, nm)
        log(f"  {nm} ({c.get('type')}) voice={v}")
        log(f"    appearance: {c.get('appearance')}")
        log(f"    locked:     {tok}")

    log("\n== PER-SHOT (speaker / voice / narration) ==")
    for sh in s["shots"]:
        v = vc.resolve_voice(s, sh.get("speaker"))
        log(f"  [{sh['shot_number']}] {sh['beat']}/{sh['shot_type']} spk={sh['speaker']:<10} voice={v} :: {sh['narration'][:55]}")

    log("\n== GENERATING PROMPTS ==")
    state = pap.generate_all_prompts(s, progress=lambda m, i, n: None)
    for k in state["prompts"]:
        state["prompts"][k]["approved"] = True
    pap._save(state)

    log("\n== CHARACTER-CONSISTENCY AUDIT (image prompts) ==")
    for k in sorted(state["prompts"], key=lambda x: int(x)):
        p = state["prompts"][k]
        img = p.get("image_prompt", "")
        present = [nm for nm, tok in cast.items()
                   if nm and nm.lower() in img.lower()]
        # token consistency: for each named char in the prompt, is its locked
        # token (or a strong fragment) present verbatim?
        flags = []
        for nm in present:
            frag = cast[nm].split(",")[0][:30].strip().lower()
            if frag and frag not in img.lower():
                flags.append(f"{nm}:token-missing")
        # gender pronoun sanity (word-boundary so 'female' != 'male')
        for nm in present:
            tok = cast[nm].lower()
            is_male = bool(re.search(r'\bmale\b', tok))
            is_female = bool(re.search(r'\bfemale\b', tok))
            if is_male and re.search(r'\b(she|her|hers)\b', img.lower()):
                flags.append(f"{nm}:she-vs-male")
            if is_female and re.search(r'\b(he|his|him)\b', img.lower()):
                flags.append(f"{nm}:he-vs-female")
        mark = "  ⚠ " + ",".join(flags) if flags else "  ok"
        log(f"  shot {k}: chars={present}{mark}")
    log(f"\nDONE. script_id={sid}")


if __name__ == "__main__":
    main()
