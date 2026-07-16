---
name: image-supervisor
description: >
  Supervise the lesson/facts image pipeline: LOOK at every generated still, find every
  visual flaw, name each with its root cause and a concrete fix, apply the fix (a guard in
  code, preferred, or a re-roll), run the test suite, re-render, and LOOP until there are
  zero defects across all images. Use when the user says "check the images", "find the
  flaws", "supervise/QC the render", "fix the bot until the images are perfect", "run in a
  loop until 0 mistakes", or after any lesson/facts render that needs review.
---

# Image Supervisor — find every flaw, fix the bot, loop to zero

You are the quality gate for the AI animation pipeline in `E:\Rexjaw_VFX\02_Agent`. A
vision model **cannot** be trusted to catch subtle defects (proven: qwen2.5vl 7b AND 32b
both passed a face-on-a-ball and a hallucinated doll — see `IMAGE_FEEDBACK.md`). **YOU are
the QC** — you read the actual pixels with the Read tool and judge them against the narration
line and the known defect ledger.

## The loop (repeat until 0 defects)

1. **Render / locate the stills.** Lesson stills live at
   `04_Outputs/lessons/<id>/stills/still_NN.png` (0-indexed). To generate: run
   `prepare_lesson(lw.load_lesson(id), progress_cb, redo=True)` from `modules.lesson_pipeline`
   (rewrites scenes + draws + conforms props). One still: `redraw_still(id, i, cb)`. Needs
   ComfyUI up on `127.0.0.1:8188`. Full quality ≈ 100s/still; run in the background.
2. **Build a contact sheet and LOOK at every shot** — Read the sheet, then Read individual
   full-res stills for anything suspect. Never judge from the 104px thumbnail; open the image.
3. **For EACH shot, check it against BOTH:**
   - its **narration line + the lesson topic** — does the picture show what the line says,
     simply and on-topic? (a "running = moving" line must NOT show a static toy);
   - the **defect checklist** below (every row is a real shipped bug from `IMAGE_FEEDBACK.md`).
4. **Report each flaw as one line:** `shot N: <symptom> — cause: <why> — fix: <the guard or
   re-roll>`. No praise, no hedging.
5. **Apply the fix.** GUARD-FIRST (systematic, permanent) over re-roll (stochastic, one-off):
   - a guard change is an edit to `modules/mascot.py` (a negative in `NEGATIVE_TEACHING` /
     `NEGATIVE_PRESENTER`, a rule in `_TEACHING_SYS`, or a guard fn in the
     `clean_scene_for_the_mascot` chain), `modules/lesson_writer.py` (writer rules /
     `_canonical_desc`), or `modules/lesson_pipeline.py` (conform / QC pass);
   - a re-roll is `redraw_still(id, i)` (a NEW seed — stochastic, only when the defect is a
     one-off render glitch a guard can't stop, e.g. a two-person scale roll).
6. **Run the tests:** `venv\Scripts\python -m pytest tests -q`. Fix any test the guard broke
   (update the assertion to the new intended behaviour — do not weaken the guard to pass a
   stale test). Never commit red.
7. **Re-render the affected shots** (or the whole lesson if the fix was a writer/scene rule).
8. **Re-inspect.** If any defect remains, GOTO 3. Stop only when **every** shot is clean AND
   on-topic AND the suite is green.

## The defect checklist (LOOK for each — every one has shipped)

Character identity
- Species/anatomy blend: girl with **dog ears / snout / tail**, dog with **human hands**
  (from feeding the mascot ref + an animal ref together). Fix: anti-blend negatives.
- **Hair drift**: a running/motion pose loosening a tied-up top-knot into a **ponytail**.
  Fix: "same hair and hairstyle" in the identity clause + ban ponytail/loose-hair.
- **Hands**: fused / melted / dissolved / extra / missing fingers. Fix: hand-artifact negatives.
- **Wrong child** entirely: a costume or a cap over her hair. Fix: ordinary clothes, nothing
  over the hair; face+hair are all the artist has.
- **Angry/snarling/blank** face, **heart/star eyes**, **bare midriff/crop top**. Fix: warm_face,
  no-symbol-eyes, no-bare-skin guards.

Props / living-vs-non-living (lesson pedagogy)
- **Face on an inanimate object** (googly eyes / smile on a ball, block, sun, earth). THE
  cardinal lesson sin. Fix: `no_face_on_objects` negatives; nothing inanimate gets a face.
- **Doll with a face / creepy doll / living-child doll**. Fix: the non-living toy is a
  **faceless building block** (`FACELESS_TOY_DESC`) — non-humanoid, no face to turn creepy.
- **Prop drifts** shot to shot (a different toy each time). Fix: `lesson_objects` pin +
  desc-lock + `conform_props` (Kontext edit, keeps angle/scale).
- **Toy version of a LIVING example** (a plush horse for "real horses run"). Fix: living
  animal is REAL and BESIDE her, never held, never a toy.
- **Off-topic / phantom prop**: an object the LINE never named, often mangled (disembodied
  doll head, colour smear). Fix: `no_phantom_object` (prop-less scenes say hands empty).

Composition / scale / style
- **Two-person scale**: mother and child the **same size**. Fix: explicit "adult ~2× the
  child's height, child's head reaches the chest" clause. (Weakest spot — may need a re-roll.)
- **Style mismatch**: stylised chibi child next to a **realistic** adult = "mascot beside a
  human". Fix: "both the same 3d pixar cartoon style".
- **Extra person** (a twin, a third child). Fix: `other_people_are_other_people` + extra-char
  negatives (imperfect — re-roll if it slips).
- **Mannequin**: the same stiff symmetrical pose every shot. Fix: varied natural `_BUSY_POSES`.
- **Busy / confusing frame**: too many elements for a child to read. Fix: the SIMPLICITY rule
  in `_TEACHING_SYS` — one clear idea per shot, only what the line names.
- **Mascot missing** from a shot. Fix: "the mascot child is ALWAYS the main subject".

## Doctrine (learned the hard way — obey)

- **The prompt is the mechanism; the CHECK is the feature.** A model told "don't do X" still
  does X. The guard that ENFORCES it (a negative, a scene rewrite, a pin reference) is the fix.
- **Guard text that a detector scans must NOT contain the detector's trigger words.** The
  self-sabotage bug: `no_phantom_object` said "holding no **doll**", `lesson_objects.detect`
  matched "doll", fed the block reference in, and a MOVING shot grew an off-topic block. When
  you write a clause, ask: does anything downstream pattern-match its words?
- **Guard-first, re-roll-last.** A guard fixes the whole class forever; a re-roll fixes one
  image once and the next render may regress differently.
- **Prefer simple, on-topic frames.** The picture is for a child who cannot read; one clear
  idea beats a busy contrast.
- **Two-person shots and rare benign faceless props (box, tablet) are model limits** — guard
  what you can, and accept a gate re-roll for the residual. "Zero defects" means zero HARMFUL
  defects (faces on objects, blends, off-topic props, wrong scale) — a benign faceless box is
  not a harmful defect.
- **Never commit red.** Every guard change updates or adds a test in `tests/` (real ffmpeg /
  real string checks, no mocks that prove nothing). Restart the bot after `.py` edits.

## Running it as an autonomous loop

When asked to "loop until 0 mistakes": drive steps 1–8 yourself each iteration, in the
background where renders are long, reporting each iteration's flaws + fixes + test result.
Do NOT stop at "looks mostly fine" — continue until a full-lesson fresh render (redo=True,
not just targeted re-rolls) is clean by your own eyes AND the suite is green. Commit each
round's guards with a message naming the defect it kills.
