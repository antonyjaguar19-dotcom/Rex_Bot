# IMAGE_FEEDBACK.md — every image defect Jeffy has flagged, and how the bot is stopped from repeating it

> Living ledger. Every time a rendered image is wrong, add a row here: **symptom → root
> cause → the fix that ENFORCES it (code, not just a prompt) → status**. The doctrine of this
> project is *the prompt is the mechanism, the CHECK is the feature* — so each entry names the
> guard/negative/lock that makes the mistake impossible, not just the wording that discourages
> it. Read this before touching `mascot.py`, `lesson_writer.py`, `lesson_objects.py`, or any
> prompt that renders a person/prop/scene.

Enforcement lives in:
- `mascot.py` — `NEGATIVE_PRESENTER` / `NEGATIVE_TEACHING` (bans), `_TEACHING_SYS` /
  `STYLE_TEACHING` (the teaching prompt), and the guard chain `clean_scene_for_the_mascot`
  (`never_empty_handed`, `warm_face`, `one_focus`, `props_fit_in_a_hand`, `toys_look_like_toys`,
  `alive_looks_alive`, `other_people_are_other_people`, …).
- `lesson_objects.py` — recurring props: description-lock + pin-once reference.
- `lesson_writer.py` — `_recurring_objects` / `_canonical_desc` (forces a prop's canonical look),
  `_framing_for` / `sequences` (shot framing).
- `shot_framing.py` — crop ladder that actually delivers the framing.

Status key: **CODE** = enforced by a guard/negative/lock (a test pins it) · **PROMPT** = steered
by wording only · **OPEN** = not yet fixed.

---

## Character identity (the mascot must stay ONE recognisable child)

| # | Symptom (what shipped) | Root cause | Fix / enforcement | Status |
|---|---|---|---|---|
| 1 | **The T-pose** — arms spread, hands doing nothing (four separate routes to it) | The scene gave her hands nothing to hold → arms default to spread | `never_empty_handed()` gives idle hands an asymmetric job | CODE |
| 2 | **Mouse ears** on the girl | `STYLE_PRESENTER` said "cub / paws" (jaguar-cub leftovers) → model reasons back to the creature paws belong to | Build PINNED to the reference: same face, same species; only clothes/props change | CODE |
| 3 | **A completely different child** in some shots | A head-to-toe costume + a cap over her hair — face and hair are all the artist has to recognise her by | Ordinary everyday clothes only; NEVER a cap/hood/helmet or anything over her hair; never "dressed as" | CODE (`_TEACHING_SYS` + negatives) |
| 4 | **Snarling / angry** six-year-old | Scene named no expression → model took the mood from the words | `warm_face()` — a face is never left to the renderer | CODE |
| 5 | **Heart / star eyes** (emoji symbols for eyes) | Decorative eye symbols | Negative bans symbols where an eye should be; guard strips them | CODE |
| 6 | **Crop top / bare midriff** on a child ("belly expanding comically") | Literal reading of a growth line | No bare skin / midriff negative | CODE |
| 7 | **Bobblehead** — head bigger than the torso | "big head, small body" language | Banned in negatives | CODE |
| 8 | **Box over her face** (occlusion) — "box of chicks" put the box on her face | Held object drawn in front of the face | Face-occlusion guard in `_TEACHING_SYS` + `NEGATIVE_TEACHING` | CODE |

## Living vs non-living props (the lesson's whole point)

| # | Symptom | Root cause | Fix / enforcement | Status |
|---|---|---|---|---|
| 9 | **A grinning Earth / smiling sun / happy cloud / face on a rock** — in a lesson about what is ALIVE | Drawn with the FACTS prompt (spectacle) and faces put on objects | `teaching=True`: the picture shows what the LINE says; nothing inanimate gets a face | CODE |
| 10 | **The doll came back a LIVING CHILD** with the mascot's own face, dangling by the wrist | A doll is HUMAN-SHAPED = exactly the reference's shape, so it captured the mascot | (superseded by #11) | CODE |
| 11 | **The doll looked like a horror-film doll** (stitched face, button eyes) — *jeffy, 2026-07-16* | A rag doll with a face is creepy; a faceless human-shaped doll reads as a voodoo doll. The shape is the problem | **The non-living toy is now a FACELESS BUILDING BLOCK** (non-humanoid → no face to turn creepy, no human shape to copy). `mascot.FACELESS_TOY_DESC`, `_DOLL_HEAD/_TAIL`, `_canonical_desc`. Narration may say "doll"; the picture is a block | CODE |
| 12 | **The puppy was a different toy/breed each shot**; doll+puppy → **TWO PLUSH TOY DOGS** | No breed lock; the one contrast the line rests on collapsed | Puppy locked to **`mascot.LABRADOR_PUPPY_DESC`** (golden Labrador) via `_canonical_desc`; `alive_looks_alive` marks the live animal ALIVE vs the lifeless toy — *jeffy, 2026-07-16* | CODE |
| 13 | **Two puppies** (a puppy cloned into a spare hand) | Qwen fills an empty hand by copying a subject | Negative: "more than one puppy, cloned pet, extra animals" | CODE |
| 14 | **Doll + puppy FUSED** into one chimera (puppy head on a cloth body) | Two similar objects melted together | Negative: "merged creature, doll fused with an animal, chimera" | CODE |
| 15 | **A recurring prop drifted** (plush rabbit → peg doll → pink bear across shots) | No prop identity | `lesson_objects.py`: description-lock (free) + pin-once reference (`objects/<key>.png`) reused on every shot that names it | CODE |

## Framing & look (a lesson is a shot list, not thirteen identical frames)

| # | Symptom | Root cause | Fix / enforcement | Status |
|---|---|---|---|---|
| 16 | **Every shot the same frame** — full-body, camera-facing, thirteen times | No per-shot framing | `lesson_writer._framing_for` (intro→establishing, teach→medium, example→wide, check→closeup…) + `shot_framing.py` crop ladder (Qwen-Edit ignores the framing clause, so the still is cropped) | CODE |
| 17 | **Backdrop colour changed every shot** (blue/purple/beige/grey) — watched like four different videos | `STYLE_PRESENTER` only said "solid color background" | `BACKDROPS[hash(lesson_id)]` — one stable colour per lesson | CODE |
| 18 | **A four-action montage crammed into one still** | Scene named a sequence ("then… finally…") | `one_focus()` / one-moment rule keeps a single action | CODE |
| 19 | **A second person came back as a TWIN** of the mascot | One identity reference copied onto every human | `other_people_are_other_people()` — a person we have no reference for is kept OUT; a person we CAN draw (cast.py) gets their own ref | CODE |

---

## The next layer — visual QC (proposed, not built)

Every fix above is a guard applied BEFORE the render, from a known failure. It cannot catch a
*new* defect in a *specific* image. The proposal Jeffy raised (2026-07-16): after a still
renders, have a vision model LOOK at it and check it against this ledger (face present and warm?
hands busy? no face on an inanimate object? the block is faceless? the puppy is a Labrador?),
and re-roll or flag when it fails.

History to respect: an earlier Qwen2.5-VL-7B QC pass was removed for being net-negative (it
hallucinated defects that were not there — a false "bad" costs a real re-render). So this needs
a **bigger / stronger vision model** than 7B to be trustworthy, which is a model download + a
VRAM decision on the 16 GB card (a 32B quantized VLM is tight; 72B will not fit locally). This
is tracked as an OPEN decision — see the plan file and the session notes.
