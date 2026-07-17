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
| 17 | **Backdrop colour changed every shot** (blue/purple/beige/grey) — watched like four different videos | `STYLE_PRESENTER` only said "solid color background" | `lesson_pipeline.SETTINGS` + `setting_for(topic)` — one real place per lesson, keyword-matched. **Keyed on the TOPIC, not the lesson_id**: two lessons on the same topic share a setting, and that is fine. (This row used to say `BACKDROPS[hash(lesson_id)]`. No such symbol has ever existed — see the doc-drift note at the bottom.) | CODE |
| 18 | **A four-action montage crammed into one still** | Scene named a sequence ("then… finally…") | `one_focus()` / one-moment rule keeps a single action | CODE |
| 19 | **A second person came back as a TWIN** of the mascot | One identity reference copied onto every human | `other_people_are_other_people()` — a person we have no reference for is kept OUT; a person we CAN draw (cast.py) gets their own ref | CODE |

| 20 | **THE PHANTOM APPLE** — a small red apple in her free hand on shots whose scene names no fruit. **3 of the 9 shots** in `20260716_000517` (still_01/04/06); still_06 had an injected pose ("her other hand resting on her hip") and Qwen filled the resting hand anyway | **NOT the model's prior, and NOT the negative. `STYLE_TEACHING` said it.** The prop-scale clause — *"the props are small toys a child can hold — **about the size of an apple**, small enough to sit in her palm, **clearly held in her hand**"* — was appended to EVERY teaching prompt, including shots with no props. It asserts, in the present tense, that props ARE held in her hand; the model believes the style over the scene. And the size metaphor is drawn as an object. **Then it got worse**: `no_phantom_object` wrote "no fruit and no apple" to say her hands were empty, `_HELD_PROP` matched those very words, and `build_presenter_prompt` concluded the shot HELD something — *the guard against phantom props was manufacturing the phantom prop* | `mascot._PROP_SCALE_CLAUSE`, appended **only** when `mascot.holds_a_prop(scene)` — one **negation-aware** detector shared with `no_phantom_object`, so the two can never disagree. The size names no object. `no_phantom_object` names nothing at all. Pinned by `test_lesson_prompts.py` + `test_mascot_pose.py` | **CODE — measured fixed** |
| 21 | **THE NEGATIVE IS NOT SENT.** The default lesson backend is USO, which wires `ConditioningZeroOut` into the sampler's negative input and whose `generate()` takes no negative at all. All ~430 words of `NEGATIVE_TEACHING` are DISCARDED — the bans on faces-on-objects, emoji eyes, the bobblehead, the crop top, the box-over-the-face and the phantom apple have **never run on the backend that draws lessons**. This is why rows marked CODE keep shipping defects | `rs.get_lesson_image_backend()` defaults `"uso"`; only Qwen-Edit honours a negative, and only off the Lightning path (cfg 1.0 = no CFG = no negative) | `lesson_pipeline.negative_is_sent()` records it per shot; the gate's 🔧 Prompts panel labels the negative **NOT SENT** instead of printing 430 words it implies were enforced. Pinned by `test_uso_reports_that_the_negative_is_not_sent`. **The backend choice itself is still open** — USO is the default for a measured reason (UNO multi-subject: mascot + mother + prop each hold their own reference; it is the twin fix) | CODE (surfaced) / OPEN (the choice) |
| 22 | **A face on the ball** — `20260716_000517` still_00, whose scene says, in the POSITIVE, "a small bright blue rubber play ball … **with no face**" | Saying it in the scene text was not enough, and the negative that also bans it was discarded (#21) | none yet — the strongest lever available is `_canonical_desc` forcing the faceless block, which this shot already had | OPEN |

---

## The phantom apple, solved by ablation (2026-07-17) — read this before writing any guard

Beat 4 of `20260716_000517`. nakshu, qwen-edit, **one seed, one variable at a time**, rendered
and LOOKED AT each time. The scene says: *"running in a playground, a horse running beside her,
her hands empty and open, not holding anything, no fruit and no apple, both palms relaxed and
clearly empty."*

| what changed | what came back |
|---|---|
| nothing (as shipped) | 🍎 an apple in her hand |
| "no fruit and no apple" deleted from the scene | 🍎 an apple |
| scene rewritten: "both hands clasped behind her back" | 🍎 an apple, hands in front |
| USO instead of Qwen-Edit (no negative at all) | 🍎 an apple |
| **`"size of an apple"` → `"size of a small plum"` in the STYLE** | 🫐 **a PLUM** |
| the fruit noun deleted, prop clause kept | 🏠 a toy house |
| **the prop clause dropped entirely** | ✅ **empty, open hands** |

Five lessons, each of which cost a defect somewhere in this ledger already:

1. **A NOUN IN A SIZE METAPHOR IS A NOUN THE MODEL DRAWS.** "About the size of an apple" is
   not a measurement to a diffusion model. It is an apple. Give scale without naming a thing.
2. **THE STYLE OUTRANKS THE SCENE.** The clause claimed props ARE "clearly held in her hand"
   on every teaching shot. The scene said her hands were empty. The style won, every time.
   A blanket clause is a claim about EVERY shot — only say it when it is true of THIS one.
3. **NEGATION DOES NOT REMOVE ANYTHING.** "No fruit and no apple" in the positive → an apple.
   The same ban in the negative → an apple. Naming a thing to forbid it is naming it.
4. **A GUARD'S OWN WORDS ARE READ BY THE NEXT DETECTOR.** `no_phantom_object` wrote "no fruit
   and no apple"; `_HELD_PROP` matched it; the prop clause was added; the apple appeared. The
   guard against phantom props MANUFACTURED the phantom prop. This is the second time this
   exact class has bitten (the first: naming "doll" fed `lesson_objects.detect` a pinned
   block). Its own docstring said "name no prop noun" while its last line named two.
5. **THE BACKEND WAS A RED HERRING.** Both backends drew the apple. The negative void (#21) is
   real, but it is not why the apples were there — and switching to Qwen-Edit would not have
   fixed a single one of them. *The obvious suspect was measured and acquitted.*

The fix is a CHECK, not wording: `mascot.holds_a_prop()` — one negation-aware detector, shared
by the guard and the prompt builder, so they can never disagree about whether this shot holds
anything. Verified through the real code path on the exact shots that had the apple: beat 4
and beat 6 come back with empty, open hands and the pose the scene asked for.

---

## The next layer — visual QC: BUILT, MEASURED, DELETED (2026-07-17)

**Do not build this a third time without reading this section.** It has now been built twice
and deleted twice, and the reasons recorded the first time were wrong.

The idea: after a still renders, have a vision model LOOK at it and check it against this
ledger. Built (`image_qc.py`, `lesson_pipeline.run_visual_qc`), fully unit-tested, wired to
nothing, and left in the tree with three documents disagreeing about whether it worked.

### It was convicted on a parser bug

The checklist was sent numbered — `1. [one_child_human] <question>` — and **neither model
answers under the bare key**:

| model | answers under |
|---|---|
| qwen2.5vl:7b | `"[one_child_human]"` — the label, brackets and all |
| qwen2.5vl:32b | `"1"`, `"2"`, `"3"` — the number |

The lookup was a plain `got.get(key)`. **Every verdict missed, on every reply, on every image,
for both models, from the day the file was written.** It was invisible because the bias was
fail-open: a missing key defaulted to `"pass"`, so every shot came back clean and the pass
looked like it was working and finding nothing. **That is why the record said the 32b "passed
a face on a ball": the model said `fail` and the code never read it.**

*(A parser that cannot fail loudly fails silently in whichever direction it is pointed. Under
a strict bias the same bug inverts: every key missing = all seven checks flagged = pure noise.
Both were observed, hours apart, from one bug.)*

### The real numbers, with the parser fixed

Measured on 10 stills from `20260714_113840` and `20260716_000517`, labelled by looking at the
pixels and cross-checked against each shot's own `mascot_scene` (a prop is only phantom if the
scene did not ask for it — that cannot be judged from the picture alone):

| check | 7b recall | 32b recall | 32b false alarm |
|---|---|---|---|
| face on an object (incl. the ball) | **0/3** | **2/3** | 0/7 |
| the T-pose | **0/1** | **1/1** | 0/9 |
| doll with a face | **0/1** | **1/1** | 3/9 |
| phantom apple | **0/3** | 1/3 | 3/7 |

- **The 7b is not uncertain — it is blind and sure.** Of a ball with a face on it, it replied:
  *"No faces on objects."* An explicit, confident, wrong pass. No strictness setting can
  rescue that: a strict bias routes UNCERTAINTY to a human, and there is none to route.
- **The 32b genuinely works** on the two defects that matter most. It costs 60-140s an image
  (~20 min added to a 13-shot prepare) plus a 21 GB unload/reload of ComfyUI on the 16 GB card.

### Why it was deleted anyway (jeffy, 2026-07-17)

Not worth ~20 minutes and a VRAM dance per lesson for a hint. And a hint is all it could ever
be: it must never auto-redraw after the gate (`redraw_still` clears the tick, so a false flag
throws away a picture you already confirmed) and it must never show a green badge (a "clean"
tick beside a face-on-a-ball at a 104px thumbnail lowers scrutiny at the one moment scrutiny
works — strictly worse than no QC).

**The checker is `.claude/skills/image-supervisor` — a person looking at the pixels.** What
survives, and what actually helps that person, is `modules/lesson_contract.py`: the shot's
promises, printed beside the picture at the gate. It turns "does this look OK?" (unanswerable
at a glance) into "is she holding the blue block and nothing else?" (anyone can).

### If you build it again

- Assert the parse. A QC whose verdicts are all "missing" must say so LOUDLY, not default.
- The 32b is the floor. The 7b cannot see these defects at all.
- Ask CLOSED questions filled in per shot (`lesson_contract.contract_for` still produces
  exactly the facts needed). Open questions — "is she holding only things that belong?" —
  need judgement, which is what got the first VLM pass deleted for hallucinating.
- **The self-sabotage rule**: a guard/repair that NAMES a prop in order to forbid it teaches
  `lesson_objects.detect` that the prop is there, which pins that prop's reference INTO the
  shot. `no_phantom_object` says "hands empty and open" and names no noun, on purpose
  (`mascot.py`, the scar is preserved in-line).

---

## Doc drift found 2026-07-17

- `BACKDROPS[hash(lesson_id)]` (row 17, and CLAUDE.md) **never existed**. It is
  `lesson_pipeline.SETTINGS` + `setting_for(topic)`, keyed on the topic.
- The phantom-apple guard was live and tested for a day with no row here (now #20).
- This document said visual QC was "proposed, not built" while it was built, tested and
  sitting in the tree; `image_qc.py`'s header argued the 32b was the answer; the
  image-supervisor skill said the 32b had been measured failing. Three documents, three
  states of the world, all about the same dead code.
