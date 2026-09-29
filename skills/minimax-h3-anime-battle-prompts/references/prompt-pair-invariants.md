# Prompt-pair invariants

Use this before delivering authored x/y preference prompts or revising an existing pair. Apply the character-fact checks in [source and staging boundaries](source-and-staging.md); this reference checks the shot and the controlled edit. The short examples below are original generic staging fragments, not complete H3 prompts or demonstrated video improvements.

## Identify the comparison

| Purpose | What may differ | What the result means |
| --- | --- | --- |
| Authored preference data | One declared text policy: contact, path, effect layering, or consequence readability | The author prefers y under that policy; label `authoring_rubric`. Text consistency does not establish video benefit. |
| Prompt-only video comparison | Complete x versus complete y, both using base H3 with matched settings and seed | Evidence about that text edit; use the [feedback record](prompt-feedback-admission.md) for the evidence actually collected. |
| Semantic Bridge A/B | Only the bridge arm and its recorded application contract; prompt bytes, reference inputs, seed and generation settings stay identical | Evidence about the bridge on that exact prompt; use the [A/B record](ab-evaluation.md). |

Do not give base H3 x and the bridge y and call the difference a bridge gain. Revising the text creates a separate prompt version, even if the seed stays fixed. Keep revisions, camera variants and seeds in their original scene family and split.

## Check the frame against the evidence

List the objects a reviewer must see at setup, contact and aftermath. Check the proposed camera against each object, including the surface where the result persists.

- A shoulder-up view cannot also establish a boot pressing a groove into the floor. Frame the boots and that floor patch in both arms, or choose a different already-supported result when authoring a new scene.
- Keep the result's relation to its source readable: for example, a dent remains on the shield that was struck, rather than appearing on an unrelated panel after a cut.
- Reserve enough of the existing shot duration for the aftermath. Adding a hold must not silently remove a required wind-up, contact or recovery from one arm.
- Full-body framing gives more spatial coverage but fewer pixels per face, hand and contact edge. Use the closest view that contains the required evidence; check whether the intended contact remains distinguishable at the chosen output size.
- If the wider view exposes new outfit details, apply the existing era-matched appearance checks. Do not fill those details from memory.
- For an occlusion followed by a direct clash, check the post-occlusion **positions of both fighters, the obstacle and the contact point** in the same frame. If the intended clash is beyond the obstacle, state the shared open side and keep the obstacle behind the fighters and outside both weapon paths. An actor emerging on the requested side alone does not establish that the later contact is unobstructed.

Shared repair example: replace an incompatible crop in **both** prompts with “A fixed side view includes both fighters, the complete held staff, and the floor patch beneath their boots.” This repairs framing only; it does not add a new strike, groove or fragment, or prove that the renderer will preserve them.

## Preserve the shared shot while editing

Before rewriting, extract a compact shared checklist from the complete x prompt:

| Shared item | Check in both complete prompts |
| --- | --- |
| Identity and equipment | Same characters, era/form, visible identity features, held weapon count, carried equipment and ownership. “Empty hands” alone does not constrain carried weapons. |
| Geography and camera | Same positions, background anchor, target, framing and intended camera movement, except a separately documented repair shared by both arms. |
| Event and duration | Same attack count, attack/response order, contact participants, damaged object, resulting pieces, recovery and time budget. |
| Effects | Same ability and emitter, visual treatment, effect inventory and intended impact scale; change only the declared layer or visibility property. |
| Readability constraints | Preserve existing brief-contact timing, contact outlines, hand/weapon attachment, visible target boundary and aftermath constraints. |
| Audio | Same timed action sounds, soundscape and music. Check sound instructions inside the visual timeline as well as the dedicated audio fields. |

Use explicit shot-specific equipment wording when needed, without turning it into a universal canon claim. Distinguish “the two silhouettes remain readable” from “the staff-to-shield contact edge remains readable”; one does not imply the other.

For an effect-layer edit, do not accidentally remove an existing **brief** flash or **contact outline** while adding spectacle. Keep the effect inventory explicit, then change its depth or placement:

- Shared: “A brief white rim traces the staff-to-shield contact edge; radial sparks and a dust fan accompany the impact. The contact edge and both silhouettes remain readable.”
- x layer clause: “The sparks and dust occupy the same midground layer.”
- y layer clause: “The sparks spread behind the shield while the same dust fan crosses the foreground below the contact.”

Only the layer clause changes. The same strike, rim, sparks, dust, aftermath and sound remain. These clauses illustrate a controlled author preference, not a guaranteed better render. If an experiment intentionally changes effect quantity, declare that quantity change as its policy instead of calling it layer separation.

## Deliver a checked pair

1. Write both complete prompts in the workflow's required H3 fields; fragments alone are not a deliverable.
2. State the one policy and quote the exact x/y replacement clauses outside the copy-ready prompts. Inspect the full diff, not only the intended edit.
3. Compare all shared items above. Remove accidental changes in equipment, timing, audio, event or effects. Record an unresolved mismatch rather than silently treating it as controlled.
4. For a shared repair, list that same replacement separately from the x-to-y policy. Verify that reversing the shared replacement reconstructs each original prompt without other edits.
5. Save the exact prompt bytes and hashes with the version and scene-family ID. Keep original versions intact; a repaired draft is not an additional independent family.
6. Report the authoring checks and unresolved readability tradeoffs briefly. Keep source/text findings, observed video behavior and measured bridge effects separate.

One controlled policy makes a comparison interpretable; it does not establish a learnable mapping, video improvement, or a causal explanation for a prior failure. Finish with **“Prompt drafted; generated-video quality unverified”** until actual video evidence supports an updated statement.
