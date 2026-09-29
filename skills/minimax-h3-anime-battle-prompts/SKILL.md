---
name: minimax-h3-anime-battle-prompts
description: Write and revise MiniMax H3 prompts for spectacular anime or game-character battles, with source-grounded identities, eras, weapons and abilities. Use for fight choreography, cinematic effects and the appropriate H3 text or reference format; apply controlled-comparison procedures only when an experiment is requested.
---

# MiniMax H3 anime and game battle prompts

Turn the user's battle idea into a copy-ready English H3 prompt with a clear action progression and spectacular, readable effects. Preserve their cast, duration, interaction type, style and intended outcome. Canon facts and newly invented choreography are separate. This skill writes prompts; text quality does not establish generated-video quality. A Semantic Bridge is a text-conditioning residual adapter whose primary goal is better execution of complex battle semantics. It is not a video or character LoRA and does not promise more spectacular effects; effect design remains part of this prompt-writing skill.

## Ground the cast without losing the requested scene

Lock the work, era/form, visible appearance, weapons and abilities actually used. Use directly inspected primary material: official anime/game, creator, publisher, studio or technique documentation. A traceable existing source record may be reused for the same fact and period; retain its URL, observed support, medium and access limits. Inspect the relevant image before asserting a new visual detail. Search snippets, clip captions and memory are leads, not verification. Current profiles may combine different eras.

Put a few verified, distinguishing features in the English prompt, sized to the composition; a name alone is not an appearance guarantee. A visible costume feature must not grow until it hides the action. Do not invent hidden clothing or weapon details. Keep unsupported claims outside the usable prompt; narrow an uncertain detail or propose a source-supported alternative without silently changing the user's named character or technique. Read [source and staging boundaries](references/source-and-staging.md) when combining media, periods or alternate routes.

The project's 20-work roster is Dragon Ball, Naruto, One Piece, Bleach, Demon Slayer, Jujutsu Kaisen, My Hero Academia, Hunter × Hunter, Fullmetal Alchemist: Brotherhood, Fairy Tail, Black Clover, Attack on Titan, JoJo's Bizarre Adventure, Saint Seiya, Yu Yu Hakusho, Rurouni Kenshin, InuYasha, One-Punch Man, Mob Psycho 100 and Chainsaw Man. This is a coverage roster, not a popularity ranking or a video-verified ability library. For game characters, apply the same checks to the requested game/version; do not substitute anime facts. If the cast is open, choose a compatible source-supported pairing. Keep a requested crossover explicitly hypothetical and its characters' abilities distinct.

For one of these works, start with the narrow [20-work source routes](references/twenty-work-source-routes.md), then reopen the relevant official pages and inspect any image whose visual details enter the prompt. The routes are fact leads for specific casts and periods, not a license to use every form, costume or move in the franchise.

Once identity, ability and equipment are supported, the setting, ordinary props, route, encounter and choreography may be original. They do not need an identical canon episode. Do not infer a new power from effect color, convert a shot-specific omission into a universal limitation, or treat a legitimate supernatural ability as a physics error.

## Build action and spectacle together

Use the requested duration. When executing, confirm the active workflow's actual timing and supported inputs; if it cannot meet the request, state the mismatch rather than silently shortening it. Local 124-frame/24fps probes are not an H3 duration limit or a required shot structure. For a very short request, favor one dominant exchange; longer scenes can build through combinations, reversals, travel and multiple shots with enough time for each action to read.

Choose the visible relationship that matters to this battle:

| Interaction | What the audience should be able to follow |
|---|---|
| Melee, weapon bind or counter | Approach, the relevant guard/contact or near miss, redirected motion and the next stance or separation |
| Beam, projectile or area attack | Verified emitter, attack travel/expansion, the defender's response, and the intended collision, deflection, miss or dissipation |
| Flight, traversal or pursuit | Departure, a coherent route past landmarks/obstacles, relative positions and the intended arrival or continuing chase |
| Combination or multi-shot battle | Who acts and responds at each beat; positions, equipment and already-established changes carried into the next shot |

Damage is optional. A resolved dodge, a checked blade, an extinguished attack, a changed formation or a landing can complete the action. Do not add a paving slab, ground scar, persistent damage or a contact event merely to make the scene resemble a local test. When damage is requested, show its cause and keep its location/state consistent afterward. When evasion is requested, show the clearance instead of inventing a hit.

Make spectacle concrete: decide where the attack gathers pressure, how its speed or scale changes, what the defender does, and where the visual peak occurs. Layer the source-supported ability, directional motion accents, atmospheric light and any earned environmental response across depth. Not every scene needs all these layers. Preserve recognizable silhouettes and the decisive spatial relationship through the peak; a large explosion can sit behind or beyond the critical actors rather than washing out every edge. Let the next beat become readable as the peak fades.

For a longer continuous battle, first make a small beat ledger outside the copy-ready prompt: fixed landmark; each actor's position and facing before and after each move; who holds each weapon or emits each effect; what an occluder hides and where the same actor reappears; and any state that must persist. Map positions relative to the landmark, especially across a camera cut, instead of assuming screen-left means the same world side after a reverse angle. If the next beat is a direct clash after passing an obstacle, place both fighters and the contact point in the intended open area on the same side of that obstacle before the clash; saying only “clear of the obstacle” may still leave it between their blades. Give each major move enough screen time to finish before adding the next. Use the ledger to write positive, visible actions; it is a planning aid, not extra H3 syntax. The [continuous-action example](references/examples.md#naruto-a-longer-occlusion-and-side-exchange) shows one possible 12-second sequence, not a required duration or terrain.

When the user wants to test whether a Semantic Bridge understands battle semantics, make the evaluation prompt a feasible continuous sequence rather than only one attack and reaction. Include a spatial transition and, when the duration permits, a temporary occlusion or exit/re-entry; reidentify the correct actor/weapon/effect source afterward and carry the consequence into the next beat. Effect source, ownership, path and timing belong to semantic execution; spectacle and attractiveness are separate, secondary observations. Use identical prompt bytes, seed and generation settings with only the bridge switched off/on to establish bridge benefit. If both arms already handle the sequence, record a semantic tie; an effects preference is not a semantic win. Do not manufacture a deficient base prompt. Follow the [A/B record](references/ab-evaluation.md) for controlled Bridge comparisons.

Attribute diegetic energy/fire/chakra/etc. to an established ability. Keep speed lines, impact stars and chosen color accents identifiable as animation treatment. A named breathing style does not by itself establish literal elemental damage. Strong effects are compatible with clear action; reducing every battle to a gentle near-contact exercise is not the goal.

Choose camera motion and cuts for the action: reveal scale, follow a route, expose an exchange, or show a changed situation. Keep screen geography intelligible through that movement; do not require a static camera or uninterrupted single shot unless the user or chosen mode calls for it. Express the necessary visible actions positively instead of burying the scene in a long list of prohibitions. Time material-specific action sounds and musical changes to the intended beats, without promising that the model will synchronize them.

## Use the actual H3 input format

Read the relevant official guide when producing the final prompt:

- [Base guide: T2VA / I2VA / FL2VA / L2VA](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md). Preserve the field order `integrated_multimodal_description`, `overall_soundscape`, `non_diegetic_music`. T2VA starts with those fields. For supplied first/last frames, put the guide's exact mode-specific alignment instruction first, with actual picture labels and duration. Do not use that image instruction for text-only input.
- [Full-reference guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md). Preserve `subject_definitions`, `summary`, `retention_analysis`, `detailed_description`, `overall_soundscape`, `non_diegetic_music` in that order. Inside `detailed_description`, establish the style before the opening shot marker; T2VA introduces it after that marker. Use the guide's reference roles, summary task prefix and retention markers, not a T2VA body with invented reference tags.

The opening marker is `[Shot 1]` without a time; later cuts use `[Shot N] At MM:SS.mmm, ...` at increasing times within the requested duration. Put timed diegetic sound in the action timeline and summarize the soundscape separately; audience-only score belongs in `non_diegetic_music`. Use `N/A` for no score, and for `overall_soundscape` only when complete silence is requested. Follow the guide for dialogue, vocal IDs and supplied text rather than inventing another syntax.

A research image is not generator input. Use `<Subject N>` for a reusable referenced character and tie it to the actual supplied asset; standalone picture definitions are for frame/composition anchors. Never invent media labels. Check that the installed workflow actually accepts the requested reference mode. A first-frame anchor is different from reusable character references, and neither guarantees action execution. For local ComfyUI conditioning, bridge compatibility or an actual controlled comparison, read [local input and evaluation boundaries](references/local-conditioning-and-evaluation.md); ordinary prompt drafting does not require an experiment dossier.

When a user asks to pair the prompt with a Semantic Bridge, read the candidate weight's current model card, encoder hash, application contract and acceptance result. Apply a candidate only within its measured input mode and fixed settings; an unpassed or unevaluated candidate is an experiment, not an established enhancement. A good prompt works as a base-H3 deliverable even when no bridge has passed.

## Deliver a usable prompt

Unless the user specifies otherwise, give:

1. **Copy-ready English prompt** in the selected official structure, including action progression, camera, layered effects and sound. Keep commentary and citations outside it.
2. **Brief Chinese timeline** explaining the attack/response, visual peak and intended result or continuation. It need not manufacture damage or a physical contact.
3. **Compact canon ledger** with direct primary URLs, the facts used, period/medium limits and excluded uncertainties. Label location, choreography, camera and effect treatment as original staging. State whether references were research-only or actually supplied.
4. **Review targets appropriate to the scene:** identity/era, equipment, ability ownership, the relevant action relationship and readable spectacle. Add playback motion and sound/sync only if actually evaluated. Include a run note only when executing a workflow.

Until reviewed, say **“Prompt drafted; generated-video quality unverified.”** Report source/text review, inspected stills, all-frame chronology, normal-speed playback and listening separately. Reference-correct identity alone does not establish correct action, separation or effect readability. Do not present an attractive still, a training loss, or a proposed prompt revision as evidence of video improvement.

For an authorized authored pair or video/bridge comparison, use the conditional procedures in [local input and evaluation boundaries](references/local-conditioning-and-evaluation.md). Preserve same-prompt/fixed-contract comparisons and distinguish author preference from video-feedback evidence. The [worked examples](references/examples.md) illustrate particular source/staging choices; their durations, stone targets and contact outcomes are not defaults for new requests.
