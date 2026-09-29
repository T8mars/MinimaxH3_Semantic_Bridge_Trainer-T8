# Local conditioning and controlled comparisons

Read this only when using the local ComfyUI implementation, a Semantic Bridge, authored preference pairs or a real comparison. It is not a required checklist for ordinary prompt writing, nor a statement of universal H3 product limits.

## Input mode and actual reference assets

Inspect installed node inputs and the submitted graph. Where `MiniMaxH3ReferenceToVideo` exists, real character references belong in its `ref_images` inputs; record hashes and order. Reference-image research alone does not supply those inputs. Use the official full-reference prompt structure and asset roles. A first-frame image on `MiniMaxH3ImageToVideo` anchors a frame/composition and is not interchangeable with separate reusable character references.

The current text-only trained bridge has no established mixed image/text conditioning compatibility. Native first/last-frame I2VA and reference modes use mixed conditioning in the inspected implementation; width 5120 alone does not establish adapter support. Do not alter alpha or token selection to imply compatibility.

An experimental alternative encodes text alone, applies the bridge or bypass, then appends a latent keyframe through `MiniMaxH3AddGuide`. This preserves text-only input at the bridge boundary. It is neither native I2VA nor evidence that identity or action will be preserved. Compare no-bridge text plus guide against the identical text, fixed-contract bridge and identical guide. Native I2VA without a bridge is a separately labeled input-mode comparison. Hold actual image/order, seed and other settings fixed.

## Keep three questions separate

| Question | Allowed intended difference | Evidence scope |
|---|---|---|
| Does a prompt revision help? | The declared text edit; base H3 and other inputs fixed | Prompt-only evidence, not bridge benefit |
| Does a bridge execute complex semantics better? | Bridge arm and its frozen application contract; prompt bytes, seed, settings and all reference inputs unchanged | Same-prompt semantic bridge evidence; FX appearance recorded separately as secondary |
| Can a bridge repair a deficient prompt? | Identical deficient text with/without bridge; separately rendered corrected text as reference | Repair evidence only if base outputs actually exhibit a scorable defect and a reachable correction |

Freeze UTF-8 prompt bytes/hash, scene-family split and exact model/encoder/bridge versions, seeds, frames/fps and settings. Check exact/near-duplicate and scene-family overlap before calling a case held out. Do not choose settings from final test outputs or change a frozen control after a failure. Record every submitted bridge option. For this project's node, use `chunk_tokens=0`; train and inference need the same encoder weight hash. Fixed-alpha token-objective weights must use their exported alpha/magnitude/token contract. Legacy weights use settings selected on validation. Do not override either contract to obtain an attractive output.

For this project, RunningHub cannot load the candidate Semantic Bridge weight, so it is not a valid bridge-on arm. Run the current bridge-off/on video comparisons in local H3 with the exact candidate weight. A remote render may be useful for unrelated prompt exploration, but cannot establish this adapter's effect unless that service actually loads the same bridge and application contract.

For no-bridge runs, feed original H3 `CONDITIONING` to the guider. Only a designated bridge arm routes conditioning through the bridge, with the latent path to the sampler retained. Verify actual socket names rather than assuming node IDs. JEV is separate and optional. Follow [the A/B record](ab-evaluation.md) for run receipts, blinding and preserved denominators; do not duplicate that procedure in a normal writing deliverable.

## Authoring is not video-feedback admission

For authored pairs, use complete official-format prompts and one declared editing policy. Preserve shared identity/era, abilities, event, time budget, camera, effects and audio except the declared edit; use [pair invariants](prompt-pair-invariants.md). Keep variants in one scene family. Label the intended preference `authoring_rubric`; it does not establish a learnable mapping or video improvement.

For video-feedback supervision, use a separate base-H3 prompt-only comparison and [feedback admission](prompt-feedback-admission.md). Its full audiovisual contract requires actual playback/listening and independent evidence; a static-frame preference does not satisfy it. A separately scoped frame-sequence study remains valid for the visible facts it inspected, with playback and unheard audio N/A. Preserve any stricter frozen protocol. Do not convert a nicer-sounding prompt, a few favorable frames, or lower training loss into quality scores.

## Review the requested battle, not every local diagnostic

Record identity/era, equipment, ability attribution, action continuity and spectacle. Choose action-specific targets: a clash needs contact visibility; an evasion needs clearance; flight needs a traceable route/arrival. If damage was not requested, its persistence is N/A. Where counting frames, keep inspected IDs/count separate from decoded total; sampled still counts are not full-clip rates or proof of normal-speed movement. Use actual listening for sound/sync claims. Keep primary observations, regressions, evidence scope and feedback admission separate.

Local evidence caution supplied with this draft: one train-derived guide comparison, two arms, 124-frame static-only review, was `both_fail`. Identities were visible with the same guide, but the contact flare obscured the exchange and the actors did not separate. This supports only the warning that an appearance reference does not establish action readability or separation. It is not unseen-test evidence, a general guide-mode verdict, or proof that a particular prompting change works. Do not prescribe reduced spectacle, a fixed camera, a five-second limit or a compulsory ground scar from that result.
