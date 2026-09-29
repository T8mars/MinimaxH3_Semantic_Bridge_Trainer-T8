# Semantic Bridge effect check for MiniMax H3

This is a video evaluation record, not a video LoRA training procedure. The primary Bridge goal is execution of complex battle semantics: continuous action, spatial transitions, occlusion/reidentification and correct actor, weapon and effect ownership across beats. Effect source, path and timing are semantic criteria; visual spectacle is a separate secondary observation, not a promised Bridge gain. A prompt that reads well and a lower bridge training loss do not establish better H3 video.

## Freeze before generating

Use development cases to choose bridge alpha and other application settings. Set aside final cases by **scene group**, and check exact text, shared long phrases, paraphrases, and shared source footage against every training group. A lexical search alone is incomplete. Keep the source-backed character and ability ledger with the case; label the encounter and choreography as original if they are invented.

For each case, save the exact UTF-8 prompt bytes in a separate `.txt` file and a record such as:

```json
{
  "case_id": "example-001",
  "role": "validation-or-locked-test",
  "prompt_sha256": "64 lowercase hex characters",
  "prompt_utf8_bytes": 0,
  "training_overlap_review": "pending-or-reviewed",
  "source_urls": [],
  "seeds": [1234, 5678],
  "h3_model_sha256": "64 lowercase hex characters",
  "h3_encoder_sha256": "64 lowercase hex characters",
  "workflow_common_sha256": "64 lowercase hex characters",
  "frame_count": 124,
  "fps": 24,
  "arms": [
    {"id": "base", "bridge_sha256": null, "alpha": 0},
    {"id": "candidate", "bridge_sha256": "64 lowercase hex characters", "alpha": 0.05,
     "magnitude_match": "per_token", "chunk_tokens": 0,
     "token_span": "all", "tail_ratio": 1.0, "allow_dim_mismatch": false,
     "judge": "none", "auto_alpha": false, "guard": false}
  ]
}
```

The frame count and alpha above are **example values**, not recommended settings. The candidate node fields illustrate reproducibility; record every field actually submitted, including judge and `auto_alpha` if used. For this project's bridge, use `chunk_tokens=0` and the same exact encoder weight used for training. For every case/seed pair, run both arms with the same prompt bytes, H3 model, encoder, sampler, scheduler, frame count, resolution, and seed. Record each submitted workflow JSON hash, output media path/hash, successful full decode, actual frames/fps/audio stream, and the one permitted bridge difference. If a run fails, preserve its receipt; do not quietly replace just that arm or seed. A prior bridge can be a third arm, with its own fixed validation-selected alpha.

Keep prompt wording comparisons in a separate experiment. A better new prompt cannot be counted as a bridge win if the base arm received the old prompt.

Match the comparison to the trained input role. A bridge trained to repair a deficient prompt while preserving a good one is not automatically an enhancer for already-complete Skill prompts. Test those uses separately. For a repair-mechanism check, compare the identical deficient prompt with and without the bridge, and use a separately rendered corrected prompt only as a reference. First establish that the base renders actually exhibit the intended defect and a reachable correction; otherwise that case is non-diagnostic for the proposed repair. This additional experiment must not erase a failed complete-prompt comparison or be described as its successful continuation.

For a claim about semantic understanding, freeze stress cases that require a feasible **continuous** sequence within the chosen clip duration. The scene should identify ordered beats, at least one actor or camera move to a new spatial relationship, a temporary occlusion or exit/re-entry, and an observable consequence that persists into the next beat. Record which character and weapon own each effect before and after occlusion, where an effect travels, and whether reaction follows contact. A single clear attack-and-hit can verify the basic route but cannot alone establish temporal or spatial understanding. Include enough continuity cases across works to support the claimed scope; do not stretch a short clip into an implausible number of beats.

## Choose the review scope before generating

The distributed experimental model card records the completed **ten-pair mixed-profile visual screen**: three previously exposed 124-frame/25-step development pairs plus seven new 294-frame/Turbo4 pairs. At the user's direction, the two relative semantic improvements in the older pairs **do count** in that screen. Its result is 2 wins, 8 ties and 0 material regressions, below the previously requested **3 of 10** target. The cohorts are not a homogeneous holdout, and one reviewer inspected ordered frames without normal-speed playback or audio review. Read the model card and bound video report before citing that result; neither effects preference nor the completed screen establishes a general video-quality gain.

For any **new prospective** sampled validation, freeze the scene set, exact bridge weight, paired prompt/seed, input mode, H3 backend, sampler and attention profile, review scope and pass rule before generating. Count a scene as an improvement only when its paired evidence shows clearer execution of its predefined continuous-action, spatial or occlusion target; prettier effects alone do not count. Report the full frozen denominator, including ties, failures, regressions and unreviewed domains. Do not silently add old development clips to a newly declared homogeneous holdout or treat a later profile change as the same experiment. A new 3/10 pass claim requires its own preregistered evidence; it cannot be inferred from the experimental card's 2/10.

Match the evidence to the requested claim. A frame-sequence study can assess visible identity/equipment, contact ordering, occlusion and lasting consequences. It cannot establish normal-speed motion quality, listening quality or audiovisual synchronization. Full audiovisual review includes normal-speed playback and actual listening. Missing playback or audio does not prevent a separately scoped visual study, but those axes remain unreviewed and its outcome must not be called a full audiovisual pass.

For a visual semantic-execution experiment, actual audio listening is required only if audio or audiovisual synchronization is part of the claimed outcome. Keep an unlistened track explicitly unreviewed; it does not invalidate supported visual observations. A separately defined full-audiovisual training-feedback contract still requires its own evidence.

Declare the scope and success criteria before looking at the outputs. Keep existing frozen protocols and their results unchanged; a later visual-only protocol cannot retroactively pass a failed or incomplete full audiovisual run. Source/text preferences and static-frame results also do not satisfy a trainer's separately defined `video_feedback_v1` full-audiovisual admission contract.

## Blind review

Assign opaque A/B labels before review and conceal the arm mapping. For full audiovisual review, each reviewer watches each full video at normal speed and listens to its audio when audio is expected, then may inspect frames for contact and persistence counts. For a frame-sequence study, each reviewer inspects all decoded frames in chronological contact sheets and opens original-resolution frames at decisive events or suspected defects; record unknown details instead of guessing from thumbnails. Require at least two independent review records before revealing the labels. A contact sheet alone cannot judge motion or sync. Record an intentionally silent workflow as audio N/A; if an audio-generating workflow's track was not actually heard, mark audio unscorable and do not call that review full audiovisual evidence.

If using a machine video observer as supplementary evidence, first check the capabilities being claimed on controlled clips with known motion and presence events. Preserve failures: correct object naming does not establish correct trajectory judgment, and supplying every frame does not prove that the observer interpreted them correctly. Bind observations to the exact video, model, preprocessing and instruction versions; verify claimed events against the actual frames. Label machine observations separately from playback and listening. If temporal or audio evidence is unavailable, report those axes unscorable rather than upgrading still-image preference to a full-video win. For a deliberately silent workflow, audio is not applicable.

For each video, record `identity_and_era`, `weapon_continuity`, `ability_source`, `emitter_to_target_path`, `visible_contact`, `defender_response`, `effect_layers_readable`, `obscuration`, `lasting_consequence`, `motion_continuity`, and `audio_sync` as pass/fail/unscorable with a short observed reason. For continuous-sequence cases also record `beat_order`, `spatial_transition`, `occlusion_reidentification`, `effect_owner_after_reentry`, and `consequence_persistence`, citing frames before, during and after each transition. Judge the requested FX richness/readability separately from semantic continuity: when both arms preserve actor identity, order and causality, an FX preference is not a semantic win. Contact-, emitter-, or damage-specific fields may be `not_applicable` when that phenomenon is absent from the requested scene, while every observation required by an already-frozen protocol remains mandatory. Record `primary_target_observation`, `regressions`, `evidence_scope` and `feedback_admission` separately. Count frames with visible decisive contact, frames where the actor or weapon silhouette is obscured, and frames that retain the resulting damage. Log `inspected_frame_ids`, `inspected_count` and `decoded_total` with each count: sampled frames are not full-video rates. Choose numeric thresholds on validation cases, **before** looking at final test outcomes. If no damage is requested, mark persistence not applicable instead of inventing one.

Ask each reviewer for a paired `A better / tie / B better / inconclusive / both fail` verdict on complex semantic execution, a separate secondary FX-appearance verdict, and a canon-regression flag. Preserve a semantic tie when both arms execute the scene correctly; do not demand a win in every case or convert a prettier effect into semantic improvement. Keep the originally planned case/seed denominator, including technical failures and unreviewed cases. Disagreements stay visible; do not resolve them by selecting only attractive still frames. After both records are fixed, reveal the mapping and summarize wins, ties, losses, failure modes and which works/seeds were tested. A bridge is ready for the final package only when same-prompt, same-seed bridge-off/on cases support the declared semantic benefit without a material identity or ability regression, and the model card states exactly what was and was not evaluated. A visual-only result cannot support claims about playback or audio. Do not extrapolate one franchise or one seed to all twenty.
