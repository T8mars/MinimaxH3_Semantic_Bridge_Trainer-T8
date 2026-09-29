# H3 prompt-only feedback admission for Semantic Bridge training

This record decides whether a **text edit** has video-supported preference evidence. Both arms run **base H3 without a bridge**. It does not measure a Semantic Bridge, and a still-frame preference alone is not a full-video training label. Record the preference's actual axis: complex semantic execution (including effect source, ownership and timing) or secondary FX appearance. An FX-only preference must not be relabeled as semantic improvement or treated as proof of the Bridge's primary goal; native X/Y remains training-target diagnosis.

## Freeze the case

Record the work, version/era, pair of characters, official URL and exact fact supported for every costume, weapon and ability in either prompt. Mark the encounter, terrain, action order, camera, effect treatment and outcome as original hypothetical staging unless a primary source actually depicts them. Keep the prompt as the exact deployable MiniMax H3 structure, not a deliberately incomplete straw-man. Save both UTF-8 prompt files and their SHA-256 values before generating. Describe the bounded edit and the defect it is meant to fix; if other attributes change, list them as confounds before seeing output. Keep each scene family, its revisions, seeds and later bridge experiments in one train/validation/test group.

```json
{
  "case_id": "example-only",
  "group_id": "work-era-character-pair-scene-family",
  "source_ledger": [{"fact": "visible character or ability fact", "url": "primary source URL", "observed_support": "exact page or image observation"}],
  "staging_status": "original hypothetical shot",
  "appearance_control": "text_only_identity",
  "prompt_base_sha256": "64 lowercase hex characters",
  "prompt_edit_sha256": "64 lowercase hex characters",
  "bounded_edit": "exact changed clause and intended visual defect",
  "confounds": [],
  "seeds": [1234, 5678],
  "h3_model_sha256": "64 lowercase hex characters",
  "encoder_sha256": "64 lowercase hex characters",
  "workflow_common_sha256": "64 lowercase hex characters",
  "media_sha256_by_prompt_and_seed": {},
  "review_status": "pending",
  "inspected_frame_ids": [],
  "inspected_count": 0,
  "decoded_total": 0,
  "primary_target_observation": "pending",
  "regressions": [],
  "evidence_scope": "unreviewed",
  "feedback_admission": "not_admitted"
}
```

Use at least two fixed seeds for a proposed positive training pair; the numeric values above are examples. Capture actual submitted prompt bytes, workflow JSON, H3 model/encoder/VAEs, sampler, resolution, length, seed, decoded frame count/fps, audio presence and media SHA for each arm. The prompts are the sole intended difference. If actual image references are used, record ordered image SHA values and keep them identical across arms; do not inherit text-only bridge conclusions.

## Review and decision

Randomize opaque A/B media labels before review; do not expose prompt identity to the reviewers. Two independent reviewers watch **each full clip at normal speed and listen to its audio when audio is expected**, then use sampled frames for detail counts. For each seed and arm, log time-indexed observations for source-correct identity/era, palm or weapon attachment, attack path, defender clearance, visible contact, effect occlusion, body/anatomy legibility, persistent result, continuous movement and sound/sync. Mark unreviewed dimensions `N/A`; an audio stream in the MP4 is not evidence it was heard. If the frozen workflow is intentionally silent, record audio as not applicable. If it generates audio but nobody actually listened, do not admit the pair under a full audiovisual feedback protocol.

Pre-register the primary defect and no-regression criteria before any output. Admit an edited prompt as a **video-feedback preference** only when two reviewers agree that the intended defect visibly improves across the declared seeds and no serious identity, causality, contact or anatomy regression appears. Preserve ties, losses, failures and disagreements. A promising scar in isolated frames with a hidden hand-to-target contact is a static observation, not a positive training pair. If frames were sampled, keep `inspected_frame_ids` and `inspected_count` distinct from `decoded_total`; sampled counts are not full-clip rates or proof of continuous causality. Record the answer key only after both reviews are sealed. This admission does not prove that a future bridge trained on the text pair improves H3: that requires a separate **same-prompt** base-versus-bridge A/B on validation and unseen cases.
