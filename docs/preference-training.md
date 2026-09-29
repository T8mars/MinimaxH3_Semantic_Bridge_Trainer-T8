# Text preference bridge training

This CLI supports original authored text preferences without inventing numeric
quality labels. They train the conditioning bridge only. They do not train a
video quality judge or establish that generated videos improve.

Each JSONL record needs nonempty `bad`, `good`, `group_id`,
`pair_preference: "good"`, and `label_type: "authoring_rubric"`.
Omit `bad_score` and `good_score` entirely. Do not mix these records with numeric
score records in the same dataset. Optional `supervision_type` must be
`preference_only`, and optional `score_label_type` must be `none`.

Use a separate split manifest:

```json
{
  "schema": 1,
  "split_unit": "group_id",
  "splits": {
    "train": ["scene_a", "scene_b"],
    "validation": ["scene_c"],
    "test": ["scene_d"]
  }
}
```

List every scene group exactly once, including all held-out groups. At least
eight pairs are required by the CLI. All variants/translations of a scene share
its group ID. `val` aliases `validation`; `challenge` and `challenge-test` alias
`test`. Do not supply two aliases for the same partition. A bridge-only explicit
split needs no calibration partition. Judges additionally need a separate,
nonempty calibration partition and actual numeric labels.

```powershell
python tools/trainer.py inspect pairs.jsonl --split-manifest splits.json
python tools/trainer.py encode pairs.jsonl --comfy-root COMFY_ROOT --encoder ENCODER.safetensors --output pairs.npz
python tools/trainer.py train pairs.npz --stage bridge --split-manifest splits.json --config configs/author_v1.json --output runs/NEW_RUN
python tools/trainer.py evaluate pairs.npz --run runs/NEW_RUN --split validation --alpha 0 0.05 0.08 0.12 0.2 --magnitude-mode per_token
python tools/trainer.py evaluate pairs.npz --run runs/NEW_RUN
```

Replace placeholder paths with your actual local paths. `inspect` checks group
coverage and shared-text leakage before loading H3. `train` also rejects identical
embeddings that cross partitions. Encoding caches can be reused; split assignment
does not affect encoding. The complete original split manifest is copied into the
run, hashed into its identity/signature, and checked again by `evaluate`.
Resume requires the same `--split-manifest` and configuration.

Select inference alpha and magnitude handling on validation before looking at
test results. `evaluate --split validation` writes a separate validation report;
the default remains the test partition at alpha 0/0.12/0.20 for compatibility.
Use `--magnitude-mode none` to compare the training blend without token magnitude
matching. After choosing settings, supply only the selected alpha (and 0 for
the identity baseline) to the final test evaluation. These are representation
diagnostics; video comparisons require their own fixed prompts and seeds.

The encoded NPZ stores NaN only as an internal missing-score sentinel. Its JSON
metadata explicitly states `supervision_type=preference_only` and
`score_label_type=none`; JSON records contain no NaN or fabricated labels. Loading
rejects scores that contradict this metadata. Judge entry points reject
preference-only data before starting optimization. Existing numeric-label
datasets retain their legacy behavior and explicit provenance description.

The rebuilt GUI supports this format: select the preference-only training mode,
your JSONL pairs, explicit group split manifest, encoder and configuration
(`configs/preference_v1.json` is a starting point). Its recorded command plan
runs strict supervision/split preflight, encoding or cache identity checks,
bridge-only training, bridge-only weight verification, then validation diagnostics
for `none` and `per_token`. It does not select parameters or evaluate the test
partition automatically. `verify --bridge-only --bridge <file>` never requires a
judge. Numeric-score GUI mode retains the original bridge+judge workflow.
The PowerShell one-click script remains the legacy numeric-score workflow.
Old distributed EXEs do not gain these features without rebuilding; use the new
`WushuBridge-Portable-Preference` package.

Representation metrics measure embedding movement toward preferred text. Report
them separately from actual video A/B evaluation and character/canon fidelity.

The experimental `semantic_objective="aligned_token_identity_v1"` trains exact
position-aligned token targets plus a good-prompt identity pass. It requires a
fixed `alpha_min==alpha_max`, `magnitude_mode="per_token"`, `residual_skip=true`,
and `anchor_weight=0`. Every record must include verified `token_alignment`
metadata tied to its text hashes and encoder identity; equal total sequence
length alone is insufficient. Insert/delete edit blocks and truncation are
rejected. The new residual output projection starts at zero; resume restores the
saved weights. Validation selects the correction-plus-identity objective, and
training defers test metrics. This mode is a text-supervised research experiment,
not a demonstrated H3 quality improvement.

The current launcher source supports the explicit **experimental aligned-token
+ identity** mode, using `configs/experimental_aligned_token_v1.json`. Keep the
JSONL, encoded NPZ, encoder and split fields pointed at the original source
files; choose a separate prepared-data directory. The GUI checks or encodes
the source, prepares eligible pairs, and trains only the bridge on the derived
NPZ and split. It verifies the exported weight and runs a **fixed-alpha,
per-token validation diagnostic**, with no `none` sweep or automatic test
evaluation. This pooled representation diagnostic is not the training
correction-plus-identity loss and does not select a new alpha.

An existing prepared directory is reused only after its source, artifact,
preparation-code and tokenizer receipts match. Resume also uses the training
engine's original run-signature checks; it never regenerates the prepared
data. Interrupted preparation needs a new output directory if an incomplete
directory exists. The original files and author weights are preserved.

Old distributed EXEs do not acquire this mode automatically. The updated source
has been compiled and checked locally; the final integrated package is still
pending model evaluation. The CLI below is also available. Neither GUI support
nor a representation gain establishes that the bridge improves an
already-complete Skill prompt or generated video.

Start with your original JSONL, its genuine H3 encoded NPZ and encoding receipt,
the same encoder/tokenizer environment, and the original group split manifest:

```powershell
python tools/trainer.py prepare-aligned SOURCE.npz --pairs SOURCE.jsonl --comfy-root COMFY_ROOT --encoder ENCODER.safetensors --split-manifest SOURCE_SPLITS.json --output-dir PREPARED
python tools/trainer.py check-dataset PREPARED/pairs.npz --pairs PREPARED/pairs.jsonl --encoder ENCODER.safetensors
python tools/trainer.py train PREPARED/pairs.npz --stage bridge --split-manifest PREPARED/split_manifest.json --config configs/experimental_aligned_token_v1.json --output runs/NEW_EXPERIMENT
python tools/trainer.py verify --bridge runs/NEW_EXPERIMENT/bridge.safetensors --bridge-only
python tools/trainer.py evaluate PREPARED/pairs.npz --run runs/NEW_EXPERIMENT --split validation --alpha 0.2 --magnitude-mode per_token
```

Preparation verifies the original JSONL/NPZ/encoder receipt and actual tokenizer
source fingerprints and package versions. It tokenizes on CPU without loading
H3 weights, copies exact native float16 rows, and preserves source labels. Each
record needs a stable unique `id`. Only equal-length replacement blocks qualify;
equal total length alone is insufficient. Overlong records are reported and
excluded without truncation. For legacy authored preferences, one eligible
record per `group_id` is selected by the lowest SHA256 of its ID, independent
of metrics. The opt-in video-feedback path keeps every unique admitted scene
family within its whole-work `group_id`. Output is ordered by group/ID.
For authored datasets that intentionally contain multiple reviewed variants
per group, pass `--selection-policy all_rows` to retain every eligible row in
that same split. The default remains `one_per_group`.
In the Windows EXE, the experimental training modes expose the same opt-in
as the “全变体” checkbox next to the training-mode selector. A saved prepared
directory is accepted only when its receipt has the selected policy; change
the policy in a fresh prepared directory and run directory. Check the reported
`eligible_count`, `selected_count`, and per-split pair counts before training;
`all_rows` does not make variants independent scenes or video-quality labels.
Original split assignment and leakage checks are retained. At least
eight selected records and nonempty train/validation/test partitions are needed.

The new directory contains pairs JSONL/NPZ/metadata, a compatible encoding
receipt, the derived split, and a preparation receipt with all hashes and
selection/rejection reasons. It must not already exist; interruption leaves no
completed directory. Preparation computes no test metrics or target-quality
scores and never chooses rows based on an observed training or video result.
To resume training, reuse this unchanged prepared dataset, configuration and
split manifest; do not regenerate or overwrite it. Keep the original inputs.
Train and evaluate operate on the prepared NPZ, not the original full dataset.

Preparation runs in a fresh CPU CLI process. Some ComfyUI builds import optional
GPU attention backends even under `--cpu`; this command temporarily disables
imports of `triton`, `sageattention`, `sageattn3`, `flash_attn`, and `xformers`
while constructing the real tokenizer, and rejects any attempted CUDA
initialization. It does not replace tokenizer code or token IDs. An already
GPU-initialized host process is rejected; launch the command separately.

The optional `token_id_monotone_transport_v1` preparation method accepts native
H3 text pairs with unequal token counts. It derives a deterministic,
order-preserving map from verified tokenizer IDs: equal blocks pair one-to-one,
unequal replacements use cell overlap, and insertions/deletions attach to the
left boundary (or the right boundary at the start). The
`monotone_token_identity_v1` objective distils good-prompt embeddings into the
existing bad-prompt token count and separately penalizes drift on already-good
prompts. It refuses truncation, stale text/embedding/encoder hashes, uncovered
transport rows or columns, near-zero transport vectors, and zero correction
energy. The default equal-length preparation and objective are unchanged.

To exercise this **text CONDITIONING software path** with a separately prepared
dataset, select the method and its matching config:

```powershell
python tools/trainer.py prepare-aligned SOURCE.npz --pairs SOURCE.jsonl --comfy-root COMFY_ROOT --encoder ENCODER.safetensors --split-manifest SOURCE_SPLITS.json --output-dir PREPARED_VARIABLE --method token_id_monotone_transport_v1
python tools/trainer.py check-dataset PREPARED_VARIABLE/pairs.npz --pairs PREPARED_VARIABLE/pairs.jsonl --encoder ENCODER.safetensors
python tools/trainer.py train PREPARED_VARIABLE/pairs.npz --stage bridge --split-manifest PREPARED_VARIABLE/split_manifest.json --config configs/experimental_monotone_token_v1.json --output runs/NEW_VARIABLE_EXPERIMENT
```

The JSONL reader now has a strict, explicit `video_feedback_v1` path. Supply
`--video-feedback-root PRIVATE_EVIDENCE_ROOT` to `inspect`, `encode`,
`prepare-aligned`, and `train`; omitting it rejects video-feedback rows. Each
row must bind a hashed external admission receipt to the exact frozen x/y
bytes, source-QA scope, two fixed seeds, two sealed frame-by-frame visual
precursor reports and two distinct sealed full-video-and-audio review reports
per seed, anonymous assignment and review locks. Ties, static-only
reports, missing evidence and changed hashes fail closed. Training rechecks
the receipts and requires an explicit whole-work split manifest. Authored
`authoring_rubric` preferences keep their previous default behavior and cannot
be mixed with video-feedback rows. These software checks validate claims and
file integrity; they cannot independently prove that someone actually watched
or heard the clips.

The currently frozen anime pilot has **zero admitted video labels**: its second
seed is deliberately unauthorized until a first-seed full-AV positive result.
The independent second-seed path keeps the original plan/index/manifests intact.
After the first seed's two full-AV reviews are locked and unanimously positive,
`local/anime_battle_formal/positive_pilot_v1/authorize_second_seed.py` can
create a one-time authorization for an explicit scene subset. A separate
second-seed runner and integrity verifier bind that authorization to the
original frozen manifests and actual execution receipts. The v1 admission
gate requires this chain; a different plan hash cannot masquerade as the
original. No real authorization or second-seed label has been created. Do not
edit the frozen plan or train on authored drafts to bypass this gate. The
transport helper is reusable after genuine evidence is admitted.
For the token-identity objectives, validation `token_target` diagnostics now
separate source-to-target token MSE, corrected MSE, already-good identity MSE,
pairwise correction improvement rate, and the theoretical minimum MSE imposed
by fixed alpha and per-token magnitude matching. The lower bound assumes an
arbitrary predictor; it is neither a trained-bridge result nor a video-quality
score. Keep `chunk_tokens=0` in base/bridge inference comparisons and pin the
exact H3 text-encoder SHA-256 across arms.
The bridge still outputs
exactly the source token count, so inserted concepts may be impossible to
recover. Equal tokenizer IDs do not guarantee identical contextual embeddings;
the objective does not enforce exact identity on unchanged bad-prompt spans.
Only same-prompt, same-seed H3 video review can establish visible benefit.
