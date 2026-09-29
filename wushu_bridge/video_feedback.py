"""Opt-in, CPU-only provenance gate for externally reviewed H3 text preferences.

Hash and schema checks cannot prove that a reviewer actually watched/listened;
they reject missing, stale, static-only and disagreeing evidence before ingest.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


LABEL = "video_feedback_v1"
SCOPE = "normal_speed_full_picture_and_listened_audio"


def _need(value, message):
    if not value:
        raise ValueError("video_feedback_v1: " + message)


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _hex(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _file(root, reference):
    _need(isinstance(reference, dict) and _hex(reference.get("sha256")), "missing bound SHA-256")
    raw = reference.get("path")
    _need(isinstance(raw, str) and raw and not Path(raw).is_absolute(), "evidence paths must be relative")
    root = Path(root).resolve(strict=True)
    _need(root.is_dir(), "missing evidence root")
    current = root
    for part in Path(raw).parts:
        _need(part not in ("", ".", "..") and ":" not in part, "unsafe evidence path")
        current = current / part
        _need(not current.is_symlink(), "linked evidence is forbidden")
    path = current.resolve(strict=True)
    _need(path.is_relative_to(root) and path.is_file(), "evidence file missing/outside root")
    _need(_sha(path) == reference["sha256"], "evidence hash mismatch: " + raw)
    return path


def _json(root, reference):
    return json.loads(_file(root, reference).read_text(encoding="utf-8-sig"))


def _index_file(root, absolute_path, expected_sha):
    """Existing pilot index uses absolute source-QA paths; pin them inside root."""
    _need(_hex(expected_sha) and isinstance(absolute_path, str), "source-QA binding missing")
    root = Path(root).resolve(strict=True)
    path = Path(absolute_path)
    _need(path.is_absolute(), "source-QA path must be frozen absolute path")
    for candidate in (path, *path.parents):
        if candidate == root:
            break
        _need(not candidate.is_symlink(), "linked source-QA evidence is forbidden")
    path = path.resolve(strict=True)
    _need(path.is_relative_to(root) and path.is_file() and _sha(path) == expected_sha,
          "bound absolute evidence changed or outside evidence root")
    return path


def verify_reviewed_seed(root, record, scene_id, prompt_hashes, plan_sha, index_sha,
                         second_authorization_sha=None, expected_manifests=None,
                         second_runner_sha=None, second_verifier_sha=None):
    """Check one base-H3 seed and two sealed full-AV reports; no media perception."""
    seed = record.get("seed")
    integrity_ref = record.get("integrity")
    integrity = _json(root, integrity_ref)
    _need(integrity.get("schema") == 1 and integrity.get("kind") == "positive_pilot_prompt_pair_integrity"
          and integrity.get("passed") is True and integrity.get("scene_id") == scene_id
          and integrity.get("seed") == seed
          and integrity.get("plan", {}).get("sha256") == plan_sha
          and integrity.get("input_index", {}).get("sha256") == index_sha
          and integrity.get("video_quality_verified") is False,
          "missing/mismatched base-H3 integrity report")
    if second_authorization_sha is not None:
        _need(integrity.get("second_seed_authorization_sha256") == second_authorization_sha,
              "second-seed integrity omits independent authorization hash")
        _need(integrity.get("verifier_sha256") == second_verifier_sha,
              "second-seed integrity verifier differs from authorized version")
        executions = record.get("second_seed_execution")
        _need(isinstance(executions, dict) and set(executions) == {"x", "y"},
              "second-seed execution receipts for x/y are required")
        for arm in ("x", "y"):
            execution = _json(root, executions[arm])
            output_receipt = execution.get("receipt", {})
            _need(execution.get("schema") == 1
                  and execution.get("kind") == "positive_pilot_second_seed_execution_v1"
                  and execution.get("status") == "generated_native_integrity_pass_quality_unreviewed"
                  and execution.get("scene_id") == scene_id and execution.get("seed") == seed
                  and execution.get("arm") == arm and execution.get("base_only") is True
                  and execution.get("authorization_sha256") == second_authorization_sha
                  and execution.get("runner_sha256") == second_runner_sha
                  and execution.get("plan_sha256") == plan_sha
                  and execution.get("index_sha256") == index_sha
                  and execution.get("manifest_sha256") == expected_manifests[arm]["sha256"]
                  and execution.get("prompt_sha256") == prompt_hashes[arm]
                  and all(_hex(output_receipt.get(k)) for k in
                          ("sample_latent_sha256", "decoded_mp4_sha256",
                           "submitted_graph_sha256", "executed_history_sha256"))
                  and output_receipt.get("decoded_mp4_sha256") ==
                  integrity.get("media", {}).get(arm + "_base", {}).get("sha256")
                  and integrity.get("second_seed_execution_sha256", {}).get(arm) == executions[arm]["sha256"],
                  "second-seed execution receipt or frozen manifest mismatch")
    for arm in ("x", "y"):
        _need(integrity.get("arms", {}).get(arm + "_base", {}).get("prompt_sha256") == prompt_hashes[arm],
              "integrity prompt mismatch")
        if expected_manifests is not None:
            _need(integrity.get("arms", {}).get(arm + "_base", {}).get("manifest_sha256")
                  == expected_manifests[arm]["sha256"], "frozen manifest mismatch")
        media = integrity.get("media", {}).get(arm + "_base", {})
        original_path = _index_file(root, media.get("copy"), media.get("sha256"))
        _need(type(media.get("bytes")) is int and media["bytes"] > 0
              and original_path.stat().st_size == media["bytes"],
              "original media byte count mismatch")
        validation = media.get("media_validation", {})
        _need(validation.get("decoded_video_frames") == 124
              and validation.get("decoded_audio_samples", 0) > 0,
              "incomplete native AV in integrity receipt")
    anonymous_path = _file(root, record.get("anonymous_manifest"))
    anonymous = json.loads(anonymous_path.read_text(encoding="utf-8-sig"))
    fidelity = _json(root, record.get("fidelity"))
    key = _json(root, record.get("sealed_key"))
    _need(anonymous.get("original_request_texts_disclosed") is False
          and anonymous.get("continuous_playback_reviewed") is False
          and anonymous.get("quality_review_performed") is False
          and fidelity.get("verified_report_sha256") == integrity_ref["sha256"]
          and key.get("status") == "assignment_sealed_before_asset_preparation"
          and key.get("verified_report_sha256") == integrity_ref["sha256"]
          and set(key.get("assignment", {})) == {"A", "B"}
          and set(key["assignment"].values()) == {"x_base", "y_base"},
          "anonymous preparation/fidelity/answer key mismatch")
    _need(set(anonymous.get("candidates", {})) == {"A", "B"}
          and set(fidelity.get("candidates", {})) == {"A", "B"},
          "anonymous x/y media receipts missing")
    for label in ("A", "B"):
        item = anonymous["candidates"][label]
        media_name = item.get("media")
        _need(media_name == label + ".mp4" and _hex(item.get("media_sha256")),
              "anonymous media binding missing")
        media_path = anonymous_path.parent / media_name
        receipt = anonymous.get("file_receipts", {}).get(media_name, {})
        _need(media_path.is_file() and not media_path.is_symlink()
              and type(receipt.get("bytes")) is int and receipt["bytes"] > 0
              and media_path.stat().st_size == receipt["bytes"]
              and receipt.get("sha256") == item["media_sha256"]
              and _sha(media_path) == item["media_sha256"],
              "anonymous media changed after review preparation")
        original = integrity["media"][key["assignment"][label]]
        _need(fidelity["candidates"][label].get("source_sha256") == original["sha256"]
              and fidelity["candidates"][label].get("anonymous_sha256") == item["media_sha256"]
              and fidelity["candidates"][label].get("passed") is True,
              "source-to-anonymous media fidelity mismatch")
    visual_refs = record.get("visual_reviewers")
    _need(isinstance(visual_refs, list) and len(visual_refs) == 2,
          "two locked independent static-visual precursor reports required")
    visual_reviews = [_json(root, item) for item in visual_refs]
    visual_ids = []
    for review in visual_reviews:
        _need(review.get("schema") == 1 and review.get("kind") == "independent_static_visual_review_v1"
              and review.get("scene_id") == scene_id and review.get("seed") == seed
              and review.get("integrity_sha256") == integrity_ref["sha256"]
              and review.get("anonymous_manifest_sha256") == record["anonymous_manifest"]["sha256"]
              and review.get("answer_key_visible_during_review") is False
              and review.get("static_visual_candidate") is True
              and review.get("serious_regression") is False
              and review.get("full_av_reviewed") is False
              and review.get("verdict") in ("A_better", "B_better")
              and isinstance(review.get("primary_target_observation"), str)
              and review["primary_target_observation"].strip()
              and all(review.get("inspected_frame_ids", {}).get(label) == list(range(124))
                      for label in ("A", "B")),
              "first-stage visual candidate was not independently locked")
        visual_ids.append(review.get("reviewer_id"))
        _need(key["assignment"][review["verdict"][0]] == "y_base",
              "visual precursor does not prefer y")
    _need(all(isinstance(i, str) and i for i in visual_ids) and visual_ids[0] != visual_ids[1],
          "visual reviewers must be distinct")
    visual_lock = _json(root, record.get("visual_lock"))
    _need(visual_lock.get("schema") == 1
          and visual_lock.get("kind") == "independent_static_visual_review_lock_v1"
          and visual_lock.get("scene_id") == scene_id and visual_lock.get("seed") == seed
          and visual_lock.get("reports_locked_before_key_reveal") is True
          and visual_lock.get("reviewer_report_sha256") == [r["sha256"] for r in visual_refs]
          and visual_lock.get("sealed_key_sha256") == record["sealed_key"]["sha256"],
          "visual precursor reports not bound before key reveal")
    reviewer_refs = record.get("reviewers")
    _need(isinstance(reviewer_refs, list) and len(reviewer_refs) == 2,
          "two independent full-AV reviewer receipts required")
    reviewers = [_json(root, item) for item in reviewer_refs]
    ids = []
    for reviewer in reviewers:
        _need(reviewer.get("schema") == 1 and reviewer.get("kind") == "independent_full_av_review_v1"
              and reviewer.get("scene_id") == scene_id and reviewer.get("seed") == seed
              and reviewer.get("integrity_sha256") == integrity_ref["sha256"]
              and reviewer.get("anonymous_manifest_sha256") == record["anonymous_manifest"]["sha256"]
              and reviewer.get("visual_lock_sha256") == record["visual_lock"]["sha256"]
              and reviewer.get("review_scope") == SCOPE
              and reviewer.get("both_full_clips_normal_speed_watched") is True
              and reviewer.get("both_audio_tracks_listened") is True
              and reviewer.get("answer_key_visible_during_review") is False
              and reviewer.get("serious_regression") is False
              and reviewer.get("verdict") in ("A_better", "B_better")
              and isinstance(reviewer.get("time_indexed_av_observations"), dict)
              and all(isinstance(reviewer["time_indexed_av_observations"].get(c), list)
                      and reviewer["time_indexed_av_observations"][c] for c in ("A", "B")),
              "static-only, unreviewed, tied or regressed review")
        ids.append(reviewer.get("reviewer_id"))
        _need(key["assignment"][reviewer["verdict"][0]] == "y_base", "review does not prefer y")
    _need(all(isinstance(i, str) and i for i in ids) and ids[0] != ids[1],
          "reviewers must be distinct")
    lock = _json(root, record.get("review_lock"))
    _need(lock.get("schema") == 1 and lock.get("kind") == "independent_full_av_review_lock_v1"
          and lock.get("scene_id") == scene_id and lock.get("seed") == seed
          and lock.get("reports_locked_before_key_reveal") is True
          and lock.get("reviewer_report_sha256") == [r["sha256"] for r in reviewer_refs]
          and lock.get("sealed_key_sha256") == record["sealed_key"]["sha256"],
          "independent reports not bound before answer-key reveal")
    return integrity


def verify_second_seed_authorization(root, reference, plan_ref, index_ref, plan, index):
    """Validate a separate post-first-seed authorization without modifying freeze."""
    authorization = _json(root, reference)
    _need(authorization.get("schema") == 1
          and authorization.get("kind") == "positive_pilot_second_seed_authorization_v1"
          and authorization.get("status") == "authorized_for_listed_scenes_only"
          and authorization.get("original_plan") == plan_ref
          and authorization.get("original_index") == index_ref
          and authorization.get("original_plan_sha256") == plan_ref["sha256"]
          and authorization.get("original_index_sha256") == index_ref["sha256"]
          and authorization.get("first_seed") == plan.get("first_seed")
          and authorization.get("second_seed") == plan.get("conditional_second_seed")
          and plan.get("conditional_second_seed_generation_authorized") is False
          and index.get("conditional_second_seed_generation_authorized") is False,
          "second-seed authorization does not extend original frozen plan")
    _file(root, authorization.get("second_seed_runner"))
    _file(root, authorization.get("second_seed_verifier"))
    allowed = authorization.get("allowed_scenes")
    _need(isinstance(allowed, list) and allowed and len(allowed) <= len(index.get("scenes", [])),
          "empty or oversized second-seed scene subset")
    seen = set()
    for entry in allowed:
        scene_id = entry.get("scene_id")
        _need(isinstance(scene_id, str) and scene_id not in seen, "duplicate/invalid authorized scene")
        seen.add(scene_id)
        matches = [s for s in index["scenes"] if s.get("scene_id") == scene_id]
        _need(len(matches) == 1, "authorized scene absent from original denominator")
        scene = matches[0]
        manifests = entry.get("second_seed_manifests")
        _need(isinstance(manifests, dict) and set(manifests) == {"x", "y"},
              "both original second-seed manifests are required")
        prompt_hashes, first_manifests = {}, {}
        for arm in ("x", "y"):
            expected = scene["arms"][arm]["manifests_by_seed"][str(plan["conditional_second_seed"])]
            _need(manifests[arm]["sha256"] == expected["sha256"],
                  "second-seed manifest differs from original frozen index")
            manifest_path = _file(root, manifests[arm])
            original_manifest = (_file(root, index_ref).parent / expected["path"]).resolve()
            _need(manifest_path == original_manifest,
                  "second-seed manifest is not the original frozen file")
            prompt_hashes[arm] = scene["arms"][arm]["frozen_prompt_sha256"]
            first_path = _file(root, index_ref).parent / scene["arms"][arm]["manifest_path"]
            _need(first_path.is_file() and _sha(first_path) == scene["arms"][arm]["manifest_sha256"],
                  "original first-seed manifest changed")
            first_manifests[arm] = {"sha256": scene["arms"][arm]["manifest_sha256"]}
        first = entry.get("first_seed_evidence")
        _need(isinstance(first, dict) and first.get("seed") == plan["first_seed"],
              "first-seed positive evidence missing")
        verify_reviewed_seed(root, first, scene_id, prompt_hashes,
                             plan_ref["sha256"], index_ref["sha256"],
                             expected_manifests=first_manifests)
    _need(authorization.get("allowed_scene_ids") == [e["scene_id"] for e in allowed],
          "allowed-scene subset order/count mismatch")
    return authorization


def verify_video_feedback_row(row, evidence_root):
    """Validate a positive preference against frozen inputs and four full-AV reviews.

    This is an evidence *integrity* gate, not independent validation of human
    perception or video quality. It intentionally requires external files.
    """
    _need(evidence_root is not None, "explicit evidence root is required")
    _need(row.get("label_type") == LABEL and row.get("supervision_type") == "preference_only"
          and row.get("pair_preference") == "good" and row.get("score_label_type") == "none",
          "wrong preference label metadata")
    _need("bad_score" not in row and "good_score" not in row, "numeric scores are forbidden")
    binding = row.get("video_feedback")
    _need(isinstance(binding, dict), "missing external admission binding")
    receipt = _json(evidence_root, binding)
    _need(receipt.get("schema") == 1 and receipt.get("kind") == "semantic_bridge_video_feedback_v1"
          and receipt.get("status") == "two_seed_video_preference_admitted"
          and receipt.get("review_scope") == SCOPE
          and receipt.get("bridge_applied_in_label_videos") is False,
          "missing full-AV admitted receipt")
    _need(all(isinstance(row.get(k), str) and row[k] == receipt.get(k) and row[k]
              for k in ("scene_id", "scene_family_id", "work_id")), "scene/work identity mismatch")
    _need(row.get("group_id") == row["work_id"], "group_id must be the whole work")
    freeze = receipt.get("freeze")
    _need(isinstance(freeze, dict), "missing frozen source binding")
    plan = _json(evidence_root, freeze.get("plan"))
    index = _json(evidence_root, freeze.get("index"))
    _need(plan.get("schema") == index.get("schema") == 1
          and plan.get("status") == "frozen_before_any_h3_render"
          and plan.get("input_index_sha256") == freeze["index"]["sha256"]
          and index.get("bridge_applied") is False,
          "frozen plan/index mismatch")
    authorization_ref = receipt.get("second_seed_authorization")
    if plan.get("conditional_second_seed_generation_authorized") is True:
        _need(authorization_ref is None, "pre-authorized plan cannot claim a second authorization")
        authorization = None
    else:
        _need(authorization_ref is not None,
              "second seed was not frozen-authorized; independent authorization required")
        authorization = verify_second_seed_authorization(
            evidence_root, authorization_ref, freeze["plan"], freeze["index"], plan, index)
    _need(_hex(plan.get("protocol_sha256")), "missing frozen protocol hash")
    protocol = freeze.get("protocol")
    _need(isinstance(protocol, dict) and protocol.get("sha256") == plan["protocol_sha256"],
          "frozen protocol binding mismatch")
    _file(evidence_root, protocol)
    scenes = [s for s in index.get("scenes", []) if s.get("scene_id") == row["scene_id"]]
    _need(len(scenes) == 1, "scene absent or duplicated in frozen index")
    scene = scenes[0]
    _need(scene.get("franchise_id") == row["work_id"]
          and scene.get("source_backed_scope") == receipt.get("source_backed_scope")
          and scene.get("original_staging_scope") == receipt.get("original_staging_scope")
          and all(isinstance(scene.get(k), str) and scene[k] for k in
                  ("source_backed_scope", "original_staging_scope")),
          "source-QA or hypothetical staging scope mismatch")
    for kind in ("source_qa", "source_ledger"):
        _need(freeze.get(kind) == {"path": scene[kind + "_path"],
                                    "sha256": scene[kind + "_sha256"]},
              "frozen source-QA/ledger reference mismatch")
        _index_file(evidence_root, scene[kind + "_path"], scene[kind + "_sha256"])
    frozen_dir = _file(evidence_root, freeze["index"]).parent
    prompt_hashes = {}
    for arm, field in (("x", "bad"), ("y", "good")):
        info = scene.get("arms", {}).get(arm, {})
        target = frozen_dir / info.get("frozen_prompt_path", "")
        _need(target.is_file() and target.resolve().is_relative_to(frozen_dir.resolve())
              and not target.is_symlink(), "missing frozen prompt")
        digest = _sha(target)
        _need(digest == info.get("frozen_prompt_sha256")
              and digest == hashlib.sha256(row[field].encode("utf-8")).hexdigest()
              and target.read_bytes() == row[field].encode("utf-8")
              and receipt.get("prompt_sha256", {}).get(arm) == digest,
              "exact x/y frozen prompt bytes mismatch")
        prompt_hashes[arm] = digest
    _need(row["bad"] != row["good"], "identical x/y text")
    _need(receipt.get("target") == scene.get("target"), "predeclared target mismatch")
    seeds = receipt.get("seeds")
    expected_seeds = [plan.get("first_seed"), plan.get("conditional_second_seed")]
    _need(isinstance(seeds, list) and len(seeds) == 2
          and [s.get("seed") for s in seeds] == expected_seeds
          and len(set(expected_seeds)) == 2, "exact two frozen seeds required")
    if authorization is not None:
        allowed = {entry["scene_id"]: entry for entry in authorization["allowed_scenes"]}
        _need(row["scene_id"] in allowed, "scene was not released for second seed")
        _need(seeds[0] == allowed[row["scene_id"]]["first_seed_evidence"],
              "admission first seed differs from independently authorized positive")
    for position, record in enumerate(seeds):
        manifests = {arm: {"sha256": scene["arms"][arm]["manifest_sha256"]}
                     for arm in ("x", "y")}
        authorization_sha = None
        if position == 1:
            manifests = (allowed[row["scene_id"]]["second_seed_manifests"]
                         if authorization is not None else
                         {arm: scene["arms"][arm]["manifests_by_seed"][str(record["seed"])]
                          for arm in ("x", "y")})
            if authorization is not None:
                authorization_sha = authorization_ref["sha256"]
        verify_reviewed_seed(evidence_root, record, row["scene_id"], prompt_hashes,
                             freeze["plan"]["sha256"], freeze["index"]["sha256"],
                             authorization_sha, manifests,
                             authorization["second_seed_runner"]["sha256"] if authorization_sha else None,
                             authorization["second_seed_verifier"]["sha256"] if authorization_sha else None)
    return receipt
