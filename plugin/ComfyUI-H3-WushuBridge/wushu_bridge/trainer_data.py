"""Validated text pairs, resumable H3 encoding and leakage-aware partitions."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from .dataset import PairDataset, PairMeta
from .video_feedback import LABEL as VIDEO_FEEDBACK_LABEL, verify_video_feedback_row


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def text_key(text):
    return hashlib.sha256(text.replace("\r\n", "\n").strip().encode("utf-8")).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def read_pairs(path, video_feedback_root=None):
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("bad", "good"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Line {line_no}: missing nonempty {key}")
        preference = "bad_score" not in row and "good_score" not in row
        if row.get("supervision_type", "preference_only" if preference else "numeric_scores") != ("preference_only" if preference else "numeric_scores"):
            raise ValueError(f"Line {line_no}: supervision_type conflicts with score fields")
        if preference:
            if row.get("pair_preference") != "good":
                raise ValueError(f"Line {line_no}: preference-only rows require pair_preference=good")
            if row.get("label_type") == VIDEO_FEEDBACK_LABEL:
                verify_video_feedback_row(row, video_feedback_root)
            elif row.get("label_type") != "authoring_rubric":
                raise ValueError(f"Line {line_no}: unsupported preference label_type")
            if row.get("score_label_type", "none") != "none":
                raise ValueError(f"Line {line_no}: preference-only rows cannot claim score labels")
        else:
            for key in ("bad_score", "good_score"):
                value = row.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"Line {line_no}: {key} must be finite and in [0, 1]")
            if row.get("score_label_type") == "none":
                raise ValueError(f"Line {line_no}: numeric scores conflict with score_label_type=none")
        if text_key(row["bad"]) == text_key(row["good"]):
            raise ValueError(f"Line {line_no}: identical bad/good text")
        rows.append(row)
    if not rows:
        raise ValueError("No text pairs")
    pair_supervision(rows)
    labels = {r.get("label_type") for r in rows if "bad_score" not in r and "good_score" not in r}
    if len(labels) > 1:
        raise ValueError("Do not mix authoring and verified video-feedback preferences")
    return rows


def pair_supervision(rows):
    kinds = {"numeric_scores" if "bad_score" in r or "good_score" in r else "preference_only" for r in rows}
    if len(kinds) != 1:
        raise ValueError("Do not mix numeric-score and preference-only pairs in one dataset")
    return next(iter(kinds))


def inspect_pairs(rows):
    scored = pair_supervision(rows) == "numeric_scores"
    labels = {r.get("label_type") for r in rows if "bad_score" not in r and "good_score" not in r}
    return {
        "pairs": len(rows),
        "unique_good": len({text_key(r["good"]) for r in rows}),
        "unique_texts": len({text_key(r[k]) for r in rows for k in ("bad", "good")}),
        "supervision_type": pair_supervision(rows),
        "label_type": next(iter(labels)) if len(labels) == 1 else None,
        "non_increasing_labels": sum(r["bad_score"] >= r["good_score"] for r in rows) if scored else None,
        "same_binary_class": sum((r["bad_score"] > .5) == (r["good_score"] > .5) for r in rows) if scored else None,
        "provenance": ("numeric text labels; not necessarily video judgments" if scored else
                       "external two-seed, two-reviewer full-AV receipt hashes verified; human observation not machine-proven"
                       if labels == {VIDEO_FEEDBACK_LABEL} else
                       "authored text preference; no numeric or video quality labels"),
    }


def validate_dataset(ds, expected_dim=5120):
    ds.validate_supervision()
    if len(ds) < 8 or ds.meta.dim != expected_dim:
        raise ValueError(f"Need >=8 pairs with dim={expected_dim}; got {len(ds)}, dim={ds.meta.dim}")
    if not (len(ds.x) == len(ds.y) == len(ds.x_score) == len(ds.y_score) == len(ds.records)):
        raise ValueError("Mismatched pair/score/record counts")
    if ds.extra:
        raise ValueError("Unpaired samples require explicit source groups; this trainer accepts paired data only")
    for a, b, sa, sb in zip(ds.x, ds.y, ds.x_score, ds.y_score):
        for t in (a, b):
            if t.ndim != 2 or not len(t) or t.shape[1] != expected_dim or not np.isfinite(t).all():
                raise ValueError("Invalid, empty, nonfinite or inconsistent embedding")
        if ds.meta.supervision_type != "preference_only" and not all(math.isfinite(s) and 0 <= s <= 1 for s in (sa, sb)):
            raise ValueError("Invalid soft label")


def connected_groups(ds):
    """Keep shared text, identical embeddings and explicit source groups together."""
    parents = list(range(len(ds)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen = {}
    for i, record in enumerate(ds.records):
        keys = []
        for name, emb in (("bad", ds.x[i]), ("good", ds.y[i])):
            if record.get(name):
                keys.append("text:" + text_key(record[name]))
            if emb is not None:
                keys.append("embedding:" + hashlib.sha256(np.ascontiguousarray(emb).tobytes()).hexdigest())
        if record.get("group_id"):
            keys.append("group:" + str(record["group_id"]))
        for key in keys:
            if key in seen:
                parents[root(i)] = root(seen[key])
            else:
                seen[key] = i
    groups = {}
    for i in range(len(ds)):
        groups.setdefault(root(i), []).append(i)
    return list(groups.values())


def partition(ds, seed=1234, split_manifest=None):
    values = connected_groups(ds)
    if split_manifest is not None:
        if isinstance(split_manifest, (str, Path)):
            split_manifest = json.loads(Path(split_manifest).read_text(encoding="utf-8-sig"))
        if not isinstance(split_manifest, dict) or type(split_manifest.get("schema")) is not int or split_manifest.get("schema") != 1 or split_manifest.get("split_unit") != "group_id":
            raise ValueError("Split manifest requires schema=1, split_unit=group_id and splits")
        raw = split_manifest.get("splits")
        if not isinstance(raw, dict):
            raise ValueError("Split manifest splits must be an object")
        aliases = {"validation":"val", "challenge":"test", "challenge-test":"test"}
        assignments, parts = {}, {}
        for name, groups in raw.items():
            name = aliases.get(name, name)
            if name not in ("train", "val", "test", "calibration") or name in parts or not isinstance(groups, list):
                raise ValueError("Invalid or duplicate split name")
            parts[name] = []
            for group in groups:
                if not isinstance(group, str) or not group or group in assignments:
                    raise ValueError("Split group IDs must be unique nonempty strings; overlap is forbidden")
                assignments[group] = name
        actual = {r.get("group_id") for r in ds.records}
        if None in actual or "" in actual or actual != set(assignments):
            raise ValueError("Split manifest must cover every dataset group_id exactly, with no unknown groups")
        for connected in values:
            destinations = {assignments[ds.records[i]["group_id"]] for i in connected}
            if len(destinations) != 1:
                raise ValueError("Split leakage: shared text, identical embedding or group crosses partitions")
            parts[next(iter(destinations))].extend(connected)
        if any(not parts.get(k) for k in ("train", "val", "test")):
            raise ValueError("Explicit train, validation and test splits must be nonempty")
        return {k:sorted(v) for k,v in parts.items()}
    if len(values) < 8:
        raise ValueError("Need at least 8 independent groups for train/val/calibration/test")
    rng = np.random.default_rng(seed)
    rng.shuffle(values)
    count = max(1, int(round(len(values) * .1)))
    pieces = {"test": values[:count], "calibration": values[count:2*count],
              "val": values[2*count:3*count], "train": values[3*count:]}
    return {key: sorted(i for group in part for i in group) for key, part in pieces.items()}


def partition_text_pairs(rows, split_manifest):
    """Preflight group/text leakage without loading or inventing embeddings."""
    class TextPairs:
        records = rows
        x = y = [None] * len(rows)
        def __len__(self):
            return len(self.records)
    return partition(TextPairs(), split_manifest=split_manifest)


def encode_pairs(rows, clip, cache_dir, output, identity, max_tokens=0,
                 video_feedback_root=None):
    """One atomic cache entry per exact text. Survives encoder interruptions."""
    import torch
    from .nodes import _conditioning_tensor

    if any(r.get("label_type") == VIDEO_FEEDBACK_LABEL for r in rows):
        for row in rows:
            verify_video_feedback_row(row, video_feedback_root)

    identity = {**identity, "max_tokens": max_tokens}
    fingerprint = text_key(json.dumps(identity, sort_keys=True))
    cache = Path(cache_dir) / fingerprint
    cache.mkdir(parents=True, exist_ok=True)
    write_json(cache / "encoder.json", identity)
    supervision = pair_supervision(rows)
    ds = PairDataset(PairMeta(mode="t2v", encoder=json.dumps(identity, sort_keys=True),
                              source="text_pairs_jsonl", supervision_type=supervision,
                              score_label_type="none" if supervision == "preference_only" else "numeric_text_labels"))
    encoded = {}
    for i, row in enumerate(rows):
        pair = []
        for key in ("bad", "good"):
            text = row[key]
            # Cache the exact bytes: normalization only applies to split grouping.
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            path = cache / (digest + ".npy")
            if digest not in encoded:
                if path.exists():
                    arr = np.load(path, allow_pickle=False, mmap_mode="r")
                else:
                    with torch.inference_mode():
                        cond = clip.encode_from_tokens_scheduled(clip.tokenize(text))
                        tensor = _conditioning_tensor(cond)
                        if tensor.shape[0] != 1:
                            raise ValueError("Encoder must return a single sequence")
                        arr = tensor[0].detach().float().cpu().numpy()
                    if max_tokens:
                        arr = arr[:max_tokens]
                    arr = arr.astype(np.float16)
                    if arr.ndim != 2 or arr.shape[1] != 5120 or not len(arr) or not np.isfinite(arr).all():
                        raise ValueError("Expected finite H3 token embeddings [T,5120]")
                    with open(str(path) + ".tmp", "wb") as f:
                        np.save(f, arr, allow_pickle=False)
                    os.replace(str(path) + ".tmp", path)
                    arr = np.load(path, allow_pickle=False, mmap_mode="r")
                if arr.ndim != 2 or arr.shape[1] != 5120 or not len(arr) or not np.isfinite(arr).all():
                    raise ValueError(f"Corrupt encoder cache: {path}")
                encoded[digest] = arr
            pair.append(encoded[digest])
        ds.add(*pair, row.get("bad_score", float("nan")), row.get("good_score", float("nan")), row)
        print(f"[encode] {i+1}/{len(rows)} pairs", flush=True)
    validate_dataset(ds)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(f"Refusing to overwrite dataset: {output}")
    temp = output.with_name(output.stem + ".partial.npz")
    ds.save(str(temp))
    os.replace(temp.with_suffix(".json"), output.with_suffix(".json"))
    os.replace(temp, output)
    return ds
