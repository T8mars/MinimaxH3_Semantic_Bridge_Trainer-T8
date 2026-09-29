"""Explicit train-only representation fit. No discovery, dataset aggregates or H3 runtime.

Defaults to validation/dry-run. --execute is a distinct, intentional optimization action.
Uses the public project Transformer, serializer, normalizer and deployment formula.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

# CPU-only implementation, including real representation fits.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from wushu_bridge.bridge_model import BridgeConfig, build_bridge, load_bridge, save_bridge
from wushu_bridge.apply import apply_bridge_to_conditioning, magnitude_match, rms_normalize


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rawsha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def bound_file(binding, base=None):
    p = Path(binding["path"])
    if not p.is_absolute():
        if base is None:
            raise ValueError("Relative binding needs its owning manifest/receipt directory")
        p = Path(base) / p
    p = p.resolve()
    if not p.is_file() or sha(p) != binding["sha256"]:
        raise ValueError(f"Missing or changed bound file: {p}")
    return p


def receipt_sources(receipt, receipt_path):
    """Relative source paths belong to the receipt, independent of process cwd."""
    return {str(bound_file(dict(path=p, sha256=digest), Path(receipt_path).parent)): digest
            for p, digest in receipt.get("source_bindings", {}).items()}


def write_json(path, data):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


@dataclass
class Config:
    steps: int = 300
    seed: int = 1234
    lr: float = 3e-4
    weight_decay: float = .01
    batch_size: int = 4
    hidden: int = 256
    layers: int = 2
    heads: int = 4
    max_tokens: int = 4096
    alpha: float = 1.
    y_identity_weight: float = 1.
    x_identity_weight: float = 0.
    x_identity_mask: str | None = None
    threads: int = 2

    def validate(self):
        for name in ("steps", "seed", "batch_size", "hidden", "layers", "heads", "max_tokens", "threads"):
            v = getattr(self, name)
            if type(v) is not int or v < (0 if name == "seed" else 1):
                raise ValueError(f"Invalid {name}")
        for name in ("lr", "weight_decay", "alpha", "y_identity_weight", "x_identity_weight"):
            v = getattr(self, name)
            if type(v) not in (int, float) or not math.isfinite(v) or v < 0:
                raise ValueError(f"Invalid {name}")
        if not 0 < self.alpha <= 1 or self.lr == 0 or self.hidden % self.heads or self.hidden % 2:
            raise ValueError("Invalid alpha/lr/hidden-head configuration")
        if self.x_identity_weight and not self.x_identity_mask:
            raise ValueError("x identity requires an explicit named mask")


def load_manifest(path, cfg):
    """Only open declared train singleton archives after checking text/context/receipt."""
    path = Path(path).resolve()
    manifest = read_json(path)
    if manifest.get("schema") != 1 or manifest.get("kind") != "explicit_train_target_manifest_v1":
        raise ValueError("Unsupported train manifest")
    if manifest.get("evidence_class") not in ("synthetic_test", "train_fit_probe"):
        raise ValueError("Explicit evidence_class required")
    records = manifest.get("pairs")
    if not isinstance(records, list) or not records:
        raise ValueError("Explicit nonempty pairs required")
    # Reject ALL split errors before any tensor archive is opened.
    if any(p.get("split") != "train" for p in records):
        raise ValueError("Only explicitly declared train pairs are accepted")
    ids, seen_x, seen_raw_x, loaded, encoder_ids = set(), set(), set(), [], set()
    for row in records:
        pid = row["pair_id"]
        if not isinstance(pid, str) or not pid or pid in ids:
            raise ValueError("Empty/duplicate pair_id")
        ids.add(pid)
        if not row.get("work_id") or not row.get("group_id"):
            raise ValueError("work_id/group_id required for honest coverage reporting")
        for key in ("prompt_x", "prompt_y", "context", "target_receipt", "arrays"):
            row[key] = {**row[key], "path": str(bound_file(row[key], path.parent))}
        context = read_json(row["context"]["path"])
        if context.get("pair_id") != pid or context.get("split") != "train":
            raise ValueError("Context pair/split mismatch")
        for side in ("x", "y"):
            if context.get(f"prompt_{side}_sha256") != row[f"prompt_{side}"]["sha256"]:
                raise ValueError("Context prompt binding mismatch")
        if not context.get("encoder_id") or not context.get("source_precision"):
            raise ValueError("Encoder identity and source precision labels required")
        encoder_ids.add(context["encoder_id"])
        receipt = read_json(row["target_receipt"]["path"])
        if receipt.get("pair_id") != pid:
            raise ValueError("Target receipt pair mismatch")
        expected_splits = {"test_fixture", "synthetic_test"} if manifest["evidence_class"] == "synthetic_test" else {"train"}
        if receipt.get("source_split") not in expected_splits:
            raise ValueError("Target receipt source split mismatch")
        output = receipt["output"]
        if (bound_file(output, Path(row["target_receipt"]["path"]).parent) != Path(row["arrays"]["path"]).resolve()
                or output["sha256"] != row["arrays"]["sha256"]):
            raise ValueError("Target receipt output binding mismatch")
        keys = row.get("array_keys", {"x": "x", "y": "y", "target": "target"})
        if set(keys) != {"x", "y", "target"} or len(set(keys.values())) != 3:
            raise ValueError("Explicit distinct x/y/target array keys required")
        masks = row.get("masks", {})
        if not isinstance(masks, dict) or "all" in masks:
            raise ValueError("masks must map region names to keys; all is reserved")
        allowed = set(keys.values()) | set(masks.values()) | {"monotone_target"}
        with np.load(row["arrays"]["path"], allow_pickle=False) as z:
            # Reject aggregate archives from their directory, before loading any member.
            if set(z.files) - allowed:
                raise ValueError("Unexpected archive keys: aggregate NPZs are prohibited")
            arrays = {k: np.array(z[v], copy=True) for k, v in keys.items()}
            mask_arrays = {k: np.array(z[v], copy=True) for k, v in masks.items()}
        for name, a in arrays.items():
            if a.ndim != 2 or min(a.shape) < 1 or a.dtype not in (np.float16, np.float32) or not np.isfinite(a).all():
                raise ValueError(f"Invalid native {name} array")
            if len(a) > cfg.max_tokens:
                raise ValueError("Native sequence exceeds max_tokens; truncation prohibited")
            if rawsha(a) != row["raw_sha256"][name]:
                raise ValueError(f"Raw {name} SHA mismatch")
        x, y, target = (arrays[n] for n in ("x", "y", "target"))
        if target.dtype != np.float32 or target.shape != x.shape or y.shape[1] != x.shape[1]:
            raise ValueError("Target must be FP32 with native x shape; x/y dimensions must agree")
        if manifest["evidence_class"] != "synthetic_test" and x.shape[1] != 5120:
            raise ValueError("Real H3 train targets require dim=5120")
        for side in ("x", "y"):
            if context.get(f"{side}_raw_sha256") != row["raw_sha256"][side]:
                raise ValueError("Context native raw SHA mismatch")
            if (receipt.get(f"{side}_raw_sha256") != row["raw_sha256"][side]
                    or receipt.get(f"prompt_{side}_sha256") != row[f"prompt_{side}"]["sha256"]):
                raise ValueError("Target receipt native/prompt binding mismatch")
        if receipt.get("source_precision") != context["source_precision"]:
            raise ValueError("Receipt/context precision mismatch")
        if context["source_precision"] != {side: str(arrays[side].dtype) for side in ("x", "y")}:
            raise ValueError("Precision label disagrees with actual native arrays")
        receipt_sources(receipt, row["target_receipt"]["path"])
        fingerprint = row["prompt_x"]["sha256"]
        if fingerprint in seen_x or row["raw_sha256"]["x"] in seen_raw_x:
            raise ValueError("Duplicate x prompt; ambiguous/reweighted supervision refused")
        seen_x.add(fingerprint)
        seen_raw_x.add(row["raw_sha256"]["x"])
        for name, mask in mask_arrays.items():
            if mask.dtype != np.bool_ or mask.shape != (len(x),):
                raise ValueError(f"Mask {name} must be native-x bool vector")
        mask_arrays["all"] = np.ones(len(x), dtype=np.bool_)
        weights = row.get("region_weights", {"all": 1.})
        if not isinstance(weights, dict) or not weights:
            raise ValueError("Nonempty region weights required")
        for name, w in weights.items():
            if name not in mask_arrays or type(w) not in (int, float) or not math.isfinite(w) or w < 0:
                raise ValueError("Invalid region weight")
            if w > 0 and not mask_arrays[name].any():
                raise ValueError("Positive-weight empty region")
        if not any(w > 0 for w in weights.values()):
            raise ValueError("At least one active target region required")
        if cfg.x_identity_weight and (cfg.x_identity_mask not in mask_arrays or not mask_arrays[cfg.x_identity_mask].any()):
            raise ValueError("Missing/empty x identity mask")
        baseline = float(np.mean((target.astype(np.float64) - x.astype(np.float64)) ** 2))
        if baseline <= 1e-8:
            raise ValueError("Nonzero target correction required")
        loaded.append(dict(row=row, arrays={k: torch.from_numpy(a.astype(np.float32)) for k, a in arrays.items()},
                           masks={k: torch.from_numpy(a) for k, a in mask_arrays.items()},
                           weights=weights, baseline=baseline, context=context,
                           source_dtypes={k: str(a.dtype) for k, a in arrays.items()}))
    if len(encoder_ids) != 1 or len({p["arrays"]["x"].shape[1] for p in loaded}) != 1:
        raise ValueError("A shared network requires one encoder identity and embedding dimension")
    return manifest, loaded


def contract(alpha):
    return dict(schema=1, alpha_min=alpha, alpha_max=alpha, magnitude_match="per_token",
                token_span="all", tail_ratio=1., chunk_tokens=0, auto_alpha=False,
                guard=False, allow_dim_mismatch=False)


def create_model(dim, cfg, extra=None):
    torch.manual_seed(cfg.seed)
    model = build_bridge(BridgeConfig(arch="trans", dim=dim, hidden=cfg.hidden,
        layers=cfg.layers, heads=cfg.heads, dropout=0., max_tokens=cfg.max_tokens,
        residual_skip=True, residual_scale=1., name="explicit_train_target_fit",
        extra={**(extra or {}), "application_contract": contract(cfg.alpha)}))
    torch.nn.init.zeros_(model.net.out_proj.weight)
    torch.nn.init.zeros_(model.net.out_proj.bias)
    return model.cpu().float()


def deployed(model, native, alpha):
    h = native.unsqueeze(0)
    pred = magnitude_match(model(rms_normalize(h)), h, "per_token")
    return (h + alpha * (pred-h))[0]


def region_mse(pred, target, mask):
    if mask.dtype != torch.bool or mask.shape != pred.shape[:1] or not bool(mask.any()):
        raise ValueError("Loss requires a nonempty bool token mask")
    return (pred[mask] - target[mask]).square().mean()


def pair_loss(model, pair, cfg):
    a = pair["arrays"]
    pred = deployed(model, a["x"], cfg.alpha)
    good = deployed(model, a["y"], cfg.alpha)
    denominator = max(pair["baseline"], 1e-4)
    target_loss = sum(w * region_mse(pred, a["target"], pair["masks"][name])
                      for name, w in pair["weights"].items() if w > 0)
    identity = (good-a["y"]).square().mean()
    protect = (region_mse(pred, a["x"], pair["masks"][cfg.x_identity_mask])
               if cfg.x_identity_weight else pred.sum()*0.)
    loss = (target_loss + cfg.y_identity_weight*identity + cfg.x_identity_weight*protect) / denominator
    metrics = dict(pair_id=pair["row"]["pair_id"], baseline_mse=pair["baseline"],
        target_mse=float((pred-a["target"]).square().mean().detach()),
        y_identity_mse=float(identity.detach()), x_protection_mse=float(protect.detach()),
        normalized_loss=float(loss.detach()),
        regions={name: dict(target_mse=float(region_mse(pred, a["target"], m).detach()),
                            x_drift_mse=float(region_mse(pred, a["x"], m).detach()), tokens=int(m.sum()))
                 for name, m in pair["masks"].items() if bool(m.any())})
    metrics["normalized_target_error"] = metrics["target_mse"] / denominator
    metrics["normalized_y_identity_error"] = metrics["y_identity_mse"] / denominator
    return loss, metrics


def evaluate(model, pairs, cfg):
    model.eval()
    with torch.no_grad():
        return [pair_loss(model, p, cfg)[1] for p in pairs]


def input_bindings(manifest, manifest_path, manifest_sha):
    bindings = {str(Path(manifest_path).resolve()): manifest_sha}
    for row in manifest["pairs"]:
        for name in ("prompt_x", "prompt_y", "context", "target_receipt", "arrays"):
            b = row[name]
            bindings[str(bound_file(b).resolve())] = b["sha256"]
        receipt = read_json(row["target_receipt"]["path"])
        bindings.update(receipt_sources(receipt, row["target_receipt"]["path"]))
    return bindings


def verify_bindings(bindings):
    for path, digest in bindings.items():
        bound_file(dict(path=path, sha256=digest))


def run(manifest_path, cfg, output=None, execute=False, manifest_sha=None):
    cfg.validate()
    torch.set_num_threads(cfg.threads)
    torch.use_deterministic_algorithms(True)
    frozen_manifest_sha = sha(manifest_path)
    if manifest_sha is not None and frozen_manifest_sha != manifest_sha:
        raise ValueError("Explicit manifest SHA mismatch")
    manifest, pairs = load_manifest(manifest_path, cfg)
    sources = input_bindings(manifest, manifest_path, frozen_manifest_sha)
    verify_bindings(sources)
    code = {str(p): sha(p) for p in (Path(__file__), ROOT/"wushu_bridge/bridge_model.py", ROOT/"wushu_bridge/apply.py")}
    plan = dict(kind="explicit_train_only_fit", evidence_class=manifest["evidence_class"],
        manifest=dict(path=str(Path(manifest_path).resolve()), sha256=frozen_manifest_sha), code=code,
        input_bindings=sources,
        config=asdict(cfg), device="cpu", dtype="float32", pair_count=len(pairs),
        work_count=len({p["row"]["work_id"] for p in pairs}), val_test_loaded=False,
        aggregate_npz_loaded=False, target_construction_performed=False, video_quality_evidence=False,
        pairs=[dict(pair_id=p["row"]["pair_id"], shape=list(p["arrays"]["x"].shape),
                    y_shape=list(p["arrays"]["y"].shape), source_dtypes=p["source_dtypes"],
                    baseline_mse=p["baseline"], region_weights=p["weights"],
                    x_identity_target_conflict_mse=(float(region_mse(p["arrays"]["target"], p["arrays"]["x"],
                         p["masks"][cfg.x_identity_mask])) if cfg.x_identity_weight else None)) for p in pairs])
    if not execute:
        return {**plan, "status": "dry_run_validated_no_model_or_optimizer_created"}
    if output is None:
        raise ValueError("--output is required with --execute")
    folder = Path(output).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder/"PLAN.json", plan)
    # Provenance metadata contains hashes/config only, never training arrays or retrieval keys.
    model = create_model(pairs[0]["arrays"]["x"].shape[1], cfg,
                         dict(semantic_objective="explicit_region_target_identity_v1", evidence_class=manifest["evidence_class"],
                              manifest_sha256=frozen_manifest_sha, video_ab_required=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    rng = random.Random(cfg.seed)
    order, cursor = [], 0
    checkpoints = {0, 50, 100, 200, cfg.steps}
    write_json(folder/"STEP_000000.json", dict(step=0, pairs=evaluate(model, pairs, cfg)))
    with (folder/"OPTIMIZATION.jsonl").open("x", encoding="utf-8") as log:
        for step in range(1, cfg.steps+1):
            if cursor >= len(order):
                order = list(range(len(pairs)))
                rng.shuffle(order)
                cursor = 0
            selected = order[cursor:cursor+cfg.batch_size]
            cursor += len(selected)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total = 0.
            # Microbatch per native sequence avoids padding, preserves token context.
            for i in selected:
                loss, _ = pair_loss(model, pairs[i], cfg)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite loss; no final model exported")
                (loss / len(selected)).backward()
                total += float(loss.detach()) / len(selected)
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            log.write(json.dumps(dict(step=step, pair_ids=[pairs[i]["row"]["pair_id"] for i in selected],
                                      loss=total, grad_norm=float(grad), lr=cfg.lr), allow_nan=False)+"\n")
            log.flush()
            if step in checkpoints:
                write_json(folder/f"STEP_{step:06d}.json", dict(step=step, pairs=evaluate(model, pairs, cfg)))
    model.eval()
    # Arrays were held in RAM; do not export against inputs/code changed during optimization.
    verify_bindings({**sources, **code})
    weight = folder/"bridge.safetensors"
    save_bridge(str(weight), model, {"steps": cfg.steps, "selection": "fixed_final_step_no_validation_selection"})
    reloaded = load_bridge(str(weight), device="cpu", dtype=torch.float32)
    errors = []
    for pair in pairs:
        native = pair["arrays"]["x"].unsqueeze(0)
        with torch.no_grad():
            expected = deployed(model, native[0], cfg.alpha).unsqueeze(0)
        actual, _ = apply_bridge_to_conditioning([[native, {"sentinel": "retained"}]], reloaded, alpha=cfg.alpha)
        error = float((expected-actual[0][0]).abs().max())
        if error > 1e-6 or actual[0][0].shape != native.shape or actual[0][1]["sentinel"] != "retained":
            raise AssertionError("Saved-weight deployment reload mismatch")
        errors.append(dict(pair_id=pair["row"]["pair_id"], max_abs_error=error))
    result = dict(status="completed_train_fit_only", evidence_class=manifest["evidence_class"],
                  steps=cfg.steps, weights=dict(path=str(weight), sha256=sha(weight)),
                  reload_checks=errors, final_pairs=evaluate(reloaded, pairs, cfg),
                  video_quality_evidence=False, generalization_evidence=False, val_test_loaded=False)
    write_json(folder/"RESULT.json", result)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--manifest-sha", help="Optional explicit preview-to-execute manifest SHA256 gate")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Run CPU optimization; default only validates declared inputs")
    mode.add_argument("--dry-run", action="store_true", help="Explicitly select validation only (the default)")
    for name, default in asdict(Config()).items():
        ap.add_argument("--"+name.replace("_", "-"), type=(str if default is None else type(default)), default=default)
    args = vars(ap.parse_args())
    args.pop("dry_run")
    manifest, output, execute, manifest_sha = (args.pop(k) for k in ("manifest", "output", "execute", "manifest_sha"))
    print(json.dumps(run(manifest, Config(**args), output, execute, manifest_sha), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
