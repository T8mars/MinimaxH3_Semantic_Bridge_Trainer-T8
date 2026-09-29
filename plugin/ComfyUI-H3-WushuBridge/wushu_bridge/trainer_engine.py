"""Repeatable small-model training with grouped holdouts and epoch checkpoints.

The bridge objective and architectures come from the author's train.py. This
runner corrects validation leakage, padding and resume behavior; it does not
claim bit-for-bit reproduction of the published metrics.
"""
from __future__ import annotations

import json
import hashlib
import math
import os
import struct
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .apply import magnitude_match, rms_normalize
from .bridge_model import BridgeConfig, build_bridge, save_bridge
from .judge import JudgeConfig, build_judge, calibrate_temperature, expected_calibration_error, save_judge
from .train import _auc, _pad_batch, masked_mean
from .trainer_data import partition, write_json
from .token_alignment import transport_edges


@dataclass
class RunConfig:
    arch: str = "trans"
    hidden: int = 256
    layers: int = 2
    heads: int = 4
    bridge_epochs: int = 150
    judge_epochs: int = 200
    batch_size: int = 8
    judge_batch_size: int = 32
    bridge_lr: float = 0.0003
    judge_lr: float = 0.001
    weight_decay: float = 0.01
    anchor_weight: float = 0.15
    abs_weight: float = 0.3
    alpha_min: float = 0.05
    alpha_max: float = 0.35
    max_seq_tokens: int = 2048
    patience: int = 12
    judge_patience: int = 20
    seed: int = 1234
    device: str = "auto"
    amp: bool = True
    magnitude_mode: str = "none"
    residual_skip: bool = False
    residual_scale: float = 0.1
    bridge_val_min_delta: float = 1e-5
    semantic_pool_scope: str = "all_valid_tokens"
    semantic_objective: str = "direction_cosine_v1"
    deployed_target_fraction: float = 0.10
    deployed_good_identity_weight: float = 1.0

    def validate(self):
        for name in ("hidden", "layers", "heads", "bridge_epochs", "judge_epochs", "batch_size", "judge_batch_size", "max_seq_tokens", "patience", "judge_patience"):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden % self.heads or self.arch not in ("trans", "mlp"):
            raise ValueError("Invalid architecture or hidden/heads")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed <= 2**32-1:
            raise ValueError("seed must be an integer in [0, 2**32-1]")
        if not isinstance(self.amp, bool):
            raise ValueError("amp must be boolean")
        for name in ("alpha_min", "alpha_max", "abs_weight", "bridge_lr", "judge_lr", "weight_decay", "anchor_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if not 0 < self.alpha_min <= self.alpha_max <= 1 or not 0 <= self.abs_weight <= 1:
            raise ValueError("Invalid alpha or loss weights")
        if self.magnitude_mode not in ("none", "per_token"):
            raise ValueError("magnitude_mode must be none or per_token")
        if self.semantic_pool_scope not in ("all_valid_tokens", "exclude_shared_first"):
            raise ValueError("semantic_pool_scope must be all_valid_tokens or exclude_shared_first")
        if self.semantic_objective not in ("direction_cosine_v1", "vector_target_v1", "aligned_token_identity_v1", "monotone_token_identity_v1", "deployed_vector_v1"):
            raise ValueError("Unknown semantic_objective")
        for name in ("deployed_target_fraction", "deployed_good_identity_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if not 0 < self.deployed_target_fraction <= 1:
            raise ValueError("deployed_target_fraction must be in (0, 1]")
        if not 0 < self.deployed_good_identity_weight <= 10:
            raise ValueError("deployed_good_identity_weight must be in (0, 10]")
        if self.semantic_objective != "deployed_vector_v1" and (
                self.deployed_target_fraction != 0.10 or self.deployed_good_identity_weight != 1.0):
            raise ValueError("deployed_* parameters require deployed_vector_v1")
        if self.semantic_objective == "deployed_vector_v1" and (
                self.alpha_min != .12 or self.alpha_max != .12
                or self.magnitude_mode != "per_token" or not self.residual_skip
                or self.anchor_weight != 0):
            raise ValueError("deployed_vector_v1 requires alpha_min=alpha_max=0.12, per_token magnitude, residual_skip and anchor_weight=0")
        if self.semantic_objective in ("aligned_token_identity_v1", "monotone_token_identity_v1") and (
                self.alpha_min != self.alpha_max or self.magnitude_mode != "per_token"
                or not self.residual_skip or self.anchor_weight != 0):
            raise ValueError("Token identity objectives require fixed alpha, per_token magnitude, residual_skip and anchor_weight=0")
        if not isinstance(self.residual_skip, bool):
            raise ValueError("residual_skip must be boolean")
        if isinstance(self.residual_scale, bool) or not isinstance(self.residual_scale, (int, float)) or not math.isfinite(self.residual_scale) or not 0 < self.residual_scale <= 1:
            raise ValueError("residual_scale must be finite and in (0, 1]")
        if isinstance(self.bridge_val_min_delta, bool) or not isinstance(self.bridge_val_min_delta, (int, float)) or not math.isfinite(self.bridge_val_min_delta) or not 0 <= self.bridge_val_min_delta < 1:
            raise ValueError("bridge_val_min_delta must be finite and in [0, 1)")
        for name in ("bridge_lr", "judge_lr", "weight_decay", "anchor_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or (name.endswith("lr") and value == 0):
                raise ValueError(f"Invalid {name}")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu or cuda")


def batch(seqs, device, limit):
    size = min(limit, max(len(s) for s in seqs))
    x, mask = _pad_batch(seqs, size)
    return x.to(device), mask.to(device)


def semantic_pool(x, mask, scope):
    """Pool the semantic target; inference still receives the full token stream."""
    if scope == "exclude_shared_first":
        if x.shape[1] < 2 or not bool(mask[:, 1:].any(dim=1).all()):
            raise ValueError("exclude_shared_first requires at least two valid tokens per sequence")
        return masked_mean(x[:, 1:], mask[:, 1:])
    if scope != "all_valid_tokens":
        raise ValueError(f"Unknown semantic pool scope: {scope}")
    return masked_mean(x, mask)


def numerically_shared_first_token(bad_embedding, good_embedding):
    """Treat one-FP16-step encoder variation as a shared first instruction."""
    if len(bad_embedding) < 2 or len(good_embedding) < 2:
        return False
    bad = bad_embedding[0].astype(np.float32)
    good = good_embedding[0].astype(np.float32)
    delta = bad - good
    max_abs = float(np.max(np.abs(delta)))
    delta_rms = float(np.sqrt(np.mean(delta * delta)))
    reference_rms = max(float(np.sqrt(np.mean(bad * bad))),
                        float(np.sqrt(np.mean(good * good))), 1.0)
    return max_abs <= .002 and delta_rms / reference_rms <= 1e-6


def transported_target(ds, indices, x, y, xm):
    """Distil native good-token vectors onto existing bad-token positions."""
    target = torch.zeros_like(x)
    for row, index in enumerate(indices):
        tx, ty = len(ds.x[index]), len(ds.y[index])
        info = ds.records[index]["token_alignment"]
        if (len(info["bad_ids"]) != tx or len(info["good_ids"]) != ty
                or tx > x.shape[1] or ty > y.shape[1]):
            raise ValueError(f"Pair {index}: transport IDs/native embedding lengths disagree")
        edges = transport_edges(info["bad_ids"], info["good_ids"])
        ii = torch.tensor([e[0] for e in edges], device=x.device, dtype=torch.long)
        jj = torch.tensor([e[1] for e in edges], device=x.device, dtype=torch.long)
        weights = torch.tensor([e[2] for e in edges], device=x.device, dtype=torch.float32)
        barycenter = torch.zeros((tx, x.shape[-1]), device=x.device, dtype=torch.float32)
        barycenter.index_add_(0, ii, rms_normalize(y[row, jj]) * weights[:, None])
        if not torch.isfinite(barycenter).all() or bool((barycenter.square().mean(-1) <= 1e-8).any()):
            raise ValueError(f"Pair {index}: near-zero or nonfinite transport barycenter")
        target[row, :tx] = magnitude_match(barycenter, x[row, :tx], "per_token")
    if not torch.isfinite(target[xm]).all():
        raise FloatingPointError("Nonfinite transported target")
    return target


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def bridge_values(model, ds, indices, cfg, alpha, magnitude_mode=None):
    if cfg.semantic_objective == "monotone_token_identity_v1" and any(
            max(len(ds.x[i]), len(ds.y[i])) > cfg.max_seq_tokens for i in indices):
        raise ValueError("Monotone token objective refuses truncated native sequences")
    device = next(model.parameters()).device
    x, mask = batch([ds.x[i] for i in indices], device, cfg.max_seq_tokens)
    y, ym = batch([ds.y[i] for i in indices], device, cfg.max_seq_tokens)
    pred = model(rms_normalize(x), mask=mask).float()
    pred = magnitude_match(pred, x, cfg.magnitude_mode if magnitude_mode is None else magnitude_mode)
    blend = x + alpha * (pred - x)
    pb = semantic_pool(x, mask, cfg.semantic_pool_scope)
    pg = semantic_pool(y, ym, cfg.semantic_pool_scope)
    pp = semantic_pool(blend, mask, cfg.semantic_pool_scope)
    dt = pg - pb
    if cfg.semantic_objective in ("aligned_token_identity_v1", "monotone_token_identity_v1"):
        if (cfg.magnitude_mode if magnitude_mode is None else magnitude_mode) != "per_token":
            raise ValueError("Token identity objective requires per_token magnitude")
        if cfg.semantic_objective == "aligned_token_identity_v1":
            if any(len(ds.x[i]) != len(ds.y[i]) for i in indices):
                raise ValueError("Aligned token objective requires equal native sequence lengths")
            target = magnitude_match(y, x, "per_token")
        else:
            target = transported_target(ds, indices, x, y, mask)
        x_denom = mask.sum(dim=1) * x.shape[-1]
        y_denom = ym.sum(dim=1) * y.shape[-1]
        baseline = (((target-x).square() * mask.unsqueeze(-1)).sum(dim=(1, 2)) / x_denom)
        if bool((baseline <= 1e-8).any()):
            raise ValueError("Token identity objective requires a nonzero token correction")
        corrected_error = (((blend-target).square() * mask.unsqueeze(-1)).sum(dim=(1, 2)) / x_denom)
        good_pred = magnitude_match(model(rms_normalize(y), mask=ym).float(), y, "per_token")
        good_blend = y + alpha * (good_pred-y)
        identity_error = (((good_blend-y).square() * ym.unsqueeze(-1)).sum(dim=(1, 2)) / y_denom)
        # Avoid amplifying FP16 cache noise in almost-identical pairs.
        sem = ((corrected_error + identity_error) / baseline.clamp_min(1e-4)).mean()
    elif cfg.semantic_objective == "deployed_vector_v1":
        if (cfg.magnitude_mode if magnitude_mode is None else magnitude_mode) != "per_token":
            raise ValueError("deployed_vector_v1 requires per_token magnitude")
        # Fit the actual deployed blend to a conservative fraction of the
        # authored pooled displacement; normalize both terms by its energy.
        target_norm_sq = dt.square().sum(dim=-1)
        valid = target_norm_sq > 1e-6
        if not bool(valid.all()):
            raise ValueError("deployed_vector_v1 requires nonzero preferred-text displacement for every pair")
        correction = ((pp - pb - cfg.deployed_target_fraction * dt).square().sum(dim=-1)
                      / target_norm_sq.clamp_min(.01))
        good_pred = magnitude_match(model(rms_normalize(y), mask=ym).float(), y, "per_token")
        good_blend = y + alpha * (good_pred - y)
        good_token_mse = (((good_blend-y).square() * ym.unsqueeze(-1)).sum(dim=(1, 2))
                          / (ym.sum(dim=1) * y.shape[-1]))
        # Convert pooled-vector energy to per-coordinate energy before
        # comparing it to the tokenwise good-input reconstruction error.
        good_identity = good_token_mse * y.shape[-1] / target_norm_sq.clamp_min(.01)
        sem = (correction + cfg.deployed_good_identity_weight * good_identity).mean()
    elif cfg.semantic_objective == "vector_target_v1":
        # Pooling is linear in the blend. Fit the full-strength correction to
        # the preferred-text displacement, so both its direction and length
        # are supervised without amplifying a tiny sampled alpha.
        update = semantic_pool(pred, mask, cfg.semantic_pool_scope) - pb
        target_norm_sq = dt.square().sum(dim=-1)
        valid = target_norm_sq > 1e-6
        if not bool(valid.any()):
            raise ValueError("vector_target_v1 requires a nonzero preferred-text displacement")
        sem = ((update[valid] - dt[valid]).square().sum(dim=-1)
               / target_norm_sq[valid].clamp_min(0.01)).mean()
    else:
        absolute = 1 - F.cosine_similarity(pp, pg, dim=-1).mean()
        dp = pp - pb
        valid = dt.norm(dim=-1) > 1e-3
        direction = 1 - F.cosine_similarity(dp[valid], dt[valid], dim=-1).mean() if valid.any() else absolute
        sem = cfg.abs_weight * absolute + (1-cfg.abs_weight) * direction
    anchor = ((pred - x).square() * mask.unsqueeze(-1)).sum() / (mask.sum() * x.shape[-1])
    return sem + cfg.anchor_weight * anchor, sem, anchor, (pb, pg, pp)


@torch.no_grad()
def evaluate_bridge(model, ds, indices, cfg, alpha=.2, magnitude_mode="per_token"):
    model.eval()
    before, after, drift, directions = [], [], [], []
    objective_sum = 0.
    for start in range(0, len(indices), cfg.batch_size):
        ids = indices[start:start+cfg.batch_size]
        _, sem, _, (pb, pg, pp) = bridge_values(model, ds, ids, cfg, alpha, magnitude_mode)
        if cfg.semantic_objective in ("vector_target_v1", "aligned_token_identity_v1", "monotone_token_identity_v1", "deployed_vector_v1"):
            objective_sum += float(sem) * len(ids)
        before.extend(F.cosine_similarity(pb, pg, dim=-1).cpu().tolist())
        after.extend(F.cosine_similarity(pp, pg, dim=-1).cpu().tolist())
        drift.extend((1-F.cosine_similarity(pb, pp, dim=-1)).cpu().tolist())
        valid = (pg-pb).norm(dim=-1) > 1e-3
        if alpha and valid.any():
            directions.extend(F.cosine_similarity((pp-pb)[valid], (pg-pb)[valid], dim=-1).cpu().tolist())
    result = {"alpha": alpha, "magnitude_mode": magnitude_mode,
              "semantic_pool_scope": cfg.semantic_pool_scope,
              "cos_before": float(np.mean(before)), "cos_after": float(np.mean(after)),
              "relative_gain": float(np.mean(after)-np.mean(before)) if alpha else 0.0,
              "drift": float(np.mean(drift)) if alpha else 0.0,
              "direction_alignment": float(np.mean(directions)) if directions else None}
    if cfg.semantic_objective in ("vector_target_v1", "aligned_token_identity_v1", "monotone_token_identity_v1", "deployed_vector_v1"):
        result["objective_loss"] = objective_sum / len(indices)
    for value in result.values():
        if isinstance(value, float) and not math.isfinite(value):
            raise FloatingPointError("Nonfinite bridge evaluation")
    return result


def judge_batch(ds, sample_ids, cfg, device):
    seqs = [ds.x[i//2] if i % 2 == 0 else ds.y[i//2] for i in sample_ids]
    labels = [ds.x_score[i//2] if i % 2 == 0 else ds.y_score[i//2] for i in sample_ids]
    x, mask = batch(seqs, device, cfg.max_seq_tokens)
    return x, mask, torch.tensor(labels, device=device)


@torch.no_grad()
def judge_predictions(model, ds, pair_ids, cfg):
    model.eval()
    sample_ids = [2*i+j for i in pair_ids for j in (0, 1)]
    logits, labels = [], []
    for start in range(0, len(sample_ids), cfg.judge_batch_size):
        x, mask, y = judge_batch(ds, sample_ids[start:start+cfg.judge_batch_size], cfg, next(model.parameters()).device)
        logits.append(model(x, mask=mask).float().cpu())
        labels.append(y.cpu())
    return torch.cat(logits), torch.cat(labels)


def judge_metrics(logits, labels, temperature=1., bias=0.):
    probs = torch.sigmoid(logits / temperature + bias)
    auc = _auc(probs, labels)
    return {"bce": float(F.binary_cross_entropy_with_logits(logits / temperature + bias, labels)),
            "accuracy_at_0_5": float(((probs > .5) == (labels > .5)).float().mean()),
            "auc_against_rule_threshold": auc if math.isfinite(auc) else None,
            "soft_label_ece": expected_calibration_error(probs, labels),
            "pairwise_good_above_bad": float((probs[1::2] > probs[::2]).float().mean()),
            "brier_against_soft_labels": float((probs-labels).square().mean()),
            "samples": len(labels)}


def save_checkpoint(path, state):
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    os.replace(temp, path)


def run_signature(cfg, identity):
    values = asdict(cfg)
    # Runs written before the opt-in residual path did not contain these keys.
    # Keep their signatures verifiable and resumable with the newer trainer.
    if not cfg.residual_skip and cfg.residual_scale == 0.1:
        values.pop("residual_skip")
        values.pop("residual_scale")
    if cfg.bridge_val_min_delta == 1e-5:
        values.pop("bridge_val_min_delta")
    if cfg.semantic_pool_scope == "all_valid_tokens":
        values.pop("semantic_pool_scope")
    if cfg.semantic_objective == "direction_cosine_v1":
        values.pop("semantic_objective")
    if cfg.deployed_target_fraction == .10:
        values.pop("deployed_target_fraction")
    if cfg.deployed_good_identity_weight == 1.0:
        values.pop("deployed_good_identity_weight")
    return hashlib.sha256(json.dumps({"config": values, "identity": identity}, sort_keys=True).encode()).hexdigest()


def validation_improves(value, selected_best, min_delta):
    """Whether an observed validation loss is enough to replace the checkpoint."""
    return value < selected_best - min_delta


def bridge_validation_loss(cfg, metrics):
    """Use each opt-in objective's deployed validation loss for selection."""
    if cfg.semantic_objective in ("vector_target_v1", "aligned_token_identity_v1",
                                  "monotone_token_identity_v1", "deployed_vector_v1"):
        return metrics["objective_loss"]
    return 1 - metrics["cos_after"]


def prepare_bridge_initialization(path, expected_sha256, cfg, dim, mode):
    """Validate one hashed CPU byte snapshot without migration or optimizer state."""
    from safetensors.torch import load
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_sha256)):
        raise ValueError("init-bridge-sha256 must be a lowercase SHA256 digest")
    path = Path(path).resolve()
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError("init-bridge SHA256 mismatch")
    # Metadata and tensors come from the exact same hashed bytes.
    state = load(raw)
    size = struct.unpack("<Q", raw[:8])[0]
    meta = json.loads(raw[8:8+size]).get("__metadata__", {})
    target = BridgeConfig(arch=cfg.arch, dim=dim, hidden=cfg.hidden, layers=cfg.layers,
                         heads=cfg.heads, mode=mode, residual_skip=cfg.residual_skip,
                         residual_scale=cfg.residual_scale)
    fields = ("arch", "dim", "hidden", "layers", "heads", "mode", "schema", "dropout", "max_tokens",
              "residual_skip", "residual_scale")
    expected = asdict(target)
    for field in fields:
        # Author v1 predates the residual options; these are its actual defaults.
        default = {"residual_skip": "False", "residual_scale": "0.1"}.get(field)
        value = meta.get(field, default)
        if value is None:
            raise ValueError(f"init-bridge missing architecture metadata: {field}")
        try:
            if field in ("dim", "hidden", "layers", "heads", "max_tokens", "schema"):
                value = int(value)
            elif field in ("dropout", "residual_scale"):
                value = float(value)
            elif field == "residual_skip":
                if value not in ("True", "False", "true", "false"):
                    raise ValueError("invalid boolean")
                value = value.lower() == "true"
        except (ValueError, TypeError) as exc:
            raise ValueError(f"init-bridge invalid architecture metadata: {field}") from exc
        if value != expected[field]:
            raise ValueError(f"init-bridge incompatible {field}: source={value!r}, required={expected[field]!r}")
    extra = json.loads(meta.get("extra_json", "{}"))
    if not isinstance(extra, dict):
        raise ValueError("init-bridge extra_json must be an object")
    if extra.get("application_contract"):
        raise ValueError("init-bridge has an application contract; migration is not supported")
    if any(not t.is_floating_point() or not bool(torch.isfinite(t).all()) for t in state.values()):
        raise ValueError("init-bridge weights must be finite floating-point tensors")
    # Strict loading checks names/shapes without changing the training RNG.
    with torch.random.fork_rng(devices=[]):
        probe = build_bridge(target)
        try:
            probe.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise ValueError(f"init-bridge incompatible weight keys/shapes: {exc}") from exc
    binding = {"path": str(path), "sha256": digest, "bytes": len(raw),
               "architecture": {key: expected[key] for key in fields},
               "method": "strict_weights_only_new_optimizer"}
    return cpu_state(probe), binding


def fit_stage(kind, ds, splits, cfg, folder, signature, resume=False, stop_after_epochs=None,
              initial_state=None, initialization=None):
    if initial_state is not None and (kind != "bridge" or resume or initialization is None):
        raise ValueError("Bridge initialization requires a new bridge stage and bound source")
    if kind == "judge":
        ds.require_score_labels()
        if not splits.get("calibration"):
            raise ValueError("Judge training requires a separate nonempty calibration split")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    is_bridge = kind == "bridge"
    model = (build_bridge(BridgeConfig(arch=cfg.arch, dim=ds.meta.dim, hidden=cfg.hidden,
                                      layers=cfg.layers, heads=cfg.heads, mode=ds.meta.mode,
                                      residual_skip=cfg.residual_skip, residual_scale=cfg.residual_scale)) if is_bridge
             else build_judge(JudgeConfig(dim=ds.meta.dim, hidden=cfg.hidden, heads=cfg.heads, dropout=.05))).to(device)
    if initial_state is not None:
        model.load_state_dict(initial_state, strict=True)  # copy into new trainable parameters
        model.requires_grad_(True)
    elif is_bridge and (cfg.semantic_objective in ("aligned_token_identity_v1", "monotone_token_identity_v1", "deployed_vector_v1")
                        or (cfg.residual_skip and cfg.semantic_objective == "vector_target_v1")):
        # A residual vector target starts at exact identity. Random output
        # projections otherwise overwhelm the tiny X→Y displacement in
        # long, nearly identical H3 prompt pairs before learning begins.
        output_layer = model.net.out_proj if cfg.arch == "trans" else model.net.fc3
        torch.nn.init.zeros_(output_layer.weight)
        torch.nn.init.zeros_(output_layer.bias)
    epochs = cfg.bridge_epochs if is_bridge else cfg.judge_epochs
    bs = cfg.batch_size if is_bridge else cfg.judge_batch_size
    train_ids = splits["train"] if is_bridge else [2*i+j for i in splits["train"] for j in (0, 1)]
    steps = math.ceil(len(train_ids) / bs)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.bridge_lr if is_bridge else cfg.judge_lr, weight_decay=cfg.weight_decay)
    if is_bridge:
        warmup = max(1, int(epochs * steps * .05))
        def lr_scale(step):
            if step < warmup:
                return step / warmup
            return .5 * (1 + math.cos(math.pi * (step-warmup) / max(1, epochs*steps-warmup)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    amp = bool(cfg.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp and amp_dtype == torch.float16)
    checkpoint = folder / f"{kind}_last.pt"
    start, bad, best, best_state, history = 0, 0, float("inf"), None, []
    selected_epoch, raw_best, raw_best_epoch = None, float("inf"), None
    if checkpoint.exists():
        if not resume:
            raise FileExistsError(f"Use --resume for {checkpoint}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["signature"] != signature or state["kind"] != kind or state["splits"] != splits:
            raise ValueError("Checkpoint does not match data/config/code/splits")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        rng.bit_generator.state = state["numpy_rng"]
        start, bad, best = state["epoch"], state["bad_epochs"], state["best"]
        best_state, history = state["best_state"], state["history"]
        if history:
            raw_row = min(history, key=lambda row: row["val_loss"])
            raw_best, raw_best_epoch = raw_row["val_loss"], raw_row["epoch"]
            selected_epoch = state.get("selected_epoch")
            if selected_epoch is None:
                historical_best = float("inf")
                min_delta = cfg.bridge_val_min_delta if is_bridge else 1e-6
                for row in history:
                    if validation_improves(row["val_loss"], historical_best, min_delta):
                        historical_best, selected_epoch = row["val_loss"], row["epoch"]
    patience = cfg.patience if is_bridge else cfg.judge_patience
    for epoch in range(start, epochs):
        if bad >= patience:
            break
        model.train()
        order = rng.permutation(train_ids).tolist()
        totals = {"loss": 0., "sem": 0., "anchor": 0.}
        for offset in range(0, len(order), bs):
            ids = order[offset:offset+bs]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                if is_bridge:
                    alpha = .12 if cfg.semantic_objective == "deployed_vector_v1" else float(rng.uniform(cfg.alpha_min, cfg.alpha_max))
                    loss, sem, anchor, _ = bridge_values(model, ds, ids, cfg, alpha)
                else:
                    x, mask, labels = judge_batch(ds, ids, cfg, device)
                    loss = F.binary_cross_entropy_with_logits(model(x, mask=mask).float(), labels)
                    sem, anchor = loss, loss.detach()*0
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{kind} epoch {epoch+1}: nonfinite loss; no model exported")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            # FP16 GradScaler handles overflow by skipping the optimizer update
            # and reducing its scale; FP32/BF16 nonfinite gradients are errors.
            scale_before = scaler.get_scale()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=not scaler.is_enabled())
            scaler.step(opt)
            scaler.update()
            if is_bridge and scaler.get_scale() >= scale_before:
                scheduler.step()
            for key, value in (("loss", loss), ("sem", sem), ("anchor", anchor)):
                totals[key] += float(value.detach()) * len(ids)
        if not is_bridge:
            scheduler.step()
        if is_bridge:
            alpha = .12 if cfg.semantic_objective == "deployed_vector_v1" else (cfg.alpha_min+cfg.alpha_max)/2
            metrics = evaluate_bridge(model, ds, splits["val"], cfg, alpha, cfg.magnitude_mode)
            val = bridge_validation_loss(cfg, metrics)
        else:
            logits, labels = judge_predictions(model, ds, splits["val"], cfg)
            val = float(F.binary_cross_entropy_with_logits(logits, labels))
        if not math.isfinite(val):
            raise FloatingPointError("Nonfinite validation loss")
        history.append({"epoch": epoch+1, **{k:v/len(order) for k,v in totals.items()}, "val_loss": val})
        if val < raw_best:
            raw_best, raw_best_epoch = val, epoch + 1
        if validation_improves(val, best, cfg.bridge_val_min_delta if is_bridge else 1e-6):
            best, bad, best_state, selected_epoch = val, 0, cpu_state(model), epoch + 1
        else:
            bad += 1
        save_checkpoint(checkpoint, {
            "signature": signature, "kind": kind, "splits": splits, "epoch": epoch+1,
            "model": cpu_state(model), "optimizer": opt.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "best_state": best_state, "best": best, "bad_epochs": bad,
            "selected_epoch": selected_epoch, "raw_best": raw_best, "raw_best_epoch": raw_best_epoch,
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "numpy_rng": rng.bit_generator.state, "history": history})
        write_json(folder / f"{kind}_history.json", history)
        print(f"[{kind} {epoch+1}/{epochs}] loss={history[-1]['loss']:.5f} val={val:.5f}", flush=True)
        if stop_after_epochs and epoch+1 >= stop_after_epochs and epoch+1 < epochs and bad < patience:
            return None
    if best_state is None:
        raise RuntimeError("No validated checkpoint to export")
    model.load_state_dict(best_state)
    model.eval()
    report = {"kind": kind, "config": asdict(cfg), "best_val_loss": best, "selected_best_val_epoch": selected_epoch,
              "raw_best_val_loss": raw_best, "raw_best_val_epoch": raw_best_epoch,
              "epochs_completed": len(history),
              "supervision_type": ds.meta.supervision_type, "score_label_type": ds.meta.score_label_type,
              "split_pair_counts": {k:len(v) for k,v in splits.items()}, "provenance": ds.meta.encoder,
              "method": (cfg.semantic_objective + "_with_grouped_holdouts_and_padding_masks"
                         if is_bridge and cfg.semantic_objective in ("vector_target_v1", "aligned_token_identity_v1", "monotone_token_identity_v1", "deployed_vector_v1")
                         else "author_objective_with_grouped_holdouts_and_padding_masks"),
              "signature": signature}
    if initialization is not None:
        report["initialization"] = initialization
    if is_bridge:
        if cfg.semantic_objective in ("aligned_token_identity_v1", "monotone_token_identity_v1"):
            report["objective_definition"] = {
                "target": ("token_id_monotone_transported_good_tokens_magnitude_matched_to_bad" if cfg.semantic_objective == "monotone_token_identity_v1"
                           else "position_aligned_good_tokens_magnitude_matched_to_bad"),
                "good_identity_weight": 1., "pair_energy_floor": 1e-4,
                "zero_energy_reject_threshold": 1e-8,
                "initialization": ("strict_pretrained_weights" if initialization is not None
                                   else "zero_residual_output_projection"),
                "training_alpha": cfg.alpha_min,
                "label_warning": ("Transported token targets are representation supervision, not proof of H3 video quality"
                                  if cfg.semantic_objective == "monotone_token_identity_v1"
                                  else "Authored token targets are not video quality labels")}
        if cfg.semantic_objective == "deployed_vector_v1":
            report["objective_definition"] = {
                "target": "deployed_pooled_displacement_equals_fraction_of_authored_good_minus_bad",
                "target_fraction": cfg.deployed_target_fraction,
                "good_identity_weight": cfg.deployed_good_identity_weight,
                "pair_energy_floor": .01, "zero_energy_reject_threshold": 1e-6,
                "training_and_selection_alpha": .12,
                "initialization": ("strict_pretrained_weights" if initialization is not None
                                   else "zero_residual_output_projection"),
                "label_warning": "Text-representation supervision is not evidence of H3 video quality"}
        if cfg.semantic_objective in ("vector_target_v1", "aligned_token_identity_v1", "monotone_token_identity_v1", "deployed_vector_v1"):
            report["test_deferred"] = True
        else:
            report["test"] = [evaluate_bridge(model, ds, splits["test"], cfg, a, "per_token") for a in (0., .12, .2)]
            report["test_training_formula"] = evaluate_bridge(model, ds, splits["test"], cfg, .2, cfg.magnitude_mode)
            report["representation_gate_passed"] = all(r["relative_gain"] > 0 and r["drift"] < .15 for r in report["test"][1:])
        report["video_ab_required"] = True
        model.cfg.extra = {"signature": signature, "encoder": ds.meta.encoder, "video_ab_required": True,
                           "supervision_type": ds.meta.supervision_type, "score_label_type": ds.meta.score_label_type,
                           "semantic_objective": cfg.semantic_objective}
        if cfg.semantic_objective == "deployed_vector_v1":
            model.cfg.extra["deployed_target_fraction"] = cfg.deployed_target_fraction
            model.cfg.extra["deployed_good_identity_weight"] = cfg.deployed_good_identity_weight
        if initialization is not None:
            model.cfg.extra["initialization"] = initialization
        if cfg.semantic_objective in ("aligned_token_identity_v1", "monotone_token_identity_v1", "deployed_vector_v1"):
            # These objectives fit the exact deployed blend at one fixed alpha.
            # Older weights have no contract and retain their previous behavior.
            model.cfg.extra["application_contract"] = {
                "schema": 1, "alpha_min": cfg.alpha_min, "alpha_max": cfg.alpha_max,
                "magnitude_match": "per_token", "token_span": "all", "tail_ratio": 1.0,
                "chunk_tokens": 0, "auto_alpha": False, "guard": False,
                "allow_dim_mismatch": False,
            }
        save_bridge(str(folder / "bridge.partial.safetensors"), model)
        os.replace(folder / "bridge.partial.safetensors", folder / "bridge.safetensors")
    else:
        logits, labels = judge_predictions(model, ds, splits["calibration"], cfg)
        temp, bias, _ = calibrate_temperature(logits, labels)
        model.cfg.temperature, model.cfg.bias = temp, bias
        test_logits, test_labels = judge_predictions(model, ds, splits["test"], cfg)
        report["test_before_calibration"] = judge_metrics(test_logits, test_labels)
        report["test"] = judge_metrics(test_logits, test_labels, temp, bias)
        report["temperature"], report["bias"] = temp, bias
        report["label_warning"] = "Scores measure agreement with rule soft labels, not probability of good generated video"
        save_judge(str(folder / "judge.partial.safetensors"), model, {"signature": signature, "encoder": ds.meta.encoder})
        os.replace(folder / "judge.partial.safetensors", folder / "judge.safetensors")
    write_json(folder / f"{kind}_report.json", report)
    return report


def train_run(ds, cfg, folder, identity, resume=False, stage="both", stop_after_epochs=None,
              split_manifest=None, video_feedback_root=None, init_bridge=None, init_bridge_sha256=None):
    cfg.validate()
    if (init_bridge is None) != (init_bridge_sha256 is None):
        raise ValueError("init_bridge and init_bridge_sha256 must be supplied together")
    initial_state, initialization = None, None
    if init_bridge is not None:
        if resume or stage != "bridge":
            raise ValueError("init_bridge requires a new bridge-only run; resume is incompatible")
        if Path(folder).exists():
            raise FileExistsError("init_bridge requires a new, nonexistent output directory")
        initial_state, initialization = prepare_bridge_initialization(
            init_bridge, init_bridge_sha256, cfg, ds.meta.dim, ds.meta.mode)
        identity = {**identity, "bridge_initialization": initialization}
    if stage not in ("bridge", "judge", "both"):
        raise ValueError("stage must be bridge, judge or both")
    if stop_after_epochs is not None and (isinstance(stop_after_epochs, bool) or not isinstance(stop_after_epochs, int) or stop_after_epochs < 1):
        raise ValueError("stop_after_epochs must be a positive integer")
    ds.validate_supervision()
    from .video_feedback import LABEL as VIDEO_FEEDBACK_LABEL, verify_video_feedback_row
    if any(r.get("label_type") == VIDEO_FEEDBACK_LABEL for r in ds.records):
        if split_manifest is None:
            raise ValueError("video_feedback_v1 requires an explicit whole-work split manifest")
        for row in ds.records:
            verify_video_feedback_row(row, video_feedback_root)
    if stage in ("bridge", "both") and cfg.semantic_objective in ("aligned_token_identity_v1", "monotone_token_identity_v1"):
        from .token_alignment import METHOD, TRANSPORT_METHOD, validate_token_alignment
        method = TRANSPORT_METHOD if cfg.semantic_objective == "monotone_token_identity_v1" else METHOD
        validate_token_alignment(ds, cfg.max_seq_tokens, method=method)
    if stage in ("bridge", "both") and cfg.semantic_pool_scope == "exclude_shared_first":
        for i, (bad_embedding, good_embedding) in enumerate(zip(ds.x, ds.y)):
            if not numerically_shared_first_token(bad_embedding, good_embedding):
                raise ValueError(f"exclude_shared_first requires numerically shared first H3 token and >=2 tokens in pair {i}")
    if stage in ("judge", "both"):
        ds.require_score_labels()
    if isinstance(split_manifest, (str, Path)):
        split_manifest = json.loads(Path(split_manifest).read_text(encoding="utf-8-sig"))
    splits = partition(ds, cfg.seed, split_manifest)
    if stage in ("judge", "both") and not splits.get("calibration"):
        raise ValueError("Judge training requires a separate nonempty calibration split")
    if split_manifest is not None:
        identity = {**identity, "split_manifest_sha256": hashlib.sha256(json.dumps(split_manifest, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
    folder = Path(folder)
    # --resume restores its own checkpoint, never re-applies initial weights.
    # Keep the original initialization provenance in the identical-run signature.
    if resume and (folder / "run.json").is_file():
        stored = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        initialization = stored.get("identity", {}).get("bridge_initialization")
        if initialization is not None:
            if stage != "bridge":
                raise ValueError("An initialized bridge run must resume as bridge-only")
            identity = {**identity, "bridge_initialization": initialization}
    folder.mkdir(parents=True, exist_ok=init_bridge is None)
    signature = run_signature(cfg, identity)
    manifest_path = folder / "run.json"
    previous = {}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not resume or previous["signature"] != signature:
            raise ValueError("Run directory exists; resume requires identical data, configuration and code")
    write_json(folder / "splits.json", splits)
    if split_manifest is not None:
        write_json(folder / "source_split_manifest.json", split_manifest)
    requested = ["bridge", "judge"] if stage == "both" else [stage]
    manifest = {"signature":signature, "config":asdict(cfg), "identity":identity, "status":"running",
                "requested_stages":requested,
                "completed_stages":previous.get("completed_stages", []),
                "split_policy":"explicit group_id manifest with connected text/embedding leakage checks" if split_manifest is not None else "70/10/10/10 by connected text/embedding/source group; source independence unknown",
                "supervision_type":ds.meta.supervision_type, "score_label_type":ds.meta.score_label_type,
                "truncated_sequences":sum(len(a)>cfg.max_seq_tokens for a in ds.x+ds.y),
                "reports": previous.get("reports", {})}
    write_json(manifest_path, manifest)
    try:
        for kind in (("bridge", "judge") if stage == "both" else (stage,)):
            report = fit_stage(kind, ds, splits, cfg, folder, signature, resume, stop_after_epochs,
                               initial_state=initial_state, initialization=initialization)
            if report is None:
                manifest["status"] = "paused_at_epoch_boundary"
                break
            manifest["reports"][kind] = f"{kind}_report.json"
            if kind not in manifest["completed_stages"]:
                manifest["completed_stages"].append(kind)
        else:
            manifest["status"] = "completed"
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["error"] = str(exc)
        raise
    finally:
        write_json(manifest_path, manifest)
    return manifest
