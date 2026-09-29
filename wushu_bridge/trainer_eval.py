"""Descriptive holdout diagnostics; these do not establish video quality."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .apply import magnitude_match, rms_normalize


def _percentiles(values):
    return {"p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95))}


@torch.no_grad()
def bridge_diagnostics(model, ds, indices, cfg, alpha, mode):
    """Report pair gains, token drift, and damage to already-good prompts.

    The bootstrap resamples pairs, not unknown originating videos. Its interval
    describes the provided holdout only and must not be a release criterion.
    """
    from .trainer_engine import batch, evaluate_bridge, semantic_pool, transported_target

    if not indices:
        raise ValueError("Diagnostics require nonempty holdout indices")
    summary = evaluate_bridge(model, ds, indices, cfg, alpha, mode)
    device = next(model.parameters()).device
    gains, token_drifts, good_drifts, good_token_drifts = [], [], [], []
    token_baselines, token_corrected, token_identity, token_floors = [], [], [], []
    token_objective = cfg.semantic_objective in (
        "aligned_token_identity_v1", "monotone_token_identity_v1")

    def blend(x, mask):
        if alpha == 0:
            return x
        pred = model(rms_normalize(x), mask=mask).float()
        return x + alpha * (magnitude_match(pred, x, mode) - x)

    for start in range(0, len(indices), cfg.batch_size):
        ids = indices[start:start + cfg.batch_size]
        x, xm = batch([ds.x[i] for i in ids], device, cfg.max_seq_tokens)
        y, ym = batch([ds.y[i] for i in ids], device, cfg.max_seq_tokens)
        xb, yb = blend(x, xm), blend(y, ym)
        if token_objective:
            target = (magnitude_match(y, x, "per_token")
                      if cfg.semantic_objective == "aligned_token_identity_v1"
                      else transported_target(ds, ids, x, y, xm))
            x_denom = xm.sum(dim=1) * x.shape[-1]
            y_denom = ym.sum(dim=1) * y.shape[-1]
            baseline = (((target - x).square() * xm[..., None]).sum(dim=(1, 2)) / x_denom)
            corrected = (((xb - target).square() * xm[..., None]).sum(dim=(1, 2)) / x_denom)
            identity = (((yb - y).square() * ym[..., None]).sum(dim=(1, 2)) / y_denom)
            # At fixed alpha, per-token magnitude matching constrains each
            # predicted token to the source token's norm. This is the best
            # possible target error with an unconstrained predictor; the
            # actual bridge can only do worse.
            desired = target - (1.0 - alpha) * x
            floor_per_token = (desired.norm(dim=-1) - alpha * x.norm(dim=-1)).square() / x.shape[-1]
            floor = (floor_per_token * xm).sum(dim=1) / xm.sum(dim=1)
            token_baselines.extend(baseline.cpu().tolist())
            token_corrected.extend(corrected.cpu().tolist())
            token_identity.extend(identity.cpu().tolist())
            token_floors.extend(floor.cpu().tolist())
        px = semantic_pool(x, xm, cfg.semantic_pool_scope)
        py = semantic_pool(y, ym, cfg.semantic_pool_scope)
        if alpha == 0:
            gains.extend([0.] * len(ids))
            good_drifts.extend([0.] * len(ids))
            token_drifts.extend([0.] * int(xm.sum()))
            good_token_drifts.extend([0.] * int(ym.sum()))
            continue
        gains.extend((F.cosine_similarity(semantic_pool(xb, xm, cfg.semantic_pool_scope), py, dim=-1)
                      - F.cosine_similarity(px, py, dim=-1)).cpu().tolist())
        token_drifts.extend((1 - F.cosine_similarity(x, xb, dim=-1))[xm].clamp_min(0).cpu().tolist())
        good_drifts.extend((1 - F.cosine_similarity(py, semantic_pool(yb, ym, cfg.semantic_pool_scope), dim=-1)).clamp_min(0).cpu().tolist())
        good_token_drifts.extend((1 - F.cosine_similarity(y, yb, dim=-1))[ym].clamp_min(0).cpu().tolist())
    gains = np.asarray(gains, dtype=np.float64)
    if not all(np.isfinite(values).all() for values in (gains, token_drifts, good_drifts, good_token_drifts)):
        raise FloatingPointError("Nonfinite holdout diagnostics")
    rng = np.random.default_rng(cfg.seed)
    bootstrap = gains[rng.integers(len(gains), size=(1000, len(gains)))].mean(axis=1)
    summary.update({
        "pairs": len(gains), "pairwise_gain_win_rate": float((gains > 0).mean()),
        "mean_gain_bootstrap_95ci": np.percentile(bootstrap, [2.5, 97.5]).tolist(),
        "bootstrap_unit": "pair; originating-video independence is unknown",
        "token_cosine_drift": _percentiles(token_drifts),
        "good_identity_pooled_drift": _percentiles(good_drifts),
        "good_identity_token_drift": _percentiles(good_token_drifts),
        "interpretation": "Descriptive representation diagnostics only; same-seed video A/B remains required",
    })
    if token_objective:
        baseline = np.asarray(token_baselines, dtype=np.float64)
        corrected = np.asarray(token_corrected, dtype=np.float64)
        identity = np.asarray(token_identity, dtype=np.float64)
        floor = np.asarray(token_floors, dtype=np.float64)
        if not all(np.isfinite(values).all() for values in (baseline, corrected, identity, floor)):
            raise FloatingPointError("Nonfinite token target diagnostics")
        denom = np.maximum(baseline, 1e-4)  # Same floor as the training objective.
        summary["token_target"] = {
            "baseline_token_mse": float(np.mean(baseline)),
            "corrected_token_mse": float(np.mean(corrected)),
            "good_identity_token_mse": float(np.mean(identity)),
            "theoretical_min_token_mse": float(np.mean(floor)),
            "baseline_normalized_error": float(np.mean(baseline / denom)),
            "corrected_normalized_error": float(np.mean(corrected / denom)),
            "good_identity_normalized_error": float(np.mean(identity / denom)),
            "theoretical_min_normalized_error": float(np.mean(floor / denom)),
            "pairwise_correction_improvement_rate": float(np.mean(corrected < baseline)),
            "definition": "Per-pair token MSE divided by max(source-to-target token MSE, 1e-4); theoretical minimum assumes arbitrary per-token predictions with the source norm at this alpha, not an achievable bridge or video-quality bound",
        }
    return summary
