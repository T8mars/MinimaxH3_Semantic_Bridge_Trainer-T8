"""原位训练：先用 H3 自己的文本编码器产 embedding，再训桥与评分头。

训练配方（这里和社区版残差桥的关键差别）
----------------------------------------
社区版是把 ``bridge(x)`` 直接对齐到目标语义，然后推理时再乘一个 alpha 做残差。
这有个隐性错配：训练时模型没见过 alpha 混合后的结果。

本插件默认按 **推理时的真实公式** 训练::

    blend = x + alpha * (bridge(x) - x),     alpha ~ U(alpha_min, alpha_max)
    loss  = 1 - cos(pool(blend), pool(y))            # 目标语义
          + w_anchor * MSE(bridge(x), x)             # 不许跑太远
          + w_mmd    * MMD(bridge(x), y)             # 可选：分布对齐

这样学到的是一条"在该强度下最优"的修正方向，而不是一个端点后再打折。
"""

from __future__ import annotations

import functools
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .bridge_model import BridgeConfig, build_bridge, save_bridge
from .dataset import PairDataset
from .judge import JudgeConfig, build_judge, calibrate_temperature, save_judge


def _grad_context(fn):
    """让训练函数在 ComfyUI 的 ``torch.inference_mode()`` 包装下仍能求导。

    为什么必须这么做（云主机实测）：ComfyUI 的 ``execution.py`` 用
    ``with torch.inference_mode():`` 执行节点，里面的张量默认不追踪梯度，
    而且 ``torch.enable_grad()`` **无法**抵消 inference_mode —— 只有
    ``torch.inference_mode(False)`` 能退出。不加这一层，两个训练节点在真
    ComfyUI 里会直接报
    ``element 0 of tensors does not require grad and does not have a grad_fn``。
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with torch.inference_mode(False), torch.enable_grad():
            return fn(*args, **kwargs)

    return wrapper



@dataclass
class TrainConfig:
    arch: str = "trans"
    hidden: int = 512
    layers: int = 2
    heads: int = 4
    dropout: float = 0.0
    max_tokens: int = 4096
    mode: str = "t2v"
    name: str = "wushu_bridge_v1"

    epochs: int = 60
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    anchor_weight: float = 0.15
    mmd_weight: float = 0.0
    # 语义损失里「绝对靠拢」与「方向对齐」的配比。真 H3 embedding 上
    # 正负例的池化余弦常常已经 0.99+，绝对项会饱和，方向项才是有效信号。
    abs_weight: float = 0.3
    alpha_min: float = 0.05
    alpha_max: float = 0.35
    val_ratio: float = 0.1
    seed: int = 1234
    device: str = "auto"
    amp: bool = True
    ema_decay: float = 0.0          # >0 时启用权重 EMA
    patience: int = 12              # 验证 loss 连续多少轮不降就早停
    max_seq_tokens: int = 2048      # 训练时截断到该长度（防爆显存）


@dataclass
class JudgeTrainConfig:
    hidden: int = 256
    heads: int = 4
    dropout: float = 0.05
    name: str = "wushu_jev_head_v1"
    epochs: int = 80
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.01
    val_ratio: float = 0.15
    seed: int = 1234
    device: str = "auto"


def _pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pad_batch(seqs: List[np.ndarray], max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    dim = seqs[0].shape[-1]
    n = len(seqs)
    out = torch.zeros((n, max_len, dim), dtype=torch.float32)
    mask = torch.zeros((n, max_len), dtype=torch.bool)
    for i, s in enumerate(seqs):
        s = s[:max_len]
        out[i, : s.shape[0]] = torch.from_numpy(np.asarray(s, dtype=np.float32))
        mask[i, : s.shape[0]] = True
    return out, mask


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


def _mmd_rbf(a: torch.Tensor, b: torch.Tensor, mask_a: torch.Tensor, mask_b: torch.Tensor) -> torch.Tensor:
    """无偏 RBF-MMD（子序列采样，够用且便宜）。"""
    def _sub(x, mask, k=64):
        idx = []
        for i in range(x.shape[0]):
            pos = torch.nonzero(mask[i], as_tuple=False).flatten()
            if pos.numel() == 0:
                continue
            take = pos[torch.randint(0, pos.numel(), (min(k, pos.numel()),), device=x.device)]
            idx.append(x[i, take])
        return torch.cat(idx, dim=0) if idx else x.reshape(-1, x.shape[-1])

    xa, xb = _sub(a, mask_a), _sub(b, mask_b)
    if xa.shape[0] < 2 or xb.shape[0] < 2:
        return a.new_zeros(())

    def _k(u, v):
        d = torch.cdist(u, v).pow(2)
        scale = d.detach().median().clamp_min(1e-6)
        return torch.exp(-d / scale)

    return _k(xa, xa).mean() + _k(xb, xb).mean() - 2 * _k(xa, xb).mean()


@_grad_context
def train_bridge(
    ds: PairDataset,
    cfg: TrainConfig,
    out_path: str,
    log=print,
) -> Dict[str, Any]:
    """训练残差桥并导出 safetensors，返回训练报告。"""
    ds.validate_supervision()
    if ds.meta.supervision_type == "preference_only":
        raise ValueError("Preference-only datasets require the CLI trainer with grouped splits; use --stage bridge --split-manifest")
    if len(ds) < 4:
        raise ValueError(f"样本太少（{len(ds)} 对），至少需要 4 对。建议 ≥ 200 对。")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = _pick_device(cfg.device)

    train_ds, val_ds = ds.split(cfg.val_ratio, cfg.seed)
    dim = ds.meta.dim
    bridge_cfg = BridgeConfig(
        arch=cfg.arch, dim=dim, hidden=cfg.hidden, layers=cfg.layers, heads=cfg.heads,
        dropout=cfg.dropout, max_tokens=cfg.max_tokens, mode=cfg.mode, name=cfg.name,
    )
    model = build_bridge(bridge_cfg).to(device)
    if cfg.amp and device.type == "cuda":
        model = model.to(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, math.ceil(len(train_ds) / cfg.batch_size))
    total_steps = steps_per_epoch * cfg.epochs
    warmup = max(1, int(total_steps * cfg.warmup_ratio))

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    ema = None
    if cfg.ema_decay > 0:
        ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    history: List[Dict[str, float]] = []
    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    bad_epochs = 0
    t0 = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        order = np.random.permutation(len(train_ds))
        ep_loss = ep_sem = ep_anchor = 0.0
        nb = 0
        for b in range(steps_per_epoch):
            idx = order[b * cfg.batch_size : (b + 1) * cfg.batch_size]
            if len(idx) == 0:
                continue
            xs = [train_ds.x[i] for i in idx]
            ys = [train_ds.y[i] for i in idx]
            max_len = min(cfg.max_seq_tokens, max(max(s.shape[0] for s in xs), max(s.shape[0] for s in ys)))
            xb, xm = _pad_batch(xs, max_len)
            yb, ym = _pad_batch(ys, max_len)
            xb, xm, yb, ym = xb.to(device), xm.to(device), yb.to(device), ym.to(device)

            alpha = float(np.random.uniform(cfg.alpha_min, cfg.alpha_max))
            xin = xb / torch.sqrt(xb.pow(2).mean(dim=-1, keepdim=True) + 1e-6)

            pred = model(xin.to(next(model.parameters()).dtype)).float()
            blend = xb + alpha * (pred - xb)

            # 语义损失 = 绝对项 + 方向项
            #   绝对项：池化后靠近正例（正负例本来就几乎相同时会饱和）
            #   方向项：沿 (good - bad) 方向移动的比例（这才是真数据上的有效信号）
            pool_bad = masked_mean(xb, xm)
            pool_good = masked_mean(yb, ym)
            pool_pred = masked_mean(blend, xm)
            abs_term = 1.0 - F.cosine_similarity(pool_pred, pool_good, dim=-1).mean()
            d_target = pool_good - pool_bad
            d_pred = pool_pred - pool_bad
            gap = d_target.norm(dim=-1)
            valid = gap > 1e-3
            if bool(valid.any()):
                dir_term = 1.0 - F.cosine_similarity(
                    d_pred[valid], d_target[valid], dim=-1).mean()
            else:
                dir_term = abs_term
            sem = cfg.abs_weight * abs_term + (1.0 - cfg.abs_weight) * dir_term
            anchor = F.mse_loss(pred, xb)
            loss = sem + cfg.anchor_weight * anchor
            mmd_val = 0.0
            if cfg.mmd_weight > 0:
                mmd_val = _mmd_rbf(blend, yb, xm, ym)
                loss = loss + cfg.mmd_weight * mmd_val

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()

            if ema is not None:
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        if v.dtype.is_floating_point:
                            ema[k].mul_(cfg.ema_decay).add_(v.detach().float(), alpha=1 - cfg.ema_decay)

            ep_loss += float(loss.item())
            ep_sem += float(sem.item())
            ep_anchor += float(anchor.item())
            nb += 1

        val_metrics = evaluate_bridge(model, val_ds, cfg) if len(val_ds) else {"val_sem": float("nan")}
        val_loss = val_metrics.get("val_sem", float("nan"))
        history.append(
            {
                "epoch": epoch + 1,
                "loss": ep_loss / max(nb, 1),
                "sem": ep_sem / max(nb, 1),
                "anchor": ep_anchor / max(nb, 1),
                **val_metrics,
            }
        )
        log(
            f"[epoch {epoch+1}/{cfg.epochs}] loss={ep_loss/max(nb,1):.4f} "
            f"sem={ep_sem/max(nb,1):.4f} anchor={ep_anchor/max(nb,1):.4f} "
            f"val_sem={val_loss:.4f}"
        )

        if val_loss == val_loss and val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                log(f"[early stop] 验证集 {cfg.patience} 轮无改善，停在 epoch {epoch+1}")
                break

    if ema is not None:
        model.load_state_dict({k: v.to(best_state[k].dtype) for k, v in ema.items()})
    else:
        model.load_state_dict(best_state)
    model.float().eval()

    # 最终指标：真正关心的两件事——语义是否更靠近正例、漂移是否可控
    final_train = evaluate_bridge(model, train_ds, cfg)
    final_val = evaluate_bridge(model, val_ds, cfg) if len(val_ds) else {}
    report = {
        "kind": "wushu_bridge_train",
        "arch": cfg.arch,
        "params": model.num_params(),
        "size_mb_fp32": round(model.num_params() * 4 / 1024 / 1024, 2),
        "samples": len(ds),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "dim": dim,
        "device": str(device),
        "seconds": round(time.time() - t0, 1),
        "best_val_sem": None if best_val == float("inf") else round(best_val, 5),
        "final_train": final_train,
        "final_val": final_val,
        "history_tail": history[-5:],
        "config": asdict(cfg),
    }
    save_bridge(out_path, model, extra_meta={"report": json.dumps(report, ensure_ascii=False)[:2000]})
    with open(os.path.splitext(out_path)[0] + "_report.json", "w", encoding="utf-8") as f:
        json.dump({"report": report, "history": history}, f, ensure_ascii=False, indent=2)
    log(f"[done] {out_path}  参数量={model.num_params():,}（fp32 {report['size_mb_fp32']}MB）")
    return report


@torch.no_grad()
def evaluate_bridge(model, ds: PairDataset, cfg: TrainConfig, alpha: Optional[float] = None) -> Dict[str, float]:
    """在给定 alpha 下评测：桥输出与正例的池化余弦、以及相对原序列的漂移。"""
    if len(ds) == 0:
        return {}
    model.eval()
    device = next(model.parameters()).device
    a = alpha if alpha is not None else (cfg.alpha_min + cfg.alpha_max) / 2

    sims, before, drift = [], [], []
    for i in range(0, len(ds), cfg.batch_size):
        xs = ds.x[i : i + cfg.batch_size]
        ys = ds.y[i : i + cfg.batch_size]
        max_len = min(cfg.max_seq_tokens, max(s.shape[0] for s in xs + ys))
        xb, xm = _pad_batch(xs, max_len)
        yb, ym = _pad_batch(ys, max_len)
        xb, xm, yb, ym = xb.to(device), xm.to(device), yb.to(device), ym.to(device)
        xin = xb / torch.sqrt(xb.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        pred = model(xin.to(next(model.parameters()).dtype)).float()
        blend = xb + a * (pred - xb)
        py = masked_mean(yb, ym)
        sims.append(F.cosine_similarity(masked_mean(blend, xm), py, dim=-1).cpu())
        before.append(F.cosine_similarity(masked_mean(xb, xm), py, dim=-1).cpu())
        drift.append(F.cosine_similarity(masked_mean(blend, xm), masked_mean(xb, xm), dim=-1).cpu())

    sim = torch.cat(sims)
    bef = torch.cat(before)
    dri = torch.cat(drift)
    # 正负例之间的实际差距（池化余弦）——这个数越接近 1，说明降级对 embedding
    # 的影响越小、桥能学的东西越少。真 H3 上实测常见 0.99+。
    gaps = []
    dirs = []
    for i in range(0, len(ds), cfg.batch_size):
        xs = ds.x[i : i + cfg.batch_size]
        ys = ds.y[i : i + cfg.batch_size]
        max_len = min(cfg.max_seq_tokens, max(s.shape[0] for s in xs + ys))
        xb, xm = _pad_batch(xs, max_len)
        yb, ym = _pad_batch(ys, max_len)
        xb, xm, yb, ym = xb.to(device), xm.to(device), yb.to(device), ym.to(device)
        xin = xb / torch.sqrt(xb.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        pred2 = model(xin.to(next(model.parameters()).dtype)).float()
        blend2 = xb + a * (pred2 - xb)
        pb, pg, pp = masked_mean(xb, xm), masked_mean(yb, ym), masked_mean(blend2, xm)
        gaps.append(torch.nn.functional.cosine_similarity(pb, pg, dim=-1).cpu())
        dt, dp = pg - pb, pp - pb
        ok = dt.norm(dim=-1) > 1e-3
        if bool(ok.any()):
            dirs.append(torch.nn.functional.cosine_similarity(dp[ok], dt[ok], dim=-1).cpu())
    gap_mean = float(torch.cat(gaps).mean()) if gaps else float("nan")
    dir_mean = float(torch.cat(dirs).mean()) if dirs else float("nan")
    cos_before = float(bef.mean())
    cos_after = float(sim.mean())
    return {
        "alpha": round(float(a), 4),
        "cos_to_target_before": round(cos_before, 5),
        "cos_to_target_after": round(cos_after, 5),
        "relative_gain": round(cos_after - cos_before, 5),
        "drift": round(float((1 - dri).mean()), 5),
        # 训练用的验证损失：语义项 1-cos（越小越好）
        "val_sem": round(1.0 - cos_after, 5),
        # bad 与 good 之间的池化余弦：越接近 1 说明降级对 embedding 影响越小
        "bad_good_similarity": round(gap_mean, 5),
        # 桥的移动方向与 (good-bad) 方向的一致度：这才是真数据上的有效指标
        "direction_alignment": None if dir_mean != dir_mean else round(dir_mean, 5),
    }


@torch.no_grad()
def _auc(probs: torch.Tensor, targets: torch.Tensor) -> float:
    """正负例之间的排序质量（0.5=瞎猜，1.0=完全分开）。"""
    pos = probs[targets > 0.5]
    neg = probs[targets <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    diff = pos.unsqueeze(1) - neg.unsqueeze(0)
    return float(((diff > 0).float() + 0.5 * (diff == 0).float()).mean().item())


@_grad_context
def train_judge(ds: PairDataset, cfg: JudgeTrainConfig, out_path: str, log=print) -> Dict[str, Any]:
    """训练 JEV 式评分头：负例 -> 低分，正例 -> 高分（软标签来自规则校验器）。

    直接在 **token 序列**上训练（不是池化向量），这样推理时那套"注意力池化盯住
    最违规 token"的机制在训练时真的被用上了，训练与推理的特征分布一致。
    """
    ds.require_score_labels()
    if len(ds) < 8:
        raise ValueError(f"样本太少（{len(ds)} 对），评分头至少需要 8 对。")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = _pick_device(cfg.device)
    dim = ds.meta.dim

    seqs: List[np.ndarray] = list(ds.x) + list(ds.y)
    labels = np.asarray(list(ds.x_score) + list(ds.y_score), dtype=np.float32)
    # 不成对的额外样本（例如真实被筛掉的片段对应的 prompt）只喂评分头
    if ds.extra:
        seqs += list(ds.extra)
        labels = np.concatenate([labels, np.asarray(ds.extra_score, dtype=np.float32)])
    n = len(seqs)

    perm = np.random.default_rng(cfg.seed).permutation(n)
    n_val = max(1, int(round(n * cfg.val_ratio)))
    val_idx = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()
    if not train_idx:
        raise ValueError("训练集为空，请增加样本量或降低 val_ratio。")

    model = build_judge(JudgeConfig(dim=dim, hidden=cfg.hidden, heads=cfg.heads,
                                    dropout=cfg.dropout, name=cfg.name)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    def _batch(idxs: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        max_len = max(seqs[i].shape[0] for i in idxs)
        xb = torch.zeros((len(idxs), max_len, dim), dtype=torch.float32)
        for k, i in enumerate(idxs):
            s = seqs[i]
            xb[k, : s.shape[0]] = torch.from_numpy(np.asarray(s, dtype=np.float32))
        yb = torch.from_numpy(labels[idxs])
        return xb.to(device), yb.to(device)

    best_val, best_state, bad = float("inf"), None, 0
    history: List[Dict[str, float]] = []
    for epoch in range(cfg.epochs):
        model.train()
        order = np.random.permutation(len(train_idx))
        ep = 0.0
        nb = 0
        for b in range(0, len(order), cfg.batch_size):
            idxs = [train_idx[j] for j in order[b : b + cfg.batch_size]]
            xb, yb = _batch(idxs)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep += float(loss.item())
            nb += 1
        sched.step()
        model.eval()
        with torch.no_grad():
            vx, vy = _batch(val_idx)
            vl = float(F.binary_cross_entropy_with_logits(model(vx), vy).item())
        history.append({"epoch": epoch + 1, "train": ep / max(nb, 1), "val": vl})
        if vl < best_val - 1e-6:
            best_val, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 20:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        vx, vy = _batch(val_idx)
        logits_va = model(vx)
        temp, bias, ece = calibrate_temperature(logits_va, vy)
        probs = torch.sigmoid(logits_va / temp + bias)
        acc = float(((probs > 0.5).float() == (vy > 0.5).float()).float().mean().item())
        sep = (
            float((probs[vy > 0.5].mean() - probs[vy <= 0.5].mean()).item())
            if (vy > 0.5).any() and (vy <= 0.5).any()
            else float("nan")
        )
        # 排序质量：正例分数是否普遍高于负例
        auc = _auc(probs, vy)

    model.cfg.temperature = temp
    model.cfg.bias = bias

    report = {
        "kind": "wushu_judge_train",
        "params": model.num_params(),
        "size_mb_fp32": round(model.num_params() * 4 / 1024 / 1024, 3),
        "samples": int(n),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "dim": dim,
        "best_val_bce": round(best_val, 5),
        "val_accuracy": round(acc, 4),
        "val_auc": None if auc != auc else round(auc, 4),
        "val_pos_neg_gap": None if sep != sep else round(sep, 4),
        "val_ece_after_calibration": round(ece, 4),
        "temperature": round(temp, 4),
        "bias": round(bias, 4),
        "history_tail": history[-5:],
        "config": asdict(cfg),
    }
    save_judge(out_path, model, extra_meta={"val_accuracy": report["val_accuracy"], "temp": temp})
    with open(os.path.splitext(out_path)[0] + "_report.json", "w", encoding="utf-8") as f:
        json.dump({"report": report, "history": history}, f, ensure_ascii=False, indent=2)
    log(
        f"[done] {out_path}  参数量={model.num_params():,} val_acc={report['val_accuracy']} "
        f"AUC={report['val_auc']} 温度={temp:.3f} ECE={ece:.4f}"
    )
    return report
