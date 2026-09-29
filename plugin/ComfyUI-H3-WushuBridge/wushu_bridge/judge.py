"""JEV 式评分头：直接从 H3 的 conditioning 判断「这段武打逻辑合不合格」。

为什么不用现成的开源 JEV 复刻（NanoJev / Verdict / reflex）
----------------------------------------------------------
它们都是「文本 state + 类型化问题 -> 概率」的通用决策模型，最小的也在
150M~0.8B 之间，而且没见过 H3 这套 5120 维条件空间。本插件里的评分头是
**在 H3 自己的文本编码器输出上原位训练**的：

* 参数量约 1.3M（fp32 ≈ 5MB），比最小的开源 JEV 复刻还小两个数量级；
* 读的是 H3 真正吃进去的那段语义（同一个 embedding 空间），不是转述文本；
* 训练标签来自本插件自带的武打规则校验器，可随用户的规则库一起更新。

它干的事和 Jev 一样：**不吐字，直接给概率**。用途有三个：
1. 给提示词打分（0~1），让用户知道这段逻辑够不够；
2. 自动驱动桥的 alpha（分低就多修一点，分高就少动）；
3. Best-of-N：多份候选提示词各编码一次，选分最高的那份送进采样器。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

JUDGE_SCHEMA_VERSION = 1


@dataclass
class JudgeConfig:
    dim: int = 5120
    hidden: int = 256
    heads: int = 4
    dropout: float = 0.0
    schema: int = JUDGE_SCHEMA_VERSION
    name: str = "wushu_jev_head"
    temperature: float = 1.0
    bias: float = 0.0
    note: str = ""

    def __post_init__(self) -> None:
        if self.hidden % self.heads != 0:
            raise ValueError("hidden 必须能被 heads 整除")


class _AttentionPool(nn.Module):
    """学一个 query，让评分头能盯住序列里"最违规的那几个 token"。"""

    def __init__(self, hidden: int, heads: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.query.expand(x.shape[0], -1, -1)
        out, _ = self.attn(q, x, x, key_padding_mask=None if mask is None else ~mask, need_weights=False)
        return self.norm(out[:, 0])


class WushuJudge(nn.Module):
    """conditioning -> P(武打逻辑合格)，单次前向，无解码。

    输入是 ``[B,T,D]`` 的 token 序列（也可以给 ``[B,D]`` 的池化向量，此时
    注意力池化退化成同一路）。先把 D=5120 投影到 ``hidden`` 再做注意力池化：
    直接在 5120 维上做 MHA 光 in_proj 就 7800 万参数，纯属浪费——投影后再算，
    整个评分头只有约 1.6M 参数（fp32 ≈ 6.5MB）。
    """

    def __init__(self, cfg: JudgeConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.dim, cfg.hidden)
        self.pool = _AttentionPool(cfg.hidden, cfg.heads)
        self.norm = nn.LayerNorm(cfg.hidden)
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden * 2, cfg.hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """返回 logits ``[B]``。"""
        if x.shape[-1] != self.cfg.dim:
            raise ValueError(
                f"embedding 维度不匹配：评分头 dim={self.cfg.dim}，输入 {x.shape[-1]}"
            )
        if x.ndim == 3:
            if mask is not None:
                if mask.shape != x.shape[:2] or mask.dtype != torch.bool:
                    raise ValueError("mask 必须是 [B,T] bool，True 表示有效 token")
                mask = mask.to(device=x.device)
                if not bool(mask.any(dim=1).all()):
                    raise ValueError("每条序列至少需要一个有效 token")
            z = self.proj(x)
            if mask is None:
                mean_pool = z.mean(dim=1)
            else:
                valid = mask.unsqueeze(-1).to(z.dtype)
                mean_pool = (z * valid).sum(dim=1) / valid.sum(dim=1)
            attn_pool = self.pool(z, mask=mask)
        elif x.ndim == 2:
            if mask is not None:
                raise ValueError("池化向量 [B,D] 不接受 token mask")
            mean_pool = self.proj(x)
            attn_pool = mean_pool
        else:
            raise ValueError(f"期望 [B,T,D] 或 [B,D]，得到 {tuple(x.shape)}")
        return self.head(torch.cat([mean_pool, attn_pool], dim=-1)).squeeze(-1)

    def probability(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """带温度校准的概率输出（Jev 式的"直接给概率"）。"""
        logits = self.forward(x, mask=mask) / max(self.cfg.temperature, 1e-3) + self.cfg.bias
        return torch.sigmoid(logits)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_judge(cfg: JudgeConfig) -> WushuJudge:
    return WushuJudge(cfg)


def save_judge(path: str, model: WushuJudge, extra_meta: Optional[Dict[str, Any]] = None) -> str:
    from safetensors.torch import save_file

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    meta = {k: str(v) for k, v in asdict(model.cfg).items()}
    if extra_meta:
        for k, v in extra_meta.items():
            meta[f"train_{k}"] = str(v)
    tensors = {k: v.detach().to(torch.float32).contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(tensors, path, metadata=meta)
    return path


def load_judge(path: str, device: str = "cpu", dtype: torch.dtype = torch.float32) -> WushuJudge:
    from safetensors import safe_open
    from safetensors.torch import load_file

    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到评分头权重：{path}")
    with safe_open(path, framework="pt", device="cpu") as f:
        meta = dict(f.metadata() or {})
    state = load_file(path, device="cpu")
    if "proj.weight" not in state:
        raise RuntimeError(
            "评分头权重格式不匹配（缺少 proj.weight）。"
            "请用本插件的 'H3 武打桥 训练 JEV 评分头' 节点重新训练一个。"
        )
    hidden = int(state["proj.weight"].shape[0])
    dim = int(state["proj.weight"].shape[1])
    cfg = JudgeConfig(
        dim=dim,
        hidden=hidden,
        heads=int(meta.get("heads", 4)) if meta.get("heads") else 4,
        dropout=float(meta.get("dropout", 0.0)),
        name=meta.get("name", os.path.basename(path)),
        temperature=float(meta.get("temperature", 1.0)),
        bias=float(meta.get("bias", 0.0)),
        note=meta.get("note", ""),
    )
    while cfg.hidden % cfg.heads != 0 and cfg.heads > 1:
        cfg.heads -= 1
    model = WushuJudge(cfg)
    model.load_state_dict(state, strict=True)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def score_conditioning(
    model: WushuJudge,
    conditioning,
    aggregate: str = "mean",
) -> Tuple[float, list]:
    """对 CONDITIONING 打分，返回 (分数, 每段分数)。"""
    device = next(model.parameters()).device
    scores = []
    for item in conditioning:
        native = item[0]
        if not torch.is_tensor(native) or native.ndim != 3:
            continue
        if native.shape[-1] != model.cfg.dim:
            continue
        p = model.probability(native.float().to(device))
        scores.append(float(p.mean().item()))
    if not scores:
        return 0.0, []
    if aggregate == "min":
        return min(scores), scores
    if aggregate == "max":
        return max(scores), scores
    return sum(scores) / len(scores), scores


@torch.inference_mode(False)
@torch.enable_grad()
def calibrate_temperature(
    logits: torch.Tensor,
    targets: torch.Tensor,
    steps: int = 200,
    lr: float = 0.05,
) -> Tuple[float, float, float]:
    """在验证集上做温度+偏置校准，返回 (temperature, bias, 校准后 ECE)。

    Jev 的卖点之一是"概率可信"，不校准的 sigmoid 分数没法当阈值用。
    """
    # Clone inside the grad context: ComfyUI may supply inference tensors, which
    # autograd cannot save for the calibration backward pass.
    logits = logits.detach().to(dtype=torch.float32).clone().flatten()
    targets = targets.detach().to(device=logits.device, dtype=torch.float32).clone().flatten()
    if logits.shape != targets.shape or not logits.numel():
        raise ValueError("校准 logits/targets 必须非空且形状一致")
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(targets).all()):
        raise ValueError("校准数据包含非有限值")
    if not bool(((targets >= 0) & (targets <= 1)).all()):
        raise ValueError("校准 targets 必须位于 [0,1]")
    t = torch.ones((), device=logits.device, requires_grad=True)
    b = torch.zeros((), device=logits.device, requires_grad=True)
    opt = torch.optim.LBFGS([t, b], lr=lr, max_iter=steps)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits / t.clamp_min(1e-3) + b, targets
        )
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        temp = float(t.clamp_min(1e-3).item())
        bias = float(b.item())
        probs = torch.sigmoid(logits / temp + bias)
        ece = expected_calibration_error(probs, targets)
    return temp, bias, float(ece)


def expected_calibration_error(probs: torch.Tensor, targets: torch.Tensor, bins: int = 10) -> float:
    edges = torch.linspace(0.0, 1.0, bins + 1, device=probs.device)
    targets = targets.to(device=probs.device)
    ece = 0.0
    n = probs.numel()
    if n == 0:
        return 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs > lo) & (probs <= hi) if i > 0 else (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        conf = probs[mask].mean()
        acc = targets[mask].mean()
        ece += (mask.sum().item() / n) * abs(conf - acc).item()
    return float(ece)


def judge_metadata(model: WushuJudge) -> str:
    return json.dumps(asdict(model.cfg), ensure_ascii=False)
