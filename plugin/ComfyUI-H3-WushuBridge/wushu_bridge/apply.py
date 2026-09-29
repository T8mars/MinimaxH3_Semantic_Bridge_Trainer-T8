"""把残差桥作用到 ComfyUI 的 CONDITIONING 上。

ComfyUI 的 CONDITIONING 结构是::

    [[embedding: Tensor[B, T, D], metadata: dict], ...]        # 正常情况 1 项

MiniMax H3 的 D = 5120，metadata 里可能带 ``minimax_keyframes`` /
``minimax_frame_count`` / ``minimax_refs`` 等字段（首尾帧、参考图、参考音频）。
本模块**只替换 embedding 张量，原样保留 metadata 的每一个键**——这是能安全
"接在 conditioning 之间"的前提，否则会把参考图/首尾帧的 latent 弄丢。

应用公式（与社区版 Semantic Bridge 一致）::

    x_norm    = rms_normalize(h)
    pred      = bridge(x_norm)
    pred      = magnitude_match(pred, h)        # per_token / global / none
    hybrid    = h + alpha * (pred - h)

``alpha`` 小（0.05~0.25）时等价于沿"武打逻辑完备"方向做温和引导；alpha=1 则
完全替换，容易跑出文本编码器原生流形，不推荐。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .bridge_model import WushuBridge

MAGNITUDE_MODES = ("per_token", "global", "none")
TEXT_SPAN_MODES = ("all", "tail")

META_KEY_PREFIX = "wushu_bridge_"


def rms_normalize(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    return x / rms


def magnitude_match(source: torch.Tensor, target: torch.Tensor, mode: str = "per_token") -> torch.Tensor:
    """把 source 的幅度对齐到 target，避免替换后整体能量漂移。"""
    source = source.float()
    target = target.float()
    if mode == "none":
        return source
    if mode == "per_token":
        s = torch.sqrt(source.pow(2).mean(dim=-1, keepdim=True) + 1e-8)
        t = torch.sqrt(target.pow(2).mean(dim=-1, keepdim=True) + 1e-8)
        return source * (t / s)
    if mode == "global":
        s = torch.sqrt(source.pow(2).mean() + 1e-8)
        t = torch.sqrt(target.pow(2).mean() + 1e-8)
        return source * (t / s)
    raise ValueError(f"未知 magnitude_match 模式：{mode}，可选 {MAGNITUDE_MODES}")


@dataclass
class ApplyReport:
    """一次应用的可观测信息，直接抛到 ComfyUI 的 STRING 输出里给用户看。"""

    applied: int = 0
    skipped: int = 0
    dims: List[int] = field(default_factory=list)
    tokens: List[int] = field(default_factory=list)
    drift: List[float] = field(default_factory=list)      # 相对原 embedding 的 1-cos
    ref_like: List[bool] = field(default_factory=list)     # 是否像参考图/首尾帧 conditioning
    messages: List[str] = field(default_factory=list)

    def to_text(self) -> str:
        lines = [
            f"武打语义桥：应用 {self.applied} 段 conditioning，跳过 {self.skipped} 段",
        ]
        for i, (d, t, dr) in enumerate(zip(self.dims, self.tokens, self.drift)):
            flag = "(含参考图/首尾帧 token)" if i < len(self.ref_like) and self.ref_like[i] else ""
            lines.append(f"  #{i}: dim={d} tokens={t} 语义漂移(1-cos)={dr:.4f}{flag}")
        lines.extend(self.messages)
        return "\n".join(lines)


def _looks_reference_conditioned(metadata: Dict[str, Any]) -> bool:
    for key in ("minimax_refs", "minimax_frame_count", "minimax_keyframes"):
        if metadata.get(key):
            return True
    return False


def _token_slice(tokens: int, span: str, tail_ratio: float) -> Tuple[int, int]:
    """选择要施加桥的 token 区间。

    H3 打包序列里，参考图/参考音频的 token 与文本 token 混在一起，且位置不稳定。
    ``all`` 全序列（与社区版一致）；``tail`` 只改末尾 ``tail_ratio`` 比例，用于
    参考图模式下"尽量只动文本尾部"的保守策略。
    """
    if span == "all" or tail_ratio >= 1.0:
        return 0, tokens
    start = max(0, int(round(tokens * (1.0 - max(0.0, tail_ratio)))))
    return start, tokens


def application_contract(bridge: WushuBridge) -> Optional[Dict[str, Any]]:
    """Return an opt-in fixed application contract; old weights have none."""
    extra = bridge.cfg.extra
    contract = extra.get("application_contract") if isinstance(extra, dict) else None
    if contract is None:
        return None
    if not isinstance(contract, dict) or contract.get("schema") != 1:
        raise ValueError("Bridge application contract has an unsupported schema")
    required = ("alpha_min", "alpha_max", "magnitude_match", "token_span",
                "tail_ratio", "chunk_tokens", "auto_alpha", "guard", "allow_dim_mismatch")
    if any(key not in contract for key in required):
        raise ValueError("Bridge application contract is incomplete")
    alpha_min, alpha_max = contract["alpha_min"], contract["alpha_max"]
    if (type(alpha_min) not in (int, float) or type(alpha_max) not in (int, float)
            or not math.isfinite(alpha_min) or not math.isfinite(alpha_max)
            or not 0 < alpha_min <= 1 or alpha_min != alpha_max):
        raise ValueError("Bridge application contract requires one finite fixed alpha")
    if (contract["magnitude_match"] != "per_token" or contract["token_span"] != "all"
            or type(contract["tail_ratio"]) not in (int, float)
            or contract["tail_ratio"] != 1.0 or type(contract["chunk_tokens"]) is not int
            or contract["chunk_tokens"] != 0
            or any(contract[key] is not False for key in
                   ("auto_alpha", "guard", "allow_dim_mismatch"))):
        raise ValueError("Bridge application contract has unsupported fixed settings")
    return contract


def validate_application_contract(
    bridge: WushuBridge, *, alpha: float, magnitude_match_mode: str,
    token_span: str, tail_ratio: float, max_tokens_per_chunk: int,
    allow_dim_mismatch: bool, auto_alpha: bool = False, guard: bool = False,
) -> None:
    """Reject a different inference formula for newly exported token weights."""
    contract = application_contract(bridge)
    if contract is None:
        return
    actual = {
        "magnitude_match": magnitude_match_mode, "token_span": token_span,
        "chunk_tokens": max_tokens_per_chunk, "auto_alpha": auto_alpha,
        "guard": guard, "allow_dim_mismatch": allow_dim_mismatch,
    }
    mismatches = []
    if not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not math.isclose(
            alpha, contract["alpha_min"], rel_tol=0.0, abs_tol=1e-7):
        mismatches.append(f"alpha={alpha!r} (required {contract['alpha_min']!r})")
    if not isinstance(tail_ratio, (int, float)) or not math.isfinite(tail_ratio) or not math.isclose(
            tail_ratio, contract["tail_ratio"], rel_tol=0.0, abs_tol=1e-7):
        mismatches.append(f"tail_ratio={tail_ratio!r} (required {contract['tail_ratio']!r})")
    for key, value in actual.items():
        if value != contract[key]:
            mismatches.append(f"{key}={value!r} (required {contract[key]!r})")
    if mismatches:
        raise ValueError("Bridge application contract mismatch: " + ", ".join(mismatches))


def apply_bridge_to_conditioning(
    conditioning: Sequence[Sequence[Any]],
    bridge: WushuBridge,
    alpha: float = 0.12,
    magnitude_match_mode: str = "per_token",
    token_span: str = "all",
    tail_ratio: float = 1.0,
    max_tokens_per_chunk: int = 0,
    allow_dim_mismatch: bool = False,
) -> Tuple[List[List[Any]], ApplyReport]:
    """对 conditioning 逐段施加残差桥，返回 (新 conditioning, 报告)。

    参数
    ----
    conditioning
        ComfyUI 的 CONDITIONING（list of [tensor, dict]）。
    alpha
        融合强度。推荐 0.05~0.25；参考图模式建议 ≤0.12。
    token_span
        ``all`` = 全序列（文生视频/首尾帧同社区版）；``tail`` = 只改尾部
        ``tail_ratio`` 比例的 token（参考图模式的保守做法）。
    max_tokens_per_chunk
        >0 时对超长序列分块前向（Transformer 桥在 CPU/小显存上更稳），
        块与块之间不重叠，逐块独立处理。
    allow_dim_mismatch
        False（默认）时遇到维度不符直接报错；True 时跳过该段并在报告里说明。
    """
    if magnitude_match_mode not in MAGNITUDE_MODES:
        raise ValueError(f"magnitude_match 必须是 {MAGNITUDE_MODES} 之一")
    if token_span not in TEXT_SPAN_MODES:
        raise ValueError(f"token_span 必须是 {TEXT_SPAN_MODES} 之一")
    validate_application_contract(
        bridge, alpha=alpha, magnitude_match_mode=magnitude_match_mode,
        token_span=token_span, tail_ratio=tail_ratio,
        max_tokens_per_chunk=max_tokens_per_chunk,
        allow_dim_mismatch=allow_dim_mismatch,
    )

    result: List[List[Any]] = []
    report = ApplyReport()
    device = next(bridge.parameters()).device

    for idx, item in enumerate(conditioning):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            raise RuntimeError("CONDITIONING 结构异常：应为 [[tensor, metadata], ...]")

        native, metadata = item[0], item[1]
        if not torch.is_tensor(native) or native.ndim != 3:
            raise RuntimeError(
                f"CONDITIONING embedding 期望 [B,T,D] 张量，得到 {type(native)}"
                + (f" shape={tuple(native.shape)}" if torch.is_tensor(native) else "")
            )
        if native.shape[-1] != bridge.cfg.dim:
            msg = (
                f"#{idx} 维度不匹配：conditioning dim={native.shape[-1]}，"
                f"桥 dim={bridge.cfg.dim}"
            )
            if not allow_dim_mismatch:
                raise RuntimeError(
                    msg + "。请用与训练时同一套 H3 文本编码器，或换用对应维度的桥权重。"
                )
            report.skipped += 1
            report.messages.append("  [!] " + msg + "（已跳过）")
            result.append([native, dict(metadata)])
            continue

        is_ref = _looks_reference_conditioned(metadata if isinstance(metadata, dict) else {})
        h = native.float()
        # 设备对齐：ComfyUI 的 conditioning 一般在 CPU，而桥权重在 GPU。
        # 不显式搬过去就会报 "found at least two devices, cuda:0 and cpu!"（云主机实测）。
        h_dev = h.to(device=device)
        start, end = _token_slice(h.shape[1], token_span, tail_ratio)
        if end <= start:
            report.skipped += 1
            report.messages.append(f"  [!] #{idx} 选定 token 区间为空（{start}:{end}），已跳过")
            result.append([native, dict(metadata)])
            continue

        with torch.inference_mode():
            x = rms_normalize(h_dev[:, start:end, :]).to(dtype=torch.float32)
            if max_tokens_per_chunk and x.shape[1] > max_tokens_per_chunk:
                preds = []
                for s in range(0, x.shape[1], max_tokens_per_chunk):
                    preds.append(bridge(x[:, s : s + max_tokens_per_chunk, :]))
                pred = torch.cat(preds, dim=1)
            else:
                pred = bridge(x)
            pred = magnitude_match(pred, h_dev[:, start:end, :], magnitude_match_mode)

            hybrid = h_dev.clone()
            hybrid[:, start:end, :] = h_dev[:, start:end, :] + float(alpha) * (pred - h_dev[:, start:end, :])

        # 语义漂移：池化后的余弦距离，用来判断这次改动是否过猛。
        # 两边都要在**同一设备**上算：h 在 CPU、hybrid 在桥所在的 GPU，
        # 混着算会让 F.cosine_similarity 内部的 clamp_min 报设备不一致（云主机实测）。
        with torch.inference_mode():
            a = h_dev.mean(dim=1)
            b = hybrid.mean(dim=1)
            cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()
        report.drift.append(float(1.0 - cos))

        new_meta = dict(metadata) if isinstance(metadata, dict) else {}
        new_meta[META_KEY_PREFIX + "applied"] = True
        new_meta[META_KEY_PREFIX + "alpha"] = float(alpha)
        new_meta[META_KEY_PREFIX + "magnitude_match"] = magnitude_match_mode
        new_meta[META_KEY_PREFIX + "name"] = bridge.cfg.name
        new_meta[META_KEY_PREFIX + "arch"] = bridge.cfg.arch
        new_meta[META_KEY_PREFIX + "token_span"] = f"{start}:{end}"

        # 搬回原设备与 dtype：conditioning 必须保持 ComfyUI 期望的形态
        result.append([hybrid.to(device=native.device, dtype=native.dtype), new_meta])
        report.applied += 1
        report.dims.append(int(native.shape[-1]))
        report.tokens.append(int(native.shape[1]))
        report.ref_like.append(bool(is_ref))

    if any(report.ref_like):
        report.messages.append(
            "  [i] 检测到参考图/首尾帧 conditioning：社区版 v1 明确不支持 Ref2VA，"
            "本插件默认同样保守——建议 alpha <= 0.12、token_span 用 tail，并单独训练 ref2v 适配器再比对。"
        )
    return result, report
