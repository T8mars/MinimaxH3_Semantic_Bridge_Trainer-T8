"""武打语义桥（Wushu Semantic Bridge）的模型架构与权重读写。

本模块只依赖 torch，不依赖 ComfyUI，方便脱离 ComfyUI 单独训练 / 单测。

两种架构
--------
``mlp``
    与社区版 MiniMax-H3-Semantic-Bridge v1 完全同构：逐 token 的
    ``D -> H -> H -> D`` 全连接残差头。用来做「同架构对照」以及直接加载
    社区那 11MB 适配器做 A/B（权重键名一致：fc1/fc2/fc3）。

``trans``
    跨 token 的轻量 Transformer 残差桥。武打逻辑的关键在于「整段是否自洽」
    —— 刀和拳不能同时出现在同一只手、格距 2 格不能拳脚命中、死线前必须
    有结果 —— 这些约束只有看到整条 token 序列才可能建模，逐 token MLP
    在原理上做不到。默认 2 层 / d=256 / 4 头，参数量约 3.2M（fp32 ≈ 13MB），
    与社区版同量级。

两个架构共享同一条应用公式（见 ``apply.py``）：
``hybrid = h + alpha * (bridge(h) - h)``，alpha 小、幅度对齐，保证不跑出
H3 文本编码器的原生流形。
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

SCHEMA_VERSION = 1
ARCH_MLP = "mlp"
ARCH_TRANS = "trans"
SUPPORTED_ARCHS = (ARCH_MLP, ARCH_TRANS)


@dataclass
class BridgeConfig:
    """残差桥的结构与训练来源元信息。"""

    arch: str = ARCH_TRANS
    dim: int = 5120
    hidden: int = 512
    layers: int = 2
    heads: int = 4
    dropout: float = 0.0
    max_tokens: int = 4096
    # Opt-in identity path for new experiments; older author/project weights
    # keep the original full-prediction behavior when this metadata is absent.
    residual_skip: bool = False
    residual_scale: float = 0.1
    # 训练来源信息（仅记录，不影响前向计算）
    mode: str = "t2v"           # t2v / ref2v / any
    schema: int = SCHEMA_VERSION
    name: str = "wushu_bridge"
    note: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.arch not in SUPPORTED_ARCHS:
            raise ValueError(f"未知 arch: {self.arch}，可选 {SUPPORTED_ARCHS}")
        if not isinstance(self.residual_skip, bool):
            raise ValueError("residual_skip 必须是布尔值")
        if not math.isfinite(self.residual_scale) or not 0 < self.residual_scale <= 1:
            raise ValueError("residual_scale 必须在 (0, 1] 内")
        if self.arch == ARCH_MLP:
            # 与社区版同构：中间层宽度=hidden，层数固定 2 个隐层
            if self.hidden <= 0:
                raise ValueError("hidden 必须为正整数")
        else:
            if self.hidden % self.heads != 0:
                raise ValueError(
                    f"hidden({self.hidden}) 必须能被 heads({self.heads}) 整除"
                )

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _sinusoidal_positions(max_tokens: int, hidden: int) -> torch.Tensor:
    pos = torch.arange(max_tokens, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, hidden, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden))
    pe = torch.zeros(max_tokens, hidden, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)[:, : pe[:, 1::2].shape[1]]
    return pe.unsqueeze(0)                                   # [1, L, H]


class _ResidualMLP(nn.Module):
    """社区版同构：逐 token MLP。"""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, hidden, bias=True)
        self.fc3 = nn.Linear(hidden, dim, bias=True)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)


class _TokenTransformer(nn.Module):
    """跨 token 轻量 Transformer。"""

    def __init__(
        self,
        dim: int,
        hidden: int,
        layers: int,
        heads: int,
        dropout: float,
        max_tokens: int,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(dim, hidden, bias=True)
        # 用固定正弦位置编码而不是可学习的位置表：省掉 max_tokens*hidden 参数
        # （4096*512 就是 200 万参数），而且天然支持任意长度插值。
        self.register_buffer(
            "_pe", _sinusoidal_positions(max_tokens, hidden), persistent=False
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, dim, bias=True)
        self.max_tokens = int(max_tokens)

    def _positions(self, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        pe = self._pe
        if n <= pe.shape[1]:
            return pe[:, :n].to(device=device, dtype=dtype)
        # 超长序列：对位置嵌入做线性插值，保证不在长 conditioning 上崩掉
        p = pe.transpose(1, 2)                              # [1, H, L]
        p = torch.nn.functional.interpolate(
            p, size=n, mode="linear", align_corners=False
        )
        return p.transpose(1, 2).to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, T, D]
        h = self.in_proj(x)
        h = h + self._positions(h.shape[1], h.device, h.dtype)
        h = self.encoder(h, src_key_padding_mask=None if mask is None else ~mask)
        h = self.norm(h)
        return self.out_proj(h)


class WushuBridge(nn.Module):
    """统一封装：``forward(x)`` 返回与输入同形状的修正后 embedding。"""

    def __init__(self, cfg: BridgeConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.arch == ARCH_MLP:
            self.net: nn.Module = _ResidualMLP(cfg.dim, cfg.hidden)
        else:
            self.net = _TokenTransformer(
                cfg.dim, cfg.hidden, cfg.layers, cfg.heads, cfg.dropout, cfg.max_tokens
            )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"期望 [B,T,D] 输入，得到 {tuple(x.shape)}")
        if x.shape[-1] != self.cfg.dim:
            raise ValueError(
                f"embedding 维度不匹配：模型 dim={self.cfg.dim}，输入 {x.shape[-1]}。"
                "请用与训练时相同的 H3 文本编码器。"
            )
        if mask is not None:
            if mask.shape != x.shape[:2] or mask.dtype != torch.bool:
                raise ValueError("mask 必须是 [B,T] bool，True 表示有效 token")
            mask = mask.to(device=x.device)
            if not bool(mask.any(dim=1).all()):
                raise ValueError("每条序列至少需要一个有效 token")
        correction = self.net(x, mask=mask) if self.cfg.arch == ARCH_TRANS else self.net(x)
        return x + self.cfg.residual_scale * correction if self.cfg.residual_skip else correction

    # ---- 统计 ----
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def approx_bytes(self, dtype: str = "fp32") -> int:
        width = {"fp32": 4, "fp16": 2, "bf16": 2}[dtype]
        return self.num_params() * width


def build_bridge(cfg: BridgeConfig) -> WushuBridge:
    return WushuBridge(cfg)


def save_bridge(path: str, model: WushuBridge, extra_meta: Optional[Dict[str, Any]] = None) -> str:
    """把桥存成 safetensors（含 config 元信息），供 ComfyUI 侧加载。"""
    from safetensors.torch import save_file

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    cfg = model.cfg
    meta = {k: str(v) for k, v in asdict(cfg).items() if k != "extra"}
    meta["extra_json"] = json.dumps(cfg.extra, ensure_ascii=False)
    if extra_meta:
        for k, v in extra_meta.items():
            meta[f"train_{k}"] = str(v)
    tensors = {k: v.detach().to(torch.float32).contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(tensors, path, metadata=meta)
    return path


def config_from_metadata(meta: Dict[str, str], state: Dict[str, torch.Tensor]) -> BridgeConfig:
    """从 safetensors 元信息推断结构（元信息缺失时按张量形状兜底）。"""
    meta = meta or {}
    if "fc1.weight" in state or "net.fc1.weight" in state:  # 社区版 / 本插件 MLP
        prefix = "" if "fc1.weight" in state else "net."
        dim = int(state[prefix + "fc1.weight"].shape[1])
        hidden = int(state[prefix + "fc1.weight"].shape[0])
        if prefix + "fc3.weight" in state and state[prefix + "fc3.weight"].shape[0] != dim:
            raise ValueError("MLP 输入和输出 embedding 维度不一致")
        return BridgeConfig(
            arch=ARCH_MLP,
            dim=dim,
            hidden=hidden,
            layers=2,
            mode=meta.get("mode", "any"),
            name=meta.get("name", "semantic_bridge_v1"),
            note="按社区 MiniMax-H3-Semantic-Bridge 权重形状推断",
        )
    if "net.in_proj.weight" in state:
        hidden = int(state["net.in_proj.weight"].shape[0])
        dim = int(state["net.in_proj.weight"].shape[1])
        layers = 0
        while f"net.encoder.layers.{layers}.norm1.weight" in state:
            layers += 1
        heads = int(meta.get("heads", 4)) if meta.get("heads") else 4
        while hidden % heads != 0 and heads > 1:
            heads -= 1
        return BridgeConfig(
            arch=ARCH_TRANS,
            dim=dim,
            hidden=hidden,
            layers=max(layers, 1),
            heads=heads,
            max_tokens=int(meta.get("max_tokens", 4096)),
            residual_skip=meta.get("residual_skip", "False").lower() == "true",
            residual_scale=float(meta.get("residual_scale", 0.1)),
            mode=meta.get("mode", "any"),
            name=meta.get("name", "wushu_trans_bridge"),
        )
    raise ValueError("无法识别的桥权重：既不是 MLP 也不是 Transformer 结构。")


def load_bridge(path: str, device: str = "cpu", dtype: torch.dtype = torch.float32) -> WushuBridge:
    """加载残差桥。兼容社区版 ``MiniMaxH3_SemanticBridge_v1.safetensors``。"""
    from safetensors import safe_open
    from safetensors.torch import load_file

    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到残差桥权重：{path}")

    meta: Dict[str, str] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        meta = dict(f.metadata() or {})
    state = load_file(path, device="cpu")

    if meta.get("arch") in SUPPORTED_ARCHS:
        cfg = BridgeConfig(
            arch=meta["arch"],
            dim=int(meta.get("dim", 5120)),
            hidden=int(meta.get("hidden", 512)),
            layers=int(meta.get("layers", 2)),
            heads=int(meta.get("heads", 4)),
            dropout=float(meta.get("dropout", 0.0)),
            max_tokens=int(meta.get("max_tokens", 4096)),
            residual_skip=meta.get("residual_skip", "False").lower() == "true",
            residual_scale=float(meta.get("residual_scale", 0.1)),
            mode=meta.get("mode", "any"),
            name=meta.get("name", os.path.basename(path)),
            note=meta.get("note", ""),
        )
        if meta.get("extra_json"):
            try:
                cfg.extra = json.loads(meta["extra_json"])
            except Exception:
                cfg.extra = {}
    else:
        cfg = config_from_metadata(meta, state)

    if cfg.arch == ARCH_MLP and "fc1.weight" in state:
        state = {"net." + k: v for k, v in state.items()}
    model = WushuBridge(cfg)
    model.load_state_dict(state, strict=True)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
