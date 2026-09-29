"""训练数据集的磁盘格式（.npz + 同名 .json 元信息）。

为什么要自己定格式
------------------
训练桥需要的是 **H3 文本编码器输出的 5120 维 token 序列**。这只能在装了
H3 的 ComfyUI 里跑（外部设备跑不动那个编码器）。所以流程是：

1. 在 ComfyUI 里用 ``H3 Wushu Build Dataset`` 节点把 (负例文本, 正例文本)
   成对编码，落盘成 ``*_pairs.npz``；
2. 再用 ``H3 Wushu Train Bridge`` 节点训练并导出 safetensors；
3. 之后推理时就不再需要编码器了（桥只吃 conditioning）。

为了让 924 条语料在 5120 维下不至于爆盘，序列用变长紧凑存储：
``x_flat / x_len`` 拼在一起，token 用 float16。

同时存一份**池化向量**给评分头用（``x_pool / y_pool``，float32），
以及每条样本的规则分（``x_score / y_score``，0~1 软标签）。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .video_feedback import LABEL as VIDEO_FEEDBACK_LABEL


@dataclass
class PairMeta:
    kind: str = "wushu_bridge_pairs"
    schema: int = 1
    dim: int = 5120
    count: int = 0
    mode: str = "t2v"
    encoder: str = "unknown"
    source: str = ""
    created_at: str = ""
    supervision_type: str = "numeric_scores"
    score_label_type: str = "legacy_numeric_labels"
    records: List[Dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False, indent=2, allow_nan=False)


def _pack(seqs: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    lens = np.asarray([s.shape[0] for s in seqs], dtype=np.int32)
    if len(seqs) == 0:
        return np.zeros((0, 0), dtype=np.float16), lens
    flat = np.concatenate(seqs, axis=0)
    return flat, lens


def _unpack(flat: np.ndarray, lens: np.ndarray) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    off = 0
    for n in lens:
        n = int(n)
        out.append(flat[off : off + n])
        off += n
    return out


class PairDataset:
    """内存中的成对数据集：``x`` = 负例（逻辑欠缺），``y`` = 正例（武打逻辑完备）。"""

    def __init__(self, meta: Optional[PairMeta] = None) -> None:
        self.meta = meta or PairMeta()
        self.x: List[np.ndarray] = []
        self.y: List[np.ndarray] = []
        self.x_score: List[float] = []
        self.y_score: List[float] = []
        self.records: List[Dict[str, Any]] = []
        # 不成对的样本：只给评分头当额外负例用（例如真实被筛掉的片段对应的
        # prompt）。桥训练需要"同一事件的正反对"，用不上这些，会跳过。
        self.extra: List[np.ndarray] = []
        self.extra_score: List[float] = []
        self.extra_records: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.x)

    def add(
        self,
        x_tokens: np.ndarray,
        y_tokens: np.ndarray,
        x_score: float = 0.0,
        y_score: float = 1.0,
        record: Optional[Dict[str, Any]] = None,
    ) -> None:
        if x_tokens.ndim != 2 or y_tokens.ndim != 2:
            raise ValueError("token 序列必须是 [T, D]")
        if x_tokens.shape[-1] != y_tokens.shape[-1]:
            raise ValueError("正负例维度不一致，说明用了不同的文本编码器")
        self.x.append(np.asarray(x_tokens, dtype=np.float16))
        self.y.append(np.asarray(y_tokens, dtype=np.float16))
        self.x_score.append(float(x_score))
        self.y_score.append(float(y_score))
        self.records.append(record or {})
        self.meta.dim = int(x_tokens.shape[-1])
        self.meta.count = len(self.x)
        self.meta.records = self.records

    def add_unpaired(
        self,
        tokens: np.ndarray,
        score: float = 0.0,
        record: Optional[Dict[str, Any]] = None,
    ) -> None:
        """加一条"没有配对"的样本，仅用于训练评分头（桥训练会忽略）。"""
        if tokens.ndim != 2:
            raise ValueError("token 序列必须是 [T, D]")
        self.extra.append(np.asarray(tokens, dtype=np.float16))
        self.extra_score.append(float(score))
        self.extra_records.append(record or {})
        self.meta.dim = int(tokens.shape[-1])
        if not self.x:
            self.meta.count = max(self.meta.count, len(self.extra))

    @property
    def pooled_x(self) -> np.ndarray:
        return np.stack([s.astype(np.float32).mean(axis=0) for s in self.x], axis=0)

    @property
    def pooled_y(self) -> np.ndarray:
        return np.stack([s.astype(np.float32).mean(axis=0) for s in self.y], axis=0)

    def save(self, path: str) -> str:
        self.validate_supervision()
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        x_flat, x_len = _pack(self.x)
        y_flat, y_len = _pack(self.y)
        e_flat, e_len = _pack(self.extra)
        # 池化向量只是给诊断/快速统计用，用 fp16 存，省一半磁盘
        np.savez_compressed(
            path,
            x_flat=x_flat,
            x_len=x_len,
            y_flat=y_flat,
            y_len=y_len,
            e_flat=e_flat,
            e_len=e_len,
            x_pool=self.pooled_x.astype(np.float16) if len(self.x) else np.zeros((0, self.meta.dim), np.float16),
            y_pool=self.pooled_y.astype(np.float16) if len(self.y) else np.zeros((0, self.meta.dim), np.float16),
            x_score=np.asarray(self.x_score, dtype=np.float32),
            y_score=np.asarray(self.y_score, dtype=np.float32),
            e_score=np.asarray(self.extra_score, dtype=np.float32),
        )
        with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as f:
            f.write(self.meta.to_json())
        return path

    @classmethod
    def load(cls, path: str) -> "PairDataset":
        if not os.path.isfile(path):
            raise FileNotFoundError(f"找不到数据集：{path}")
        meta_path = os.path.splitext(path)[0] + ".json"
        raw_meta: Dict[str, Any] = {}
        meta = PairMeta()
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                raw_meta = json.load(f)
            if not isinstance(raw_meta, dict):
                raise ValueError("数据集元信息必须是 JSON 对象")
            if raw_meta.get("schema", 1) != 1 or isinstance(raw_meta.get("schema"), bool):
                raise ValueError("不支持的数据集 schema")
            if raw_meta.get("kind", "wushu_bridge_pairs") != "wushu_bridge_pairs":
                raise ValueError("数据集 kind 不匹配")
            meta = PairMeta(**raw_meta)
            if isinstance(meta.dim, bool) or not isinstance(meta.dim, int) or meta.dim <= 0:
                raise ValueError("数据集 dim 必须为正整数")
            if isinstance(meta.count, bool) or not isinstance(meta.count, int) or meta.count < 0:
                raise ValueError("数据集 count 必须为非负整数")
            if not isinstance(meta.records, list) or not all(isinstance(r, dict) for r in meta.records):
                raise ValueError("数据集 records 必须为对象列表")

        # Validate the on-disk arrays before pairing them: zip() alone silently
        # discards unmatched sequences, while unchecked lengths discard tokens.
        with np.load(path, allow_pickle=False) as z:
            packed = {}
            for prefix in ("x", "y", "e"):
                flat_key, len_key = prefix + "_flat", prefix + "_len"
                if flat_key not in z and len_key not in z and prefix == "e":
                    packed[prefix] = (np.zeros((0, 0), np.float16), np.zeros(0, np.int32))
                    continue
                if flat_key not in z or len_key not in z:
                    raise ValueError(f"缺少数据集数组 {flat_key}/{len_key}")
                flat, lengths = z[flat_key], z[len_key]
                if flat.ndim != 2 or flat.dtype.kind not in "fiu" or not np.isfinite(flat).all():
                    raise ValueError(f"{flat_key} 必须是有限实数的二维数组")
                if lengths.ndim != 1 or lengths.dtype.kind not in "iu" or (lengths <= 0).any():
                    raise ValueError(f"{len_key} 必须是一维正整数序列")
                if sum(int(n) for n in lengths) != len(flat):
                    raise ValueError(f"{len_key} 总长度与 {flat_key} 不匹配")
                packed[prefix] = flat, lengths
            if len(packed["x"][1]) != len(packed["y"][1]):
                raise ValueError("正负例序列数量不一致")
            dims = {flat.shape[1] for flat, lengths in packed.values() if len(lengths)}
            if len(dims) > 1 or (dims and next(iter(dims)) <= 0):
                raise ValueError("token embedding 维度不一致或为空")
            inferred_dim = next(iter(dims), meta.dim)
            if "dim" in raw_meta and inferred_dim != meta.dim:
                raise ValueError("元信息 dim 与 embedding 维度不一致")
            meta.dim = inferred_dim
            for prefix, (flat, lengths) in packed.items():
                if flat.shape[1] not in (meta.dim, 0) or (len(lengths) and flat.shape[1] != meta.dim):
                    raise ValueError(f"{prefix}_flat embedding 维度不一致")
            pair_count = len(packed["x"][1])
            extra_count = len(packed["e"][1])
            if "count" in raw_meta and meta.count != (pair_count or extra_count):
                raise ValueError("元信息 count 与序列数量不一致")
            if meta.records and len(meta.records) != pair_count:
                raise ValueError("元信息 records 与训练对数量不一致")
            scores = {}
            for prefix, default in (("x", 0.0), ("y", 1.0), ("e", 0.0)):
                count = len(packed[prefix][1])
                values = z[prefix + "_score"] if prefix + "_score" in z else np.full(count, default)
                if meta.supervision_type == "preference_only" and prefix in ("x", "y"):
                    if prefix + "_score" not in z or values.shape != (count,) or values.dtype.kind != "f" or not np.isnan(values).all():
                        raise ValueError("preference_only requires explicit NaN score sentinels, never numeric labels")
                    scores[prefix] = values.tolist()
                    continue
                if values.shape != (count,) or values.dtype.kind not in "fiu" or not np.isfinite(values).all():
                    raise ValueError(f"{prefix}_score 数量不一致或包含非有限值")
                if ((values < 0) | (values > 1)).any():
                    raise ValueError(f"{prefix}_score 必须位于 [0,1]")
                scores[prefix] = values.tolist()
            for prefix in ("x", "y"):
                if prefix + "_pool" in z:
                    pooled = z[prefix + "_pool"]
                    if pooled.shape != (pair_count, meta.dim) or pooled.dtype.kind not in "fiu" or not np.isfinite(pooled).all():
                        raise ValueError(f"{prefix}_pool 形状不匹配或包含非有限值")
            xs, ys, es = (_unpack(*packed[prefix]) for prefix in ("x", "y", "e"))
        ds = cls(meta)
        x_score, y_score = scores["x"], scores["y"]
        for i, (a, b) in enumerate(zip(xs, ys)):
            ds.x.append(a)
            ds.y.append(b)
            ds.x_score.append(float(x_score[i]))
            ds.y_score.append(float(y_score[i]))
            ds.records.append(meta.records[i] if i < len(meta.records) else {})
        for i, a in enumerate(es):
            ds.extra.append(a)
            ds.extra_score.append(float(scores["e"][i]))
            ds.extra_records.append({"source": "unpaired"})
        ds.meta.count = len(ds.x)
        ds.validate_supervision()
        return ds

    def validate_supervision(self) -> None:
        if self.meta.supervision_type not in ("numeric_scores", "preference_only"):
            raise ValueError("Unknown supervision_type")
        if self.meta.supervision_type == "preference_only":
            if self.meta.score_label_type != "none" or self.extra:
                raise ValueError("preference_only requires score_label_type=none and no unpaired score samples")
            if len(self.records) != len(self.x) or any(
                    r.get("pair_preference") != "good" or
                    r.get("label_type") not in ("authoring_rubric", VIDEO_FEEDBACK_LABEL) or
                    "bad_score" in r or "good_score" in r for r in self.records):
                raise ValueError("preference_only requires explicit supported preferences without score fields")
            labels = {r.get("label_type") for r in self.records}
            if len(labels) != 1:
                raise ValueError("Do not mix authoring and video-feedback preferences")
            if VIDEO_FEEDBACK_LABEL in labels and any(
                    not isinstance(r.get("video_feedback"), dict) or
                    not r.get("scene_id") or not r.get("scene_family_id") or
                    not r.get("work_id") or r.get("group_id") != r.get("work_id")
                    for r in self.records):
                raise ValueError("video_feedback_v1 requires external binding and whole-work group_id")
            if len(self.x_score) != len(self.x) or len(self.y_score) != len(self.y) or not np.isnan(self.x_score + self.y_score).all():
                raise ValueError("preference_only score arrays must contain only missing-label NaN sentinels")
        else:
            if not self.meta.score_label_type or self.meta.score_label_type == "none":
                raise ValueError("numeric_scores requires an explicit score_label_type")
            values = np.asarray(self.x_score + self.y_score + self.extra_score)
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError("Invalid numeric score labels")

    def require_score_labels(self) -> None:
        self.validate_supervision()
        if self.meta.supervision_type != "numeric_scores":
            raise ValueError("Judge training requires actual numeric score labels; preference_only supports --stage bridge only")

    def split(self, val_ratio: float = 0.1, seed: int = 1234) -> Tuple["PairDataset", "PairDataset"]:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self))
        n_val = max(1, int(round(len(self) * val_ratio))) if len(self) > 4 else 0
        val_idx = set(idx[:n_val].tolist())
        train, val = PairDataset(PairMeta(**{**self.meta.__dict__, "records": []})), PairDataset(
            PairMeta(**{**self.meta.__dict__, "records": []})
        )
        for i in range(len(self)):
            dst = val if i in val_idx else train
            dst.add(self.x[i], self.y[i], self.x_score[i], self.y_score[i], self.records[i])
        return train, val

    def stats(self) -> Dict[str, Any]:
        if not self.x:
            return {"count": 0}
        tx = [s.shape[0] for s in self.x]
        ty = [s.shape[0] for s in self.y]
        return {
            "count": len(self.x),
            "dim": self.meta.dim,
            "unpaired_extra": len(self.extra),
            "x_tokens": {"min": int(min(tx)), "max": int(max(tx)), "mean": float(np.mean(tx))},
            "y_tokens": {"min": int(min(ty)), "max": int(max(ty)), "mean": float(np.mean(ty))},
            "supervision_type": self.meta.supervision_type,
            "score_label_type": self.meta.score_label_type,
            "x_score_mean": float(np.mean(self.x_score)) if self.x_score and self.meta.supervision_type == "numeric_scores" else None,
            "y_score_mean": float(np.mean(self.y_score)) if self.y_score and self.meta.supervision_type == "numeric_scores" else None,
        }


def pool_tokens(flat: np.ndarray, lens: np.ndarray) -> np.ndarray:
    """兼容旧格式：把紧凑序列池化成 [N, D]。"""
    pooled = []
    off = 0
    for n in lens:
        n = int(n)
        pooled.append(flat[off : off + n].astype(np.float32).mean(axis=0))
        off += n
    return np.stack(pooled, axis=0) if pooled else np.zeros((0, flat.shape[-1]), np.float32)
