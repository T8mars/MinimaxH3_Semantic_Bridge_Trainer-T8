"""ComfyUI 节点定义：H3 武打语义逻辑翻译桥。

节点分三类
----------
**推理（接在 conditioning 之间）**
* ``H3 Wushu Semantic Bridge``  —— CONDITIONING -> CONDITIONING，施加残差桥
* ``H3 Wushu JEV Score``        —— CONDITIONING -> 武打逻辑概率（0~1）
* ``H3 Wushu Clear Cache``

**训练（在装好 H3 的那台机器上原地跑，不依赖云 GPU）**
* ``H3 Wushu Build Dataset``    —— 用 H3 自带文本编码器把正/负例编码成 embedding
* ``H3 Wushu Harvest Pair``     —— 从两段真实 CONDITIONING 直接采一对样本
* ``H3 Wushu Train Bridge``     —— 训练残差桥并导出 safetensors
* ``H3 Wushu Train JEV Head``   —— 训练评分头并导出 safetensors

**规则（不需要模型，纯 CPU）**
* ``H3 Wushu Lint Prompt``      —— 武打逻辑体检（h3lint 的 Python 版）
* ``H3 Wushu Degrade Preview``  —— 看一条正例被降级成什么样（造训练对用）
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from . import lexicons, pairs as pairs_mod
from .apply import MAGNITUDE_MODES, TEXT_SPAN_MODES, apply_bridge_to_conditioning, validate_application_contract
from .bridge_model import ARCH_MLP, ARCH_TRANS, load_bridge
from .dataset import PairDataset
from .judge import load_judge, score_conditioning
from .pairs import DegradeProfile, build_pairs, dump_pairs, load_corpus, pairs_stats, score_text

try:  # ComfyUI 环境
    import folder_paths  # type: ignore

    MODELS_ROOT = getattr(folder_paths, "models_dir", os.path.join(os.path.dirname(__file__), "..", "models"))
except Exception:  # 脱离 ComfyUI 也能 import（单测用）
    folder_paths = None
    MODELS_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

BRIDGE_DIR = os.path.join(MODELS_ROOT, "wushu_bridge")
DATASET_DIR = os.path.join(BRIDGE_DIR, "datasets")
os.makedirs(BRIDGE_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

if folder_paths is not None:
    try:
        folder_paths.add_model_folder_path("wushu_bridge", BRIDGE_DIR)
    except Exception:
        pass

_BRIDGE_CACHE: Dict[Tuple[str, str, str], Any] = {}
_JUDGE_CACHE: Dict[Tuple[str, str], Any] = {}


# ── 通用小工具 ──────────────────────────────────────────────────────────

def _list_weights() -> List[str]:
    if folder_paths is not None:
        try:
            files = folder_paths.get_filename_list("wushu_bridge")
            files = [f for f in files if f.lower().endswith(".safetensors")]
            if files:
                return sorted(files)
        except Exception:
            pass
    if os.path.isdir(BRIDGE_DIR):
        files = [f for f in os.listdir(BRIDGE_DIR) if f.lower().endswith(".safetensors")]
        if files:
            return sorted(files)
    return []


def _resolve_weight(name: str) -> str:
    if folder_paths is not None:
        try:
            path = folder_paths.get_full_path("wushu_bridge", name)
            if path and os.path.isfile(path):
                return path
        except Exception:
            pass
    direct = os.path.join(BRIDGE_DIR, name)
    if os.path.isfile(direct):
        return direct
    raise FileNotFoundError(
        f"找不到权重 {name}\n请把 safetensors 放进：\n{BRIDGE_DIR}\n"
        "（或用插件自带的训练节点先生成一个）"
    )


def _weight_choices() -> List[str]:
    files = _list_weights()
    return files if files else ["(还没有权重，请先训练或用训练节点的输出)"]


def _detect_kind(name: str) -> str:
    low = name.lower()
    if "judge" in low or "jev" in low:
        return "judge"
    return "bridge"


def _bridge_choices() -> List[str]:
    files = [f for f in _list_weights() if _detect_kind(f) == "bridge"]
    return files if files else ["(还没有桥权重，请先跑 H3 Wushu Train Bridge)"]


def _judge_choices() -> List[str]:
    files = [f for f in _list_weights() if _detect_kind(f) == "judge"]
    return ["none"] + files if files else ["none"]


def _weight_sha256(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as weight_file:
        for chunk in iter(lambda: weight_file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _get_bridge(name: str, device: str):
    path = _resolve_weight(name)
    # A trainer or package installer can atomically replace a weight while
    # ComfyUI stays open. Path/device alone would silently keep the old model.
    digest = _weight_sha256(path)
    key = (os.path.abspath(path), device, digest)
    if key not in _BRIDGE_CACHE:
        model = load_bridge(path, device=device)
        for stale in [k for k in _BRIDGE_CACHE if k[:2] == key[:2]]:
            del _BRIDGE_CACHE[stale]
        _BRIDGE_CACHE[key] = model
    return _BRIDGE_CACHE[key]


def _get_judge(name: str, device: str):
    path = _resolve_weight(name)
    key = (os.path.abspath(path), device)
    if key not in _JUDGE_CACHE:
        _JUDGE_CACHE[key] = load_judge(path, device=device)
    return _JUDGE_CACHE[key]


def _pick_torch_device(prefer: str = "auto") -> str:
    if prefer != "auto":
        return prefer
    return "cuda" if torch.cuda.is_available() else "cpu"


def _progress(total: int):
    try:
        import comfy.utils  # type: ignore

        return comfy.utils.ProgressBar(total)
    except Exception:
        class _Null:
            def update(self, *a, **k):
                return None

            def update_absolute(self, *a, **k):
                return None

        return _Null()


def _conditioning_tensor(cond) -> torch.Tensor:
    """从 CLIP 输出里取出 [B,T,D] 张量。"""
    if isinstance(cond, dict):
        cond = cond.get("cond") or cond.get("conditioning")
    if not cond:
        raise RuntimeError("CLIP 编码返回空结果")
    item = cond[0]
    tensor = item[0] if isinstance(item, (list, tuple)) else item
    if not torch.is_tensor(tensor):
        raise RuntimeError(f"无法识别的 CLIP 输出类型：{type(tensor)}")
    if tensor.ndim != 3:
        raise RuntimeError(f"CLIP 输出的 embedding 期望 [B,T,D]，得到 {tuple(tensor.shape)}")
    return tensor


def _encode_texts(clip, texts: List[str], max_tokens: int = 0, progress=None) -> List["Any"]:
    """用 ComfyUI 的 CLIP 对象批量编码文本（H3 的文本编码器就在这里）。"""
    import numpy as np

    out: List["Any"] = []
    for i, text in enumerate(texts):
        tokens = clip.tokenize(text)
        cond = clip.encode_from_tokens_scheduled(tokens)
        t = _conditioning_tensor(cond).detach().float().cpu()
        if max_tokens and t.shape[1] > max_tokens:
            t = t[:, :max_tokens, :]
        out.append(t[0].numpy().astype("float16"))
        if progress is not None:
            progress.update(1)
        if i % 64 == 63:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return out


def _ui(result, *texts):
    """给"只出报告"的节点加 `ui` 文本，result 元组保持不变。

    两个原因（都是在真 ComfyUI 上实测出来的）：
    1. 报告类节点必须声明 ``OUTPUT_NODE = True``，否则单独把它们放进图里点运行，
       ComfyUI 会直接拒执行：``prompt_no_outputs: Prompt has no outputs``；
    2. 报告要能在节点上直接看到，就得放进 ``ui["text"]``，光靠 STRING 输出
       用户还得再接一个显示节点。
    """
    lines = [str(t) for t in texts if t is not None and str(t).strip()]
    return {"ui": {"text": lines}, "result": result}



# ══════════════════════════════════════════════════════════════════════
# 推理节点
# ══════════════════════════════════════════════════════════════════════

class H3WushuSemanticBridge:
    """把武打残差桥接在 conditioning 之间，输出修正后的 CONDITIONING。

    只替换 embedding 张量，``minimax_keyframes`` / ``minimax_refs`` /
    ``minimax_frame_count`` 等 metadata 原样保留，所以首尾帧与参考图工作流
    都能安全串在这条链上。
    """

    @classmethod
    def IS_CHANGED(cls, bridge, **kwargs):
        # ComfyUI may reuse a previous CONDITIONING output without calling
        # apply(). Include the file contents in its execution fingerprint.
        return _weight_sha256(_resolve_weight(bridge))

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "bridge": (_bridge_choices(),),
                "alpha": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01}),
                "magnitude_match": (list(MAGNITUDE_MODES), {"default": "per_token"}),
                "token_span": (list(TEXT_SPAN_MODES), {"default": "all"}),
                "tail_ratio": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 1.0, "step": 0.05}),
                "device": (["auto", "cpu", "cuda"], {"default": "auto"}),
                "chunk_tokens": ("INT", {"default": 0, "min": 0, "max": 65536, "step": 256}),
                "allow_dim_mismatch": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "judge": (_judge_choices(),),
                "auto_alpha": ("BOOLEAN", {"default": False,
                                           "tooltip": "用评分头自动调 alpha：逻辑分越低修得越多"}),
                "auto_alpha_max": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "guard": ("BOOLEAN", {"default": False,
                                      "tooltip": "改动后评分没变好就回退，避免把好稿子改坏"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING", "FLOAT")
    RETURN_NAMES = ("conditioning", "report", "logic_score")
    FUNCTION = "apply"
    CATEGORY = "MiniMax H3/Wushu Bridge"
    # 桥是链路中段节点，但仍标成 output 节点：这样它的 report（漂移、改动前后逻辑分）
    # 会直接显示在节点上，用户不必再接一个文本显示节点才能看到（实测这个很影响可用性）。
    OUTPUT_NODE = True
    DESCRIPTION = "武打语义逻辑翻译：在条件空间对 CONDITIONING 做武打逻辑修正（残差桥）"

    def apply(
        self,
        conditioning,
        bridge,
        alpha,
        magnitude_match,
        token_span,
        tail_ratio,
        device,
        chunk_tokens,
        allow_dim_mismatch,
        judge="none",
        auto_alpha=False,
        auto_alpha_max=0.25,
        guard=False,
    ):
        dev = _pick_torch_device(device)
        model = _get_bridge(bridge, dev)
        validate_application_contract(
            model, alpha=float(alpha), magnitude_match_mode=magnitude_match,
            token_span=token_span, tail_ratio=float(tail_ratio),
            max_tokens_per_chunk=int(chunk_tokens),
            allow_dim_mismatch=bool(allow_dim_mismatch),
            auto_alpha=bool(auto_alpha), guard=bool(guard),
        )
        notes: List[str] = []

        judge_model = None
        score_before = None
        if judge and judge != "none":
            judge_model = _get_judge(judge, dev)

        if judge_model is not None:
            score_before, per = score_conditioning(judge_model, conditioning)
            notes.append(f"改动前武打逻辑分：{score_before:.3f}（每段 {[round(x,3) for x in per]}）")
            if auto_alpha:
                # 分数越低，修得越多；分数已经很高就几乎不动
                deficit = max(0.0, 0.75 - score_before) / 0.75
                alpha = float(min(auto_alpha_max, max(0.0, alpha + deficit * auto_alpha_max)))
                notes.append(f"自动 alpha -> {alpha:.3f}")

        result, rep = apply_bridge_to_conditioning(
            conditioning,
            model,
            alpha=float(alpha),
            magnitude_match_mode=magnitude_match,
            token_span=token_span,
            tail_ratio=float(tail_ratio),
            max_tokens_per_chunk=int(chunk_tokens),
            allow_dim_mismatch=bool(allow_dim_mismatch),
        )

        final_score = 0.0
        if judge_model is not None:
            score_after, per_after = score_conditioning(judge_model, result)
            notes.append(f"改动后武打逻辑分：{score_after:.3f}（每段 {[round(x,3) for x in per_after]}）")
            if guard and score_before is not None and score_after < score_before - 1e-3:
                notes.append("守护开启：改动后分数下降，已回退为原 conditioning。")
                result = [[item[0], dict(item[1])] for item in conditioning]
                final_score = score_before
            else:
                final_score = score_after
        # 报告只拼一次：之前这里拼了两回，导致「改动前」那行在节点上出现两次
        rep.messages = notes + rep.messages

        _txt = rep.to_text()
        return _ui((result, _txt, float(final_score)), _txt)


class H3WushuJevScore:
    """JEV 式打分：不吐字，直接对 CONDITIONING 给武打逻辑概率。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "judge": (_judge_choices(),),
                "aggregate": (["mean", "min", "max"], {"default": "mean"}),
                "device": (["auto", "cpu", "cuda"], {"default": "auto"}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING", "BOOLEAN")
    RETURN_NAMES = ("score", "report", "pass")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3/Wushu Bridge"
    OUTPUT_NODE = True
    DESCRIPTION = "直接用小型评分头读 CONDITIONING，输出武打逻辑合格概率（0~1）"

    def run(self, conditioning, judge, aggregate, device, threshold):
        if judge == "none":
            _msg = "未选择评分头权重。请先跑 H3 Wushu Train JEV Head 生成一个。"
            return _ui((0.0, _msg, False), _msg)
        dev = _pick_torch_device(device)
        model = _get_judge(judge, dev)
        score, per = score_conditioning(model, conditioning, aggregate=aggregate)
        passed = score >= float(threshold)
        report = (
            f"武打逻辑分：{score:.3f}（阈值 {threshold:.2f} -> {'通过' if passed else '不通过'}）\n"
            f"每段：{[round(x, 3) for x in per]}\n"
            f"权重：{judge}｜参数量 {model.num_params():,}"
        )
        return _ui((float(score), report, bool(passed)), report)


class H3WushuLocateTextSpan:
    """诊断用：找出参考图 conditioning 里"文本 token"落在哪一段。

    做法：把同一段文字**单独**编码一次（文本流），再和目标 conditioning
    逐 token 比对，用子序列匹配找出文本 token 的起止下标。有了它就可以
    只对文本段施加桥，避免动到参考图 token。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target": ("CONDITIONING",),
                "text_only": ("CONDITIONING",),
            }
        }

    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("start", "end", "report")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3/Wushu Bridge"
    OUTPUT_NODE = True

    def run(self, target, text_only):
        a = target[0][0][0].float()
        b = text_only[0][0][0].float()
        if a.shape[-1] != b.shape[-1]:
            raise RuntimeError("两段 conditioning 维度不同，无法比对")
        n, m = a.shape[0], b.shape[0]
        # 子序列匹配：在 a 里找 b[0] 的最佳匹配起点，然后顺序推进
        best = None
        for start in range(max(1, n - m - 64), n):
            j = 0
            i = start
            hits = 0
            while i < n and j < m:
                if torch.allclose(a[i], b[j], atol=1e-3, rtol=1e-3):
                    hits += 1
                    j += 1
                i += 1
            if hits >= max(4, int(m * 0.6)):
                best = (start, i)
                break
        if best is None:
            _msg = f"没能定位文本段（target {n} tokens / text_only {m} tokens）。建议改用 token_span=all 或手动给区间。"
            return _ui((0, n, _msg), _msg)
        _msg = f"文本段定位：{best[0]} ~ {best[1]}（共 {n} tokens，文本单独编码 {m} tokens）"
        return _ui((int(best[0]), int(best[1]), _msg), _msg)


class H3WushuClearCache:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clear": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "run"
    CATEGORY = "MiniMax H3/Wushu Bridge"
    OUTPUT_NODE = True

    def run(self, clear):
        if clear:
            _BRIDGE_CACHE.clear()
            _JUDGE_CACHE.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _msg = "已清空武打桥 / 评分头缓存。"
            return _ui((_msg,), _msg)
        _msg = "未做任何操作。"
        return _ui((_msg,), _msg)


# ══════════════════════════════════════════════════════════════════════
# 训练节点
# ══════════════════════════════════════════════════════════════════════

class H3WushuBuildDataset:
    """用 H3 自带文本编码器把正/负例编码成训练数据集。

    为什么必须在 ComfyUI 里跑：训练数据是 H3 文本编码器输出的 5120 维 token
    序列，只有装了 H3 的环境拿得到。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "mode": (["t2v", "ref2v"], {"default": "t2v",
                                            "tooltip": "文生视频 / 多参考图：两套 H3 字段不同，请分别建数据集"}),
                "output_name": ("STRING", {"default": "wushu_pairs_t2v"}),
                "variants_per_source": ("INT", {"default": 3, "min": 1, "max": 12}),
                "intensity": ("INT", {"default": 3, "min": 1, "max": 5,
                                      "tooltip": "降级强度：越大负例越残缺"}),
                "min_good_score": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.05,
                                "tooltip": "正例的复合分门槛（相对值，不是绝对质量）。你的同分布语料中位约 0.54，默认 0.45 只滤掉明显不合格的稿子"}),
                "score_mode": (["composite", "logic", "lint"], {"default": "composite",
                                "tooltip": "composite=0.4*壳结构分+0.6*武打逻辑分（推荐，中英双语）；logic=只算武打逻辑；lint=只算 h3lint 壳结构分"}),
                "include_seeds": ("BOOLEAN", {"default": True}),
                "max_pairs": ("INT", {"default": 3000, "min": 4, "max": 200000}),
                "max_tokens": ("INT", {"default": 0, "min": 0, "max": 65536, "step": 128}),
            },
            "optional": {
                "corpus_path": ("STRING", {"default": "", "multiline": False,
                                           "tooltip": "你自己的提示词库：文件、目录或 ; 分隔的多个路径（只读，不会被修改）"}),
                "ops": ("STRING", {"default": "", "multiline": False,
                                   "tooltip": "降级算子，逗号分隔；留空=只动逻辑层（force_chain,distance,contact,feedback,defense,moves,ending）"}),
                "rejected_csv": ("STRING", {"default": "", "multiline": False,
                                            "tooltip": "LoRA 打标表 metadata.csv（列含 video,prompt）；配合 rejected_dir 挖出真实负例"}),
                "rejected_dir": ("STRING", {"default": "", "multiline": False,
                                            "tooltip": "被筛掉的片段目录（*.mp4）；对应 prompt 会作为额外负例喂给评分头"}),
                "include_design_drafts": ("BOOLEAN", {"default": False,
                                            "tooltip": "额外收「无 H3 壳但动作密度高」的设计稿散文（如 fight-design.md 的 7 条中文打斗稿）。它们与成品 H3 提示词不同分布，先小规模试"}),
                "overwrite": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("dataset_path", "report")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3/Wushu Bridge"
    OUTPUT_NODE = True

    def run(
        self,
        clip,
        mode,
        output_name,
        variants_per_source,
        intensity,
        min_good_score,
        score_mode,
        include_seeds,
        max_pairs,
        max_tokens,
        corpus_path="",
        ops="",
        rejected_csv="",
        rejected_dir="",
        include_design_drafts=False,
        overwrite=True,
    ):
        out_path = os.path.join(DATASET_DIR, f"{output_name}.npz")
        if os.path.isfile(out_path) and not overwrite:
            _msg = f"数据集已存在，未覆盖：{out_path}"
            return _ui((out_path, _msg), _msg)

        op_list = [s.strip() for s in ops.split(",") if s.strip()] if ops.strip() else []
        profile = DegradeProfile(ops=op_list, intensity=int(intensity))

        sources: List[str] = []
        if corpus_path.strip():
            sources = load_corpus(
                [p.strip() for p in corpus_path.split(";") if p.strip()],
                mode=mode,
                include_design=bool(include_design_drafts),
            )

        pair_list = build_pairs(
            sources=sources,
            mode=mode,
            variants_per_source=int(variants_per_source),
            profile=profile,
            include_seeds=bool(include_seeds),
            score_mode=score_mode,
            min_good_score=float(min_good_score),
            max_pairs=int(max_pairs),
        )
        if not pair_list:
            raise RuntimeError("没有生成任何训练对。请检查语料路径，或把 include_seeds 打开。")

        stats = pairs_stats(pair_list)
        texts: List[str] = []
        for p in pair_list:
            texts.append(p.bad)
            texts.append(p.good)

        # 真实负例：被筛掉的片段对应的 prompt（不成对，只给评分头用）
        rejected: List[str] = []
        if rejected_csv.strip() and rejected_dir.strip():
            rejected = pairs_mod.load_rejected_prompts(rejected_csv.strip(), rejected_dir.strip())
        texts.extend(rejected)
        if rejected:
            print(f"[WushuBridge] 额外负例（被筛片段对应 prompt）：{len(rejected)} 条")

        print(f"[WushuBridge] 开始编码 {len(texts)} 条文本（mode={mode}）…")
        bar = _progress(len(texts))
        t0 = time.time()
        embs = _encode_texts(clip, texts, max_tokens=int(max_tokens), progress=bar)
        elapsed = time.time() - t0

        ds = PairDataset()
        ds.meta.mode = mode
        ds.meta.source = corpus_path or "seeds"
        ds.meta.encoder = type(clip).__name__
        import datetime

        ds.meta.created_at = datetime.datetime.now().isoformat(timespec="seconds")
        for i, p in enumerate(pair_list):
            ds.add(embs[2 * i], embs[2 * i + 1], p.bad_score, p.good_score, p.to_record())
        for j, text in enumerate(rejected):
            ds.add_unpaired(embs[2 * len(pair_list) + j], score=0.0,
                            record={"source": "rejected", "head": text[:120]})

        ds.save(out_path)
        dump_pairs(pair_list, os.path.splitext(out_path)[0] + "_pairs.jsonl")
        st = ds.stats()
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        report = (
            f"数据集已写出：{out_path}\n"
            f"训练对 {st['count']} 条｜额外负例 {st.get('unpaired_extra', 0)} 条｜"
            f"embedding 维度 {st['dim']}｜模式 {mode}\n"
            f"负例 token 数 {st['x_tokens']}｜正例 token 数 {st['y_tokens']}\n"
            f"负例均分 {st['x_score_mean']}｜正例均分 {st['y_score_mean']}\n"
            f"文件大小 {size_mb:.1f} MB（每对约 {size_mb/max(1,st['count']):.2f} MB）｜编码耗时 {elapsed:.0f}s\n"
            f"降级算子使用情况：{json.dumps(stats.get('op_usage', {}), ensure_ascii=False)}\n"
            f"语料来源：{corpus_path or '（仅内置种子）'}｜语料条数 {len(sources)}\n"
            "提示：真实 H3 conditioning 的 token 数远大于种子文本，数据集会长得比较快；"
            "先用 max_pairs 控制在几百对做一次试训，确认效果再放大。"
        )
        print("[WushuBridge] " + report.replace("\n", " | "))
        return _ui((out_path, report), report)


class H3WushuHarvestPair:
    """从两段真实 CONDITIONING 采一对样本（参考生视频模式强烈建议用它）。

    做法：同一个工作流里放两个 CLIPTextEncode（同样的参考图），一个填"粗糙
    写法"（bad），一个填"逻辑完备写法"（good），把两条 CONDITIONING 接到本
    节点上，即可把这对 embedding 追加进数据集——这样采到的是**含参考图 token
    的真实 conditioning**，比纯文本编码更贴合 Ref2VA。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bad_conditioning": ("CONDITIONING",),
                "good_conditioning": ("CONDITIONING",),
                "dataset_path": ("STRING", {"default": "wushu_pairs_ref2v.npz"}),
                "create_if_missing": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "bad_score": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "good_score": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("dataset_path", "report")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3/Wushu Bridge"
    OUTPUT_NODE = True

    def run(self, bad_conditioning, good_conditioning, dataset_path, create_if_missing, bad_score=0.2, good_score=0.9):
        path = dataset_path if os.path.isabs(dataset_path) else os.path.join(DATASET_DIR, dataset_path)
        if not path.endswith(".npz"):
            path += ".npz"
        if os.path.isfile(path):
            ds = PairDataset.load(path)
        elif create_if_missing:
            ds = PairDataset()
            ds.meta.mode = "ref2v"
            ds.meta.source = "harvested"
        else:
            raise FileNotFoundError(f"数据集不存在：{path}")

        x = bad_conditioning[0][0][0]
        y = good_conditioning[0][0][0]
        if x.shape[-1] != y.shape[-1]:
            raise RuntimeError("两条 conditioning 维度不同，无法配对")
        ds.add(x.float().cpu().numpy().astype("float16"), y.float().cpu().numpy().astype("float16"),
               float(bad_score), float(good_score), {"source": "harvest"})
        ds.save(path)
        _msg = f"已追加 1 对：{path}\n当前数据集 {len(ds)} 对，维度 {ds.meta.dim}"
        return _ui((path, _msg), _msg)


class H3WushuTrainBridge:
    """训练武打残差桥，导出 safetensors 到 models/wushu_bridge/。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_path": ("STRING", {"default": "wushu_pairs_t2v.npz"}),
                "arch": ([ARCH_MLP, ARCH_TRANS], {"default": ARCH_TRANS,
                                                  "tooltip": "trans=跨 token（能管整段逻辑自洽）；mlp=社区版同构（可加载社区 11MB 权重做对照）"}),
                "out_name": ("STRING", {"default": "wushu_bridge_v1.safetensors"}),
                "epochs": ("INT", {"default": 60, "min": 1, "max": 2000}),
                "batch_size": ("INT", {"default": 8, "min": 1, "max": 512}),
                "lr": ("FLOAT", {"default": 0.0003, "min": 1e-6, "max": 0.1, "step": 1e-5}),
                "hidden": ("INT", {"default": 512, "min": 16, "max": 4096, "step": 16}),
                "layers": ("INT", {"default": 2, "min": 1, "max": 12}),
                "heads": ("INT", {"default": 4, "min": 1, "max": 32}),
                "anchor_weight": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 10.0, "step": 0.01}),
                "alpha_min": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "alpha_max": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "val_ratio": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.5, "step": 0.01}),
                "device": (["auto", "cpu", "cuda"], {"default": "auto"}),
            },
            "optional": {
                "mode": (["t2v", "ref2v", "any"], {"default": "t2v"}),
                "seed": ("INT", {"default": 1234, "min": 0, "max": 2**31 - 1}),
                "max_seq_tokens": ("INT", {"default": 2048, "min": 32, "max": 65536, "step": 32}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("bridge_path", "report")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3/Wushu Bridge"
    OUTPUT_NODE = True

    def run(self, dataset_path, arch, out_name, epochs, batch_size, lr, hidden, layers, heads,
            anchor_weight, alpha_min, alpha_max, val_ratio, device,
            mode="t2v", seed=1234, max_seq_tokens=2048):
        from .train import TrainConfig, train_bridge

        path = dataset_path if os.path.isabs(dataset_path) else os.path.join(DATASET_DIR, dataset_path)
        ds = PairDataset.load(path)
        out = out_name if os.path.isabs(out_name) else os.path.join(BRIDGE_DIR, out_name)
        if not out.endswith(".safetensors"):
            out += ".safetensors"

        cfg = TrainConfig(
            arch=arch, hidden=int(hidden), layers=int(layers), heads=int(heads),
            mode=mode, name=os.path.splitext(os.path.basename(out))[0],
            epochs=int(epochs), batch_size=int(batch_size), lr=float(lr),
            anchor_weight=float(anchor_weight), alpha_min=float(alpha_min), alpha_max=float(alpha_max),
            val_ratio=float(val_ratio), seed=int(seed), device=device,
            max_seq_tokens=int(max_seq_tokens),
        )
        report = train_bridge(ds, cfg, out, log=lambda s: print("[WushuBridge] " + str(s)))
        text = (
            f"桥已导出：{out}\n"
            f"架构 {report['arch']}｜参数量 {report['params']:,}（fp32 {report['size_mb_fp32']}MB）\n"
            f"样本 {report['samples']}（训练 {report['train_samples']} / 验证 {report['val_samples']}）\n"
            f"验证集：{json.dumps(report.get('final_val', {}), ensure_ascii=False)}\n"
            f"训练集：{json.dumps(report.get('final_train', {}), ensure_ascii=False)}\n"
            f"耗时 {report['seconds']}s\n"
            "说明：cos_to_target_after 高于 before 表示桥确实把 embedding 拉向了'逻辑完备'版本，"
            "relative_gain 越大越好，drift 是相对原序列的漂移（建议保持 < 0.15）。"
        )
        return _ui((out, text), text)


class H3WushuTrainJevHead:
    """训练 JEV 式评分头（直接从 CONDITIONING 给武打逻辑概率）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_path": ("STRING", {"default": "wushu_pairs_t2v.npz"}),
                "out_name": ("STRING", {"default": "wushu_jev_head_v1.safetensors"}),
                "hidden": ("INT", {"default": 256, "min": 16, "max": 4096, "step": 16}),
                "epochs": ("INT", {"default": 80, "min": 1, "max": 2000}),
                "batch_size": ("INT", {"default": 32, "min": 1, "max": 4096}),
                "lr": ("FLOAT", {"default": 0.001, "min": 1e-6, "max": 0.1, "step": 1e-5}),
                "device": (["auto", "cpu", "cuda"], {"default": "auto"}),
            },
            "optional": {
                "heads": ("INT", {"default": 4, "min": 1, "max": 32}),
                "dropout": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.9, "step": 0.01}),
                "seed": ("INT", {"default": 1234, "min": 0, "max": 2**31 - 1}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("judge_path", "report")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3/Wushu Bridge"
    OUTPUT_NODE = True

    def run(self, dataset_path, out_name, hidden, epochs, batch_size, lr, device,
            heads=4, dropout=0.05, seed=1234):
        from .train import JudgeTrainConfig, train_judge

        path = dataset_path if os.path.isabs(dataset_path) else os.path.join(DATASET_DIR, dataset_path)
        ds = PairDataset.load(path)
        out = out_name if os.path.isabs(out_name) else os.path.join(BRIDGE_DIR, out_name)
        if not out.endswith(".safetensors"):
            out += ".safetensors"

        cfg = JudgeTrainConfig(
            hidden=int(hidden), heads=int(heads), dropout=float(dropout),
            name=os.path.splitext(os.path.basename(out))[0],
            epochs=int(epochs), batch_size=int(batch_size), lr=float(lr), seed=int(seed), device=device,
        )
        report = train_judge(ds, cfg, out, log=lambda s: print("[WushuBridge] " + str(s)))
        text = (
            f"评分头已导出：{out}\n"
            f"参数量 {report['params']:,}（fp32 {report['size_mb_fp32']}MB）\n"
            f"样本 {report['samples']}（训练 {report['train_samples']} / 验证 {report['val_samples']}）\n"
            f"验证 BCE {report['best_val_bce']}｜准确率 {report['val_accuracy']}｜"
            f"正负例分差 {report['val_pos_neg_gap']}｜校准后 ECE {report['val_ece_after_calibration']}\n"
            f"温度 {report['temperature']}｜偏置 {report['bias']}\n"
            "说明：ECE 越小说明输出的概率越可信，可直接拿阈值做筛选或 Best-of-N。"
        )
        return _ui((out, text), text)


# ══════════════════════════════════════════════════════════════════════
# 规则节点（纯 CPU，不需要模型）
# ══════════════════════════════════════════════════════════════════════

class H3WushuLintPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "mode": (["final", "design"], {"default": "final",
                                               "tooltip": "final=成品稿（按 H3 两套壳判字段）；design=设计稿（只判逻辑与物理）"}),
                "shell": (["auto", "t2v", "ref2v"], {"default": "auto"}),
            },
            "optional": {
                "names": ("STRING", {"default": "", "multiline": False}),
                "score_mode": (["composite", "logic", "lint"], {"default": "composite"}),
            },
        }

    RETURN_TYPES = ("FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("score", "grade", "report")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3/Wushu Bridge"
    OUTPUT_NODE = True

    def run(self, text, mode, shell, names="", score_mode="composite"):
        from .logic_score import composite_score, format_score, score_logic

        opts = {"mode": mode, "names": [n for n in names.split(",") if n.strip()]}
        if not lexicons.is_chinese(text):
            # h3lint.js 原版只保留 CJK 做台词比对 → 纯英文台词重复检不出来。
            # 语料以英文为主，这里默认修正（见 lint/h3lint.py 的同名开关说明）。
            opts["englishAware"] = True
        logic = score_logic(text, mode)
        fmt = format_score(text, mode, opts)
        comp = composite_score(text, mode, opts)
        score = {"composite": comp, "logic": logic.score, "lint": fmt}[score_mode]
        grade = logic.grade if score_mode == "logic" else ("A" if score >= 0.9 else "B" if score >= 0.78
                                                           else "C" if score >= 0.62 else "D")

        lines = [
            f"复合分 {comp*100:.0f}/100 ｜ 壳结构分(h3lint) {fmt*100:.0f}/100 ｜ 武打逻辑分 {logic.score*100:.0f}/100",
            "",
            logic.to_text(),
        ]
        try:
            from .lint import h3lint

            res = h3lint.check(text, opts)
            lines += ["", "── h3lint 八维体检 ──", h3lint.report(res)]
        except Exception as exc:
            lines += ["", f"（规则引擎未加载：{exc}）", f"槽位统计：{json.dumps(lexicons.slots_present(text), ensure_ascii=False)}"]

        report = "\n".join(lines)
        return _ui((float(score), grade, report), report)


class H3WushuDegradePreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "good_text": ("STRING", {"multiline": True, "default": ""}),
                "intensity": ("INT", {"default": 3, "min": 1, "max": 5}),
                "seed": ("INT", {"default": 1234, "min": 0, "max": 2**31 - 1}),
            },
            "optional": {
                "ops": ("STRING", {"default": "", "multiline": False}),
                "variants": ("INT", {"default": 1, "min": 1, "max": 8}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("bad_text", "report", "all_variants_json")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3/Wushu Bridge"
    OUTPUT_NODE = True

    def run(self, good_text, intensity, seed, ops="", variants=1):
        import random

        op_list = [s.strip() for s in ops.split(",") if s.strip()] if ops.strip() else []
        profile = DegradeProfile(ops=op_list, intensity=int(intensity))
        rng = random.Random(int(seed))
        results = []
        for v in range(int(variants)):
            bad, applied = pairs_mod.degrade(good_text, profile, rng, intensity=int(intensity))
            results.append(
                {
                    "variant": v + 1,
                    "ops": applied,
                    "bad_score": round(score_text(bad), 4),
                    "good_score": round(score_text(good_text), 4),
                    "bad": bad,
                }
            )
        first = results[0]
        report = (
            f"正例分 {first['good_score']} -> 负例分 {first['bad_score']}\n"
            f"用到的降级算子：{first['ops']}\n"
            f"算子含义：{ {k: v.label for k, v in pairs_mod.OPS_BY_KEY.items()} }"
        )
        return _ui((first["bad"], report, json.dumps(results, ensure_ascii=False, indent=1)), report)


NODE_CLASS_MAPPINGS = {
    "H3WushuSemanticBridge": H3WushuSemanticBridge,
    "H3WushuJevScore": H3WushuJevScore,
    "H3WushuLocateTextSpan": H3WushuLocateTextSpan,
    "H3WushuClearCache": H3WushuClearCache,
    "H3WushuBuildDataset": H3WushuBuildDataset,
    "H3WushuHarvestPair": H3WushuHarvestPair,
    "H3WushuTrainBridge": H3WushuTrainBridge,
    "H3WushuTrainJevHead": H3WushuTrainJevHead,
    "H3WushuLintPrompt": H3WushuLintPrompt,
    "H3WushuDegradePreview": H3WushuDegradePreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3WushuSemanticBridge": "H3 武打语义逻辑桥（接在 conditioning 之间）",
    "H3WushuJevScore": "H3 武打逻辑评分（JEV 式，直接给概率）",
    "H3WushuLocateTextSpan": "H3 文本 token 段定位（参考图模式诊断）",
    "H3WushuClearCache": "H3 武打桥 清空缓存",
    "H3WushuBuildDataset": "H3 武打桥 构建数据集（用 H3 文本编码器）",
    "H3WushuHarvestPair": "H3 武打桥 采集训练对（从真实 CONDITIONING）",
    "H3WushuTrainBridge": "H3 武打桥 训练残差桥",
    "H3WushuTrainJevHead": "H3 武打桥 训练 JEV 评分头",
    "H3WushuLintPrompt": "H3 武打提示词体检（规则引擎）",
    "H3WushuDegradePreview": "H3 武打桥 降级预览（造训练对）",
}


# ── 编排层节点（武术动作导演）─────────────────────────────────────────────
# 放在这里而不是直接写进上面的字典：choreo_nodes 依赖 choreography，
# 而 choreography 不依赖 nodes，这样引入不会形成循环 import。
from .choreo_nodes import (  # noqa: E402
    NODE_CLASS_MAPPINGS as _CHOREO_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _CHOREO_NAMES,
)

NODE_CLASS_MAPPINGS.update(_CHOREO_NODES)
NODE_DISPLAY_NAME_MAPPINGS.update(_CHOREO_NAMES)
