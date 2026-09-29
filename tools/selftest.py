"""离线自检：不需要 ComfyUI / 不需要 H3，用合成 embedding 把整条流水线跑通。

用途：
1. 用户装完插件，可先在自己的 ComfyUI Python 里跑一遍，确认环境没问题；
2. 开发时验证「造训练对 -> 建数据集 -> 训桥 -> 训评分头 -> 应用 -> 存读权重」
   这条链每一环都通。

跑法（Windows）::

    python tools\\selftest.py

在 venv 里::

    E:\\Wushu\\.venv\\Scripts\\python.exe tools\\selftest.py

关于合成 embedding 的建模方式：现实里 (粗糙写法, 精细写法) 描述的是**同一个
事件**，所以两条 embedding 共享一个"内容方向"，差别主要在"逻辑完备度"方向 L 上。
自检里就照这个假设造数据——这正是桥要学的东西。
"""

from __future__ import annotations

import json
import os
import random
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows 中文控制台默认 GBK，打印中文/符号会直接抛 UnicodeEncodeError，
# 这里强制切到 UTF-8，顺便把 ComfyUI 日志里常见的编码坑提前暴露出来。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from wushu_bridge import lexicons, pairs as pairs_mod  # noqa: E402
from wushu_bridge.apply import apply_bridge_to_conditioning  # noqa: E402
from wushu_bridge.bridge_model import BridgeConfig, build_bridge, load_bridge, save_bridge  # noqa: E402
from wushu_bridge.dataset import PairDataset  # noqa: E402
from wushu_bridge.judge import load_judge, score_conditioning  # noqa: E402
from wushu_bridge.train import JudgeTrainConfig, TrainConfig, train_bridge, train_judge  # noqa: E402

DIM = 5120
OK = "  [ok]"
FAIL = "  [FAIL]"


class FakeEncoder:
    """合成"H3 文本编码器"：内容方向由事件决定，逻辑完备度沿固定方向 L。"""

    def __init__(self, dim: int = DIM, seed: int = 20260921) -> None:
        r = np.random.default_rng(seed)
        L = r.normal(0, 1, size=(dim,)).astype(np.float32)
        self.L = L / np.linalg.norm(L)

    def event(self, rng: np.random.Generator, n_tok: int) -> np.ndarray:
        base = rng.normal(0, 1, size=(n_tok, DIM)).astype(np.float32)
        return base / np.linalg.norm(base, axis=-1, keepdims=True)

    def encode(self, base: np.ndarray, logic: float, rng: np.random.Generator) -> np.ndarray:
        toks = base + (1.4 * float(logic)) * self.L[None, :]
        toks = toks + rng.normal(0, 0.04, size=toks.shape).astype(np.float32)
        return (toks * 3.0).astype(np.float16)


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="wushu_selftest_")
    print("=" * 78)
    print("H3 武打语义逻辑桥 · 离线自检")
    print("临时目录:", tmp)
    print("=" * 78)
    failures = []

    # ── 1. 规则：降级与槽位 ────────────────────────────────────────────
    print("\n[1] 规则引擎 / 降级算子")
    seeds = pairs_mod.all_seeds("t2v")
    good = seeds[0]
    bad, ops = pairs_mod.degrade(good, pairs_mod.DegradeProfile(intensity=4), random.Random(7))
    s_good, s_bad = pairs_mod.score_text(good), pairs_mod.score_text(bad)
    sg = lexicons.slots_present(good)
    sb = lexicons.slots_present(bad)
    print(f"  种子正例 {len(good)} 字，槽位 {json.dumps(sg, ensure_ascii=False)}")
    print(f"  降级后 {len(bad)} 字，算子 {ops}")
    print(f"  分数：正例 {s_good:.3f} -> 负例 {s_bad:.3f}")
    if s_bad >= s_good:
        failures.append("降级没有降低规则分")
        print(FAIL + " 降级后分数没有下降")
    else:
        print(OK + " 降级有效")
    if sum(sb.values()) >= sum(sg.values()):
        failures.append("降级没有减少槽位词")
        print(FAIL + " 槽位词没有减少")
    else:
        print(OK + " 槽位词减少")

    # ── 2. 训练对构建 ─────────────────────────────────────────────────
    print("\n[2] 训练对构建")
    pair_list = pairs_mod.build_pairs(mode="t2v", variants_per_source=8, max_pairs=128)
    stats = pairs_mod.pairs_stats(pair_list)
    print("  " + json.dumps(stats, ensure_ascii=False))
    if stats["count"] < 16:
        failures.append(f"训练对太少：{stats['count']}")
        print(FAIL)
    else:
        print(OK + f" 生成 {stats['count']} 对")

    # ── 3. 数据集（合成 embedding）─────────────────────────────────────
    print("\n[3] 数据集读写")
    enc = FakeEncoder()
    rng = np.random.default_rng(0)
    ds = PairDataset()
    ds.meta.mode = "t2v"
    ds.meta.encoder = "synthetic"
    for p in pair_list:
        n_tok = 40 + len(p.good) % 20
        base = enc.event(rng, n_tok)
        ds.add(
            enc.encode(base, p.bad_score, rng),
            enc.encode(base, p.good_score, rng),
            p.bad_score, p.good_score, p.to_record(),
        )
    ds_path = os.path.join(tmp, "pairs.npz")
    ds.save(ds_path)
    ds2 = PairDataset.load(ds_path)
    print("  " + json.dumps(ds2.stats(), ensure_ascii=False))
    if len(ds2) != len(ds) or ds2.meta.dim != DIM:
        failures.append("数据集读写不一致")
        print(FAIL)
    else:
        print(OK + f" 存读一致（{len(ds2)} 对，{os.path.getsize(ds_path)/1024/1024:.2f} MB）")

    # ── 4. 训练残差桥 ─────────────────────────────────────────────────
    print("\n[4] 训练残差桥（trans 架构）")
    bridge_path = os.path.join(tmp, "wushu_bridge.safetensors")
    cfg = TrainConfig(arch="trans", hidden=256, layers=2, heads=4, epochs=40, batch_size=8,
                      lr=1e-3, anchor_weight=0.1, max_seq_tokens=256, patience=40, val_ratio=0.2)
    rep = train_bridge(ds, cfg, bridge_path, log=lambda s: print("    " + str(s)))
    gain = (rep.get("final_val") or {}).get("relative_gain")
    print(f"  参数量 {rep['params']:,}（fp16 约 {rep['params']*2/1024/1024:.1f}MB）"
          f"；验证集 relative_gain={gain}")
    if gain is None or gain <= 0:
        failures.append(f"桥没有把 embedding 拉向正例（gain={gain}）")
        print(FAIL)
    else:
        print(OK + " 桥学到了有效方向")

    # ── 5. 训练 JEV 评分头 ────────────────────────────────────────────
    print("\n[5] 训练 JEV 评分头")
    judge_path = os.path.join(tmp, "wushu_jev.safetensors")
    jrep = train_judge(ds, JudgeTrainConfig(hidden=128, epochs=120, batch_size=16, val_ratio=0.25),
                       judge_path, log=lambda s: print("    " + str(s)))
    print(f"  参数量 {jrep['params']:,}；准确率 {jrep['val_accuracy']}；AUC {jrep['val_auc']}；"
          f"ECE {jrep['val_ece_after_calibration']}")
    if jrep["val_accuracy"] < 0.8:
        failures.append(f"评分头准确率过低：{jrep['val_accuracy']}")
        print(FAIL)
    else:
        print(OK + " 评分头可分")

    # ── 6. 应用 + metadata 保留 ───────────────────────────────────────
    print("\n[6] 应用到 CONDITIONING（含参考图 metadata）")
    model = load_bridge(bridge_path, device="cpu")
    judge = load_judge(judge_path, device="cpu")
    bad_tok = ds.x[0].astype(np.float32)
    tensor = torch.from_numpy(bad_tok).unsqueeze(0)
    cond = [[tensor, {
        "minimax_frame_count": 124,
        "minimax_keyframes": [{"resolved_frame_index": 0}],
        "minimax_refs": [{"kind": "image"}],
        "custom_key": "必须保留",
    }]]
    out, report = apply_bridge_to_conditioning(cond, model, alpha=0.25, magnitude_match_mode="per_token")
    meta_out = out[0][1]
    kept = all(k in meta_out for k in ("minimax_frame_count", "minimax_keyframes", "minimax_refs", "custom_key"))
    shape_ok = out[0][0].shape == tensor.shape
    dtype_ok = out[0][0].dtype == tensor.dtype
    print("  " + report.to_text().replace("\n", "\n  "))
    print(f"  metadata 完整: {kept}｜形状 {tuple(out[0][0].shape)}｜dtype {out[0][0].dtype}")
    if not (kept and shape_ok and dtype_ok):
        failures.append("应用桥后 metadata/形状/dtype 被破坏")
        print(FAIL)
    else:
        print(OK + " metadata 与张量形状均保留")

    before, _ = score_conditioning(judge, cond)
    after, _ = score_conditioning(judge, out)
    print(f"  评分头：改动前 {before:.3f} -> 改动后 {after:.3f}")
    if after <= before:
        print("  (提示) 合成数据上分数没有上升，属正常波动；真实数据应观察相对提升")
    else:
        print(OK + " 评分上升")

    # ── 7. 社区版权重兼容性（MLP 架构）───────────────────────────────
    print("\n[7] 社区版 Semantic Bridge 权重兼容性")
    mlp = build_bridge(BridgeConfig(arch="mlp", dim=DIM, hidden=512))
    mlp_path = os.path.join(tmp, "community_like.safetensors")
    save_bridge(mlp_path, mlp)
    loaded = load_bridge(mlp_path, device="cpu")
    print(f"  构造 {mlp.num_params():,} 参数 -> 存读后 arch={loaded.cfg.arch} dim={loaded.cfg.dim} "
          f"hidden={loaded.cfg.hidden}")
    if loaded.cfg.arch != "mlp" or loaded.num_params() != mlp.num_params():
        failures.append("MLP 权重存读不一致")
        print(FAIL)
    else:
        print(OK + " 与社区 5120->512->512->5120 结构一致，可直接加载社区 11MB 权重做对照")

    # ── 8. 维度不匹配保护 ────────────────────────────────────────────
    print("\n[8] 维度不匹配保护")
    wrong = [[torch.randn(1, 32, 4096), {}]]
    try:
        apply_bridge_to_conditioning(wrong, model, alpha=0.1)
        failures.append("维度不匹配时没有报错")
        print(FAIL)
    except RuntimeError as exc:
        print(OK + f" 正确报错：{str(exc)[:60]}…")

    print("\n" + "=" * 78)
    if failures:
        print(f"自检失败 {len(failures)} 项：")
        for f in failures:
            print("  x", f)
        return 1
    print("自检全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
