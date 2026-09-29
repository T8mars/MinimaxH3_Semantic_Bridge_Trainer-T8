"""一键检查：把插件能离线自证的东西全跑一遍，输出一张通过/失败看板。

不需要 H3、不需要显卡、不需要 ComfyUI。装完插件先跑这个。

用法::

    python tools\\run_all_checks.py
    python tools\\run_all_checks.py --corpus "<自己的metadata.csv>"
    python tools\\run_all_checks.py --skip-train        # 跳过合成训练（更快）

在自建的 venv 里::

    <虚拟环境>\\Scripts\\python.exe tools\\run_all_checks.py

七项检查
--------
1. **依赖**：torch / numpy / safetensors 是否就位
2. **节点注册**：10 个节点的 INPUT_TYPES 是否都能构造（ComfyUI 加载时会做的事）
3. **规则引擎**：h3lint 移植版在自带的 45 条断言对上是否通过
4. **端到端流水线**：合成 embedding → 造训练对 → 建数据集 → 训桥 → 训评分头 →
   应用到 CONDITIONING → 校验 metadata 是否原样保留
5. **权重兼容性**：社区同构 MLP 架构能否存读（用于加载社区 11MB 权重做对照）
6. **语料探针**（可选，给了 --corpus 才跑）：能抽出多少正例、降级后分数是否真的下降
7. **语言/时间码两处实用修正**：英文台词重复能检出、Ref2VA 引用式镜头数不再数错
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

RESULTS: list = []


def record(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"  {'[PASS]' if ok else '[FAIL]'} {name}" + (f"  — {detail}" if detail else ""))
    return ok


def check_deps():
    print("\n[1] 依赖")
    ok = True
    for mod in ("torch", "numpy", "safetensors"):
        try:
            m = __import__(mod)
            record(f"{mod} {getattr(m, '__version__', '?')}", True)
        except Exception as exc:
            ok = record(f"{mod} 缺失", False, str(exc))
    return ok


def check_nodes():
    print("\n[2] 节点注册（ComfyUI 加载时会做的事）")
    try:
        from wushu_bridge import nodes
    except Exception:
        return record("import wushu_bridge.nodes", False, traceback.format_exc(limit=2))
    bad = []
    for name, cls in nodes.NODE_CLASS_MAPPINGS.items():
        try:
            it = cls.INPUT_TYPES()
            assert "required" in it or "optional" in it
        except Exception as exc:
            bad.append(f"{name}: {exc}")
    record(f"{len(nodes.NODE_CLASS_MAPPINGS)} 个节点 INPUT_TYPES", not bad, "; ".join(bad)[:160])
    record("显示名齐全",
           len(nodes.NODE_DISPLAY_NAME_MAPPINGS) == len(nodes.NODE_CLASS_MAPPINGS))
    return not bad


def check_lint():
    print("\n[3] 规则引擎（h3lint 移植版）")
    try:
        from wushu_bridge.lint import h3lint
        from wushu_bridge.seeds import all_seeds
    except Exception:
        return record("import lint", False, traceback.format_exc(limit=2))
    record(f"VERSION {h3lint.VERSION}", True)

    ok = True
    for i, s in enumerate(all_seeds("t2v")):
        r = h3lint.check(s, {"mode": "final"})
        good = r["stats"]["errors"] == 0 and r["score"] >= 75
        ok = record(f"内置种子 #{i} 零 error 且 >=75 分", good,
                    f"score={r['score']} shots={r['stats']['shots']} "
                    f"timecodes={r['stats']['timecodes']} errors={r['stats']['errors']}") and ok

    # 语言修正：英文台词重复
    en = ("wushu_action, 8 seconds, 200 frames, 16:9, 24fps, 832x480.\n"
          "integrated_multimodal_description:\n"
          "[Shot 1] The fighter (S1) says: <d>[English] Hold your ground.</d> Steel rings out.\n"
          "overall_soundscape: Feet on wet stone, and he says Hold your ground over the clash.\n"
          "non_diegetic_music: None.")
    raw = {i["id"] for i in h3lint.check(en, {"mode": "final", "englishAware": False})["items"]}
    fix = {i["id"] for i in h3lint.check(en, {"mode": "final", "englishAware": True})["items"]}
    ok = record("englishAware：英文台词重复可检出",
                "sound-dialogue-repeat" not in raw and "sound-dialogue-repeat" in fix) and ok

    # 时间码修正：引用式写法不再数错镜头
    ref = ("subject_definitions:\n<Subject 1> in <Picture 1>.\nsummary:\ntwo fighters.\n"
           "retention_analysis:\n<Subject 1> (appears in [Shot 1]): fully_preserved.\n"
           "detailed_description:\n[Shot 1] A low tracking shot.\n"
           "overall_soundscape: wind.\nnon_diegetic_music: None.")
    a = h3lint.check(ref, {"mode": "final", "shotCounting": "raw"})["stats"]["shots"]
    b = h3lint.check(ref, {"mode": "final", "shotCounting": "auto"})["stats"]["shots"]
    ok = record("shotCounting：引用式镜头数不再数错", a > b and b == 1, f"raw={a} auto={b}") and ok
    return ok


def check_pipeline(skip_train=False):
    print("\n[4] 端到端流水线（合成 embedding，不需要 H3）")
    import numpy as np
    import torch

    from wushu_bridge.apply import apply_bridge_to_conditioning
    from wushu_bridge.bridge_model import BridgeConfig, build_bridge, load_bridge, save_bridge
    from wushu_bridge.dataset import PairDataset
    from wushu_bridge.judge import load_judge, score_conditioning
    from wushu_bridge import pairs as pairs_mod

    ok = True
    tmp = tempfile.mkdtemp(prefix="wushu_checks_")
    DIM = 5120

    pair_list = pairs_mod.build_pairs(mode="t2v", variants_per_source=4, max_pairs=24)
    st = pairs_mod.pairs_stats(pair_list)
    ok = record("造训练对", st.get("count", 0) >= 8,
                f"{st.get('count')} 对，正例均分 {st.get('mean_good_score')} "
                f"负例均分 {st.get('mean_bad_score')}") and ok

    rl = np.random.default_rng(20260921)
    L = rl.normal(0, 1, size=(DIM,)).astype(np.float32)
    L /= np.linalg.norm(L)
    rng = np.random.default_rng(0)

    def enc(base, logic):
        t = base + (1.4 * float(logic)) * L[None, :] + rng.normal(0, 0.04, size=base.shape).astype(np.float32)
        return (t * 3.0).astype(np.float16)

    ds = PairDataset()
    for p in pair_list:
        base = rng.normal(0, 1, size=(40 + len(p.good) % 20, DIM)).astype(np.float32)
        base /= np.linalg.norm(base, axis=-1, keepdims=True)
        ds.add(enc(base, p.bad_score), enc(base, p.good_score), p.bad_score, p.good_score, p.to_record())
    ds_path = os.path.join(tmp, "pairs.npz")
    ds.save(ds_path)
    back = PairDataset.load(ds_path)
    ok = record("数据集存读", len(back) == len(ds) and back.meta.dim == DIM,
                f"{len(back)} 对，dim={back.meta.dim}") and ok

    if skip_train:
        print("  [skip] --skip-train：跳过训练与权重检查")
        return ok

    from wushu_bridge.train import JudgeTrainConfig, TrainConfig, train_bridge, train_judge

    bridge_path = os.path.join(tmp, "wushu_bridge.safetensors")
    rep = train_bridge(ds, TrainConfig(arch="trans", hidden=128, layers=1, heads=4, epochs=25,
                                       batch_size=8, lr=1e-3, anchor_weight=0.1, val_ratio=0.2,
                                       max_seq_tokens=128, patience=25), bridge_path,
                       log=lambda s: None)
    gain = (rep.get("final_val") or {}).get("relative_gain")
    ok = record("训练残差桥：语义被拉向正例", gain is not None and gain > 0,
                f"参数量 {rep['params']:,}；relative_gain={gain}") and ok

    judge_path = os.path.join(tmp, "wushu_jev.safetensors")
    jrep = train_judge(ds, JudgeTrainConfig(hidden=64, epochs=100, batch_size=16, val_ratio=0.25),
                       judge_path, log=lambda s: None)
    ok = record("训练 JEV 评分头：可分", jrep["val_accuracy"] >= 0.8,
                f"参数量 {jrep['params']:,}；acc={jrep['val_accuracy']} AUC={jrep['val_auc']} "
                f"ECE={jrep['val_ece_after_calibration']}") and ok

    model = load_bridge(bridge_path, device="cpu")
    judge = load_judge(judge_path, device="cpu")
    tensor = torch.from_numpy(ds.x[0].astype(np.float32)).unsqueeze(0)
    cond = [[tensor, {"minimax_frame_count": 124, "minimax_refs": [{"kind": "image"}],
                      "custom_key": "必须保留"}]]
    out, report = apply_bridge_to_conditioning(cond, model, alpha=0.2)
    meta = out[0][1]
    keep = all(k in meta for k in ("minimax_frame_count", "minimax_refs", "custom_key"))
    ok = record("应用后 metadata 与形状保留",
                keep and out[0][0].shape == tensor.shape and out[0][0].dtype == tensor.dtype,
                f"漂移={report.drift[0]:.4f}") and ok

    before, _ = score_conditioning(judge, cond)
    after, _ = score_conditioning(judge, out)
    print(f"     评分头：改动前 {before:.3f} -> 改动后 {after:.3f}")
    ok = record("评分头可对 CONDITIONING 打分", 0.0 <= before <= 1.0 and 0.0 <= after <= 1.0) and ok

    mlp_path = os.path.join(tmp, "community_like.safetensors")
    save_bridge(mlp_path, build_bridge(BridgeConfig(arch="mlp", dim=DIM, hidden=512)))
    loaded = load_bridge(mlp_path, device="cpu")
    ok = record("社区同构 MLP 权重存读（11MB/fp16 那套）",
                loaded.cfg.arch == "mlp" and loaded.cfg.dim == DIM and loaded.cfg.hidden == 512,
                f"参数量 {loaded.num_params():,}") and ok

    try:
        apply_bridge_to_conditioning([[torch.randn(1, 8, 4096), {}]], model, alpha=0.1)
        ok = record("维度不匹配时报错", False, "没有报错") and ok
    except RuntimeError:
        record("维度不匹配时报错", True)
    return ok


def check_corpus(path):
    print(f"\n[6] 语料探针：{path}")
    from wushu_bridge.pairs import DegradeProfile, degrade, load_corpus, score_text

    prompts = load_corpus([path], mode="t2v", max_items=20000)
    ok = record("抽出候选正例", len(prompts) > 0, f"{len(prompts)} 条")
    if not prompts:
        return ok
    rng = random.Random(1234)
    gaps = []
    for good in prompts[:30]:
        bad, _ = degrade(good, DegradeProfile(intensity=3), rng)
        gaps.append(score_text(good, "final") - score_text(bad, "final"))
    ratio = sum(1 for g in gaps if g > 0) / len(gaps)
    ok = record("降级后分数确实下降", ratio >= 0.7,
                f"{ratio*100:.0f}% 的样本负例更低，平均分差 {sum(gaps)/len(gaps):.3f}") and ok
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="", help="可选：跑一次语料探针")
    ap.add_argument("--skip-train", action="store_true", help="跳过合成训练（更快）")
    args = ap.parse_args()

    print("=" * 78)
    print("ComfyUI-H3-WushuBridge · 一键检查")
    print(f"插件目录: {ROOT}")
    print("=" * 78)

    try:
        check_deps()
    except Exception:
        traceback.print_exc()
    for fn in (check_nodes, check_lint):
        try:
            fn()
        except Exception:
            record(fn.__name__, False, traceback.format_exc(limit=2))
    try:
        check_pipeline(skip_train=args.skip_train)
    except Exception:
        record("check_pipeline", False, traceback.format_exc(limit=3))
    if args.corpus:
        try:
            check_corpus(args.corpus)
        except Exception:
            record("check_corpus", False, traceback.format_exc(limit=2))

    npass = sum(1 for _, ok, _ in RESULTS if ok)
    nfail = len(RESULTS) - npass
    print("\n" + "=" * 78)
    print(f"看板：{npass} 项通过，{nfail} 项失败，共 {len(RESULTS)} 项")
    if nfail:
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  FAIL  {name}  {detail[:200]}")
    else:
        print("全部通过 —— 插件在本机可用；下一步去装了 H3 的 ComfyUI 上建数据集、训练。")
    print("=" * 78)
    return 1 if nfail else 0


if __name__ == "__main__":
    raise SystemExit(main())
