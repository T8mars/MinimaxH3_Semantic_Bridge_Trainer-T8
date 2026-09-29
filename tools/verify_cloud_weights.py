"""本地验收：加载从云主机下载回来的权重，确认真能用。

不依赖 ComfyUI/H3 —— 用 5120 维合成 conditioning 走一遍桥与评分头，
再把训练报告里的关键指标打出来。
"""

from __future__ import annotations

import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from wushu_bridge.apply import apply_bridge_to_conditioning  # noqa: E402
from wushu_bridge.bridge_model import load_bridge  # noqa: E402
from wushu_bridge.judge import load_judge, score_conditioning  # noqa: E402

W = os.path.join(ROOT, "models", "wushu_bridge")
DEFAULT_BRIDGE = os.path.join(W, "wushu_bridge_wushu_v1.safetensors")
DEFAULT_JUDGE = os.path.join(W, "wushu_jev_wushu_v1.safetensors")


def show_report(path: str) -> None:
    if not os.path.isfile(path):
        return
    data = json.load(open(path, encoding="utf-8"))
    r = data.get("report", data)
    print(f"\n=== {os.path.basename(path)}")
    print(f"  参数量 {r.get('params'):,}  样本 {r.get('samples')}  耗时 {r.get('seconds')}s"
          if r.get("params") else "  (无 report)")
    fv, ft = r.get("final_val") or {}, r.get("final_train") or {}
    if fv:
        print(f"  val   : bad/good 相似度={fv.get('bad_good_similarity')} "
              f"方向对齐={fv.get('direction_alignment')} gain={fv.get('relative_gain')} "
              f"漂移={fv.get('drift')}")
    if ft:
        print(f"  train : bad/good 相似度={ft.get('bad_good_similarity')} "
              f"方向对齐={ft.get('direction_alignment')} gain={ft.get('relative_gain')} "
              f"漂移={ft.get('drift')}")
    for k, label in (("val_accuracy", "验证准确率"), ("val_auc", "AUC"),
                     ("val_ece_after_calibration", "校准后 ECE"), ("temperature", "温度")):
        if r.get(k) is not None:
            print(f"  {label} = {r[k]}")


def main() -> int:
    bridge_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BRIDGE
    judge_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_JUDGE
    print("=" * 78)
    print("云端权重 · 本地验收")
    print("=" * 78)

    ok = True
    if not os.path.isfile(bridge_path):
        print(f"[FAIL] 找不到桥权重：{bridge_path}")
        return 1
    print(f"\n桥：{bridge_path}  {os.path.getsize(bridge_path)/1024/1024:.2f} MB")
    bridge = load_bridge(bridge_path, device="cpu")
    print(f"  架构={bridge.cfg.arch} dim={bridge.cfg.dim} hidden={bridge.cfg.hidden} "
          f"layers={bridge.cfg.layers} heads={bridge.cfg.heads} mode={bridge.cfg.mode}")
    print(f"  参数量 {bridge.num_params():,}")

    judge = None
    if os.path.isfile(judge_path):
        print(f"\n评分头：{judge_path}  {os.path.getsize(judge_path)/1024/1024:.2f} MB")
        judge = load_judge(judge_path, device="cpu")
        print(f"  dim={judge.cfg.dim} hidden={judge.cfg.hidden} 温度={judge.cfg.temperature} "
              f"偏置={judge.cfg.bias}")
        print(f"  参数量 {judge.num_params():,}")
    else:
        print(f"\n[warn] 找不到评分头：{judge_path}")

    # ── 用一个"缺逻辑"的合成 conditioning 走一遍 ────────────────────
    # 注意：这里喂的是**随机向量**，不是真 H3 embedding。它只用来验证
    # "权重能加载、能应用、metadata 不丢"，**不能**用来判断桥的行为是否合理：
    # 随机向量在训练分布之外，桥会推得很猛（实测漂移 0.40）。
    # 真 H3 embedding 上的实测参考：漂移 0.0104、武打逻辑分 0.524 -> 0.714。
    print("\n[说明] 下面用的是随机合成向量，漂移偏大属正常；"
          "真 H3 embedding 上实测漂移约 0.010。")
    torch.manual_seed(0)
    cond = [[torch.randn(1, 220, 5120) * 3.0,
             {"minimax_frame_count": 124, "minimax_refs": [{"kind": "image"}], "keep": "必须保留"}]]
    before_score = score_conditioning(judge, cond)[0] if judge else None

    out, rep = apply_bridge_to_conditioning(cond, bridge, alpha=0.15,
                                            magnitude_match_mode="per_token")
    meta = out[0][1]
    kept = all(k in meta for k in ("minimax_frame_count", "minimax_refs", "keep"))
    shape_ok = out[0][0].shape == cond[0][0].shape and out[0][0].dtype == cond[0][0].dtype
    print("\n--- 应用结果 ---")
    print("  " + rep.to_text().replace("\n", "\n  "))
    print(f"  metadata 保留: {kept}｜形状/dtype 保持: {shape_ok}")
    if not (kept and shape_ok):
        ok = False

    if judge:
        after_score, _ = score_conditioning(judge, out)
        print(f"  评分头：改动前 {before_score:.3f} -> 改动后 {after_score:.3f}")

    show_report(os.path.splitext(bridge_path)[0] + "_report.json")
    show_report(os.path.splitext(judge_path)[0] + "_report.json")

    print("\n" + "=" * 78)
    print("验收通过：权重可加载、可应用、metadata 完整。" if ok else "验收失败。")
    print("把这两个 safetensors 放进 ComfyUI/models/wushu_bridge/ 就能在节点里选到。")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
