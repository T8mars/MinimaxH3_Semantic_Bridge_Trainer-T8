"""语料探针：看看你的提示词库里能挖出多少条可用正例 / 真实负例。

只读，不改动任何文件。

用法::

    python tools\\probe_corpus.py <路径1> [<路径2> ...]
    python tools\\probe_corpus.py "<自己的提示词目录>" "<自己的metadata.csv>"
    python tools\\probe_corpus.py --negatives "路径\\metadata.csv" "路径\\rejected"

输出：每个路径抽到多少条候选正例、平均长度、规则分分布，以及（给了
--negatives 时）能匹配出多少条"被筛片段对应的 prompt"真实负例。
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from wushu_bridge.pairs import load_corpus, load_rejected_prompts, score_text  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="文件或目录")
    ap.add_argument("--mode", default="t2v", choices=["t2v", "ref2v"])
    ap.add_argument("--negatives", nargs=2, metavar=("METADATA_CSV", "REJECTED_DIR"),
                    help="从打标表 + 被筛片段目录里挖真实负例")
    ap.add_argument("--degrade", type=int, default=0, metavar="N",
                    help="拿前 N 条候选正例做降级演示，看训练对长什么样")
    ap.add_argument("--include-design", action="store_true",
                    help="额外收「无 H3 壳但动作密度高」的设计稿散文（第1步产物）")
    ap.add_argument("--hygiene", action="store_true",
                    help="对目录做语料卫生检查：非 UTF-8 编码、同内容重复、LTX/H3 交叉污染、空壳文件")
    ap.add_argument("--max", type=int, default=20000)
    args = ap.parse_args()

    total = 0
    collected: list = []
    for p in args.paths:
        prompts = load_corpus([p], mode=args.mode, max_items=args.max,
                              include_design=args.include_design)
        total += len(prompts)
        collected.extend(prompts)
        print(f"\n=== {p}")
        print(f"  抽到候选提示词：{len(prompts)} 条")
        if prompts:
            lens = [len(x) for x in prompts]
            scores = [score_text(x, "final") for x in prompts[:200]]
            print(f"  长度：min={min(lens)} max={max(lens)} 中位={int(statistics.median(lens))}")
            print(f"  规则分（前 {len(scores)} 条）：min={min(scores):.3f} "
                  f"中位={statistics.median(scores):.3f} max={max(scores):.3f}")
            print(f"  最低分样例开头：{prompts[scores.index(min(scores))][:100]!r}")
    print(f"\n合计候选正例：{total} 条")

    if args.negatives:
        csv_path, rej_dir = args.negatives
        negs = load_rejected_prompts(csv_path, rej_dir)
        print(f"\n=== 真实负例（被筛片段对应的 prompt）")
        print(f"  匹配到：{len(negs)} 条")
        if negs:
            sc = [score_text(x, "final") for x in negs[:200]]
            print(f"  规则分：min={min(sc):.3f} 中位={statistics.median(sc):.3f} max={max(sc):.3f}")

    if args.degrade and collected:
        import random

        from wushu_bridge.pairs import DegradeProfile, degrade

        print(f"\n=== 降级演示（前 {args.degrade} 条）")
        rng = random.Random(1234)
        for i, good in enumerate(collected[: args.degrade]):
            bad, ops = degrade(good, DegradeProfile(intensity=3), rng)
            print(f"\n--- 样本 {i+1}（{len(good)} 字）")
            print(f"  算子：{ops}")
            print(f"  正例分 {score_text(good, 'final'):.3f} -> 负例分 {score_text(bad, 'final'):.3f}")
            print(f"  正例开头：{good[:150]!r}")
            print(f"  负例开头：{bad[:150]!r}")

    if args.hygiene:
        for p in args.paths:
            if os.path.isdir(p):
                hygiene_report(p)
    return 0


# ── 语料卫生检查 ────────────────────────────────────────────────────────
# 依据用户侧盘点的 §4.13「语料卫生与跨库注意事项」实测条目：
#   * 部分文件是 GBK/CP936，按 UTF-8 读会乱码
#   * 存在同内容重复文件（MD5 相同），不去重会虚增语料量
#   * H3 与 LTX 是**硬隔离**的：LTX 侧把 H3 字段名列入黑名单，命中即报错，
#     所以 H3 语料别直接喂 LTX，LTX 稿也别混进 H3 数据集
#   * 有 404 占位文件（`seedance_skill.txt` 只有 14 字节，内容是 `404: Not Found`）

_H3_FIELDS = ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music",
              "detailed_description", "subject_definitions", "summary")


def hygiene_report(root: str) -> None:
    import hashlib

    seen: dict = {}
    non_utf8, dups, ltx_like, tiny, cross = [], [], [], [], []
    n_files = 0
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            p = os.path.join(dirpath, name)
            try:
                with open(p, "rb") as f:
                    raw = f.read()
            except Exception:
                continue
            n_files += 1
            try:
                text = raw.decode("utf-8")
            except Exception:
                non_utf8.append(p)
                try:
                    text = raw.decode("gb18030")
                except Exception:
                    text = raw.decode("utf-8", errors="ignore")
            h = hashlib.md5(raw).hexdigest()
            if h in seen:
                dups.append((p, seen[h]))
            else:
                seen[h] = p
            if len(raw) < 64:
                tiny.append((p, text.strip()[:40]))
            low_path = p.lower()
            is_ltx = ("ltx" in low_path) or ("ltx2" in text.lower()) or ("LTX 2.5" in text)
            if is_ltx:
                if any(f in text for f in _H3_FIELDS):
                    cross.append(p)
                else:
                    ltx_like.append(p)

    print(f"\n=== 语料卫生检查：{root}")
    print(f"  文件总数 {n_files}")
    print(f"  非 UTF-8 编码 {len(non_utf8)} 个（读取时已按 GB18030 回退）")
    for p in non_utf8[:8]:
        print(f"     - {os.path.relpath(p, root)}")
    print(f"  同内容重复 {len(dups)} 组（load_corpus 已按内容哈希自动去重）")
    for a, b in dups[:5]:
        print(f"     - {os.path.relpath(a, root)}  ==  {os.path.relpath(b, root)}")
    print(f"  LTX 稿 {len(ltx_like)} 个；其中**同时含 H3 字段名的 {len(cross)} 个**（交叉污染风险）")
    for p in cross[:5]:
        print(f"     ⚠ {os.path.relpath(p, root)}")
    print(f"  疑似空壳/404 文件 {len(tiny)} 个")
    for p, head in tiny[:5]:
        print(f"     - {os.path.relpath(p, root)}: {head!r}")
    print("  提醒：H3 与 LTX 硬隔离 —— LTX 侧把 H3 字段名列入黑名单，命中即报错；"
          "H3 数据集不要混入 LTX 稿。")


if __name__ == "__main__":
    raise SystemExit(main())
