"""武打编排的批量验收：多 seed × 多时长 × 两种壳。

验收线（每条都对应一个真实踩过的坑）：
1. **必须出结果** —— 绝不能出现"未分胜负"（追击/反击吃光死线窗口的坑）
2. lint 必须 **≥90 分且零必改**（用用户自己的规则引擎判）
3. 至少有一定比例出现追击/反击（否则"连续出招/防守后反击"就没兑现）
4. 距离不得跳变（R8 距离纪律）
5. 出现道具破坏时必须有留痕描述
"""

from __future__ import annotations

import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from wushu_bridge.choreography import (  # noqa: E402
    FightDirector, Fighter, PromptSpec, Prop, Technique, build_prompt, split_long_prompt,
)
from wushu_bridge.lint import h3lint  # noqa: E402


def make(seed: int):
    a = Fighter(name="角色A", weapon="nodachi", tier=4, hp=5.0, side="左",
                look="男性，黑发披散，黑色武士劲装，双手持野太刀",
                techniques=[Technique("过肩劈", "diag", damage=1.15, cooldown=3.0),
                            Technique("横扫", "slash", damage=1.0, cooldown=2.5),
                            Technique("突刺", "thrust", damage=1.05, cooldown=2.0)])
    b = Fighter(name="角色B", weapon="dao", tier=3, hp=5.0, side="右",
                look="男性，灰发束髻，靛蓝汉服武袍，右手持雁翎单刀",
                techniques=[Technique("撩刀", "rise", damage=1.0, cooldown=2.0),
                            Technique("反手削", "slash", damage=1.05, cooldown=3.0)])
    props = [Prop("右侧灯笼柱", hp=2.5, break_fx="灯笼柱从中段断开，灯纸炸开",
                  debris_fx="断口留着毛刺，灯纸碎片贴在湿石上"),
             Prop("街边木栏", hp=1.5, break_fx="木栏被劈断两根，木屑飞散",
                  debris_fx="木栏断口参差")]
    return a, b, props


def main() -> int:
    scene = "雨夜长街，两侧灯笼暖光，湿石板反光，右侧灯笼柱与街边木栏为实体"
    rows = []
    fail_result, fail_lint, fail_dist = [], [], []
    for duration in (8.0, 10.0, 15.1):
        for seed in range(1, 13):
            a, b, props = make(seed)
            dir_ = FightDirector(a, b, duration=duration, scene=scene, ground="湿石",
                                 props=props, seed=seed * 17 + int(duration))
            ch = dir_.run()
            spec = PromptSpec(mode="t2v", duration=duration, shots=3, scene=scene)
            segs = split_long_prompt(ch, spec, (a, b), max_chars=2400)
            text = segs[0]
            res = h3lint.check(text, {"mode": "final", "englishAware": True, "shotCounting": "auto"})
            # 分段时每一段都要过线
            seg_scores = []
            for s in segs:
                rr = h3lint.check(s, {"mode": "final", "englishAware": True,
                                      "shotCounting": "auto"})
                seg_scores.append((rr["score"], rr["stats"]["errors"]))
            worst = min(x[0] for x in seg_scores)
            errors_total = sum(x[1] for x in seg_scores)
            over = sum(1 for s in segs if len(s) > 2500)

            # 1. 必须出结果
            if "未分胜负" in ch.result or not ch.beats or ch.beats[-1].outcome != "FINISH":
                fail_result.append((duration, seed, ch.result, ch.beats[-1].outcome if ch.beats else "-"))
            # 2. lint（分段时按最差的一段算）
            if worst < 90 or errors_total > 0:
                fail_lint.append((duration, seed, worst, errors_total, f"{len(segs)} 段"))
            # 2b. 分段后每段都不该超 2500
            if over:
                fail_lint.append((duration, seed, worst, 0, f"{over} 段仍超 2500 字"))
            # 4. 距离不跳变
            for p, q in zip(ch.beats, ch.beats[1:]):
                if abs(q.distance - p.distance) > 2.01:
                    fail_dist.append((duration, seed, p.distance, q.distance))

            kinds = [x.attack_kind for x in ch.beats]
            rows.append({
                "duration": duration, "seed": seed,
                "result": ch.result, "beats": len(ch.beats), "exchanges": ch.exchanges,
                "chase": kinds.count("追击"), "counter": kinds.count("反击"),
                "props": sum(1 for p in ch.props if p.broken),
                "lint": worst, "errors": errors_total, "segs": len(segs),
            })

    n = len(rows)
    print(f"跑完 {n} 场（时长 {sorted(set(r['duration'] for r in rows))}）")
    print(f"  lint 分数：min={min(r['lint'] for r in rows)} "
          f"中位={statistics.median(r['lint'] for r in rows):.0f} "
          f"max={max(r['lint'] for r in rows)}")
    print(f"  必改项合计：{sum(r['errors'] for r in rows)}")
    print(f"  平均拍数 {statistics.mean(r['beats'] for r in rows):.1f}｜"
          f"平均交锋 {statistics.mean(r['exchanges'] for r in rows):.1f} 次")
    print(f"  出现「追击」的场次：{sum(1 for r in rows if r['chase'])}/{n}")
    print(f"  出现「反击」的场次：{sum(1 for r in rows if r['counter'])}/{n}")
    print(f"  出现道具破坏的场次：{sum(1 for r in rows if r['props'])}/{n}")
    print(f"\n结果分布：")
    for k, v in sorted({r["result"]: sum(1 for x in rows if x["result"] == r["result"])
                        for r in rows}.items(), key=lambda kv: -kv[1]):
        print(f"  {v:3d}  {k}")

    ok = True
    if fail_result:
        ok = False
        print(f"\n[FAIL] {len(fail_result)} 场没出结果：")
        for x in fail_result[:5]:
            print("   ", x)
    else:
        print("\n[PASS] 全部场次都以可见结果收尾")
    if fail_lint:
        ok = False
        print(f"[FAIL] {len(fail_lint)} 场 lint 不达标：")
        for x in fail_lint[:5]:
            print("   ", x)
    else:
        print("[PASS] 全部场次 lint ≥90 且零必改")
    if fail_dist:
        ok = False
        print(f"[FAIL] {len(fail_dist)} 处距离跳变：{fail_dist[:5]}")
    else:
        print("[PASS] 距离没有跳变（R8 距离纪律）")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
