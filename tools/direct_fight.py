"""武打编排的自检：生成 → 跑 h3lint → 看分数与必改项。

用法::

    python tools/direct_fight.py                 # 默认对打，打印提示词 + 体检
    python tools/direct_fight.py --seed 7 --mode ref2v
    python tools/direct_fight.py --json          # 只出 beats JSON
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from wushu_bridge.choreography import (  # noqa: E402
    FightDirector, Fighter, PromptSpec, Prop, Technique, build_prompt,
)
from wushu_bridge.lint import h3lint  # noqa: E402


def make_fighters() -> tuple:
    a = Fighter(
        name="角色A", weapon="nodachi", tier=4, hp=5.0, side="左",
        look="男性，黑发披散，黑色武士劲装，双手持野太刀",
        techniques=[
            Technique("过肩劈", variant="diag", damage=1.15, cooldown=3, note="从肩到地走完整弧"),
            Technique("横扫", variant="slash", damage=1.0, cooldown=2.5),
            Technique("突刺", variant="thrust", damage=1.05, cooldown=2.0),
        ],
    )
    b = Fighter(
        name="角色B", weapon="dao", tier=3, hp=5.0, side="右",
        look="男性，灰发束髻，靛蓝汉服武袍，右手持雁翎单刀",
        techniques=[
            Technique("撩刀", variant="rise", damage=1.0, cooldown=2.0),
            Technique("反手削", variant="slash", damage=1.05, cooldown=3.0),
            Technique("缠头裹脑", variant="slash", damage=1.1, cooldown=3.5),
        ],
    )
    return a, b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--mode", default="t2v", choices=["t2v", "ref2v"])
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--winner", default="")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    a, b = make_fighters()
    props = [
        Prop("右侧灯笼柱", hp=3.0, material="木质", break_fx="灯笼柱从中段断开，灯纸炸开",
             debris_fx="断口留着毛刺，灯纸碎片贴在湿石上"),
        Prop("街边木栏", hp=2.0, material="木质", break_fx="木栏被劈断两根，木屑飞散",
             debris_fx="木栏断口参差，横档挂在半空"),
    ]
    scene = "雨夜长街，两侧灯笼暖光，湿石板反光，右侧灯笼柱与街边木栏为实体"
    director = FightDirector(
        a, b, duration=args.duration, start_distance=2.0, scene=scene, ground="湿石",
        props=props, winner=args.winner, seed=args.seed,
    )
    choreo = director.run()

    if args.json:
        print(choreo.to_json())
        return 0

    spec = PromptSpec(mode=args.mode, duration=args.duration, shots=args.shots, scene=scene)
    prompt = build_prompt(choreo, spec, (a, b))

    print("=" * 78)
    print(f"编排结果：{choreo.result}｜交锋 {choreo.exchanges} 次｜拍数 {len(choreo.beats)}")
    for bt in choreo.beats:
        print(f"  [{bt.t0:5.2f}-{bt.t1:5.2f}] {bt.distance:g}格 {bt.attacker}->{bt.defender} "
              f"{bt.attack_kind}「{bt.technique}」 {bt.outcome:12s} {bt.defense:10s} "
              f"{'道具:' + bt.prop_event[:14] if bt.prop_event else ''}")
    print("=" * 78)
    print(prompt)
    print("=" * 78)

    res = h3lint.check(prompt, {"mode": "final", "englishAware": True, "shotCounting": "auto"})
    print(f"h3lint 体检：{res['grade']} {res['score']}/100  "
          f"必改 {res['stats']['errors']}｜建议 {res['stats']['warns']}｜提示 {res['stats']['infos']}")
    for it in res["items"]:
        if it["level"] in ("error", "warn"):
            print(f"  [{it['level']}] {it['id']}: {it['msg'][:90]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
