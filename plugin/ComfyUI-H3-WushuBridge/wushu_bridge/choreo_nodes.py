"""编排层的 ComfyUI 节点：把"武术动作导演"接进工作流。

与语义桥的分工：

* ``H3 武打编排（动作导演）`` —— 产出**打什么**：完整打斗编排 + H3 提示词
* ``H3 武打语义逻辑桥``       —— 负责**怎么说**：把 conditioning 往武打逻辑分布推

典型接法::

    [H3 武打编排] --prompt--> [CLIPTextEncode] --> [H3 武打语义逻辑桥] --> 采样器
          └--beats_json--> （可选）接文本显示节点看编排明细
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .choreography import (
    RULES_14, VARIANTS, WEAPONS, Choreography, FightDirector, Fighter, PromptSpec,
    Prop, Technique, build_prompt, build_solo_lora_prompt, solo_drill, split_long_prompt,
)
from .lora_vocab import LORA_PHYS_ZH


def _ui(result, *texts):
    """与 nodes.py 里的同名助手一致（这里自带一份，避免与 nodes.py 循环 import）。"""
    lines = [str(t) for t in texts if t is not None and str(t).strip()]
    return {"ui": {"text": lines}, "result": result}


GROUNDS = ["湿石", "夯土", "木地板", "雪地", "石砖"]

# 招式名 → 变体的关键词映射（用户只写招式名，这里自动配扇形/高度/倍率）
_VARIANT_HINTS = [
    (("劈", "斩", "砍", "剁"), "diag"),
    (("扫", "堂", "旋", "回身"), "sweep"),
    (("刺", "扎", "点"), "thrust"),
    (("撩", "挑", "挂", "崩"), "rise"),
    (("撞", "摔", "肘", "膝", "崩拳", "重"), "heavy"),
    (("横", "削", "抹", "缠", "裹"), "slash"),
]


def _variant_for(name: str, idx: int) -> str:
    for keys, key in _VARIANT_HINTS:
        if any(k in name for k in keys):
            return key
    return ["slash", "diag", "thrust", "rise"][idx % 4]


def parse_fighter(spec: str, default_name: str, side: str) -> Fighter:
    """解析角色卡。格式（``|`` 分隔，后面的可省）::

        武器|等级|外貌|招式1,招式2,招式3

    例：``nodachi|4|男性，黑发披散，黑色武士劲装，双手持野太刀|过肩劈,横扫,突刺``
    也可以先写中文名：``太刀|4|…|过肩劈,横扫``（武器名会自动映射到引擎的 key）
    """
    parts = [p.strip() for p in (spec or "").split("|")]
    weapon_raw = parts[0] if parts and parts[0] else "dao"
    weapon = weapon_raw.lower()
    if weapon not in WEAPONS:
        for key, meta in WEAPONS.items():
            if weapon_raw in (meta["zh"], *meta.get("alias", [])):
                weapon = key
                break
        else:
            weapon = "dao"
    tier = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 3
    look = parts[2] if len(parts) > 2 else ""
    names = [n.strip() for n in (parts[3] if len(parts) > 3 else "").replace("，", ",").split(",") if n.strip()]
    techs = [Technique(name=n, variant=_variant_for(n, i), damage=1.0 + 0.05 * (i % 3), cooldown=2.0 + i)
             for i, n in enumerate(names)]
    return Fighter(name=default_name, weapon=weapon, tier=tier, look=look, side=side, techniques=techs)


def parse_props(spec: str) -> List[Prop]:
    """解析道具。格式：``名称|hp|破坏表现; 名称|hp|破坏表现``（hp 与表现可省）。"""
    out: List[Prop] = []
    for chunk in (spec or "").replace("；", ";").replace("\n", ";").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        f = [x.strip() for x in chunk.split("|")]
        name = f[0]
        hp = float(f[1]) if len(f) > 1 and f[1].replace(".", "", 1).isdigit() else 2.5
        fx = f[2] if len(f) > 2 else f"{name}被劈断，碎屑飞散"
        out.append(Prop(name=name, hp=hp, break_fx=fx,
                        debris_fx=f"{name}的断口留着毛刺"))
    return out


class H3WushuChoreograph:
    """武打编排（武术动作导演）：产出完整打斗编排 + 可直接用的 H3 提示词。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "角色A": ("STRING", {"default": "nodachi|4|男性，黑发披散，黑色武士劲装，双手持野太刀|过肩劈,横扫,突刺",
                                    "multiline": False,
                                    "tooltip": "武器|等级|外貌|招式1,招式2 —— 武器可写中文（太刀/刀/剑/枪/棍/棒/空手）"}),
                "角色B": ("STRING", {"default": "dao|3|男性，灰发束髻，靛蓝汉服武袍，右手持雁翎单刀|撩刀,反手削",
                                    "multiline": False}),
                "场景": ("STRING", {"default": "雨夜长街，两侧灯笼暖光，湿石板反光，右侧灯笼柱与街边木栏为实体",
                                   "multiline": True}),
                "地面": (GROUNDS, {"default": "湿石"}),
                "道具": ("STRING", {"default": "右侧灯笼柱|2.5|灯笼柱从中段断开，灯纸炸开; 街边木栏|1.5|木栏被劈断两根，木屑飞散",
                                   "multiline": True,
                                   "tooltip": "名称|耐久|破坏表现，分号分隔；留空=场景没有可破坏道具"}),
                "时长秒": ("FLOAT", {"default": 10.0, "min": 3.0, "max": 60.0, "step": 0.1}),
                "分镜数": ("INT", {"default": 3, "min": 1, "max": 8,
                                  "tooltip": "武打默认 1~3 镜；4 镜起规则引擎会给警告"}),
                "模式": (["t2v", "ref2v"], {"default": "t2v",
                                           "tooltip": "文生视频 / 多参考图 —— 两套壳字段不同，绝不能混用"}),
                "编排类型": (["对打（双人）", "单人演练（对齐 wushu_h3_v2 分布）"],
                           {"default": "对打（双人）",
                            "tooltip": "wushu_h3_v2 是单人体操模型 LoRA，单人连招才是分布内；双人属外推"}),
                "招式词库": (["LoRA 实证词（wushu_h3_v2 训练集）", "自造招式名（v7 影视向）"],
                           {"default": "LoRA 实证词（wushu_h3_v2 训练集）",
                            "tooltip": "实证词 = 训练集里真实出现过的词，命中率远高于自造词"}),
                "开场间距": ("FLOAT", {"default": 2.0, "min": 0.68, "max": 6.0, "step": 0.5}),
                "指定赢家": (["自动按血量判定", "角色A", "角色B"], {"default": "自动按血量判定"}),
                "进攻性": ("FLOAT", {"default": 0.62, "min": 0.1, "max": 1.0, "step": 0.02,
                                    "tooltip": "越高越倾向放招式而不是普攻"}),
                "种子": ("INT", {"default": 1234, "min": 0, "max": 2**31 - 1}),
            },
            "optional": {
                "参考图张数": ("INT", {"default": 3, "min": 1, "max": 9,
                                      "tooltip": "仅 ref2v 模式：<Picture N> 的编号上限"}),
                "规则自检": ("BOOLEAN", {"default": True,
                                        "tooltip": "生成后用 h3lint 跑一遍，报告里给分数与必改项"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "beats_json", "report")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3/Wushu Bridge"
    OUTPUT_NODE = True
    DESCRIPTION = "武打编排（动作导演）：出招/接招/命中反馈/追击连招/防守反击/道具破坏/招式拼接"

    def run(self, 角色A, 角色B, 场景, 地面, 道具, 时长秒, 分镜数, 模式,
            开场间距, 指定赢家, 进攻性, 种子,
            编排类型="对打（双人）", 招式词库="LoRA 实证词（wushu_h3_v2 训练集）",
            参考图张数=3, 规则自检=True):
        a = parse_fighter(角色A, "角色A", "左")
        b = parse_fighter(角色B, "角色B", "右")
        props = parse_props(道具)
        winner = "" if 指定赢家 == "自动按血量判定" else 指定赢家
        style = "lora_v2" if 招式词库.startswith("LoRA") else "h3"

        # ── 单人演练：对齐 wushu_h3_v2 的单人体操模型训练分布 ──
        if 编排类型.startswith("单人演练"):
            moves = [t.name for t in a.techniques] or solo_drill(a, float(时长秒), int(种子))
            prompt = build_solo_lora_prompt(
                PromptSpec(style="lora_v2", scene=场景, duration=float(时长秒)),
                a, moves, rng=__import__("random").Random(int(种子)))
            rep = (f"单人演练（wushu_h3_v2 分布内）：{len(moves)} 个动作\n"
                   f"招式串：{'、'.join(moves)}\n"
                   f"提示词 {len(prompt)} 字｜固化段落已带上物理反馈词"
                   f"（{', '.join(LORA_PHYS_ZH)}）\n"
                   f"提醒：该 LoRA 训练分布是 39~90 帧 @30fps（1.3~3.0 秒）单view，"
                   f"长片建议分段生成")
            return _ui((prompt, "{}", rep), rep)

        director = FightDirector(
            a, b, duration=float(时长秒), start_distance=float(开场间距),
            scene=场景, ground=地面, props=props, winner=winner,
            seed=int(种子), aggressiveness=float(进攻性), style=style,
        )
        choreo: Choreography = director.run()
        spec = PromptSpec(mode=模式, duration=float(时长秒), shots=int(分镜数),
                          scene=场景, pictures=int(参考图张数), style=style)
        # 超长自动分段（规则引擎 maxLen=2500；15 秒要 6~9 次交锋，必然写不下）
        segments = split_long_prompt(choreo.beats, spec, (a, b), max_chars=2400,
                                     rng=__import__("random").Random(int(种子)))
        if len(segments) == 1:
            prompt = segments[0]
        else:
            prompt = ("\n\n" + "=" * 30 + " 分段提示词（每段单独提交，上一段尾帧作下一段首帧）"
                      + "=" * 30 + "\n\n").join(
                f"【第 {i + 1}/{len(segments)} 段】\n{s}" for i, s in enumerate(segments))

        # 编排明细报告
        lines = [f"编排结果：{choreo.result}｜交锋 {choreo.exchanges} 次｜拍数 {len(choreo.beats)}"]
        kinds: Dict[str, int] = {}
        for bt in choreo.beats:
            kinds[bt.attack_kind] = kinds.get(bt.attack_kind, 0) + 1
        lines.append("拍型统计：" + "、".join(f"{k}×{v}" for k, v in kinds.items()))
        lines.append("逐拍：")
        for bt in choreo.beats:
            extra = f"｜道具：{bt.prop_event[:24]}" if bt.prop_event else ""
            lines.append(f"  [{bt.t0:5.2f}-{bt.t1:5.2f}] {bt.distance:g}格 {bt.attacker}→{bt.defender} "
                         f"{bt.attack_kind}「{bt.technique}」 {bt.outcome}／{bt.defense or '—'}{extra}")
        if any(p.broken for p in choreo.props):
            lines.append("道具留痕：" + "、".join(
                f"{p.name}（{p.debris_fx}）" for p in choreo.props if p.broken))
        else:
            lines.append("道具留痕：本场没有道具被破坏")

        if 规则自检:
            try:
                from .lint import h3lint

                segs = [prompt] if len(segments) == 1 else segments
                lines.append("")
                worst = 100
                allerr = 0
                for i, s in enumerate(segs):
                    res = h3lint.check(s, {"mode": "final", "englishAware": True,
                                           "shotCounting": "auto"})
                    worst = min(worst, res["score"])
                    allerr += res["stats"]["errors"]
                    tag = f"第{i + 1}段 " if len(segs) > 1 else ""
                    lines.append(f"规则引擎自检 {tag}：{res['grade']} {res['score']}/100  "
                                 f"必改 {res['stats']['errors']}｜建议 {res['stats']['warns']}")
                    for it in res["items"]:
                        if it["level"] in ("error", "warn"):
                            lines.append(f"  [{it['level']}] {it['msg'][:80]}")
                if allerr == 0 and worst >= 90:
                    lines.append("  ✓ 全部段落达标（≥90 分且零必改），可直接用")
                if len(segs) > 1:
                    lines.append(f"  ℹ 因超长已自动切成 {len(segs)} 段：每段单独提交，"
                                 f"上一段尾帧作下一段首帧（道具留痕与站位已在段内保留）")
            except Exception as exc:
                lines.append(f"（规则自检跳过：{exc}）")

        report = "\n".join(lines)
        return _ui((prompt, choreo.to_json(), report), report)


NODE_CLASS_MAPPINGS = {
    "H3WushuChoreograph": H3WushuChoreograph,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3WushuChoreograph": "H3 武打编排（动作导演）",
}
