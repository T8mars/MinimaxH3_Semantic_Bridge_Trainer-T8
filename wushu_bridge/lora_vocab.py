"""wushu_h3_v2 LoRA 的**训练集实证招式词库**（引擎的用词来源）。

为什么单独建一份：这个 LoRA 的训练集是 **1668 段人体模型三视图动画**，
caption 是固定模板 + **单个招式词**（109 种组合）。也就是说：

* 训练集里**真实出现过**的词（如 ``横斩劈砍`` / ``突刺贯穿`` / ``举盾格挡`` /
  ``踉跄后退`` / ``连续连招``），命中率远高于自造词；
* ``战斗动作``(232 条) 是泛化占位词，用它会"招式不具体"，应尽量换成具体招式；
* 模板里那几个**物理反馈词必须保留**：``实时爆发力`` / ``全身发力传导`` /
  ``无慢动作`` / ``无停顿`` —— 它们参与塑造动作质量；
* LoRA 是**单人体操模型**训练，双人属外推。

本模块把这些实证词整理成可直接驱动编排引擎的表：每个词都映射到引擎的
``VARIANTS`` 变体（扇形角/攻击高度/伤害·击退倍率），所以用 LoRA 的词
**同时也保留物理模型**，不是换一套词就丢掉量化。

括号内数字是训练集里的出现次数（来自 LoRA 使用指南 §5.3，实证统计）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


@dataclass(frozen=True)
class LoraMove:
    """一条训练集实证招式词。"""

    zh: str                      # 训练集里的中文原词
    en: str                      # 训练集里的英文对照
    variant: str                 # 映射到 choreography.VARIANTS 的 key
    category: str                # 拳法/腿法/步法/受击/刀剑/棍棒/远程/防御/摔投/空中/节奏/通用
    count: int = 0               # 训练集出现次数（0 = 组合词或未计数）


# ── 训练集实证词库（逐条来自 LoRA 使用指南 §5.3，不改写、不自造）──────────
LORA_MOVES: List[LoraMove] = [
    # 拳法
    LoraMove("蹬地打直拳", "drives a straight punch", "thrust", "拳法", 37),
    LoraMove("后手横拳", "rear-hand cross", "slash", "拳法", 0),
    LoraMove("抡臂横勾", "swings a hook", "slash", "拳法", 0),
    LoraMove("上勾拳", "uppercut", "rise", "拳法", 0),
    # 腿法
    LoraMove("高侧踢", "high side kick", "sweep", "腿法", 24),
    LoraMove("提膝撞击", "knee strike", "rise", "腿法", 8),
    # 步法（这些当"间隙动作"用，不当攻击）
    LoraMove("稳步前进", "steps forward steadily", "step", "步法", 108),
    LoraMove("疾步冲刺", "dashes forward", "step", "步法", 79),
    LoraMove("转身换向", "turns and shifts direction", "step", "步法", 7),
    LoraMove("下蹲低姿", "drops into a low crouch", "step", "步法", 0),
    LoraMove("急速冲刺", "sprints in", "step", "步法", 0),
    # 受击 / 失衡（命中反馈用词）
    LoraMove("踉跄后退", "staggers backward", "react", "受击", 69),
    LoraMove("失衡踉跄", "loses balance and staggers", "react", "受击", 7),
    LoraMove("反弓跌落", "arcs backward and falls", "react", "受击", 0),
    LoraMove("团身翻滚", "tucks into a roll", "react", "受击", 0),
    # 刀剑
    LoraMove("横斩劈砍", "horizontal slash", "slash", "刀剑", 35),
    LoraMove("双持交叉斩", "dual-wield cross slash", "diag", "刀剑", 10),
    LoraMove("横扫", "sweeping slash", "sweep", "刀剑", 7),
    LoraMove("突刺贯穿", "piercing thrust", "thrust", "刀剑", 50),
    # 棍棒
    LoraMove("转棍扫击", "spins the staff into a sweep", "sweep", "棍棒", 10),
    # 远程 / 弩
    LoraMove("举弩瞄准", "raises the crossbow and aims", "aim", "远程", 0),
    LoraMove("脚踏上弦装填", "braces the crossbow and reloads", "aim", "远程", 0),
    LoraMove("松手放箭", "releases the bolt", "thrust", "远程", 0),
    # 防御（防守反应用词）
    LoraMove("举盾格挡", "raises a shield to block", "guard", "防御", 54),
    LoraMove("侧身格挡反击", "blocks from the side and counters", "guard", "防御", 0),
    LoraMove("侧闪回避", "sidesteps to evade", "dodge", "防御", 24),
    # 摔投
    LoraMove("过肩摔投", "shoulder throw", "heavy", "摔投", 7),
    LoraMove("锁颌下坠", "locks the jaw and drops", "heavy", "摔投", 0),
    # 空中
    LoraMove("腾空跃起", "leaps into the air", "leap", "空中", 67),
    # 节奏 / 待机（间隙动作用词）
    LoraMove("呼吸站姿微动", "small breathing shifts in stance", "idle", "节奏", 41),
    # 通用
    LoraMove("连续连招", "chains a combo sequence", "combo", "通用", 239),
    LoraMove("战斗动作", "combat action", "generic", "通用", 232),
]

BY_CATEGORY: Dict[str, List[LoraMove]] = {}
for _m in LORA_MOVES:
    BY_CATEGORY.setdefault(_m.category, []).append(_m)

# 训练集里就是这么写的：多招式用「、」串联（原样照抄的 5 组）
LORA_COMBOS: List[str] = [
    "蹬地打直拳、转身换向",
    "突刺贯穿、疾步冲刺",
    "举盾格挡、侧身格挡反击、转棍扫击",
    "后手横拳、举弩瞄准、脚踏上弦装填、松手放箭",
    "连续连招",
]

# **必须保留**的物理反馈词（训练集模板固定段，去掉动作质量会掉）
LORA_PHYS_ZH = ["实时爆发力", "全身发力传导", "无慢动作", "无停顿"]
LORA_PHYS_EN = ["explosive real-time force", "force transmission through whole body",
                "no slow motion", "no pause"]

# 负向（训练时负向为空，只加画质类）
LORA_QUALITY_NEG = ("blurry, distorted, low quality, jittery motion, extra limbs, "
                    "deformed hands, morphing bodies, inconsistent lighting, flickering, "
                    "static pose, frozen motion")

# 训练分布事实（用于编排时的合法性检查）
LORA_TRAIN_FACTS = {
    "videos": 1668, "single_person": True, "three_view_mannequin": True,
    "frames": (39, 90), "fps": 30, "seconds": (1.3, 3.0),
    "resolution": (960, 360), "trigger": "wushu_action",
    "strength_normal": (0.6, 0.8), "strength_mannequin": (0.9, 1.0),
}


def attacks(category: Optional[str] = None) -> List[LoraMove]:
    """可用作**攻击**的实证词（排除步法/受击/防御/节奏/通用）。"""
    out = [m for m in LORA_MOVES if m.category not in
           ("步法", "受击", "防御", "节奏", "通用")]
    if category:
        out = [m for m in out if m.category == category]
    return sorted(out, key=lambda m: -m.count)


def by_variant(variant: str) -> List[LoraMove]:
    return [m for m in LORA_MOVES if m.variant == variant]


def pick_by_variant(variant: str, rng) -> Optional[LoraMove]:
    """给定引擎选出的变体，挑一条对应的实证词（保证词与物理对得上）。

    映射逻辑：引擎按扇形角/高度选变体 → 这里回挑同变体的训练集词，
    所以"用 LoRA 的词"不会破坏量化模型。
    """
    cands = [m for m in attacks() if m.variant == variant]
    if not cands:
        cands = [m for m in attacks()]
    return rng.choice(cands) if cands else None


def defense_word(kind: str, rng) -> Optional[LoraMove]:
    """防守反应词：格挡 / 闪避。"""
    want = {"格挡": "guard", "闪避": "dodge"}.get(kind, "guard")
    cands = [m for m in LORA_MOVES if m.variant == want]
    return rng.choice(cands) if cands else None


def reaction_word(sev: str, rng) -> Optional[LoraMove]:
    """命中反馈词：轻/中/重。"""
    pool = {"light": ["失衡踉跄"], "solid": ["踉跄后退", "失衡踉跄"],
            "heavy": ["反弓跌落", "团身翻滚"], "finish": ["反弓跌落"]}.get(sev, [])
    cands = [m for m in LORA_MOVES if m.zh in pool]
    return rng.choice(cands) if cands else None


def step_words() -> List[LoraMove]:
    return [m for m in LORA_MOVES if m.category == "步法"]


def idle_words() -> List[LoraMove]:
    return [m for m in LORA_MOVES if m.variant in ("idle", "step")]


def phys_suffix(zh: bool = True) -> str:
    """训练模板里那段必须保留的物理反馈词。"""
    return "、".join(LORA_PHYS_ZH) if zh else ", ".join(LORA_PHYS_EN)


def summary() -> str:
    lines = [f"LoRA 实证招式词 {len(LORA_MOVES)} 条，分 {len(BY_CATEGORY)} 类"]
    for cat, moves in BY_CATEGORY.items():
        top = "、".join(f"{m.zh}({m.count})" if m.count else m.zh for m in moves[:6])
        lines.append(f"  {cat}：{top}")
    return "\n".join(lines)
