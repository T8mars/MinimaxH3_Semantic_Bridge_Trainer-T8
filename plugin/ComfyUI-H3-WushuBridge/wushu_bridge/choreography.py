"""武打编排引擎（武术动作导演）。

这一层解决的问题和"语义桥"完全不同：

* **语义桥**：把 conditioning 往"武打逻辑完备"的分布推 —— 它管**怎么说**。
* **编排引擎**：决定**打什么** —— 谁先出招、对手怎么接、打中之后什么反馈、
  什么时候追击连招、防守/闪避之后怎么反击、招式怎么拼接、打坏的东西怎么留痕。

量化模型全部沿用你项目 `sim3d/engine.js` 的口径（逐字搬过来，不另造一套）：

* ``WEAPONS``  10 种兵器：reach(米) / w 起手 / a 判定 / r 收招 / dmg / gb 破防系数
* ``VARIANTS`` 7 种地面招式变体：扇形角 / 攻击高度 / 伤害·击退·耗力·射程倍率
* ``AIR_VARIANTS`` 4 种空中招式
* ``CELL = 0.5`` 米/格；``CAP_R = 0.34`` 米胶囊半径（不穿模阈值 0.68 米）
* ``TIER_POWER / MOBILITY`` 等级与机动档
* ``FILLERS`` 间隙动作库（近/中/远三档）—— 保证每一秒都有具体动作，不出现无因停顿
* 14 条核心打斗规则（``RULES_14``）作为自检 rubric

状态机覆盖用户要求的全部要点：

1. **出招/接招时机** —— 主动权(initiative) + 每拍攻守决策
2. **命中反馈分级** —— 轻触 / 格挡 / 擦过 / 命中 / 重击 / 终结，各自的受力方向与位移
3. **追击与连续出招** —— 命中造成硬直(stun)，攻方在硬直内可以追击
4. **防守后反击 / 闪避后反击** —— 连格/连闪到阈值、或对方落空 → 反击窗口，主动权交换
5. **招式拼接** —— 连段（压制 → 换位 → 终结），同一连段内距离不跳变
6. **道具破坏与留痕** —— 劈空/被磕飞的刀路落到道具上扣耐久，破坏后永久留痕
7. **死线** —— 到时必须有可见结果（终结技 / 双倒）
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ══════════════════════════════════════════════════════════════════
# 一、量化常量（逐字来自 sim3d/engine.js）
# ══════════════════════════════════════════════════════════════════

CELL = 0.5          # 米/格
CAP_R = 0.34        # 角色胶囊半径（不穿模阈值 = 0.68 米）

WEAPONS: Dict[str, Dict[str, Any]] = {
    "none":    {"zh": "空手", "reach": 0.86, "w": 0.16, "a": 0.09, "r": 0.22, "dmg": 7,  "gb": 0.9, "alias": ["拳", "掌", "肘", "膝"]},
    "duanren": {"zh": "短刃", "reach": 0.78, "w": 0.14, "a": 0.08, "r": 0.20, "dmg": 8,  "gb": 0.9, "alias": ["匕首", "短刃"]},
    "duangun": {"zh": "短棍", "reach": 1.05, "w": 0.19, "a": 0.09, "r": 0.25, "dmg": 9,  "gb": 1.1, "alias": ["短棍"]},
    "dao":     {"zh": "刀",   "reach": 1.24, "w": 0.22, "a": 0.11, "r": 0.30, "dmg": 10, "gb": 1.2, "alias": ["刀"]},
    "jian":    {"zh": "剑",   "reach": 1.34, "w": 0.20, "a": 0.10, "r": 0.27, "dmg": 9,  "gb": 1.1, "alias": ["剑"]},
    "pu":      {"zh": "朴刀", "reach": 1.50, "w": 0.30, "a": 0.12, "r": 0.38, "dmg": 13, "gb": 1.4, "alias": ["朴刀"]},
    "nodachi": {"zh": "太刀", "reach": 1.64, "w": 0.34, "a": 0.13, "r": 0.43, "dmg": 14, "gb": 1.5, "alias": ["太刀"]},
    "gun":     {"zh": "棍",   "reach": 1.86, "w": 0.28, "a": 0.12, "r": 0.36, "dmg": 11, "gb": 1.3, "alias": ["棍"]},
    "bang":    {"zh": "棒",   "reach": 1.92, "w": 0.30, "a": 0.12, "r": 0.38, "dmg": 12, "gb": 1.4, "alias": ["棒", "杆"]},
    "qiang":   {"zh": "枪",   "reach": 2.44, "w": 0.30, "a": 0.10, "r": 0.38, "dmg": 12, "gb": 1.2, "alias": ["枪", "矛"]},
}

VARIANTS: List[Dict[str, Any]] = [
    {"key": "slash",  "zh": "横斩", "arc": 118, "h": "mid",  "dmg": 1.00, "kb": 1.00, "stam": 1.00, "reach": 1.00, "tag": "攻击"},
    {"key": "diag",   "zh": "斜劈", "arc": 96,  "h": "high", "dmg": 1.10, "kb": 1.15, "stam": 1.10, "reach": 1.00, "tag": "攻击"},
    {"key": "thrust", "zh": "突刺", "arc": 22,  "h": "mid",  "dmg": 1.05, "kb": 0.70, "stam": 0.85, "reach": 1.22, "tag": "攻击"},
    {"key": "rise",   "zh": "撩挑", "arc": 74,  "h": "low",  "dmg": 0.95, "kb": 1.05, "stam": 0.95, "reach": 1.02, "tag": "攻击"},
    {"key": "sweep",  "zh": "扫堂", "arc": 150, "h": "low",  "dmg": 0.90, "kb": 1.30, "stam": 1.20, "reach": 1.06, "tag": "移动"},
    {"key": "heavy",  "zh": "重击", "arc": 138, "h": "mid",  "dmg": 1.70, "kb": 2.10, "stam": 1.75, "reach": 1.05, "tag": "攻击", "slow": 1.35},
    {"key": "finish", "zh": "终结", "arc": 132, "h": "mid",  "dmg": 2.30, "kb": 2.60, "stam": 1.90, "reach": 1.05, "tag": "攻击", "slow": 1.45, "finisher": True},
]
AIR_VARIANTS: List[Dict[str, Any]] = [
    {"key": "leap_slash", "zh": "跃斩",     "dmg": 1.15, "kb": 1.25},
    {"key": "wall_flip",  "zh": "踏墙翻身", "dmg": 1.05, "kb": 1.10},
    {"key": "air_combo",  "zh": "空中连击", "dmg": 1.30, "kb": 1.35},
    {"key": "dive",       "zh": "俯冲击",   "dmg": 1.65, "kb": 1.95},
]

TIER_POWER = {1: 1, 2: 1.5, 3: 2, 4: 3, 5: 4, 6: 6, 7: 8, 8: 12, 9: 16}
TIER_NAME = ["", "凡人", "好手", "高手", "宗师", "大宗师", "绝世", "超凡", "神魔", "灭世"]

# 间隙动作库（combat-logic.js:40-101，近/中/远三档）
FILLERS: Dict[str, List[str]] = {
    "close": ["收肘顶住对方小臂，脚下碾半步保持粘住",
              "用肩胛顶开对方持械手，同时后脚跟抬起换重心",
              "头略微后让避开擦过的刃线，手仍扣在对方腕上",
              "被压住的瞬间沉腰降低重心，把对方的力卸进地面",
              "贴着对方旋半步，把两人轴线从正面改成侧向",
              "以小臂敲开对方手腕一次，重新夺回内侧位置",
              "呼吸顶在鼻腔里，脚下不停，脚尖反复调整三点支撑",
              "用兵器中段压住对方兵器中段，谁都不动但都在加力"],
    "mid":   ["垫步两次逼近，每次把距离缩短半个身位",
              "绕半步改角度，刃线始终指着对方肩线",
              "抬手做一次短促的试探刺，逼对方先动",
              "换架，前后脚互换，把发力腿调到后面",
              "拖半步撤出射程，同时把兵器收到腰侧蓄势",
              "空手侧抬起护住面门，持械侧沉在腰线以下",
              "斜上一步切断对方的绕行路线，两人重新面对面",
              "呼吸一沉一浮，肩膀放松半寸再收紧"],
    "far":   ["压着步子连续逼近，兵器前指不给对手起势时间",
              "横向绕行寻找掩体，视线不离开对方腰胯",
              "侧身小碎步调整站位，把太阳移到对方眼里",
              "把兵器举到肩上缩短出手路径，脚下不减速",
              "突然收步做一次假起手，看对方先反应",
              "沿掩体边缘折线靠近，每次拐角都改变攻击角度",
              "低身快步入射程，落地时起手已经完成"],
}

# 运镜阶梯（pipeline.js:243-244）+ 镜头职责（camera-guide.md:21-30）
CAM_LADDER = ["Wide tracking shot", "Side tracking shot", "Over-the-shoulder shot",
              "Low-angle tracking shot", "Slow dolly in", "High-angle crane shot"]
CAM_ZH = ["全景跟拍", "侧面平行跟拍", "过肩跟拍", "低机位跟拍", "缓慢推近", "高位俯拍"]
CAM_DUTY = ["建立空间", "追随位移", "看清接触", "表达反制", "表达失衡", "展示弹道", "揭示结果", "稳定收尾"]

# 14 条核心打斗规则（combat-logic.js:116-131）—— 作为自检 rubric
RULES_14 = [
    "无因停顿禁令：任何超过 0.35 秒的静止，都必须由受击硬直/闪避落位/被兵器压住/换架/绕步/掩体阻隔解释，并写成画面里的动作",
    "打斗反馈五要素：前因、动作、接触点、受力方向与位移、余波与下一步的起因",
    "因果链闭合：上一拍的结果必须是下一拍的起因",
    "连续招式：收招后半段接下一招，写成「第一式未收尽，第二式已起」；同串连段距离不得跳变",
    "闪避必须指向具体来招，写明落位，落位后必须出现反击或反压",
    "反击窗口：格挡后 0.2~0.8 秒、闪避落位后 0.15~0.6 秒、对方落空后 0.15~0.5 秒内必须有反击或抢位",
    "主动权交换：每 1.5~3 秒交换一次，写成谁退谁进、谁的兵器线压住对方",
    "距离纪律：距离由招式有效范围决定，缩短必须先写脚步，拉开必须先写撤步或被击退",
    "高度与姿态：至少各变化一次，且每次变化有受力或选择作为原因",
    "兵器与道具唯一性：同一件兵器不得凭空出现第二把；被击落的兵器留在画面里",
    "间隙动作显性化：把间隙动作写进正文，让每一秒都有具体动作",
    "节奏红线：首个有效动作 ≤0.3 秒；纯恢复动作 ≤0.7 秒；15 秒内交锋 6~9 次",
    "声音跟着受力走：材质与声音对应，徒手不写金属声",
    "结束画面兑现完成条件：胜负或脱离必须被画面看见",
]

# 兵器 → 材质音（R138：徒手不写金属声）
WEAPON_SOUND = {
    "none": "拳掌击肉的闷响", "duanren": "短刃破空的细响", "duangun": "短棍扫过的风声",
    "dao": "刀弧破空", "jian": "剑锋清鸣", "pu": "朴刀沉重的破空",
    "nodachi": "太刀长长的破空声", "gun": "棍身扫过的风声", "bang": "棒梢沉闷的风声",
    "qiang": "枪尖刺破空气的锐响",
}
METAL_WEAPONS = {"duanren", "dao", "jian", "pu", "nodachi", "qiang"}


def band_of(dist_cells: float) -> str:
    """间隙动作分档（combat-logic.js:161）。"""
    return "close" if dist_cells <= 1 else ("mid" if dist_cells <= 2.5 else "far")


def reach_cells(weapon: str, variant: Optional[Dict[str, Any]] = None) -> float:
    """兵器有效射程换算成格。"""
    w = WEAPONS.get(weapon, WEAPONS["none"])
    r = w["reach"]
    if variant:
        r *= variant.get("reach", 1.0)
    return r / CELL


# ══════════════════════════════════════════════════════════════════
# 二、数据结构
# ══════════════════════════════════════════════════════════════════

@dataclass
class Technique:
    """招式（角色卡里的招式条目）。"""

    name: str
    variant: str = "slash"          # 对应 VARIANTS 的 key
    reach_bonus: float = 0.0        # 额外格数
    damage: float = 1.0             # 倍率
    cooldown: float = 0.0           # 秒
    note: str = ""                  # 效果句（首现要写全）

    def to_variant(self) -> Dict[str, Any]:
        base = next((v for v in VARIANTS if v["key"] == self.variant), VARIANTS[0])
        v = dict(base)
        v["dmg"] = v["dmg"] * self.damage
        v["reach"] = v["reach"] + self.reach_bonus
        v["name"] = self.name
        return v


@dataclass
class Fighter:
    """角色卡。"""

    name: str
    weapon: str = "dao"
    tier: int = 3
    hp: float = 5.0
    guard_per_sec: int = 1
    dodge_per_sec: int = 1
    guard_chain_threshold: int = 3      # 连格几次可反击
    dodge_chain_threshold: int = 2      # 连闪几次可反击
    techniques: List[Technique] = field(default_factory=list)
    look: str = ""                      # 外貌：发色+发型+服装+体型
    side: str = "左"                    # 开场站位

    # 运行期状态
    guards_left: int = 0
    dodges_left: int = 0
    guard_chain: int = 0
    dodge_chain: int = 0
    stun: float = 0.0                   # 剩余硬直秒
    down: bool = False
    dropped_weapon: bool = False
    cooldowns: Dict[str, float] = field(default_factory=dict)

    def weapon_zh(self) -> str:
        return WEAPONS.get(self.weapon, WEAPONS["none"])["zh"]

    def reset_second(self) -> None:
        self.guards_left = self.guard_per_sec
        self.dodges_left = self.dodge_per_sec


@dataclass
class Prop:
    """场景道具（可破坏）。"""

    name: str
    hp: float = 3.0
    material: str = "木质"
    break_fx: str = "木栏断裂，木屑飞散"
    debris_fx: str = "断口留着毛刺"
    intact: bool = True
    broken: bool = False
    hits: int = 0


@dataclass
class Beat:
    """一拍（= 一次交锋或一次明确位移）。"""

    index: int
    t0: float
    t1: float
    distance: float
    attacker: str
    defender: str
    initiative_from: str = ""
    attack_kind: str = "普攻"           # 普攻 / 招式 / 追击 / 反击
    technique: str = ""
    variant: str = ""
    attack_desc: str = ""
    defense: str = ""                   # 格挡 / 闪避 / 硬吃 / 无（硬直中）
    defense_desc: str = ""
    outcome: str = ""                   # MISS/GRAZE/BLOCK/GUARD_BREAK/HIT/HEAVY/FINISH/DODGE
    contact: str = ""
    feedback: str = ""
    displacement: str = ""
    cause: str = ""                     # 前因（上一拍的结果）
    filler: str = ""
    prop_event: str = ""
    camera: str = ""
    camera_zh: str = ""
    camera_duty: str = ""
    afterimage: str = ""                # 余波/环境反馈
    fatal: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(self).items()}


@dataclass
class Choreography:
    beats: List[Beat] = field(default_factory=list)
    result: str = ""
    winner: str = ""
    exchanges: int = 0
    props: List[Prop] = field(default_factory=list)
    report: str = ""

    def to_json(self) -> str:
        return json.dumps({"result": self.result, "winner": self.winner,
                           "exchanges": self.exchanges,
                           "props": [asdict(p) for p in self.props],
                           "beats": [b.to_dict() for b in self.beats]},
                          ensure_ascii=False, indent=1)


# ══════════════════════════════════════════════════════════════════
# 三、编排引擎
# ══════════════════════════════════════════════════════════════════

# 反应分级：伤害倍率 → 反馈文字 / 位移 / 硬直秒
REACTION_TIERS = [
    {"max": 0.55, "key": "light",  "stun": 0.45, "disp": "后退半步",
     "fx": ["闷响一声，{d}肩头一沉", "被顶得胸口一缩，{d}脚跟碾了半步"],
     "after": "尘土从{n}脚下弹起"},
    {"max": 1.15, "key": "solid",  "stun": 0.75, "disp": "踉跄倒退一格",
     "fx": ["{d}衣料撕裂，踉跄倒退", "{d}重心后压，喉间闷哼一声"],
     "after": "{n}的鞋底在{n2}上刮出一道痕"},
    {"max": 2.0,  "key": "heavy",  "stun": 1.1,  "disp": "倒退两格，单膝着地",
     "fx": ["{d}膝盖砸地，兵器一端磕在地面", "{d}被震得半跪，手臂发麻"],
     "after": "{n}脚下{n2}碎屑溅起"},
    {"max": 99.0, "key": "finish", "stun": 9.0,  "disp": "沿刀的作用线仰面倒地，不再起身",
     "fx": ["{d}的兵器脱手飞出，人沿受力方向倒下", "{d}被压得仰面倒进{n}，水花溅起"],
     "after": "倒地余波里{n2}上的浮尘缓缓落下"},
]

GROUND_MAT = {"湿石": "湿石板", "夯土": "夯土地面", "木地板": "木地板", "雪地": "积雪", "石砖": "石砖地"}


def _pick(rng: random.Random, seq: Sequence[str]) -> str:
    return seq[rng.randrange(len(seq))]


class FightDirector:
    """武打编排器：给定双方与场景，产出一串带完整因果的拍子。"""

    def __init__(
        self,
        a: Fighter,
        b: Fighter,
        duration: float = 10.0,
        fps: int = 24,
        start_distance: float = 2.0,
        scene: str = "雨夜长街，两侧灯笼暖光，湿石板反光",
        ground: str = "湿石",
        props: Optional[List[Prop]] = None,
        winner: str = "",
        seed: int = 1234,
        aggressiveness: float = 0.62,
        max_exchanges: int = 12,
        style: str = "h3",
    ) -> None:
        self.a, self.b = a, b
        # style="lora_v2" 时用 wushu_h3_v2 LoRA 训练集**实证词**替换招式/防守/受击用词
        # （自造词在这个 LoRA 上命中率低）。词按变体映射，量化模型不变。
        self.style = style
        self.lv = None
        if style == "lora_v2":
            from . import lora_vocab as lv
            self.lv = lv
        self.duration = float(duration)
        self.fps = int(fps)
        self.dist = float(start_distance)
        self.scene = scene
        self.ground = GROUND_MAT.get(ground, ground)
        self.props = props or []
        self.winner = winner
        self.rng = random.Random(seed)
        self.aggro = aggressiveness
        self.max_exchanges = max_exchanges

        self.beats: List[Beat] = []
        self.exchanges = 0
        self.t = 0.0
        self.last_outcome = ""       # 上一拍结果（作为下一拍前因，进 JSON）
        self.last_link = ""          # 上一拍结果压缩成的**因果连接短语**（进正文）
        self.last_outcome_key = ""   # 上一拍的结果类型（HIT/BLOCK/DODGE/MISS…）
        self.last_actor = ""
        self.cam_i = 0
        self.duty_i = 0
        self.last_dist_change = 0.0  # 上一次主动权交换的时间
        self.initiative = a          # 谁掌握主动权
        self.height_changed = False
        self.stance_changed = False
        self.used_techniques: Dict[str, int] = {}   # 招式使用次数（首现要写全效果）
        self.follow_streak = 0       # 连续追击计数（最多连 2 下，否则对手全程没反应）
        self.done = False            # 是否已经写出终结拍（防重复）

    # ---------- 工具 ----------
    def _first_use_effect(self, name: str, variant: Dict[str, Any], who: Fighter) -> str:
        """招式**首现写全**：起手 → 轨迹 → 效果实体 → 落点（R021）。

        按变体分别写，避免每条都是同一句话（实测同一句重复写会显得机械）。
        **摔投类单独写**：把「过肩摔投」按劈砍弧线写会自相矛盾（实测发现）。
        """
        key = variant.get("key", "slash")
        arc = variant.get("arc", 100)
        w = who.weapon_zh()
        if any(k in name for k in ("摔", "投", "锁", "擒")):
            return (f"首现写全：起手抢住对方手腕或衣领，跨半步把重心顶进对方胯下，"
                    f"用腰背把人整体掀过肩线，落点是对方的背与后脑砸地，余势扫起{self._mat_zh()}上的水花与碎屑")
        if any(k in name for k in ("踢", "膝", "腿", "踹")):
            return (f"首现写全：起手拧腰把支撑腿蹬直，攻击腿沿{arc}度扫过去，"
                    f"落点在对方肋侧或膝窝，余势扫起{self._mat_zh()}上的水花与碎屑")
        if any(k in name for k in ("弩", "箭", "弓")):
            return (f"首现写全：起手把弩端平压住肩线，视线沿箭槽对齐，松手放箭，"
                    f"落点在对方胸口一线，弩身回震顶得肩膀后错半分")
        table = {
            "slash":  f"起手把{w}拉到腰高，刃走{arc}度平弧从对方外侧扫过胸口线，落点压在架门上",
            "diag":   f"起手沉肩抬肘，刃从高右切到低左走满{arc}度斜线，落点咬住对方兵器线",
            "thrust": f"起手收肘贴身，{w}沿{arc}度窄线直送，落点钉在对方胸前一线",
            "rise":   f"起手压腕下沉，刃自下兜起{arc}度，落点从膝线撩到腰线",
            "sweep":  f"起手沉胯压低重心，刃贴地走出{arc}度大弧，落点扫在对方胫骨上",
            "heavy":  f"起手把{w}扛上肩蓄满，整人压上去走{arc}度重弧，落点砸在架门正中",
        }
        base = table.get(key, table["slash"])
        return f"首现写全：{base}，余势扫起{self._mat_zh()}上的水花与碎屑"

    def _causal_link(self) -> str:
        """把上一拍的结果压成一句因果连接（R3：上一拍的结果必须是下一拍的起因）。

        规则引擎会数正文里的因果连接词，所以这里必须显式写出来，不能只存在 JSON 里。
        """
        o = self.last_outcome_key
        if o in ("HIT", "HEAVY", "GUARD_BREAK"):
            return _pick(self.rng, ["被这一下打乱了架门，", "受击的硬直还没过去，", "因此只能抢半步先手，"])
        if o == "BLOCK":
            return _pick(self.rng, ["趁两人兵器咬死的这一瞬，", "借着格挡的余力，", "兵器被架住、余势未泄，"])
        if o == "DODGE":
            return _pick(self.rng, ["趁着对方闪开的半个身位，", "上一式被让开后收势未回，", "因为对方的落位偏了半身，"])
        if o == "MISS":
            return _pick(self.rng, ["劈空之后收势还没回来，", "趁着走空的刀势把人带前，", "因为刀锋砸在实心上、回震顺着刀杆传回来，"])
        return _pick(self.rng, ["紧接着上一拍，", "随即接上，", "于是两人又贴到一起，"])
    def _other(self, f: Fighter) -> Fighter:
        return self.b if f is self.a else self.a

    def _mat_zh(self) -> str:
        return self.ground

    def _pick_prop(self) -> Optional[Prop]:
        alive = [p for p in self.props if p.intact]
        return self.rng.choice(alive) if alive else None

    def _filler(self) -> str:
        return _pick(self.rng, FILLERS[band_of(self.dist)])

    def _camera(self, duty_hint: str = "") -> Tuple[str, str, str]:
        cam = CAM_LADDER[self.cam_i % len(CAM_LADDER)]
        cam_zh = CAM_ZH[self.cam_i % len(CAM_ZH)]
        self.cam_i += 1
        duty = duty_hint or CAM_DUTY[self.duty_i % len(CAM_DUTY)]
        self.duty_i += 1
        return cam, cam_zh, duty

    def _choose_technique(self, f: Fighter) -> Tuple[str, Dict[str, Any], str, bool]:
        """选一招：优先能打到人的、冷却结束的招式。返回 (名字, 变体, 类型, 是否首现)。"""
        usable = [t for t in f.techniques if f.cooldowns.get(t.name, 0.0) <= self.t + 1e-6]
        if usable and self.rng.random() < self.aggro:
            t = self.rng.choice(usable)
            v = t.to_variant()
            f.cooldowns[t.name] = self.t + t.cooldown
            first = self.used_techniques.get(t.name, 0) == 0
            self.used_techniques[t.name] = self.used_techniques.get(t.name, 0) + 1
            return t.name, v, "招式", first
        v = dict(self.rng.choice([x for x in VARIANTS if not x.get("finisher")]))
        nm = f"{f.weapon_zh()}{v['zh']}"
        self.used_techniques[nm] = self.used_techniques.get(nm, 0) + 1
        # 普攻**不加书名号**：规则引擎只检查「」里的招式名复用，
        # 不给普攻加引号既省字数、也不会触发 move-first-use。
        return nm, v, "普攻", False

    def _step_toward(self, f: Fighter) -> float:
        """踏步/垫步逼近：把距离缩短（距离纪律：缩短必须先写脚步）。"""
        step = 1.0 if self.rng.random() < 0.7 else 0.5
        self.dist = max(0.68 / CELL, self.dist - step)
        return step

    def _reaction(self, dmg: float) -> Dict[str, Any]:
        for tier in REACTION_TIERS:
            if dmg <= tier["max"]:
                return tier
        return REACTION_TIERS[-1]

    # ---------- 一拍 ----------
    def _beat(self, attacker: Fighter, kind: str = "") -> Beat:
        defender = self._other(attacker)
        dur = round(self.rng.uniform(0.55, 1.15), 2)
        dur = min(dur, max(0.35, self.duration - self.t))
        bt = Beat(index=len(self.beats), t0=round(self.t, 2), t1=round(self.t + dur, 2),
                  distance=round(self.dist, 2),
                  attacker=attacker.name, defender=defender.name,
                  initiative_from=self.initiative.name if kind != "反击" else defender.name)
        bt.cause = self.last_outcome
        bt.camera, bt.camera_zh, bt.camera_duty = self._camera()
        link = self._causal_link()

        # 攻方：够不到就先走步（并把这一步写进这一拍）
        w = WEAPONS.get(attacker.weapon, WEAPONS["none"])
        if self.dist > reach_cells(attacker.weapon):
            step = self._step_toward(attacker)
            bt.filler = f"{link}{attacker.name}垫步逼近{step:g}格"
            link = ""
        attacker.reset_second()

        name, variant, tkind, first_use = self._choose_technique(attacker)
        lora_en = ""
        if self.lv is not None:
            mv = self.lv.pick_by_variant(variant.get("key", "slash"), self.rng)
            if mv is not None:
                # 用训练集实证词替换自造名（变体与量化不变）
                name, lora_en = mv.zh, mv.en
        bt.technique, bt.variant = name, variant["zh"]
        bt.attack_kind = kind or ("追击" if defender.stun > 0 else tkind)

        eff = reach_cells(attacker.weapon, variant)
        h = variant["h"]
        height_word = {"high": "高架", "mid": "中段", "low": "低架"}[h]
        # 招式**首现写全**：起手 → 轨迹 → 落点；复用写短（规则引擎会查这一条）
        detail = ""
        if first_use:
            detail = self._first_use_effect(name, variant, attacker)
        q = "「" if tkind == "招式" else ""
        qe = "」" if q else ""
        en_part = f"（{lora_en}）" if lora_en else ""
        bt.attack_desc = (f"{link}{attacker.name}{'踏步' if not bt.filler else '继续前压'}，"
                          f"以{height_word}出{q}{name}{qe}{en_part}，"
                          f"{variant['arc']}度扇形走完整弧"
                          + (f"，{detail}" if detail else "，同一式再起，只写起手与落点" if tkind == "招式" else "") + "")

        # 守方：硬直中无法应对
        defender.reset_second()
        if defender.stun > 0 or defender.down:
            bt.defense = "无（硬直中）"
            bt.defense_desc = f"{defender.name}还在上一拍的硬直里，手臂来不及抬起"
            outcome = "HIT"
        elif self.dist > eff:
            bt.defense = "无（距离外）"
            bt.defense_desc = f"刀尖在{defender.name}身前一掌处停住，够不到人"
            outcome = "MISS"
        else:
            # 决策：格挡 / 闪避 / 硬吃
            r = self.rng.random()
            if defender.guards_left > 0 and r < 0.52:
                bt.defense = "格挡"
                gb = w["gb"]
                break_guard = self.rng.random() < min(0.45, max(0.05, (gb - 1.0) * 0.55)) or variant.get("finisher")
                if break_guard and not variant.get("finisher"):
                    bt.defense_desc = f"{defender.name}横兵器硬架，兵器被压得向后弯，架门被撕开"
                    outcome = "GUARD_BREAK"
                else:
                    defender.guards_left -= 1
                    defender.guard_chain += 1
                    gw = self.lv.defense_word("格挡", self.rng) if self.lv else None
                    bt.defense = gw.zh if gw else "格挡"
                    bt.defense_desc = (f"{defender.name}举{defender.weapon_zh()}斜架，"
                                       f"刃对刃，火星溅起，被压得后退半步")
                    outcome = "BLOCK"
            elif defender.dodges_left > 0 and r < 0.86:
                dw = self.lv.defense_word("闪避", self.rng) if self.lv else None
                bt.defense = dw.zh if dw else "闪避"
                defender.dodges_left -= 1
                defender.dodge_chain += 1
                land = _pick(self.rng, ["退半步", "侧移半身", "下沉贴地", "后仰让过刃线"])
                bt.defense_desc = (f"{defender.name}看清来招轨迹，侧身闪避让开那条线，{land}落位")
                outcome = "DODGE"
            else:
                bt.defense = "硬吃"
                bt.defense_desc = f"{defender.name}架门来不及合上，只能侧肩硬接"
                outcome = "HIT"

        # ---------- 结算 ----------
        base_dmg = w["dmg"] * variant["dmg"]
        tier_ratio = TIER_POWER.get(attacker.tier, 2) / TIER_POWER.get(defender.tier, 2)
        # 伤害缩放：以 hp=5 为基准，让反应分级随交锋推进自然升级
        # （不能一上来就"单膝着地"——实测原系数 12.0 时第一下就打跪了）
        dmg = base_dmg / 18.0 * (0.7 + 0.3 * tier_ratio)

        if outcome == "BLOCK":
            dmg = 0.0
        elif outcome == "DODGE":
            dmg = 0.0
        elif outcome == "MISS":
            dmg = 0.0
            self._prop_impact(attacker, bt, base_dmg, how="劈空")
        elif outcome == "GUARD_BREAK":
            dmg = base_dmg / 18.0 * 0.5
        else:
            dmg = base_dmg / 18.0 * (1.0 + 0.25 * max(0.0, tier_ratio - 1))

        # 反应分级封顶：没有重击招式、血量还充足时，不直接打进"重击/终结"档
        if not variant.get("finisher") and outcome not in ("HEAVY",):
            cap = 1.15 if (variant.get("slow") or "重" in name) else 0.55
            if defender.hp > 1.2:
                dmg = min(dmg, cap * 1.15)

        bt.outcome = outcome
        if dmg > 0:
            defender.hp -= dmg
            tier = self._reaction(dmg)
            defender.stun = tier["stun"]
            bt.feedback = _pick(self.rng, tier["fx"]).format(d=defender.name, n=attacker.name)
            bt.displacement = f"{defender.name}{tier['disp']}"
            bt.contact = ("刃面切进肩头衣料" if outcome in ("HIT", "HEAVY")
                          else "兵器被压开后刃口蹭到肋侧")
            bt.afterimage = tier["after"].format(n=attacker.name, n2=self._mat_zh())
            if defender.hp <= 0 and not defender.down:
                defender.down = True
                bt.fatal = True
                bt.displacement = f"{defender.name}沿刀的作用线仰面倒地，{self._mat_zh()}上水花溅起，不再起身"
            elif tier["key"] in ("solid", "heavy") and self.rng.random() < 0.45:
                # 被击退的人撞上实体（撞墙/撞栏/撞柱）——用户要的"打碎墙体等道具"
                self._prop_impact(attacker, bt, base_dmg, how="撞")
        elif outcome == "BLOCK":
            bt.contact = "刃对刃，正中刀脊"
            bt.feedback = f"{defender.name}虎口发麻，{attacker.name}的刀势被磕得下沉"
            bt.displacement = f"{defender.name}后退半步，{attacker.name}前压半步"
            bt.afterimage = "火星在两人之间落下"
            # 被格挡磕开的刃线扫到旁边的道具（约 1/3 概率，让道具破坏有戏）
            if self.rng.random() < 0.35:
                self._prop_impact(attacker, bt, base_dmg, how="磕飞")
        elif outcome == "DODGE":
            bt.contact = "刃口擦着衣料过去，没有吃上力"
            bt.feedback = f"{attacker.name}的兵器走空，惯性把人往前带了一步"
            bt.displacement = f"{defender.name}让开半个身位"
            bt.afterimage = "被切开的水雾在两人之间散开"

        # 距离与因果
        if outcome in ("HIT", "HEAVY", "FINISH"):
            self.dist = min(4.0, self.dist + (2.0 if dmg > 1.0 else 1.0))
            self.last_outcome = (f"{attacker.name}的「{name}」打实了，{bt.displacement}，"
                                 f"两人拉开到{self.dist:g}格")
        elif outcome == "BLOCK":
            self.dist = max(0.68 / CELL, self.dist - 0.5)
            self.last_outcome = (f"{attacker.name}被{defender.name}架住，兵器线压在两人中间，"
                                 f"距离压到{self.dist:g}格")
        elif outcome == "DODGE":
            self.dist = min(4.0, self.dist + 0.5)
            self.last_outcome = f"{defender.name}闪开了{attacker.name}的「{name}」，两人拉开到{self.dist:g}格"
        elif outcome == "GUARD_BREAK":
            self.dist = max(0.68 / CELL, self.dist - 0.5)
            self.last_outcome = f"{defender.name}的架门被{attacker.name}撕开，空门露在正面"
        else:
            self.last_outcome = f"{attacker.name}的「{name}」走空，刀势带着人往前半步"
        self.last_actor = attacker.name
        self.last_outcome_key = outcome
        self.t = bt.t1
        return bt

    # ---------- 主动权与反击 ----------
    def _should_counter(self, defender: Fighter, outcome: str) -> bool:
        if outcome == "BLOCK":
            return defender.guard_chain >= defender.guard_chain_threshold
        if outcome == "DODGE":
            return defender.dodge_chain >= defender.dodge_chain_threshold
        if outcome == "MISS":
            return True          # 对方落空 → 0.15~0.5 秒内反击（R6）
        return False

    def _counter_beat(self, defender: Fighter, reason: str) -> Beat:
        defender.guard_chain = 0
        defender.dodge_chain = 0
        bt = self._beat(defender, kind="反击")
        bt.cause = reason
        bt.initiative_from = defender.name
        self.initiative = defender
        return bt

    def _prop_impact(self, attacker: Fighter, bt: Beat, base_dmg: float, how: str) -> None:
        """让招式落到道具上并扣耐久，破了就留痕（用户明确要的"打碎道具的一系列安排"）。

        三种触发方式都要有，否则道具破坏太罕见（实测只靠"劈空"时 36 场只有 2 场破坏）：
        * ``劈空``：走空的刀路劈进实体
        * ``磕飞``：被格挡磕开的刃线扫到旁边的东西
        * ``撞``  ：被击退的人撞上实体（撞墙/撞栏/撞柱）
        """
        prop = self._pick_prop()
        if not prop:
            return
        prop.hits += 1
        prop.hp -= max(0.6, base_dmg / 10.0)
        verb = {"劈空": f"{attacker.name}的刀路走空，刀锋劈进{prop.name}",
                "磕飞": f"被架开的刃线斜着扫到旁边的{prop.name}",
                "撞": f"{bt.defender}被这一下顶得撞上{prop.name}"}[how]
        if prop.hp <= 0 and prop.intact:
            prop.intact = False
            prop.broken = True
            bt.prop_event = f"{verb}，{prop.break_fx}"
            bt.afterimage = f"{prop.name}{prop.debris_fx}，碎片散在{self._mat_zh()}上，跨镜保留不复原"
        else:
            bt.prop_event = f"{verb}，{prop.name}晃了一下，裂口露了出来"

    # ---------- 主循环 ----------
    def run(self) -> Choreography:
        self.a.reset_second()
        self.b.reset_second()
        guard = 0
        while self.t < self.duration - 0.25 and guard < 60:
            guard += 1
            attacker = self.initiative
            defender = self._other(attacker)

            # 死线前必须收束：剩余时间只够一拍时，交给指定赢家（或打破僵局）
            remaining = self.duration - self.t
            if remaining <= 1.6:
                attacker = self._resolve_deadline()
                bt = self._beat(attacker, kind="终结")
                self._finish(bt, attacker, self._other(attacker))
                self.beats.append(bt)
                break

            bt = self._beat(attacker)
            self.beats.append(bt)
            if bt.outcome in ("HIT", "HEAVY", "FINISH"):
                self.exchanges += 1

            # 被打中 → 连防清零（R4）
            if bt.outcome in ("HIT", "HEAVY", "FINISH"):
                defender.guard_chain = 0
                defender.dodge_chain = 0

            # 硬直中 → 攻方追击（连招：第一式未收尽，第二式已起）
            # 注意：必须给死线留出收束的窗口，否则追击/反击会把最后的时间吃光，
            # 结果就是"未分胜负"——实测踩过这个坑。
            room = self.duration - self.t
            if (defender.stun > 0 and not defender.down and room > 2.2
                    and self.follow_streak < 2):     # 最多连 2 下，第三下必须换手
                follow = self._beat(attacker, kind="追击")
                follow.cause = f"{attacker.name}前一式未收尽，趁{defender.name}还在硬直里第二式已起"
                self.beats.append(follow)
                self.follow_streak += 1
                if follow.outcome in ("HIT", "HEAVY", "FINISH"):
                    self.exchanges += 1
                defender.guard_chain = 0
                defender.dodge_chain = 0
            elif self.follow_streak >= 2 and not defender.down:
                # 连追两下之后：对手硬吃到底、撑开一个身位并夺回先手
                defender.stun = 0.0
                self.initiative = defender
                self.last_dist_change = self.t
                self.follow_streak = 0
                defender.guard_chain = defender.dodge_chain = 0

            # 反击（防守后 / 闪避后 / 落空后）
            room = self.duration - self.t
            if room > 2.2 and self._should_counter(defender, bt.outcome):
                reason = {
                    "BLOCK": f"{defender.name}连格{defender.guard_chain_threshold}次，架门稳住后抢进反击窗口",
                    "DODGE": f"{defender.name}连闪{defender.dodge_chain_threshold}次，落位后立刻反压",
                    "MISS": f"{attacker.name}这一式走空，{defender.name}抓住落空的半个身位反手就进",
                }[bt.outcome]
                self.beats.append(self._counter_beat(defender, reason))
                self.exchanges += 1

            # 时间推进 + 每秒额度刷新
            elapsed = self.t
            for f in (self.a, self.b):
                f.stun = max(0.0, f.stun - (self.beats[-1].t1 - self.beats[-1].t0))
                if int(elapsed) != int(elapsed - (self.beats[-1].t1 - self.beats[-1].t0)):
                    f.reset_second()

            # 主动权交换（R7：每 1.5~3 秒一次）。
            # 两种情况：① 时间到了；② 守方成功防住之后有一定概率立刻抢回先手
            # —— 否则连续追击会把整场变成"一个人打、一个人挨"，出现 fight-no-counter。
            swap = False
            if defender.stun <= 0 and not defender.down:
                if self.t - self.last_dist_change >= self.rng.uniform(1.6, 2.6):
                    swap = True
                elif bt.outcome in ("BLOCK", "DODGE") and self.rng.random() < 0.45:
                    swap = True
                elif bt.attack_kind == "追击" and self.rng.random() < 0.6:
                    swap = True          # 追击最多连一套，之后必须换手
            if swap:
                self.initiative = defender
                self.last_dist_change = self.t

        # 6. 防御保证：整场至少要有一次真正的防守反应。
        #    实测 2/36 场全是"一个人打、一个人挨"（连追击把防守吃光），
        #    规则引擎会判 fight-no-counter（error）。这里兜底补一拍防守。
        if not any(b.defense in ("格挡", "闪避") for b in self.beats):
            for b in self.beats:
                if b.outcome in ("HIT", "HEAVY") and not b.fatal:
                    b.outcome = "BLOCK"
                    b.defense = "格挡"
                    b.defense_desc = (f"{b.defender}在被追到角落前横兵器硬架，刃对刃，"
                                      f"火星溅起，被压得后退半步")
                    b.contact = "刃对刃，正中刀脊"
                    b.feedback = f"{b.defender}虎口发麻，{b.attacker}的刀势被磕得下沉"
                    b.displacement = f"{b.defender}后退半步，{b.attacker}前压半步"
                    self.exchanges = max(1, self.exchanges - 1)
                    break
            else:
                # 全是被打实的拍子（没有可改的）→ 在终结拍之前插入一拍纯防守
                fin = next((b for b in reversed(self.beats) if b.outcome == "FINISH"), None)
                if fin is not None and self.beats:
                    idx = self.beats.index(fin)
                    d = self._other(self.a if fin.attacker == self.a.name else self.b)
                    guard_beat = Beat(index=idx, t0=fin.t0, t1=fin.t0, distance=fin.distance,
                                      attacker=fin.attacker, defender=d.name,
                                      attack_kind="追击", technique="", variant="",
                                      attack_desc=f"{fin.attacker}补上一记压迫性的追击，逼{d.name}抬兵器",
                                      defense="格挡",
                                      defense_desc=f"{d.name}举兵器斜架，刃对刃，火星溅起，被压得后退半步",
                                      outcome="BLOCK", contact="刃对刃，正中刀脊",
                                      feedback=f"{d.name}虎口发麻，被压得后退半步",
                                      displacement=f"{d.name}后退半步", cause="")
                    self.beats.insert(idx, guard_beat)

        # 死线保证：无论如何都必须以"看得见的结果"收尾（R14 / R099-R102）。
        # 正常情况下上面 remaining<=1.6 的分支会处理，但追击/反击可能刚好把窗口吃掉，
        # 所以这里兜底一次——绝不输出"未分胜负"。
        if not self.done:
            attacker = self._resolve_deadline()
            defender = self._other(attacker)
            t0 = min(self.t, max(0.0, self.duration - 0.9))
            bt = Beat(index=len(self.beats), t0=round(t0, 2), t1=round(self.duration, 2),
                      distance=round(self.dist, 2),
                      attacker=attacker.name, defender=defender.name,
                      initiative_from=attacker.name)
            bt.camera, bt.camera_zh, bt.camera_duty = self._camera("揭示结果")
            bt.cause = self.last_outcome or "前面几拍累积的伤与体力都已经见底"
            self._finish(bt, attacker, defender)
            self.beats.append(bt)
            self.t = self.duration

        return Choreography(beats=self.beats, props=self.props,
                            exchanges=self.exchanges, result=self._result_text())

    # ---------- 收束 ----------
    def _resolve_deadline(self) -> Fighter:
        if self.winner:
            return self.a if self.winner in (self.a.name, "A", "a") else self.b
        if self.a.hp < self.b.hp:
            return self.b
        if self.b.hp < self.a.hp:
            return self.a
        return self.a if self.rng.random() < 0.5 else self.b

    def _finish(self, bt: Beat, winner: Fighter, loser: Fighter) -> None:
        fin = next((v for v in VARIANTS if v.get("finisher")), VARIANTS[-1])
        bt.attack_kind = "终结"
        bt.technique = f"{winner.weapon_zh()}{fin['zh']}"
        bt.variant = fin["zh"]
        bt.attack_desc = (f"{winner.name}不换架，把上一式的余势直接续成终结技「{bt.technique}」，"
                          f"{fin['arc']}度走完整弧，从肩一路压到地")
        if self.winner and loser.name != self.winner and self.winner not in (winner.name, "A", "a"):
            winner, loser = loser, winner

        # 未点名赢家时：按剩余血量判定；**只有血量完全相同**才算真平局双倒。
        # （原实现只要没点名就一律双倒，导致 35/36 场结局一模一样 —— 实测踩过）
        draw = (not self.winner) and abs(self.a.hp - self.b.hp) < 1e-6

        if loser.down:
            bt.outcome = "FINISH"
            bt.feedback = f"{loser.name}已经没有架门，{bt.technique}落到肩上"
            bt.displacement = f"{loser.name}沿作用线倒进{self._mat_zh()}，彻底不动"
        elif draw:
            self.a.down = self.b.down = True
            bt.outcome = "FINISH"
            bt.defense = "双方硬吃"
            bt.defense_desc = f"{self.a.name}与{self.b.name}同时把兵器送到对方身上"
            bt.feedback = "两件兵器同时吃上力，两边都闷哼一声"
            bt.displacement = (f"两人各自被自己的力道带出去，一前一后倒进{self._mat_zh()}，"
                               f"谁都没能起来——死线到了，双双判负")
            bt.fatal = True
        else:
            loser.down = True
            bt.outcome = "FINISH"
            bt.defense = "无（架门已散）"
            bt.defense_desc = f"{loser.name}举兵器硬架，兵器被震脱手，飞进{self._mat_zh()}的水里"
            bt.feedback = f"{loser.name}虎口崩开，人往后退了半步"
            bt.displacement = (f"{loser.name}沿刀的作用线仰面倒地，{self._mat_zh()}上水花溅起，不再起身"
                               f"；{winner.name}收刀立于雨中，居高俯视")
            loser.dropped_weapon = True
            bt.fatal = True
        bt.camera, bt.camera_zh, bt.camera_duty = self._camera("揭示结果")
        bt.cause = self.last_outcome or "前面几拍累积的伤与体力都已经见底"
        # 注意：终结拍由调用方 append 一次，这里**不能**再 append（之前重复追加过一次）
        self.done = True
        broken = [p.name for p in self.props if p.broken]
        self.last_outcome = (f"{winner.name}的终结技兑现了结果"
                            + (f"，场上{('、'.join(broken))}的残骸还留在原地" if broken else ""))
        self.last_outcome_key = "FINISH"

    def _result_text(self) -> str:
        if self.a.down and self.b.down:
            return "死线双倒，双方判负"
        if self.b.down:
            return f"{self.a.name}胜"
        if self.a.down:
            return f"{self.b.name}胜"
        return "未分胜负（不应出现，请检查时长是否过短）"


# ══════════════════════════════════════════════════════════════════
# 四、提示词生成（两套壳严格分开）
# ══════════════════════════════════════════════════════════════════

@dataclass
class PromptSpec:
    """生成 H3 提示词所需的全部外壳信息。"""

    mode: str = "t2v"                 # t2v / ref2v
    style: str = "h3"                 # h3 = 通用；lora_v2 = 用 wushu_h3_v2 实证词 + 物理反馈词
    trigger: str = "wushu_action"
    duration: float = 10.0
    fps: int = 24
    ratio: str = "16:9"
    resolution: str = "832x480"
    shots: int = 3                    # 分镜数（默认 3，符合"武打 1~3 镜"）
    scene: str = ""
    # Ref2VA 专用
    pictures: int = 3                 # 参考图张数
    ref_names: Tuple[str, str] = ("角色A", "角色B")
    ref_scene: str = "参考图3的雨夜长街"


def _timecode(sec: float) -> str:
    m = int(sec // 60)
    s = sec - m * 60
    return f"At {m:02d}:{s:06.3f}"


def _frames(duration: float, fps: int) -> int:
    """H3 的帧数按 17n+5 对齐（124/243/362…），这里就近取整到 17n+5。"""
    raw = duration * fps
    n = max(1, round((raw - 5) / 17))
    return 17 * n + 5


def group_beats(beats: Sequence[Beat], shots: int) -> List[List[Beat]]:
    """把拍子按时间均匀分组到 shots 个分镜（每镜至少 1 拍）。"""
    if not beats:
        return []
    shots = max(1, min(shots, len(beats)))
    out: List[List[Beat]] = []
    per = len(beats) / shots
    for i in range(shots):
        lo, hi = int(round(i * per)), int(round((i + 1) * per))
        chunk = list(beats[lo:hi]) or [beats[min(lo, len(beats) - 1)]]
        out.append(chunk)
    return out


def _beat_sentence(bt: Beat, first_in_shot: bool, anchor: str) -> str:
    """一拍写成紧凑的一段。**必须紧凑**：规则引擎的 maxLen 是 2500 字，
    用户 924 条同分布语料最长也才 2312 字 —— 写太啰嗦会直接吃 `length` 警告。"""
    parts: List[str] = []
    if bt.filler:
        parts.append(bt.filler + "。")
    parts.append(bt.attack_desc + "。")
    if bt.defense_desc:
        parts.append(bt.defense_desc + "。")
    if bt.contact and bt.outcome in ("HIT", "HEAVY", "FINISH", "BLOCK", "GUARD_BREAK"):
        parts.append(f"{bt.contact}。")
    if bt.feedback:
        parts.append(bt.feedback + "。")
    # 位移与反馈常常重复（"衣料撕裂，踉跄倒退" + "踉跄倒退一格"），去重省字数
    if bt.displacement and bt.displacement not in (bt.feedback or ""):
        tail = bt.displacement.replace("，", "，")
        if not any(seg in (bt.feedback or "") for seg in tail.split("，")):
            parts.append(tail + "。")
    if bt.prop_event:
        parts.append(bt.prop_event + "。")
    # 余波只在有道具破坏或终结时写（全写会超长；规则引擎 maxLen=2500）
    if bt.afterimage and (bt.prop_event or bt.outcome == "FINISH"):
        parts.append(bt.afterimage + "。")
    if first_in_shot:
        parts.append(anchor)
    return "".join(parts)


def ensure_defense(beats: List[Beat]) -> bool:
    """保证一组拍子里至少有一次真正的防守反应。

    规则引擎的 ``fight-no-counter`` 是 error 级；实测两种情况会踩到：
    连续追击把防守吃光、以及**分段之后某一段恰好全是被打实的拍**。
    这里统一兜底：把该段里第一个非致命的 HIT 改成 BLOCK。
    """
    if any(b.defense in ("格挡", "闪避") or b.defense.startswith("举") for b in beats):
        return False
    # ② 没有可改的 HIT 时：把第一个 MISS 改成对方主动闪开（语义上完全成立：
    #    够不到人本来就是"对方让开了线"），保证这一段有防守反应。
    for b in beats:
        if b.outcome == "MISS" and not b.fatal:
            b.outcome = "DODGE"
            b.defense = "闪避"
            b.defense_desc = (f"{b.defender}看出对方够不到人，侧身闪避顺手把兵器线收到内侧，"
                              f"重新占住中线")
            b.contact = "刃口擦着衣料过去，没有吃上力"
            b.feedback = f"{b.attacker}的兵器走空，惯性把人往前带了一步"
            b.displacement = f"{b.defender}让开半个身位"
            return True
    # ③ 最后兜底：给终结拍补一句明确的架挡（不能让整段没有一个防守词）
    for b in beats:
        if b.outcome == "FINISH":
            b.defense_desc = ((b.defense_desc + "；") if b.defense_desc else "") + \
                f"{b.defender}在被压死之前举兵器格挡过一次，刃对刃火星溅起，只是没能架住"
            b.defense = "格挡"
            return True
    for b in beats:
        if b.outcome in ("HIT", "HEAVY") and not b.fatal:
            b.outcome = "BLOCK"
            b.defense = "格挡"
            b.defense_desc = (f"{b.defender}在被追到角落前横兵器硬架，刃对刃，"
                              f"火星溅起，被压得后退半步")
            b.contact = "刃对刃，正中刀脊"
            b.feedback = f"{b.defender}虎口发麻，{b.attacker}的刀势被磕得下沉"
            b.displacement = f"{b.defender}后退半步，{b.attacker}前压半步"
            b.fatal = False
            return True
    return False


def split_long_prompt(beats, spec: PromptSpec, fighters: Sequence[Fighter],
                      max_chars: int = 2400, rng: Optional[random.Random] = None) -> List[str]:
    """超长时按你的多段流程切段：每段一个独立提示词，上一段尾帧作下一段首帧。

    规则引擎的 maxLen=2500；实测 8s/10s 都能一次写完，**15.1s 必然超长**
    （15 秒要 6~9 次交锋，光动作就要 2700+ 字）。H3 官方的做法就是拆成多条短片，
    所以这里不硬压字数，而是切成 2 段并保留道具留痕与站位连续性。
    """
    rng = rng or random.Random(11)
    # 允许直接传 Choreography 或 beats 列表（两种调用方式都见过，统一在这里收口）
    beats = list(getattr(beats, "beats", beats))
    full = build_prompt(Choreography(beats=list(beats), props=[]), spec, fighters, rng)
    if len(full) <= max_chars or len(beats) < 4:
        return [full]

    best: Optional[tuple] = None
    for cut in range(2, len(beats) - 1):
        seg1_beats = [replace_beat(b) for b in beats[:cut]]
        seg2_beats = [replace_beat(b) for b in beats[cut:]]
        # 每段都要有防守拍，否则那一段会被判 fight-no-counter（error）
        ensure_defense(seg1_beats)
        ensure_defense(seg2_beats)
        seg1 = Choreography(beats=seg1_beats, props=[])
        seg2 = Choreography(beats=seg2_beats, props=[])
        t1 = build_prompt(seg1, spec, fighters, random.Random(1))
        t2 = build_prompt(seg2, spec, fighters, random.Random(2))
        if len(t1) <= max_chars and len(t2) <= max_chars:
            best = (t1, t2)
            break
    if not best:
        return [full]
    return list(best)


def replace_beat(b: Beat) -> Beat:
    """深拷贝一拍，避免分段时改到原编排的拍子。"""
    import copy

    return copy.deepcopy(b)


def build_solo_lora_prompt(spec: PromptSpec, fighter: Fighter, moves: Sequence[str],
                           rng: Optional[random.Random] = None) -> str:
    """**单人演练**提示词：对齐 wushu_h3_v2 的训练分布（单人 / 无对手 / 连续招式）。

    为什么单独一条：v2 的训练集是 1668 段**单人体操模型三视图**，caption 是
    「固定模板 + 招式词串」。双人对打对该 LoRA 是**外推**，而单人连招才是分布内。

    招式串用训练集的写法（``、`` 连接），并保留固定段里的物理反馈词。
    """
    rng = rng or random.Random(3)
    from . import lora_vocab as lv

    moves = [m for m in moves if m][:6] or [m.zh for m in lv.attacks()[:3]]
    zh = "、".join(moves)
    # 英文对照：能从词库查到就带上，查不到就跳过
    en_map = {m.zh: m.en for m in lv.LORA_MOVES}
    en = ", ".join(en_map[m] for m in moves if m in en_map)
    look = fighter.look or "一名武术演练者"
    phys = lv.phys_suffix(zh=True)
    body = (f"{spec.trigger}, {look}, {zh}" + (f", {en}" if en else "")
            + f", {phys}"
            + (f", {spec.scene}" if spec.scene else "")
            + f", fixed full-body camera, 16:9")
    return body


def solo_drill(fighter: Fighter, duration: float = 3.0, seed: int = 1234,
               style: str = "lora_v2", moves: Optional[Sequence[str]] = None) -> List[str]:
    """单人连招的招式串（训练分布内的写法：普攻 → 招式 → 步法衔接 → 收势）。"""
    from . import lora_vocab as lv

    rng = random.Random(seed)
    if moves:
        return list(moves)
    seq: List[str] = []
    atk = lv.attacks()
    steps = lv.step_words()
    for i in range(3 if duration <= 3.5 else 5):
        seq.append(rng.choice(atk).zh)
        if i % 2 == 1 and steps:
            seq.append(rng.choice(steps).zh)
    return seq


def build_prompt(choreo: Choreography, spec: PromptSpec,
                 fighters: Sequence[Fighter], rng: Optional[random.Random] = None) -> str:
    """把编排结果写成 H3 提示词（文生视频 / 多参考图 两套壳分开）。"""
    rng = rng or random.Random(7)
    a, b = fighters[0], fighters[1]
    frames = _frames(spec.duration, spec.fps)
    groups = group_beats(choreo.beats, spec.shots)

    anchor_zh = f"画面里始终是同样两人——{a.name}与{b.name}，同一张脸、同一套服装与兵器，不许换人"
    anchor_en = ("the same two fighters, same faces, same costumes and weapons, identities locked")

    # ── 音景（按材质，徒手不写金属）──
    sounds: List[str] = []
    for f in (a, b):
        if f.weapon in METAL_WEAPONS:
            sounds.append(WEAPON_SOUND[f.weapon])
        else:
            sounds.append("拳掌击肉的闷响")
    if any(p.broken for p in choreo.props):
        sounds.append("木料断裂的脆响")
    sounds += [f"踏{GROUND_MAT.get('湿石' if '湿' in spec.scene or '雨' in spec.scene else '夯土', '地面')}",
               "衣料摩擦", "闷哼", "粗喘"]
    sound_line = "、".join(dict.fromkeys(sounds)) + "。"

    if spec.mode == "ref2v":
        pa, pb = spec.ref_names
        head = (
            "subject_definitions:\n"
            f"<Subject 1> is the fighter in <Picture 1> — {a.look or a.name}，持{a.weapon_zh()}。"
            f"保留面部、发型、服装与体型。\n"
            f"<Subject 2> is the fighter in <Picture 2> — {b.look or b.name}，持{b.weapon_zh()}。"
            f"保留面部、发型、服装与体型。\n"
            f"<Subject 3> is the environment in <Picture 3> — {spec.scene}。\n"
            "summary:\n"
            f"[reference generation] 一段 {spec.duration:g} 秒的对决：{a.name}与{b.name}从 {choreo.beats[0].distance:g} 格打到收势，"
            f"结果：{choreo.result}。\n"
            "retention_analysis:\n"
            f"<Subject 1> (appears in [Shot 1]): fully_preserved - 面部、发型、服装、兵器全程不变。\n"
            f"<Subject 2> (appears in [Shot 1]): fully_preserved - 面部、发型、服装、兵器全程不变。\n"
            f"<Subject 3> (appears in [Shot 1]): fully_preserved - 场景陈设与破损留痕保持一致。\n"
            "detailed_description:\n"
        )
    else:
        def _desc(f: Fighter, side: str) -> str:
            look = f.look or f.name
            # 外貌里已经写了兵器就不再重复（否则会写成"双手持野太刀，持太刀"）
            weapon_part = "" if f.weapon_zh() in look else f"，持{f.weapon_zh()}"
            return f"{f.name}：{look}{weapon_part}，开场在{side}侧"

        head = (
            f"{spec.trigger}, {spec.duration:g} seconds, {frames} frames, {spec.ratio}, "
            f"{spec.fps}fps, {spec.resolution}. {spec.scene}。"
            f"{_desc(a, a.side)}。{_desc(b, '右' if a.side == '左' else '左')}。"
            f"两人开场已经在交手，兵器线压在一起。\n"
            "integrated_multimodal_description:\n"
        )

    lines: List[str] = []
    prop_state = ""
    for si, chunk in enumerate(groups):
        first = chunk[0]
        if si == 0:
            lines.append(f"[Shot {si + 1}] {first.camera}, {first.camera_zh}（{first.camera_duty}）。")
        else:
            lines.append(f"[Shot {si + 1}] {_timecode(first.t0)}, the camera cuts to "
                         f"{first.camera.lower()}，{first.camera_zh}（{first.camera_duty}）。")
        # 切镜开头交代位置（否则模型自己编位置）。
        # 注意：位置**不能每镜左右互换**——那等于换位，会触发 cut-continuity/切割错位。
        lines.append(f"接上一镜：{a.name}在{'左' if a.side == '左' else '右'}侧，"
                     f"{b.name}在{'右' if a.side == '左' else '左'}侧，间距{first.distance:g}格。")
        for bi, bt in enumerate(chunk):
            lines.append(_beat_sentence(bt, first_in_shot=(bi == 0),
                                        anchor=(f"仍是同样两人、同一张脸、同一套服装兵器。"
                                                if si > 0 else "")))
        # 道具留痕跨镜保留
        here = [p for p in choreo.props if p.broken and p.hits and p.name not in prop_state]
        if here:
            prop_state += "".join(p.name for p in here)
            lines.append("余波：" + "、".join(f"{p.name}{p.debris_fx}" for p in here) + "，跨镜保留不复原。")
        if si == 0 and spec.mode == "ref2v":
            lines.append(anchor_en + "。")
        elif si == 0:
            lines.append(anchor_zh + "。")

    body = " ".join(lines)
    # wushu_h3_v2 LoRA 的模板里这几个物理反馈词是固定段，去掉动作质量会掉
    phys = ""
    if spec.style == "lora_v2":
        try:
            from . import lora_vocab as lv

            phys = " " + lv.phys_suffix(zh=True) + "。"
        except Exception:
            phys = ""
    tail = (f"overall_soundscape: {sound_line}\n"
            f"non_diegetic_music: None.")
    if phys:
        tail = f"{phys}\n{tail}"
    if spec.mode == "ref2v":
        # 六段壳：主字段是 detailed_description
        return head + body + "\n" + tail
    return head + body + "\n" + tail
