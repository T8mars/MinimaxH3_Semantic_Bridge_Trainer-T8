"""武打语义词表（中英双语）。

这份词表是「语义逻辑翻译」的判据来源：哪些词代表力线、格距、防守反应、
打击反馈、招式、运镜；哪些词是逻辑漏洞（瞬移、无借力腾空、站桩对望）；
哪些词会让 H3 读不懂或稀释动作（规则说明、空泛形容、慢动作、特效剑气）。

**为什么必须双语**：用户的 924 条同分布语料
（``dataset\\h3_train\\metadata.csv``）是**英文为主**的，英文词表就是从那批
语料里统计出来的高频领域词（weight / stance / hips / momentum / gravity /
footwork / parry / thrust / whoosh / rustle …）。桥要学到的是"往训练分布
的那套写法靠"，所以判据必须覆盖那套写法。

中文侧的检查词表以 ``lint/h3lint.py``（h3lint.js 的 Python 移植）为权威版本，
本文件只补充**降级/改写**用的正向槽位词表，避免两套常量互相打架。
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence

CJK_RE = re.compile(r"[\u3400-\u9fff]")


def is_chinese(text: str, threshold: float = 0.02) -> bool:
    """粗略判断一段文本以中文为主。"""
    if not text:
        return False
    cjk = len(CJK_RE.findall(text))
    return (cjk / max(1, len(text))) >= threshold


# ── 力线 / 发力链 ──────────────────────────────────────────────────────
FORCE_CHAIN_ZH: List[str] = [
    "蹬地", "蹬后脚", "后脚蹬", "转腰", "拧腰", "腰胯", "沉胯", "坐胯",
    "力从地起", "重心前压", "重心下沉", "沉肩", "坠肘", "送肩", "送胯",
    "顺步", "跟步", "上步", "撤步", "落步", "半步", "借力", "卸力",
    "惯性", "余势", "收势", "回劲",
]
FORCE_CHAIN_EN: List[str] = [
    "weight", "center of gravity", "centre of gravity", "gravity", "hip", "hips",
    "waist", "rear leg", "back leg", "front leg", "back foot", "front foot",
    "drives from the ground", "drives off", "momentum", "explosive",
    "transfer of weight", "transfers weight", "shifts weight", "shifting weight",
    "pivots on the heel", "rotation", "rotational", "stored power", "winding up",
    "wind-up", "wind up", "ground reaction", "planted foot",
]

# ── 空间 / 格距 ────────────────────────────────────────────────────────
DISTANCE_ZH: List[str] = [
    "间距", "格距", "两格", "一格", "三格", "半格", "距离", "贴身", "近身",
    "拉近", "拉开", "退出", "退开", "逼近", "卡位", "站位", "身位",
    "左前方", "右后方", "斜侧", "正面", "背后", "高低位",
]
DISTANCE_EN: List[str] = [
    "distance", "spacing", "closes the distance", "close range", "mid-range",
    "out of range", "within range", "gap", "steps back", "step back",
    "steps forward", "advances", "closes in", "retreats", "backpedals",
    "off-balance distance", "stance width", "left side", "right side",
    "upper left", "lower right", "diagonal", "opposite side",
]

# ── 防守反应 ───────────────────────────────────────────────────────────
DEFENSE_ZH: List[str] = [
    "格挡", "格开", "架住", "封挡", "横挡", "斜挡", "举刀", "举剑", "抬臂",
    "闪避", "侧闪", "后仰", "低头", "侧身", "下潜", "翻滚", "侧滚", "前滚",
    "漏防", "硬吃", "挡不住", "来不及", "卸力", "滑步", "换步",
]
DEFENSE_EN: List[str] = [
    "guard", "guards", "blocks", "block", "blocking", "parry", "parries",
    "parried", "deflect", "deflects", "deflection", "evade", "evades",
    "evasion", "dodge", "dodges", "sidestep", "side-step", "rolls", "roll",
    "rolls low", "sway-back", "leans back", "ducks", "ducks under",
    "slips the strike", "catches the strike", "intercepts", "braces",
    "takes the hit", "fails to guard", "guard is broken", "guard opens",
]

# ── 接触点 / 命中判定 ──────────────────────────────────────────────────
CONTACT_ZH: List[str] = [
    "刃对刃", "刀脊", "刀背", "刀刃", "剑锋", "枪尖", "棍梢", "拳面", "肘尖",
    "接触点", "命中", "劈中", "刺中", "扫中", "磕中", "击中", "未命中",
    "擦过", "蹭过", "贴着", "掠过", "劈空", "落空", "打空", "刃面朝向",
]
CONTACT_EN: List[str] = [
    "impact", "impacts", "contact point", "connects", "clashes", "clang",
    "edge-on-edge", "blade meets blade", "blade bites", "lands cleanly",
    "grazes", "glancing", "skim", "misses", "whiffs", "cuts through empty air",
    "stops short", "deflected off", "hits the shoulder", "strikes the leg",
]

# ── 打击反馈 / 物理后果 ────────────────────────────────────────────────
FEEDBACK_ZH: List[str] = [
    "火星", "溅起", "碎屑", "闷响", "脆响", "颤", "震得", "发麻", "虎口",
    "踉跄", "趔趄", "倒退", "后仰", "单膝", "跪地", "半跪", "扶墙", "撑地",
    "衣料撕裂", "衣破", "血线", "渗血", "汗珠", "喘息", "粗喘", "闷哼",
    "脚下滑", "打滑", "重心不稳", "失去平衡", "倒地", "仰面", "侧倒",
]
FEEDBACK_EN: List[str] = [
    "sparks", "thud", "clang", "shudder", "jarred", "numb", "stagger",
    "staggers", "stumbles", "recoil", "recoils", "slides", "skids", "friction",
    "dust", "snow sprays", "cloth tears", "fabric rips", "blood", "sweat",
    "exhale", "sharp exhale", "grunt", "breath", "kneels", "drops to one knee",
    "loses balance", "off-balance", "falls", "collapses", "sprawls",
    "does not get up", "can't get up", "pinned",
]

# ── 招式名 ─────────────────────────────────────────────────────────────
MOVE_NAMES_ZH: List[str] = [
    "过肩劈", "斜劈", "正劈", "横扫", "突刺", "直刺", "撩刀", "挂刀", "点刺",
    "缠头裹脑", "劈剑", "撩剑", "点剑", "扎枪", "拦枪", "回马枪",
    "扫堂腿", "鞭腿", "侧踹", "直拳", "崩拳", "摆拳", "勾拳",
    "过肩摔", "别腿", "抱摔", "封喉手", "撩阴腿",
]
MOVE_NAMES_EN: List[str] = [
    "overhead chop", "overhead strike", "vertical chop", "diagonal slash",
    "horizontal sweep", "side sweep", "thrust", "reverse thrust", "backhand",
    "reverse-backhand", "upward flick", "liao", "spinning strike", "spinning back",
    "roundhouse", "side kick", "thrust kick", "palm strike", "straight punch",
    "hook punch", "uppercut", "cleave", "downward cut", "rising cut",
    "shoulder throw", "hip throw", "sweeping kick", "low sweep",
]

# ── 身法 ───────────────────────────────────────────────────────────────
FOOTWORK_ZH: List[str] = [
    "踏步", "上步", "撤步", "绕步", "侧移", "滑步", "换步", "错步",
    "垫步", "转身", "回身", "拧身", "沉身",
]
FOOTWORK_EN: List[str] = [
    "shuffle step", "slide step", "sliding step", "pivot", "pivots", "sidestep",
    "side-step", "crossover step", "crossover", "advance step", "back step",
    "circles", "circling", "turns", "turning", "spins", "spinning",
    "lands", "landing", "steps in", "steps out",
]

# ── 运镜（英文机位 + 中文说明）─────────────────────────────────────────
CAMERA_ZH: List[str] = [
    "跟拍", "中景", "全景", "低机位", "过肩", "侧向平移", "推近", "拉远",
    "短弧侧绕", "斜侧推进", "刃面特写", "手持",
]
CAMERA_EN: List[str] = [
    "wide shot", "medium shot", "close-up", "low-angle", "high-angle",
    "overhead shot", "handheld", "tracking", "dolly", "push in", "pull back",
    "over-the-shoulder", "pan", "tilt", "crane", "camera", "frame",
]

# ── 音景 ───────────────────────────────────────────────────────────────
SOUNDSCAPE_ZH: List[str] = [
    "踏湿石", "踏石", "破空", "刀风", "兵刃相交", "火星", "翻滚刮地",
    "衣裂", "闷哼", "雨", "粗喘", "喘息", "夜风", "脚步声",
]
SOUNDSCAPE_EN: List[str] = [
    "whoosh", "clang", "clangs", "thud", "footsteps", "rustling", "rustle",
    "growl", "exhale", "breath", "wind", "scrape", "crack", "crunch",
]

# ── 合并导出（中英一起用）──────────────────────────────────────────────
FORCE_CHAIN: List[str] = FORCE_CHAIN_ZH + FORCE_CHAIN_EN
DISTANCE: List[str] = DISTANCE_ZH + DISTANCE_EN
DEFENSE: List[str] = DEFENSE_ZH + DEFENSE_EN
CONTACT: List[str] = CONTACT_ZH + CONTACT_EN
FEEDBACK: List[str] = FEEDBACK_ZH + FEEDBACK_EN
MOVE_NAMES: List[str] = MOVE_NAMES_ZH + MOVE_NAMES_EN
FOOTWORK: List[str] = FOOTWORK_ZH + FOOTWORK_EN
CAMERA: List[str] = CAMERA_ZH + CAMERA_EN
SOUNDSCAPE: List[str] = SOUNDSCAPE_ZH + SOUNDSCAPE_EN

# 攻击动作词（用于"具体描写 -> 空泛形容"的定位）
ATTACK_WORDS: List[str] = [
    "攻击", "出招", "命中", "strike", "strikes", "attacks", "swings", "slashes",
    "thrusts", "kicks", "punches", "sweeps", "cuts",
]

# ── 反例词（逻辑漏洞 / 稀释动作）──────────────────────────────────────
# 注意：这些词出现在**正向提示词**里即是问题（H3 没有独立负面栏，负面项要么
# 写进 ComfyUI 的 negative prompt，要么用正向点名的方式描述）。
LOGIC_HOLES: Dict[str, List[str]] = {
    "slowmo": ["慢动作", "慢镜", "冻帧", "定格", "子弹时间",
               "slow motion", "slow-motion", "freeze frame", "bullet time", "time stop"],
    "teleport": ["瞬移", "闪现", "瞬间消失", "凭空出现", "凭空消失",
                 "teleport", "teleports", "blinks out", "vanishes", "appears out of nowhere",
                 "instantly appears"],
    "energy_fx": ["剑气", "刀气", "掌风", "掌力", "罡气", "光刃", "冲击波", "气浪",
                  "qi blast", "energy wave", "sword aura", "aura", "shockwave",
                  "energy beam", "projectile", "energy trail"],
    "ui_meta": ["血条", "UI", "分数", "得分", "扣血", "hp",
                "health bar", "hp bar", "score", "damage number", "ui overlay"],
    "idle_open": ["对峙", "凝视", "站着不动", "相互看着", "摆架",
                  "staring at each other", "face off", "face-off", "standstill",
                  "motionless", "posing", "holding a pose"],
    "empty_adj": ["很重", "极快", "非常快", "激烈", "爆发", "震撼",
                  "重重一拳", "速度很快", "威力巨大", "气势惊人",
                  "very fast", "extremely fast", "intense", "devastating",
                  "powerful", "brutal", "lightning fast", "with great force"],
    # 风格反噬词：用户文档实测结论——写实人物写这几个词会被放大成重噪点/油腻脸。
    # 「H3 无独立负面提示词栏，须正向点名」→ 写实人物应当**正向**写皮肤锁。
    "style_risk": ["film grain", "35mm", "flawless skin", "胶片颗粒", "完美皮肤"],
}

# ── 空泛替换池 ─────────────────────────────────────────────────────────
EMPTY_FILLER_ZH: List[str] = ["很重", "极快", "激烈", "威力巨大", "速度很快", "气势惊人"]
EMPTY_FILLER_EN: List[str] = [
    "extremely fast", "very powerful", "intense", "devastating", "lightning fast",
]
EMPTY_ATTACK_SENTENCE_ZH = "角色发动一次极快的攻击，威力巨大。"
EMPTY_ATTACK_SENTENCE_EN = "The fighter attacks with extremely fast and powerful movements."

# ── 站桩替换池 ─────────────────────────────────────────────────────────
IDLE_REPLACEMENTS_ZH: List[str] = [
    "角色站在原地不动，只是看着对方",
    "双方拉开对峙，相互凝视，没有出手",
    "角色摆好架势，缓缓环顾四周",
]
IDLE_REPLACEMENTS_EN: List[str] = [
    "the fighter stands motionless and watches the opponent",
    "both fighters hold their positions, staring at each other",
    "the fighter settles into a ready pose and looks around",
]

# ── 规则说明式文字：H3 读不懂，会稀释动作 ─────────────────────────────
TEMPLATE_JUNK_ZH: List[str] = [
    "注意：本场要求按下列战斗规则严格结算，普攻与招式每拍各一次。",
    "参数：24fps，832x480，CFG 1.0，LoRA 权重 0.9。",
    "说明：以上动作按第一步的电影级打斗设计稿执行，不要改动动作内容。",
]
TEMPLATE_JUNK_EN: List[str] = [
    "Note: every beat must follow the combat rules: one basic attack and one technique per second.",
    "Parameters: 24fps, 832x480, CFG 1.0, LoRA weight 0.9.",
    "Instruction: execute the action described in step one exactly, do not modify it.",
]

# ── H3 提示词骨架字段 ─────────────────────────────────────────────────
# 官方两套壳的段名（依据随包 h3-skill/base-en.txt、ref-en.txt）：
# 「文生视频」与「多参考图」是两套不同字段，绝不能混用。
BASE_SECTIONS: List[str] = [
    "integrated_multimodal_description",
    "overall_soundscape",
    "non_diegetic_music",
]
REF_SECTIONS: List[str] = [
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
]
REF_ONLY_SECTIONS: List[str] = [s for s in REF_SECTIONS if s not in BASE_SECTIONS]
BASE_ONLY_SECTIONS: List[str] = [s for s in BASE_SECTIONS if s not in REF_SECTIONS]

# 开工头（触发词 + 时长/帧数/比例/fps/分辨率）
HEADER_TOKENS: List[str] = [
    "wushu_action", "seconds", "frames", "16:9", "9:16", "24fps", "832x480",
]

_SLOT_BUCKETS: Dict[str, List[str]] = {
    "force_chain": FORCE_CHAIN,
    "distance": DISTANCE,
    "defense": DEFENSE,
    "contact": CONTACT,
    "feedback": FEEDBACK,
    "move_names": MOVE_NAMES,
    "footwork": FOOTWORK,
    "camera": CAMERA,
    "soundscape": SOUNDSCAPE,
}


def _count(text_low: str, words: Sequence[str]) -> int:
    """大小写不敏感计数（英文按原样包含匹配即可，词表里已含词组）。"""
    total = 0
    for w in words:
        if w.isascii():
            total += text_low.count(w.lower())
        else:
            total += text_low.count(w)
    return total


def slots_present(text: str) -> Dict[str, int]:
    """粗略统计各类槽位词在文本里出现的次数（用于降级前后对比、诊断）。"""
    low = text.lower()
    return {name: _count(low, words) for name, words in _SLOT_BUCKETS.items()}


def logic_hole_hits(text: str) -> Dict[str, List[str]]:
    low = text.lower()
    hits: Dict[str, List[str]] = {}
    for key, words in LOGIC_HOLES.items():
        found = [w for w in words if w.lower() in low]
        if found:
            hits[key] = found
    return hits
