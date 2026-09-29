"""训练对合成：把「武打逻辑完备」的正例，按规则定向降级成负例。

核心思想
--------
用户已有的分镜稿/提示词库是"正例"。真正让模型学会"武打逻辑"的，不是正例
本身，而是**正例与它的逻辑缺失版本之间的向量差**。这里用确定性规则做降级，
好处有三：

* 不需要任何外部大模型、不需要人工标注，离线可跑，随规则库一起更新；
* 降级是可解释的（哪个槽位被抽掉、注入了什么漏洞），能反查训练效果；
* 强度可控：同一正例可以生成 1~5 级不同残缺程度的负例，让评分头学到一条
  连续的"逻辑完备度"刻度，而不是简单的 0/1。

降级维度（与 ``lint/h3lint.py`` 的八维诊断同源）
------------------------------------------------
* force_chain  抽掉力线（蹬地/转腰/送胯/借力）
* distance     抽掉格距与站位
* defense      把防守反应换成站桩对望
* contact      抽掉接触点与命中判定
* feedback     抽掉打击反馈（火星/踉跄/衣破/喘息）
* moves        把具体招式泛化成"一次攻击"
* ending       破坏死线结局（没有结果）
* junk         掺入 H3 读不懂的规则说明
* empty        把具体描写换成空泛形容（很重/极快/激烈）
* holes        注入逻辑漏洞（慢动作/剑气/瞬移/血条 UI）
* order        打乱时间码顺序
* sound        抽掉音景段
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import lexicons
from .seeds import all_seeds

# ── 打分：优先用移植过来的官方规则引擎，缺失时退回启发式 ─────────────────


def _heuristic_score(text: str) -> float:
    s = lexicons.slots_present(text)
    weights = {
        "force_chain": 0.10, "distance": 0.08, "defense": 0.10, "contact": 0.12,
        "feedback": 0.14, "move_names": 0.08, "footwork": 0.06,
        "camera": 0.06, "soundscape": 0.06,
    }
    score = 0.10  # 底分
    for k, w in weights.items():
        score += w * min(1.0, s.get(k, 0) / 3.0)
    holes = lexicons.logic_hole_hits(text)
    score -= 0.06 * min(3, sum(len(v) for v in holes.values()))
    if "[Shot 1]" not in text:
        score -= 0.05
    return max(0.0, min(1.0, score))


def score_text(text: str, mode: str = "final", score_mode: str = "composite") -> float:
    """0~1 的武打逻辑合规分（训练标签就用它）。

    ``score_mode``：

    * ``composite``（默认）＝ ``0.4 * 壳结构分 + 0.6 * 武打逻辑分``。壳结构分来自
      ``lint/h3lint.py``（h3lint.js 的忠实移植），逻辑分来自 ``logic_score.py``
      （中英双语）。**必须复合**的原因：h3lint 是中文语境规则，在英文稿上会把
      "blocks / sidesteps / rolls" 误判成"没有防守反应"。
    * ``lint``    只取壳结构分（要和 H3武斗模拟器 的体检分数对齐时用）。
    * ``logic``   只取武打逻辑分（最贴近"桥要学的东西"）。
    """
    if score_mode == "logic":
        return _logic_score(text, mode)
    if score_mode == "lint":
        return _format_score(text, mode)
    return _composite_score(text, mode)


def _composite_score(text: str, mode: str = "final") -> float:
    try:
        from .logic_score import composite_score

        return composite_score(text, mode)
    except Exception:
        return _heuristic_score(text)


def _logic_score(text: str, mode: str = "final") -> float:
    try:
        from .logic_score import score_logic

        return float(score_logic(text, mode).score)
    except Exception:
        return _heuristic_score(text)


def _format_score(text: str, mode: str = "final") -> float:
    try:
        from .lint import lint_score_01  # type: ignore

        return float(lint_score_01(text, {"mode": mode}))
    except Exception:
        return _heuristic_score(text)


# ── 降级算子 ────────────────────────────────────────────────────────────

_EMPTY_REPLACEMENTS = lexicons.EMPTY_FILLER_ZH
_GENERIC_MOVE_ZH = "一次攻击"
_GENERIC_MOVE_EN = "a basic attack"
_IDLE_REPLACEMENTS = lexicons.IDLE_REPLACEMENTS_ZH
_IDLE_REPLACEMENTS_EN = lexicons.IDLE_REPLACEMENTS_EN

# 中英双语句子切分：中文按句号/分号切（顺带吃掉空白），英文按 .!? 但用零宽断言，
# 把空格留在下一句开头，避免重组后出现 "courtyard.A red-winged" 这种粘连。
_SENT_SPLIT = re.compile(r"(?<=[。！？；;])\s*|(?<=[.!?])(?=\s+[A-Z\"'(])")

# 防守动作的"短语级手术刀"：把「格挡 incoming strike」这类从句换成站桩，
# 句子其余部分（攻击方、位置）保持不动 —— 这样正负例仍然是同一个事件。
_DEFENSE_PHRASE_EN = re.compile(
    r"\b(?:blocks?|blocking|parr(?:y|ies|ied)|deflects?|deflection|evad(?:e|es|ed)|"
    r"dodges?|sidesteps?|side-steps?|rolls?(?:\s+low)?|sway-back|leans\s+back|"
    r"ducks?(?:\s+under)?|intercepts?|guards?)\b[^,.;]*",
    re.I,
)
_DEFENSE_PHRASE_ZH = re.compile(
    r"(?:格挡|格开|架住|封挡|横挡|斜挡|闪避|侧闪|翻滚|侧滚|前滚|硬吃|卸力|举刀|举剑)[^，。；]*"
)


def _sentences(line: str) -> List[str]:
    parts = _SENT_SPLIT.split(line)
    return [p for p in parts if p and p.strip()]


def _line_has(line: str, words: Sequence[str]) -> bool:
    """大小写不敏感的中英混合包含判断。"""
    low = line.lower()
    return any((w.lower() if w.isascii() else w) in low for w in words)


def _rand_idle(rng: random.Random, chinese: bool) -> str:
    return rng.choice(_IDLE_REPLACEMENTS if chinese else _IDLE_REPLACEMENTS_EN)


def _rand_empty(rng: random.Random, chinese: bool) -> str:
    return rng.choice(_EMPTY_REPLACEMENTS if chinese else lexicons.EMPTY_FILLER_EN)


def _operate_on_shots(text: str, fn) -> Tuple[str, bool]:
    """只对分镜行（``[Shot n]`` / ``[镜头n]``）做变换，保住头部（触发词/时长/场景/人物）不动。

    保住头部很关键：正负例必须是"同一个事件、同一个人物"，否则桥学到的是
    "换场景"而不是"补逻辑"。
    """
    lines = text.split("\n")
    changed = False
    out: List[str] = []
    for line in lines:
        if SHOT_RE.match(line.strip()):
            new, did = fn(line)
            out.append(new)
            changed = changed or did
        else:
            out.append(line)
    return "\n".join(out), changed


def _drop_sentences_with(text: str, words: Sequence[str], keep_ratio: float, rng: random.Random) -> Tuple[str, bool]:
    changed = False

    def fn(line: str) -> Tuple[str, bool]:
        nonlocal changed
        sents = _sentences(line)
        kept = []
        for s in sents:
            if _line_has(s, words) and rng.random() > keep_ratio:
                changed = True
                continue
            kept.append(s)
        if not kept:
            return line, False
        return "".join(kept), changed

    new, did = _operate_on_shots(text, fn)
    return new, did


def _replace_moves(text: str, ratio: float, rng: random.Random) -> Tuple[str, bool]:
    """把具体招式泛化成"一次攻击"——招式名是最密集的信息载体，泛化后 H3 只能瞎编。"""
    changed = False

    def fn(line: str) -> Tuple[str, bool]:
        nonlocal changed
        out = line
        chinese = lexicons.is_chinese(out)
        generic = _GENERIC_MOVE_ZH if chinese else _GENERIC_MOVE_EN
        for mv in lexicons.MOVE_NAMES:
            if rng.random() >= ratio:
                continue
            if mv.isascii():
                pat = re.compile(r"\b" + re.escape(mv) + r"\b", re.I)
                if pat.search(out):
                    out = pat.sub(generic, out)
                    changed = True
            elif mv in out:
                out = out.replace(f"「{mv}」", f"「{generic}」").replace(mv, generic)
                changed = True
        return out, changed

    return _operate_on_shots(text, fn)


def _collapse_defense(text: str, rng: random.Random) -> Tuple[str, bool]:
    """把防守反应换成站桩对望。

    中文稿常常带「防守：」标签，直接整段替换；英文稿的防守写在散文里，用正则
    只切掉防守从句，保留攻击与位置信息——正负例必须仍是同一个事件。
    """
    changed = False

    def fn(line: str) -> Tuple[str, bool]:
        nonlocal changed
        chinese = lexicons.is_chinese(line)

        if "防守：" in line:
            sents = _sentences(line)
            out = []
            for s in sents:
                if "防守：" in s and rng.random() < 0.85:
                    out.append("防守：" + _rand_idle(rng, chinese) + "。")
                    changed = True
                else:
                    out.append(s)
            return "".join(out), changed

        if rng.random() >= 0.85:
            return line, False
        if chinese:
            new, n = _DEFENSE_PHRASE_ZH.subn("站着不动", line)
        else:
            new, n = _DEFENSE_PHRASE_EN.subn("stands motionless", line)
        if n:
            changed = True
            return new, True
        return line, False

    return _operate_on_shots(text, fn)


def _empty_adjectives(text: str, ratio: float, rng: random.Random) -> Tuple[str, bool]:
    """把具体动作描写换成空泛形容——这是 H3 最常见的"写了等于没写"。"""
    changed = False

    def fn(line: str) -> Tuple[str, bool]:
        nonlocal changed
        chinese = lexicons.is_chinese(line)
        sents = _sentences(line)
        out = []
        for s in sents:
            is_attack = ("攻击：" in s or "反应：" in s or _line_has(s, lexicons.ATTACK_WORDS))
            if is_attack and len(s) > 12 and rng.random() < ratio:
                if chinese:
                    body = s.split("：", 1)[-1] if "：" in s else s
                    body = re.sub(r"「[^」]+」", "", body)[:6]
                    out.append(body + _rand_empty(rng, True) + "。")
                else:
                    out.append(lexicons.EMPTY_ATTACK_SENTENCE_EN + " ")
                changed = True
            else:
                out.append(s)
        return "".join(out), changed

    return _operate_on_shots(text, fn)


def _flatten_ending(text: str, rng: random.Random) -> Tuple[str, bool]:
    """破坏死线结局：把终结技降级成普攻、把倒地改成后退半步。"""
    lines = text.split("\n")
    shot_idx = [i for i, l in enumerate(lines) if SHOT_RE.match(l.strip())]
    if not shot_idx:
        return text, False
    last = shot_idx[-1]
    orig = lines[last]
    new = orig
    new = new.replace("终结技", "一次普攻")
    for w in ("倒地", "不再起身", "跪地", "单膝", "脱手", "失去平衡", "仰面"):
        new = new.replace(w, "后退半步")
    for pat, rep in (
        (r"\bfinisher\b", "a basic attack"),
        (r"\bfinishing (?:blow|strike|move)\b", "a basic attack"),
        (r"\bfinal strike\b", "a basic attack"),
        (r"\b(?:falls|collapses|sprawls|drops to one knee|kneels)\b", "steps back half a step"),
        (r"\bdoes not get up\b", "stays on his feet"),
        (r"\bcan't get up\b", "stays on his feet"),
    ):
        new = re.sub(pat, rep, new, flags=re.I)
    if new == orig:
        new = re.sub(r"反应：.*$", "反应：双方仍在交手，本拍没有分出结果。", orig)
        if new == orig:
            new = re.sub(r"\boverall_soundscape\b", "overall_soundscape", orig)
    lines[last] = new
    return "\n".join(lines), new != orig


def _inject_holes(text: str, rng: random.Random, k: int = 2) -> Tuple[str, bool]:
    """注入逻辑漏洞：慢动作 / 剑气 / 瞬移 / 血条 UI 等按语言挑词。"""
    chinese = lexicons.is_chinese(text)
    pool: List[str] = []
    for words in lexicons.LOGIC_HOLES.values():
        picked = [w for w in words if (not w.isascii()) == chinese]
        pool.extend(picked[:2] if picked else words[:1])
    picked = rng.sample(pool, min(k, len(pool)))
    inject = "、".join(picked)
    if chinese:
        tail = f"\n（补注：镜头里加入{inject}的处理。）"
    else:
        tail = f"\n(Note: add {', '.join(picked)} to the shot.)"
    return text.rstrip() + tail, True


def _inject_rule_junk(text: str, rng: random.Random) -> Tuple[str, bool]:
    """掺规则说明：H3 读不懂这类文字，会把动作语义稀释掉。"""
    chinese = lexicons.is_chinese(text)
    junk = rng.choice(lexicons.TEMPLATE_JUNK_ZH if chinese else lexicons.TEMPLATE_JUNK_EN)
    lines = text.split("\n")
    lines.insert(1 if lines else 0, junk)
    return "\n".join(lines), True


def _drop_soundscape(text: str, rng: random.Random) -> Tuple[str, bool]:
    if "overall_soundscape" not in text:
        return text, False
    return re.sub(r"overall_soundscape:.*", "overall_soundscape: 无特别声音设计。", text), True


def _shuffle_timecodes(text: str, rng: random.Random) -> Tuple[str, bool]:
    """把某两拍的时间码互换，制造"时间码顺序错乱"的负例。

    同时兼容两种写法：``At 00:02.100``（规则引擎唯一解析的形式）
    与 ``2.1-5.6秒``（文档里的旧写法）。
    """
    lines = text.split("\n")
    idx = [i for i, l in enumerate(lines) if SHOT_RE.match(l.strip())]
    if len(idx) < 2:
        return text, False
    a, b = rng.sample(idx, 2)
    rx = re.compile(r"(?:At\s+\d{1,2}:\d{2}\.\d{2,3}|"
                    r"\d+(?:\.\d+)?\s*[-–~]\s*\d+(?:\.\d+)?\s*秒)")
    ma, mb = rx.search(lines[a]), rx.search(lines[b])
    if not ma or not mb:
        return text, False
    lines[a] = lines[a][: ma.start()] + mb.group(0) + lines[a][ma.end() :]
    lines[b] = lines[b][: mb.start()] + ma.group(0) + lines[b][mb.end() :]
    return "\n".join(lines), True


@dataclass
class DegradeOp:
    key: str
    label: str
    weight: float
    fn: Any


_OPS: List[DegradeOp] = [
    DegradeOp("force_chain", "抽掉力线（蹬地/转腰/送胯/借力）", 1.0,
              lambda t, r, lv: _drop_sentences_with(t, lexicons.FORCE_CHAIN, 0.15, r)),
    DegradeOp("distance", "抽掉格距与站位", 0.9,
              lambda t, r, lv: _drop_sentences_with(t, lexicons.DISTANCE, 0.2, r)),
    DegradeOp("contact", "抽掉接触点与命中判定", 0.9,
              lambda t, r, lv: _drop_sentences_with(t, lexicons.CONTACT, 0.2, r)),
    DegradeOp("feedback", "抽掉打击反馈", 1.0,
              lambda t, r, lv: _drop_sentences_with(t, lexicons.FEEDBACK, 0.2, r)),
    DegradeOp("defense", "把防守换成站桩对望", 1.0,
              lambda t, r, lv: _collapse_defense(t, r)),
    DegradeOp("moves", "具体招式泛化为「一次攻击」", 0.7,
              lambda t, r, lv: _replace_moves(t, 0.8, r)),
    DegradeOp("ending", "破坏死线结局", 0.8,
              lambda t, r, lv: _flatten_ending(t, r)),
    DegradeOp("empty", "具体描写换成空泛形容", 0.8,
              lambda t, r, lv: _empty_adjectives(t, 0.5, r)),
    DegradeOp("junk", "掺入规则说明（H3 读不懂）", 0.5,
              lambda t, r, lv: _inject_rule_junk(t, r)),
    DegradeOp("holes", "注入逻辑漏洞（慢动作/剑气/瞬移/血条）", 0.6,
              lambda t, r, lv: _inject_holes(t, r, k=2 if lv < 4 else 4)),
    DegradeOp("sound", "抽掉音景", 0.4,
              lambda t, r, lv: _drop_soundscape(t, r)),
    DegradeOp("order", "打乱时间码顺序", 0.3,
              lambda t, r, lv: _shuffle_timecodes(t, r)),
]
OPS_BY_KEY: Dict[str, DegradeOp] = {op.key: op for op in _OPS}
ALL_OP_KEYS: List[str] = [op.key for op in _OPS]
# 只动"逻辑层"、不动场景与人物的算子（默认推荐集）
LOGIC_OPS: List[str] = ["force_chain", "distance", "contact", "feedback", "defense", "moves", "ending"]


@dataclass
class DegradeProfile:
    """降级配置。``ops`` 为空时用 ``LOGIC_OPS``（不动场景，专修逻辑）。"""

    ops: List[str] = field(default_factory=list)
    intensity: int = 3           # 1~5，越大越残缺
    keep_ratio_range: Tuple[float, float] = (0.0, 0.25)

    def resolved_ops(self) -> List[str]:
        keys = self.ops or LOGIC_OPS
        bad = [k for k in keys if k not in OPS_BY_KEY]
        if bad:
            raise ValueError(f"未知降级算子：{bad}，可选 {ALL_OP_KEYS}")
        return keys


def degrade(
    good_text: str,
    profile: Optional[DegradeProfile] = None,
    rng: Optional[random.Random] = None,
    intensity: Optional[int] = None,
) -> Tuple[str, List[str]]:
    """把正例降级成负例，返回 (负例文本, 实际用到的算子 key 列表)。"""
    profile = profile or DegradeProfile()
    rng = rng or random.Random(0)
    lv = int(intensity if intensity is not None else profile.intensity)
    lv = max(1, min(5, lv))

    ops = profile.resolved_ops()
    n_apply = max(1, min(len(ops), int(round(len(ops) * lv / 5.0)) + (1 if lv >= 4 else 0)))
    chosen = rng.sample(ops, n_apply)

    text = good_text
    applied: List[str] = []
    for key in chosen:
        op = OPS_BY_KEY[key]
        try:
            new_text, did = op.fn(text, rng, lv)
        except Exception:
            continue
        if did and new_text != text:
            text = new_text
            applied.append(key)

    if not applied:  # 兜底：至少注入一个逻辑漏洞，保证负例真的更差
        text, did = _inject_holes(text, rng, k=2)
        applied.append("holes")

    return text, applied


# ── 语料抽取：从用户自己的提示词库/分镜稿里挖正例 ─────────────────────────

_PROMPT_HINTS = ("wushu_action", "integrated_multimodal_description", "detailed_description")
_TEXT_EXT = {".md", ".txt", ".json", ".jsonl", ".srt", ".csv", ".js", ".mjs", ".py"}

# 分镜标记：英文 `[Shot 3]` 与中文 `[镜头3]`（用户 924 条语料里中文版用的是后者，
# 且字段名保持英文不译）。所有涉及"逐拍"的算子都用这一个正则。
SHOT_RE = re.compile(r"\[(?:Shot|镜头|鏡頭)\s*\d+\]")

# H3 壳字段必须**独占一行并以冒号结尾**才算"真提示词"。
# 只做子串匹配的话，凡是"讲怎么用这个字段"的文档段落都会被误当成提示词
# （实测 `H3武斗模拟器_使用指南.md` 会因此多出十几条噪声）。
SECTION_LINE_RE = re.compile(
    r"(?m)^\s*(?:integrated_multimodal_description|subject_definitions|summary|"
    r"retention_analysis|detailed_description|overall_soundscape|non_diegetic_music)\s*[:：]"
)
# 未替换的模板占位符 —— 有它就不是成品提示词
PLACEHOLDER_RE = re.compile(r"\{\{[^}]+\}\}")
# 【…】式占位/规则段：用户文档里"全例"其实是**带【题材】【角色A】【空间轴线】的骨架**，
# 不是成片；正文出现【…】也会被规则引擎判成"残留规则段"。实测用户 924 条同分布
# 语料里【…】出现 0 次，所以直接拒收不会有副作用。
BRACKET_PLACEHOLDER_RE = re.compile(r"【[^】\n]{1,14}】")


def _looks_like_prompt(block: str, mode: str = "t2v", include_design: bool = False) -> bool:
    if PLACEHOLDER_RE.search(block):
        return False
    if BRACKET_PLACEHOLDER_RE.search(block):
        return False
    has_shot = bool(SHOT_RE.search(block))
    has_section = bool(SECTION_LINE_RE.search(block))
    has_trigger = "wushu_action" in block[:160]

    if len(block) >= 200:
        if mode == "ref2v":
            if has_shot and has_section and (
                "detailed_description" in block or "subject_definitions" in block
            ):
                return True
        elif has_section and (has_shot or "overall_soundscape" in block):
            return True
        elif has_trigger and has_shot:
            return True

    # 设计稿散文（第1步产物）：没有 H3 壳字段，但动作密度高。
    # 注意：这类素材精度天然较低，且与成品 H3 提示词**不同分布**，
    # 默认关闭（include_design_drafts），要用户明确打开才收。
    if include_design and len(block) >= 150:
        head = block.lstrip()
        if head.startswith(("#", "|", ">", "-", "*")) or re.match(r"^\d+\s*[.、)]\s", head):
            return False   # 骨架清单 / 表格 / 编号规则，不是动作散文
        if "**" in block or "`" in block:
            return False   # Markdown 装饰 → 是文档说明，不是设计稿
        slots = lexicons.slots_present(block)
        if sum(1 for v in slots.values() if v > 0) < 2:
            return False
        action_buckets = ("force_chain", "contact", "feedback", "defense", "footwork", "move_names", "distance")
        if sum(1 for k in action_buckets if slots.get(k, 0) > 0) < 1:
            return False
        return True
    return False


def extract_prompts_from_text(
    content: str,
    mode: str = "t2v",
    include_design: bool = False,
) -> List[str]:
    """从任意文本里切出候选完整提示词。

    三种切法叠加，覆盖真实文档的排布习惯：

    1. **代码围栏**（`` ``` ``）—— 模板类文档最常见；
    2. **Markdown 标题分段**（``## [Segment 01] …``）—— 一份文档里并列几十条
       提示词的写法（例如 yajni 的 38 条 Ref2VA 提示词）；
    3. **空行切块** —— 兜底。

    再按 ``mode`` 过滤结构；``include_design=True`` 时额外收"没有 H3 壳但动作
    密度很高"的设计稿散文（第1步产物）。
    """
    candidates: List[str] = []
    # 1) 代码围栏内的内容优先（用户文档里模板通常放在 ``` 里）
    for m in re.finditer(r"```[a-zA-Z]*\n(.*?)```", content, flags=re.S):
        candidates.append(m.group(1).strip())
    # 2) 按 Markdown 标题切段（标题连同其正文算一段）
    for seg in re.split(r"(?m)^(?=#{1,4}\s)", content):
        candidates.append(seg.strip())
    # 3) 按空行切块
    for block in re.split(r"\n\s*\n", content):
        candidates.append(block.strip())

    out: List[str] = []
    seen = set()
    for block in candidates:
        if not _looks_like_prompt(block, mode, include_design):
            continue
        # 去掉行首的引用/列表装饰，但保留 H3 字段名与 [Shot n]
        block = re.sub(r"^[>\-\*\s]+", "", block, flags=re.M).strip()
        key = hashlib.md5(block.encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(block)
    return out


def _read_text(path: str) -> str:
    """读文本文件，带编码回退（UTF-8 → GB18030 → errors=ignore）。

    说明：这是**防御性**设计。用户侧盘点曾声称 `_research_prompts\\` 里
    `model_adapters.txt` / `movieflow_standard.txt` / `platform_guides.txt`
    是 GBK/CP936，但按字节实测**三个都是合法 UTF-8**，所以本仓库并不需要它；
    留着是因为别人的语料（尤其 Windows 中文环境导出的 txt）常常真是 GBK，
    按 UTF-8 硬读会得到乱码或空串，白白丢语料。
    """
    raw = None
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except Exception:
        return ""
    for enc in ("utf-8", "utf-8-sig", "gb18030", "cp936"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("utf-8", errors="ignore")


def load_corpus(
    paths: Iterable[str],
    mode: str = "t2v",
    max_items: int = 5000,
    include_design: bool = False,
) -> List[str]:
    """从文件或目录加载正例语料（**只读**，不修改任何文件）。

    按扩展名分派：``.json`` 走结构化取值（``prompt`` 字段），``.jsonl`` 逐行取值，
    ``.csv`` 取 ``prompt`` 列，其余按文本切块。**目录与文件走同一条分派**，
    否则目录模式会把 JSON 当纯文本读，白白丢掉九成语料。

    **跨文件按内容哈希去重**：用户仓库里存在同内容重复文件（`chain-director\\docs\\`
    有两组 MD5 相同的文件），不去重会虚增语料量、让同一段文本在数据集里占多份权重。
    """
    prompts: List[str] = []
    seen = set()
    for p in paths:
        if not p:
            continue
        if os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                for name in files:
                    if os.path.splitext(name)[1].lower() in _TEXT_EXT:
                        for text in _load_one(os.path.join(root, name), mode, include_design):
                            key = hashlib.md5(text.encode("utf-8")).hexdigest()
                            if key not in seen:
                                seen.add(key)
                                prompts.append(text)
        elif os.path.isfile(p):
            for text in _load_one(p, mode, include_design):
                key = hashlib.md5(text.encode("utf-8")).hexdigest()
                if key not in seen:
                    seen.add(key)
                    prompts.append(text)
        if len(prompts) >= max_items:
            break
    return prompts[:max_items]


def _load_one(path: str, mode: str, include_design: bool = False) -> List[str]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        return _load_json(path, mode, include_design)
    if ext == ".jsonl":
        return _load_jsonl(path, mode)
    if ext == ".csv":
        return [
            r["prompt"] for r in load_metadata_rows(path)
            if _looks_like_prompt(r.get("prompt", ""), mode, include_design)
        ]
    return _read_and_extract(path, mode, include_design)


def _read_and_extract(path: str, mode: str, include_design: bool = False) -> List[str]:
    content = _read_text(path)
    if not content:
        return []
    prompts = extract_prompts_from_text(content, mode, include_design)
    # "主导块"规则：一个文件里只有一条提示词时（`dataset/clips/srcNNN_segNN.txt`
    # 就是这种一文件一条），最长的那块几乎覆盖了整个文件；此时只留它，
    # 避免把同一份提示词切出的碎片也当成额外正例。
    if len(prompts) > 1:
        longest = max(prompts, key=len)
        if len(longest) >= 0.6 * len(content.strip()):
            return [longest]
    return prompts


_JSON_PROMPT_KEYS = ("prompt", "positive", "text", "content", "body", "value")


def _load_json(path: str, mode: str, include_design: bool = False) -> List[str]:
    """读结构化 JSON 语料。

    真实来源：``_template_sources\\awesome-h3\\prompts\\*.json`` —— 每个文件是一个
    15 键的 dict（``slug/title/description/category/mode/duration/...``），
    提示词原文在 ``prompt`` 字段；另有一个 ``catalog.json`` 把全部条目收在
    ``prompts`` 列表里。所以这里按"已知键优先、其余递归兜底"来取。
    """
    out: List[str] = []
    try:
        obj = json.loads(_read_text(path))
    except Exception:
        return []

    def take(v) -> None:
        if isinstance(v, str):
            if _looks_like_prompt(v, mode, include_design):
                out.append(v)
        elif isinstance(v, dict):
            for key in _JSON_PROMPT_KEYS:
                val = v.get(key)
                if isinstance(val, str) and _looks_like_prompt(val, mode, include_design):
                    out.append(val)
                    return
            for val in v.values():
                if isinstance(val, (dict, list)):
                    take(val)
        elif isinstance(v, list):
            for val in v:
                take(val)

    take(obj)
    return out


def _load_jsonl(path: str, mode: str) -> List[str]:
    out: List[str] = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                for key in ("good", "positive", "prompt", "text", "target", "output"):
                    if isinstance(obj.get(key), str) and _looks_like_prompt(obj[key], mode):
                        out.append(obj[key])
                        break
    except Exception:
        pass
    return out


# ── LoRA 打标表（metadata.csv）与"被拒片段"负例 ─────────────────────────
# 用户 924 条同分布语料的原始出处：``dataset\h3_train\metadata.csv``
# （列头 video,prompt,input_audio,frame_rate），以及 ``dataset\rejected\*.mp4``
# —— 这些是被筛掉的片段，它们对应的 prompt 就是**真实世界的负例**。
# 合成负例（规则降级）教模型"补逻辑"，真实负例教模型"什么会翻车"，两者互补。

def load_metadata_rows(csv_path: str, prompt_col: str = "prompt") -> List[Dict[str, str]]:
    """读 metadata.csv，返回 [{video, prompt, ...}]。"""
    import csv

    rows: List[Dict[str, str]] = []
    try:
        with open(csv_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return []
            for row in reader:
                if prompt_col in row and row[prompt_col]:
                    rows.append({k: (v or "") for k, v in row.items()})
    except Exception:
        return []
    return rows


def load_rejected_prompts(csv_path: str, rejected_dir: str, prompt_col: str = "prompt") -> List[str]:
    """从 metadata.csv 里挑出"对应片段被 reject 掉"的那些 prompt（真实负例）。"""
    if not os.path.isdir(rejected_dir):
        return []
    names = set()
    for root, _dirs, files in os.walk(rejected_dir):
        for name in files:
            names.add(os.path.splitext(name)[0].lower())
    if not names:
        return []
    out: List[str] = []
    for row in load_metadata_rows(csv_path, prompt_col):
        video = os.path.splitext(os.path.basename(row.get("video", "")))[0].lower()
        if video and video in names:
            out.append(row[prompt_col])
    return out


# ── 训练对构建 ─────────────────────────────────────────────────────────

@dataclass
class TrainingPair:
    bad: str
    good: str
    bad_score: float
    good_score: float
    ops: List[str] = field(default_factory=list)
    source: str = "seed"

    def to_record(self) -> Dict[str, Any]:
        return {
            "ops": self.ops,
            "source": self.source,
            "bad_score": round(self.bad_score, 4),
            "good_score": round(self.good_score, 4),
            "bad_head": self.bad[:120],
            "good_head": self.good[:120],
        }


def build_pairs(
    sources: Optional[Sequence[str]] = None,
    mode: str = "t2v",
    variants_per_source: int = 3,
    profile: Optional[DegradeProfile] = None,
    include_seeds: bool = True,
    score_fn=None,
    score_mode: str = "composite",
    min_good_score: float = 0.45,
    max_pairs: int = 20000,
    seed: int = 1234,
) -> List[TrainingPair]:
    """从语料 + 种子构建训练对。

    * ``sources`` 空则只用内置种子（先把流水线跑通）；
    * 每条正例生成 ``variants_per_source`` 个不同强度的负例；
    * 正例分数低于 ``min_good_score`` 的会被丢掉（避免拿错了的稿子当老师）；
    * ``score_mode`` 见 :func:`score_text`（默认 composite，中英双语复合分）。
    """
    rng = random.Random(seed)
    # profile 允许为 None（签名里就是 Optional）：这里补上默认档案，
    # 否则下面读 profile.intensity 会直接 AttributeError（tools/selftest.py 第 2 步实测崩过）
    if profile is None:
        profile = DegradeProfile()
    if score_fn is None:
        score_fn = lambda t, m="final": score_text(t, m, score_mode)  # noqa: E731
    goods: List[Tuple[str, str]] = []
    for s in sources or []:
        goods.append((s, "corpus"))
    if include_seeds:
        for s in all_seeds(mode):
            goods.append((s, "seed"))

    pairs: List[TrainingPair] = []
    seen = set()
    for text, source in goods:
        text = text.strip()
        if not text:
            continue
        gs = float(score_fn(text, "final"))
        if gs < min_good_score and source == "corpus":
            continue
        nv = max(1, variants_per_source)
        cap = max(1, min(5, int(profile.intensity)))
        for v in range(nv):
            # ``intensity`` 是强度上限：多变体时在 1~cap 之间铺开（轻->重），
            # 单变体时直接用满。旧写法在 nv==1 时把强度写死成 3，
            # 会让用户设的 intensity=5 失效（实测发现的 bug）。
            lv = cap if nv == 1 else max(1, int(round(cap * (v + 1) / nv)))
            bad, ops = degrade(text, profile, rng, intensity=lv)
            bs = float(score_fn(bad, "final"))
            if bs >= gs:            # 降级没降下去，丢掉
                continue
            key = hashlib.md5((bad + "||" + text).encode("utf-8")).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            pairs.append(TrainingPair(bad, text, bs, gs, ops, source))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def pairs_stats(pairs: Sequence[TrainingPair]) -> Dict[str, Any]:
    if not pairs:
        return {"count": 0}
    from collections import Counter

    ops = Counter(k for p in pairs for k in p.ops)
    return {
        "count": len(pairs),
        "sources": dict(Counter(p.source for p in pairs)),
        "mean_bad_score": round(sum(p.bad_score for p in pairs) / len(pairs), 4),
        "mean_good_score": round(sum(p.good_score for p in pairs) / len(pairs), 4),
        "mean_gap": round(sum(p.good_score - p.bad_score for p in pairs) / len(pairs), 4),
        "op_usage": dict(ops.most_common()),
    }


def dump_pairs(pairs: Sequence[TrainingPair], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps({"bad": p.bad, "good": p.good, **p.to_record()}, ensure_ascii=False) + "\n")
    return path
