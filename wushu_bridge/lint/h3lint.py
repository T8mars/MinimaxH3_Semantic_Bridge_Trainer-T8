# -*- coding: utf-8 -*-
"""h3lint.py — H3 提示词体检（Python 忠实移植版）

移植来源：用户提供的 ``h3lint.js``（385 行，UMD，VERSION=h3lint-0.3）。
本文件是**逐字段忠实**移植：检查项 id、item 字段、item 顺序、计分权重、grade 阈值、
stats 字段全部与 JS 版一致，便于用 Node 原版做差分验证。

移植对应关系（JS 行号 → 本文件）：
    L22          VERSION
    L25-L41      常量表（BASE_SECTIONS / REF_SECTIONS / LEGACY_FIELDS / 词表 / 占位符）
    L43-L70      工具函数 splitSentences / rxShots / firstShot1Segment / rxTimecodes / tcToSec
    L72-L328     check()
    L313-L317    计分与 grade
    L331-L342    DIMENSIONS
    L345-L381    diagnose / diagnose_report / report

纯标准库：不依赖 torch / numpy / comfyui。

────────────────────────────────────────────────────────────────────────────
关于 JS 第 16 行的 ``require("./sim3d/combat-logic.js")``
────────────────────────────────────────────────────────────────────────────
JS 版在 L304-L310 用 FIGHT_LOGIC.checkPrompt 做「打斗连贯性」判定并产生
``fight-*`` 系列 item（其中 4 个是 error 级）。该模块已一并移植到同目录的
``_combat_logic.py``（同样纯标准库），默认启用，行为与 Node 版一致。

可选的降级开关：
    * 环境变量 ``H3LINT_DISABLE_COMBAT_LOGIC=1`` → 跳过所有 fight-* 检查；
    * ``check(text, {"combatLogic": False})``            → 同上。
降级时「打斗连贯性」维度不会有 error，分数会偏高，模块内所有 fight-* 相关行为
都在本文件的「11. 打斗连贯性」段落里有注释说明。
"""

from __future__ import annotations

import math
import os
import re

__all__ = [
    "VERSION", "check", "report", "diagnose", "diagnose_report",
    "DIMENSIONS", "BASE_SECTIONS", "REF_SECTIONS", "LEGACY_FIELDS",
    "EMPTY_WORDS", "SLOWMO", "BLOOD", "METAL_SOUND", "IDLE_OPEN", "AIR_WORDS", "TELEPORT",
    "AIR_SUPPORT", "NEG_WORDS", "PLACEHOLDER", "MARK",
    "combat_logic_available",
]

VERSION = "h3lint-0.3"  # ← JS L22

# ── 可选依赖：打斗逻辑（JS 里是 ./sim3d/combat-logic.js）─────────────────────
try:  # 作为包的一部分导入
    from . import _combat_logic as _COMBAT  # type: ignore
except Exception:  # pragma: no cover - 单文件直接运行时
    try:
        import _combat_logic as _COMBAT  # type: ignore
    except Exception:
        _COMBAT = None


def combat_logic_available() -> bool:
    """FIGHT_LOGIC 是否可用（对应 JS 里 ``FIGHT_LOGIC && FIGHT_LOGIC.checkPrompt``）。"""
    return _COMBAT is not None and hasattr(_COMBAT, "check_prompt")


# ── 官方两套壳的段名（JS L25-L26）────────────────────────────────────────────
BASE_SECTIONS = ["integrated_multimodal_description", "overall_soundscape", "non_diegetic_music"]
REF_SECTIONS = ["subject_definitions", "summary", "retention_analysis",
                "detailed_description", "overall_soundscape", "non_diegetic_music"]
# 旧稿常见错名 → 官方名（JS L28）
LEGACY_FIELDS = [["preservation_analysis", "retention_analysis"]]

# ── 词表（JS L31-L41，与第1步提示词规则同源）────────────────────────────────
EMPTY_WORDS = ["很重", "极快", "非常快", "激烈", "爆发", "震撼", "重重一拳", "打得很快",
               "速度很快", "威力巨大", "气势惊人"]
SLOWMO = ["慢动作", "慢镜", "冻帧", "定格", "子弹时间", "slow motion", "freeze frame"]
BLOOD = ["喷血", "溅血", "断肢", "开膛", "毙命", "致命伤", "血肉横飞", "锁喉", "勒颈", "插眼"]
METAL_SOUND = ["剑鸣", "刀啸", "金属相击", "金属碰撞", "兵器相击", "叮当"]
IDLE_OPEN = ["对峙", "凝视", "静静", "站着不动", "相互看着", "缓缓", "慢慢走近", "环顾"]
AIR_WORDS = ["腾空", "凌空", "悬停", "飞天", "滞空", "半空", "腾起"]
AIR_SUPPORT = ["蹬", "踏", "借力", "起跳", "跃", "撑", "墙", "檐", "栏杆", "柱", "梁",
               "绳索", "台边", "被击飞", "弹起"]
TELEPORT = ["瞬移", "闪现", "瞬间消失", "凭空出现", "凭空消失", "瞬身"]
NEG_WORDS = ["不要", "禁止", "不许", "不得", "no ", "don't", "avoid "]

# JS 的 \b 是 ASCII 词边界（\w == [A-Za-z0-9_]）；Python 的 \b 是 Unicode 词边界，
# 中文也算 \w，会让 `\bTODO\b`、`qi\b`、`\b(?:left|right|...)\b` 在中文上下文里行为
# 和 JS 不同。这里用等价的零宽断言复刻 JS 语义（两侧都覆盖，宽度为 0）。
_JSB = r"(?:(?<![A-Za-z0-9_])(?=[A-Za-z0-9_])|(?<=[A-Za-z0-9_])(?![A-Za-z0-9_]))"

# JS L41：占位符 / 规则段（每个正则各自 exec 一次，只报第一处）
PLACEHOLDER = [
    re.compile(r"\{\{[^}]+\}\}"),
    re.compile(r"【(?:情景|特效等级|身法速度|角色与场景锁定|人物性格|角色设定|电影级打斗|镜头组合|战斗规则)】"),
    re.compile(_JSB + r"TODO" + _JSB),
    re.compile(r"XXX"),
]

_RX_FINAL_RULEBLOCK = re.compile(r"【[^】]{2,12}】")
_RX_SECTION = lambda name: re.compile(name + r"\s*:", re.I)
_RX_SHOT = re.compile(r"\[Shot\s*\d+\]", re.I)          # JS L48
_RX_TIME_RAW = re.compile(r"At\s+(\d{1,2}):(\d{2})\.(\d{2,3})", re.I)   # JS L64
_RX_TC_ANY = re.compile(r"(\d{1,2}):(\d{2})\.(\d{2,3})")               # JS L66
_RX_SHOT1 = re.compile(r"\[Shot\s*1\]", re.I)                          # JS L53


# ── 工具函数（JS L43-L70）───────────────────────────────────────────────────
def split_sentences(t):
    """按句号/感叹号/问号/分号/换行切句（JS L43）。"""
    parts = re.split(r"[。！？；!?;\n]+", "" if t is None else str(t))
    return [s.strip() for s in parts if s.strip()]


def _rx_shots(t):
    """所有 [Shot N] 标记（JS L48）。"""
    return _RX_SHOT.findall("" if t is None else str(t))


def _shots_count(t):
    return len(_rx_shots(t))  # JS L329


# 行首的真分镜块。官方 Ref2VA 写法会在 retention_analysis 里写
# `(appears in [Shot 1])` 这类**引用**，JS 原版的 _rx_shots 把引用也算进去，
# 于是一份 1 镜的稿子会被数成 4 镜、甚至 6 镜以上直接判 shot-many(error)。
# opts["shotCounting"] = "blocks" 时只认行首标记；"auto" 在检测到参考式引用
# 时自动切到 blocks；默认 "raw" 与 JS 逐位一致。
_RX_SHOT_LINE = re.compile(r"(?m)^[ \t]*\[(?:Shot|镜头|鏡頭)\s*\d+\]")
_RX_SHOT_REF = re.compile(r"(?:appears\s+in|in|见|参见)\s*[\[(（]?\s*\[Shot\s*\d+\]", re.I)


def _use_block_counting(t, opts) -> bool:
    mode = opts.get("shotCounting", "raw")
    if mode == "blocks":
        return True
    return mode == "auto" and bool(_RX_SHOT_REF.search(t))


def _shots_for_count(t, opts):
    if _use_block_counting(t, opts):
        blocks = _RX_SHOT_LINE.findall(t)
        if blocks:
            return blocks
    return _rx_shots(t)


def _shot_segments(t, opts):
    """切出各镜正文段。策略与 _shots_for_count 保持一致，否则引用式写法会让
    段头错位，进而把「切镜接续/还是这两人」判成误报。"""
    if _use_block_counting(t, opts):
        ms = list(_RX_SHOT_LINE.finditer(t))
        if ms:
            return [
                t[m.end(): (ms[i + 1].start() if i + 1 < len(ms) else len(t))]
                for i, m in enumerate(ms)
            ]
    return re.split(r"\[Shot\s*\d+\]", t, flags=re.I)[1:]


def _first_shot1_segment(t):
    """真正的第 1 镜正文片段（JS L52-L63）。

    为什么：Ref2VA 的 retention_analysis 里会出现 ``(appears in [Shot 1])`` 这类引用，
    那不是镜头块，不能拿来判断第 1 镜有没有写时间码。
    """
    for m in _RX_SHOT1.finditer(t):
        before = t[max(0, m.start() - 26):m.start()]
        after = t[m.end():m.end() + 60]
        if re.search(r"(?:appears\s+in|in|见|参见|位于)\s*$|[(（,，]\s*$", before):
            continue  # 引用式出现
        if re.match(r"^\s*[)\）,，:：\]】]", after):
            continue
        return after
    return ""


def _rx_timecodes(t):
    # JS L64：String.match 返回的是完整匹配串（不是捕获组），所以这里取 group(0)
    return [m.group(0) for m in _RX_TIME_RAW.finditer("" if t is None else str(t))]


def _tc_to_sec(s):
    """`MM:SS.mm` / `MM:SS.mmm` → 秒（JS L65）。"""
    m = _RX_TC_ANY.search(s)
    if not m:
        return float("nan")
    frac = int(m.group(3)) / (100.0 if len(m.group(3)) == 2 else 1000.0)
    return int(m.group(1)) * 60 + int(m.group(2)) + frac


def _num(v):
    """JS 的 ``+v || 0``：NaN / 空 / 非数字 → 0。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(f) or math.isinf(f):
        return 0.0
    return f


def _js_num_str(x):
    """JS 模板字符串里 Number→String 的短表示（5.0 → "5"）。"""
    if float(x) == int(x):
        return str(int(x))
    return repr(float(x))


def _js_round(x):
    """JS Math.round：floor(x + 0.5)（与 Python 的银行家舍入不同）。"""
    return math.floor(x + 0.5)


def _fix(x, n):
    """JS Number.prototype.toFixed(n)。"""
    return f"{x:.{n}f}"


def _effect_re():
    # JS L167
    return re.compile(r"剑气|刀气|掌风|掌力|拳劲|气劲|劲风|罡气|火焰|火舌|冰霜|雷|电|光刃|残影|"
                      r"冲击波|气浪|震波|碎屑|火星|水花|尘土|"
                      r"blade|energy|flame|shockwave|aura|projectile|slash|gust", re.I)


# ── 主检查（JS L72-L328）────────────────────────────────────────────────────
def check(text, opts=None):
    """H3 提示词体检。

    opts 与 JS 版一致（JS L73-L78 + L190 + L254 + L305）::

        {
          "mode": "final" | "design",   # final=成片提示词 / design=第1步设计稿
          "names": ["洪七公", "杨过"],   # 双人同框与主语检查
          "weapons": ["bang", "none"],  # 全为 "none" ⇒ 徒手不许有金属声
          "duration": 15,               # 片长（秒），用于时间码范围检查
          "maxLen": 2500,               # 长度红线，默认 2500
          "tier": 7,                    # 本场最高等级（角色卡），内力外放检查
          "scenarioZh": "追逐战",        # 情景名，用于场景落实提示
          "combatLogic": True,          # 本移植版扩展：False 时跳过 fight-* 检查
        }

    返回（与 JS L318-L327 逐字段一致）::

        {"version", "mode", "score", "grade", "items": [{"id","level","msg","hint","at"}],
         "stats": {"length","shots","sentences","timecodes","errors","warns","infos"}}
    """
    opts = opts or {}
    mode = opts.get("mode") or "final"                     # JS L74
    names = [n for n in (opts.get("names") or []) if n]     # JS L75
    weapons = opts.get("weapons") or []                     # JS L76
    max_len = opts.get("maxLen") or 2500                    # JS L77
    duration = _num(opts.get("duration"))                   # JS L78
    t = "" if text is None else str(text)                   # JS L79
    items = []

    def add(level, id_, msg, hint="", at=""):
        # JS L81：hint 缺省 ""，at 为 None/未给 → ""
        items.append({"id": id_, "level": level, "msg": msg,
                      "hint": hint or "", "at": "" if at is None else at})

    # ── 1. 残留占位符与规则段（JS L83-L88 ｜ 为什么：成片提示词里出现 {{变量}} 或
    #       【】规则段说明第1步的模板没被替换，H3 会把它们当画面文字渲染）──────────
    for rx in PLACEHOLDER:
        m = rx.search(t)
        if m:
            add("error", "placeholder", "残留占位符/规则段：" + m.group(0),
                "成片提示词里不能出现 {{变量}} 或【】规则段，请替换为实际内容。", m.group(0))
    # 为什么：H3 不按条款执行；规则段只作用于润色，写进正文会被当字幕渲染。
    if mode == "final" and _RX_FINAL_RULEBLOCK.search(t):
        add("warn", "ruleblock", "正文里出现了【…】规则段",
            "H3 不按条款执行；把规则写进动作与镜头里，规则段只作用于润色，不进正文。")

    # ── 2. 结构字段（成片模式）（JS L90-L152）──────────────────────────────
    is_ref = bool(re.search(r"subject_definitions\s*:", t, re.I))   # JS L95
    if mode == "final":
        if is_ref:
            # 为什么：多参考 Ref2VA 必须按官方六段顺序补齐，缺段模型会漏掉一致性约束。
            for f in REF_SECTIONS:
                if not _RX_SECTION(f).search(t):
                    add("error", "field-" + f, "Ref2VA 缺少段 " + f + ":",
                        "多参考模式按官方六段顺序补齐；主字段是 detailed_description，"
                        "一致性段是 retention_analysis。")
            for bad, good in LEGACY_FIELDS:      # JS L102-L105
                if _RX_SECTION(bad).search(t):
                    add("warn", "ref-legacy-field", "用了旧字段名 %s:" % bad,
                        "官方 Ref2VA 已改名为 " + good + "：（见随包 h3-skill/ref-en.txt），请改名并保留原内容。")
            # 为什么：基础模式主字段是 integrated_multimodal_description，多参考模式主字段是 detailed_description，混用会让模型按错壳解析。
            if re.search(r"integrated_multimodal_description\s*:", t, re.I) and \
               not re.search(r"detailed_description\s*:", t, re.I):
                add("warn", "ref-main-field", "Ref2VA 里用了 integrated_multimodal_description:",
                    "官方对照表：基础模式主字段是 integrated_multimodal_description，"
                    "多参考模式主字段是 detailed_description。")
        else:
            # 为什么：基础 T2VA/I2VA/FL2VA/L2VA 官方壳就这三个核心字段，缺一个模型就拿不到声音/配乐指令。
            for f in BASE_SECTIONS:
                if not _RX_SECTION(f).search(t):
                    add("error", "field-" + f, "缺少字段 " + f + ":",
                        "按 MiniMax H3 官方壳补上（第2步应输出这三个核心字段）。")

        shots = _shots_for_count(t, opts)                      # JS L113（+ 引用式计数的修正）
        if not shots:
            add("error", "no-shot", "没有 [Shot N] 分镜",
                "成片提示词必须有分镜；用「生成/润色」重跑第2步。")
        else:
            # 切镜频率（JS L116-L118）：武打默认少切镜，4 镜以上基本是切太碎
            if len(shots) >= 6:
                add("error", "shot-many", "切了 %d 镜，切得太碎" % len(shots),
                    "合并成 1~3 镜：同一场打斗连续拍，用镜头运动（推近/横移/环绕/抬升）代替切镜。")
            elif len(shots) >= 4:
                add("warn", "shot-many", "切了 %d 镜，偏多" % len(shots),
                    "武打默认 1~3 镜：能一镜到底就一镜到底，景别或小角度变化改用运镜而不是切镜。")

            # 官方：第 1 镜不写时间码（JS L119-L122）
            first_seg = _first_shot1_segment(t)
            if re.search(r"\d{1,2}:\d{2}\.\d{2,3}", first_seg) or \
               re.match(r"^\s*(?:At\s+)?\d+(?:\.\d+)?\s*[-–~至]\s*\d+", first_seg, re.I):
                add("warn", "shot1-timecode", "[Shot 1] 带了时间码",
                    "官方规定第一镜不加时间码（风格与开场构图写在 [Shot 1] 之后），"
                    "时间码从 [Shot 2] 开始写切镜时间。")

            tcs = [_tc_to_sec(x) for x in _rx_timecodes(t)]     # JS L123
            tcs = [x for x in tcs if not math.isnan(x)]
            bad = 0
            for i in range(1, len(tcs)):
                if tcs[i] <= tcs[i - 1]:
                    bad += 1
            if bad:
                # 为什么：At MM:SS.mmm 是「切入该镜的时刻」，不递增就说明分镜表算错了。
                add("error", "timecode-order", "时间码有 %d 处不递增" % bad,
                    "At MM:SS.mmm 是逐镜的切镜时间，必须严格递增。")
            if duration and tcs:                                # JS L127-L131
                last = tcs[-1]
                if last >= duration:
                    add("warn", "timecode-range",
                        "最后一个切镜时间 %ss 不小于片长 %ss" % (_fix(last, 2), _js_num_str(duration)),
                        "时间码是「切入该镜的时刻」，必须落在片长以内；不要把它写成片长或结束时间。")
                elif duration >= 6 and last < duration * 0.35:
                    add("info", "timecode-range",
                        "最后一个切镜时间 %ss 只到片长 %ss 的 %d%%"
                        % (_fix(last, 2), _js_num_str(duration), _js_round(last / duration * 100)),
                        "后面还有很长一段没有切镜；确认是有意的长镜头，或补切镜。")

            # 切镜接续（JS L132-L141）：第 2 镜起开头必须先交代双方位置/朝向/姿态，
            # 否则模型会自己编位置。
            segs = _shot_segments(t, opts)
            for i, seg in enumerate(segs):
                if i == 0 or not seg.strip():
                    continue
                # 只看这一镜开头的 170 字（不要按句号切，会切碎小数时间码）
                head = seg.strip()[:170]
                has_place = re.search(
                    r"在左|在右|左侧|右侧|左边|右边|距|身位|米处|米外|面向|朝向|背对|左前|右后|"
                    r"正前方|正后方|下位|上位|" + _JSB +
                    r"(?:left|right|facing|behind|in front|across from|distance of|metres?|meters?)" + _JSB,
                    head, re.I)
                if not has_place:
                    add("warn", "cut-continuity", "第 %d 镜开头没有交代接续位置" % (i + 1),
                        "切镜的第一句要先复述双方此刻的位置、朝向与姿态（接上一镜），"
                        "否则模型会自己编位置、出现换位或对不上。", "Shot %d" % (i + 1))
                anchor = re.search(
                    r"同一张脸|同一套|同一人|还是这两人|仍是这两人|原班|不变|"
                    r"same (?:two|face|fighters|costume|weapon|clothing)|unchanged|unaltered|"
                    r"identities locked", seg, re.I)
                if not anchor:
                    add("info", "cut-identity", "第 %d 镜没有重申「还是这两人」" % (i + 1),
                        "每个切镜镜头里补一句 same two fighters, same faces, costumes and weapons — "
                        "这是切镜后换人的主要来源。", "Shot %d" % (i + 1))

        # Ref2VA 一致性（JS L143-L151）
        if is_ref:
            # JS L145-L146：match 后 toUpperCase + replace(/\s+/g,"") 去重
            subs = {re.sub(r"\s+", "", s.upper()) for s in re.findall(r"<Subject\s*\d+>", t, re.I)}
            pics = {re.sub(r"\s+", "", s.upper()) for s in re.findall(r"<Picture\s*\d+>", t, re.I)}
            if not subs:
                add("warn", "ref-subject", "没有 <Subject N> 标签",
                    "多参考模式要用 <Subject N> 锁定两位打斗者与场景。")
            if subs and pics and len(subs) != len(pics):
                add("info", "ref-tags", "标签数量不一致：Subject %d 个 / Picture %d 个" % (len(subs), len(pics)),
                    "确认每个 Subject 都指向了正确的来源图（一个 Subject 可来自多张图）。")
            if not re.search(r"<(?:Subject|Picture|Video|Audio)\s*\d+>[\s\S]*?(?:禁止|不许|不要)换", t) \
               and re.search(r"换脸|换衣", t):
                add("info", "ref-recast", "出现了「换脸/换衣」这类否定说法",
                    "官方 Ref2VA 的做法是在 retention_analysis 里正向声明保留项（fully_preserved 等），"
                    "而不是在正文写否定句。")

    # ── 2b. 招式与特效的写法（JS L154-L198 ｜ 首现写全 · 复用写短 · 不许写定义清单）──
    # ① 为什么：视频模型不做符号引用（后面写「使用招式A」会退化成普通挥砍），
    #    而且这种标签极可能被渲染成画面字幕。
    codex = re.findall(
        r"(?:^|\n)\s*(?:招式|技能|绝招|必杀|combo|skill|move)\s*[A-Za-z0-9一二三四五六七八九十]+\s*[:：]",
        t, re.I)
    if codex:
        add("warn", "move-codebook", "正文里有 %d 处「招式A：…」这种定义清单写法" % len(codex),
            "不要写招式字典：视频模型不做符号引用（后面写「使用招式A」基本会退化成普通挥砍），"
            "而且这种标签极可能被渲染成画面字幕。改成「首现写全、复用写短」：第一次用这招时把名称与效果"
            "写进那句动作里，之后再写一句「同一…再起」即可。", codex[0].strip()[:18])
    # ② 纯编号标签泄漏：编号指代会让模型把它当成画面文字，或干脆出一记普通攻击。
    if re.search(r"(?:使用|施展|发动|打出|出)\s*(?:招式|技能|绝招|combo|skill)\s*"
                 r"[A-Za-z0-9一二三四五六七八九十]+"
                 r"(?:(?<=[A-Za-z0-9])(?![A-Za-z0-9_])|(?<=[一二三四五六七八九十])(?=[A-Za-z0-9_]))",
                 t, re.I):
        add("warn", "move-label-leak", "用编号指代招式（如「使用招式A」）",
            "改成把招式名与效果写出来；编号指代会让模型把它当成画面文字，或干脆出一记普通攻击。")
    # ③ 被反复引用的招式名（引号里的名字出现 ≥2 次）：首次出现附近必须有效果描写
    quoted = {}
    for q in re.findall(r"[「『][^」』]{2,14}[」』]", t):
        quoted[q] = quoted.get(q, 0) + 1
    reused = [q for q in quoted if quoted[q] >= 2]
    if reused:
        effect = _effect_re()
        miss = []
        for q in reused:
            idx = t.find(q)
            # 把招式名本身从窗口里去掉，否则「十字剑气」里的「剑气」会把自己判成“写了效果”
            win = t[max(0, idx - 80):idx + 160].replace(q, "")
            if not effect.search(win):
                miss.append(q)
        if miss:
            add("warn", "move-first-use", "招式名 %s 被引用了多次，但第一次出现时没写效果" % miss[0],
                "招式第一次出现要把效果写全：起手 → 轨迹 → 效果实体（形状/颜色/体积/速度）→ 落点结果 → 余波；"
                "之后复用只写「同一…再起」+ 一句效果锚点，不要重复整套描写。", miss[0])
    # ④ 特效无来源/无落点：特效必须是一次连贯实体，不许出现无源特效。
    fx = re.findall(r"剑气|刀气|掌风|掌力|气劲|火焰|火舌|冰霜|雷霆|光刃|冲击波|气浪", t, re.I)
    if fx and not re.search(
            r"从|自|由|飞向|直取|扑向|打向|落向|命中|擦过|劈在|砸在|没入|迸出|涌出|炸开|荡开|推出|扫出|"
            r"toward|into|at the|hits|strikes|bursts|erupts", t, re.I):
        add("warn", "effect-unanchored", "出现 %d 处特效词，但没有来源与落点" % len(fx),
            "特效必须是一次连贯实体：谁发出、什么形态、怎么飞、落到哪里、结果如何；不许出现无源特效。", fx[0])
    # ⑤ 多参考模式：反复出现的特效更适合定义成 <Subject N>
    fx_count = {}
    for w in fx:
        k = w.lower()
        fx_count[k] = fx_count.get(k, 0) + 1
    repeated = [w for w in fx_count if fx_count[w] >= 2]
    if is_ref and repeated:
        defined = [w for w in repeated if re.search(r"<Subject\s*\d+>[^\n]{0,80}" + re.escape(w), t, re.I)]
        if not defined:
            first = fx[[s.lower() for s in fx].index(repeated[0])]
            add("info", "effect-subject", "特效「%s」反复出现，但没有定义成 <Subject N>" % first,
                "官方 Ref2VA 的 <Subject N> 允许把「可复用的可见内容」——包括特效与风格——定义一次后在各镜引用；"
                "这才是模型认得的复用机制（自己发明的「招式A」它不认）。")
    # ⑥ 按等级要求内力外放（高手不许写成平A）：等级来自角色卡，由调用方传进来
    tier = _num(opts.get("tier"))                                  # JS L190
    inner = re.compile(r"内力|内劲|真气|真罡|罡气|气劲|掌风|掌力|剑气|刀气|拳劲|气刃|气浪|冲击波|法相|气机|"
                       r"qi" + _JSB + r"|inner energy|aura|shockwave", re.I)
    if tier >= 5 and not inner.search(t):
        add("warn", "tier-inner", "本场最高等级 %s 级，但正文里看不到任何内力/气劲/剑气外放" % _js_num_str(tier),
            "5 级起必须有离体手段（掌风/剑气/罡气），7 级起气劲要成实体化形；命中瞬间也要写气劲外放。"
            "只写兵器互击＝把高手写成平A。")
    if tier >= 7 and not re.search(r"气刃|气浪|成环|尘环|丈许|护体|真罡|冲击波|法相|裂纹|崩裂|碎石|残影|"
                                   r"shockwave|aura|ring of", t, re.I):
        add("info", "tier-inner-shape", "本场最高等级 %s 级，建议写出气劲的「形」" % _js_num_str(tier),
            "7 级以上气劲要有可看见的形状与体量（丈许气刃、护体真罡、气浪成环、地面裂纹），"
            "不要只写「内力一震」。")
    if tier >= 9 and not re.search(r"天|山|江|海|星河|法相|万象|天地|崩|裂|倒卷|异象", t, re.I):
        add("info", "tier-cataclysm", "9 级绝世档没有写出天地级异象",
            "9 级要有改变场地的一击：山石崩裂、气浪成环、江河倒卷、万千法相、身影残像；"
            "否则量级撑不住「绝世」。")

    # ── 3. 节奏量化（JS L200-L208）────────────────────────────────────────
    m_first = re.search(r"\[Shot\s*1\][\s\S]*?(?=\[Shot\s*2\]|$)", t, re.I)
    first_block = m_first.group(0) if m_first else t
    # 为什么：0.3 秒内必须出第一个有效动作，空镜/对峙开场会把最贵的前 0.5 秒浪费掉。
    idle = next((w for w in IDLE_OPEN if w in first_block), None)
    if idle:
        add("warn", "start-slow", "开场出现「%s」这类空转描写" % idle,
            "0~0.5 秒就要给对手身份＋距离兵器＋光比，0.3 秒内出第一个有效动作，禁止空镜与对峙开场。")
    sentences = split_sentences(re.sub(r"\[Shot\s*\d+\][^\n]*", "", t, flags=re.I))
    sc = _shots_count(t)
    avg_beat = (len(sentences) / max(1, sc)) if sc else float(len(sentences))
    # 为什么：每镜 2~4 个有效拍最稳；太少显得空，太多一镜装不下。
    if sc and (avg_beat < 1.5 or avg_beat > 7):
        add("info", "beat-density", "平均每镜 %s 句（一拍一句）" % _fix(avg_beat, 1),
            "每镜 2~4 个有效拍最稳；太少显得空，太多一镜装不下。")
    # 为什么：慢镜与定格全片最多 1~2 处，其余必须实时速度，否则打击感全丢。
    slow = sum(1 for w in SLOWMO if re.search(w, t, re.I))
    if slow > 2:
        add("warn", "slowmo", "慢动作/定格类词出现 %d 种" % slow,
            "慢镜与定格全片最多 1~2 处，其余实时速度。")

    # ── 4. 空泛词（JS L210-L213 ｜ 为什么：空泛强度词不可拍，要换成可见的物理事实）──
    for w in EMPTY_WORDS:
        if w in t:
            add("warn", "empty-word", "空泛强度词「%s」" % w,
                "换成可见的物理事实：接触点＋材质变化＋受力方向＋位移结果＋余波。")

    # ── 5. 主语与双人同框（JS L215-L227）──────────────────────────────────
    if names:
        shots_text = re.split(r"\[Shot\s*\d+\]", t, flags=re.I)[1:] if sc else [t]
        for i, seg in enumerate(shots_text):
            if not seg.strip():
                continue
            has = sum(1 for n in names if n in seg)
            # 为什么：双人对打每一镜都要两人同框露面，否则模型会把缺席的一方编掉。
            if has < min(2, len(names)):
                add("warn", "both-fighters", "第 %d 段只提到 %d 名角色" % (i + 1, has),
                    "双人对打每一镜都要两人同框露面：谁出招、谁在同一时间格挡/闪避/受击/反打。",
                    "Shot %d" % (i + 1))
        if any(names) and re.search(r"角色\s*[AB]", t) and any(n in t for n in names):
            add("warn", "name-mix", "同时出现「角色A/角色B」与角色真名",
                "全篇只用一个固定称谓，避免模型把人认错。")
        no_subject = [s for s in split_sentences(t)
                      if len(s) > 10 and not any(n in s for n in names)
                      and not re.match(r"^(他|她|对方|其|双方|两人)", s)
                      and not re.match(r"^[A-Za-z0-9<]", s)]
        if no_subject:
            add("warn", "no-subject", "有 %d 句看不出主语" % len(no_subject),
                "每句都要点名是谁做的（谁出招、谁挨打、谁位移）。", no_subject[0][:18] + "…")

    # ── 6. 物理逻辑：腾空要有依据、位移不许瞬移（JS L229-L235 ｜ 为什么：没有借力依据
    #       人物会凭空漂在空中；瞬移违反空间连续性，观众读不出路径）────────────
    air_word = next((w for w in AIR_WORDS if w in t), None)
    if air_word and not any(w in t for w in AIR_SUPPORT):
        add("warn", "air-unsupported", "写了「%s」但全篇看不到借力依据" % air_word,
            "腾空/悬停必须给依据：蹬墙、踏檐、借栏杆、起跳、被击飞；否则画面里人物会凭空漂在空中。",
            air_word)
    for w in TELEPORT:
        if w in t:
            add("warn", "teleport", "出现「%s」" % w,
                "位移要写出路径与耗时（几步、多快、踩到哪里），不许瞬移。")

    # ── 7. 声音与材质（JS L237-L241 ｜ 为什么：材质与声音必须对应，徒手写金属声
    #       等于给模型一个错误的材质指令）────────────────────────────────────
    unarmed_only = len(weapons) > 0 and all(w == "none" for w in weapons)
    if unarmed_only:
        for w in METAL_SOUND:
            if w in t:
                add("error", "sound-metal", "双方徒手却出现「%s」" % w,
                    "没有金属兵器就不许出现剑鸣/金属声，改成闷响、衣料、呼吸、脚步。")

    # ── 7. 平台安全（JS L243-L246 ｜ 为什么：血腥/致命直述会被平台拦，改用动作戏标准词）──
    for w in BLOOD:
        if w in t:
            add("warn", "blood", "血腥/致命直述「%s」" % w,
                "用动作戏标准词替代：震退、化解、火星溅起、衣袂破损、重心崩溃、失战。")

    # ── 8. 负面词位置 & 长度（JS L248-L251 ｜ 为什么：Runway Gen-4 只认正面描述，
    #       写 no X 反而招来 X；Kling 上限 2500 字符）────────────────────────
    neg = [w for w in NEG_WORDS if w in t.lower()]
    if neg:
        add("info", "neg-in-body", "正文出现否定式措辞（%s…）" % "、".join(neg[:3]),
            "负面要求放在负面词字段里；Runway Gen-4 只能写正面（写 no X 反而招来 X）。")
    if len(t) > max_len:
        add("warn", "length", "长度 %d 字符，超过 %s" % (len(t), _js_num_str(max_len)),
            "Kling 上限 2500 字符；把过渡拍并句、删重复受力描写。")

    # ── 9. 场景与情景落实（JS L253-L254 ｜ 为什么：情景不写进正文也行，但要确保场地
    #       与运动方式体现在动作里）────────────────────────────────────────
    scenario_zh = opts.get("scenarioZh")
    if scenario_zh and scenario_zh not in t:
        add("info", "scenario-hint", "正文没有出现情景名「%s」" % scenario_zh,
            "情景不写进正文也行，但要确保场地与运动方式体现在动作里（例如追逐战不许改成原地对打）。")

    # ── 10. 对白与说话人（JS L256-L301，官方 base-en.txt 4.4 / 4.6）────────
    d_blocks = re.findall(r"<d>[\s\S]*?</d>", t)
    d_opens = len(re.findall(r"<d>", t))
    d_closes = len(re.findall(r"</d>", t))
    # 为什么：漏闭合会把后面的正文全吃进台词里。
    if d_opens != d_closes:
        add("error", "d-unclosed", "<d> 与 </d> 数量不等（%d/%d）" % (d_opens, d_closes),
            "每句台词都要闭合：<d>[Chinese] 原句</d>；漏闭合会把后面的正文吃进台词里。")
    if d_blocks:
        # 为什么：<d> 内只放语言标签与台词原文，身份与动作写在 <d> 外面。
        for i, b in enumerate(d_blocks):
            if not re.match(r"^\s*\[[A-Za-z\- ]+\]", b[3:-4]):
                add("error", "d-lang", "第 %d 句 <d> 内缺少语言标签" % (i + 1),
                    "官方要求 <d> 内只放语言标签与台词原文（<d>[Chinese] 你来了。</d>），"
                    "身份与动作写在 <d> 外面。")
        segs = re.split(r"<d>[\s\S]*?</d>", t)
        miss_speaker = 0
        for i in range(1, len(segs)):
            if not re.search(r"\(S\d+(?:\s*,\s*S\d+)*\)", segs[i - 1][-260:]):
                miss_speaker += 1
        if miss_speaker:
            add("error", "speaker-before-d", "%d 句台词前面没有说话人编号 (S1)/(S2)" % miss_speaker,
                "把身份与编号写在 <d> 外面：the grey-bearded man with a low, raspy voice (S1) says: "
                "<d>[Chinese] …</d>。")
        speakers = []
        for m in re.findall(r"\(S\d+(?:\s*,\s*S\d+)*\)", t):
            for x in m.replace("(", "").replace(")", "").split(","):
                v = x.strip()
                if v not in speakers:
                    speakers.append(v)
        # 为什么：编号从 (S1) 起、全片固定；从不发声的角色不给编号。
        if speakers and speakers[0] != "S1":
            add("error", "speaker-first", "第一个出现的说话人是 (%s)，应为 (S1)" % speakers[0],
                "编号从 (S1) 起、全片固定；从不发声的角色不给编号。")
        vo = 0
        vo_bad = 0
        for i, b in enumerate(d_blocks):
            if re.search(r"off-screen voiceover|画外音", (segs[i] if i < len(segs) else "")[-200:], re.I):
                vo += 1
                after = (segs[i + 1] if i + 1 < len(segs) else "")[:220]
                if not re.search(r"lips?\s+(?:remain|stay|are)\s+(?:completely\s+)?closed|"
                                 r"mouth\s+(?:remains|stays)\s+closed|lips?\s+do(?:es)?\s+not\s+move|"
                                 r"嘴唇[^\n]{0,8}(?:不动|闭合)", after, re.I):
                    vo_bad += 1
        if vo_bad:
            add("error", "voiceover-lips", "%d 处画外音没有紧跟\"嘴唇保持不动\"" % vo_bad,
                "官方写法：… says in an off-screen voiceover: <d>[Language] …</d> while his lips remain "
                "completely closed.")
        # 为什么：台词跨切时要在切点两侧各写一次 <scenetrans>，说明声音跨切连续。
        st_count = len(re.findall(r"<scenetrans>", t, re.I))
        if st_count % 2:
            add("warn", "scenetrans-pair", "<scenetrans> 出现 %d 次（应为偶数）" % st_count,
                "台词跨切时要在切点两侧各写一次 <scenetrans>，并说明声音跨切连续。")
        last_shot = t[max(0, t.rfind("[Shot")):]
        m_last_d = re.search(r"<d>([\s\S]*?)</d>", last_shot)
        if m_last_d:
            inner = re.sub(r"^\s*\[[A-Za-z\- ]+\]\s*", "", m_last_d.group(1)).strip()
            if not re.search(r"[。！？.!?…][\"'」』）)]?\s*$", inner):
                add("info", "cutoff-hint", "最后一句台词没有收尾标点，像是被片尾截断",
                    "被片尾截断的台词用 <cutoff> 标出（官方 4.4）。")
        ss_idx = t.find("overall_soundscape:")   # JS L292：大小写敏感、粘连写法
        if ss_idx >= 0:
            rest = t[ss_idx + 18:]
            nx = re.search(r"\n(?:integrated_multimodal_description|non_diegetic_music|subject_definitions|"
                           r"summary|retention_analysis|detailed_description|preservation_analysis):", rest)
            ss = rest[:nx.start()] if nx else rest

            # 归一化：JS 原版只保留 CJK（`replace(/[^\u4e00-\u9fff]/g,"")`），
            # 后果是**纯英文台词永远判不出重复**。用户的 924 条语料以英文为主，
            # 所以这里加一个 opt-in 开关 opts["englishAware"]=True：额外保留拉丁字母，
            # 让英文台词也能比对。默认关闭 —— 保持与 h3lint.js 逐位一致的差分结论。
            _ea = bool(opts.get("englishAware"))

            def _cjk(x):
                return re.sub(r"[^\u4e00-\u9fff]", "", str(x))

            def _norm(x):
                if not _ea:
                    return _cjk(x)
                x = re.sub(r"<d>|</d>|<\s*/\s*d\s*>", "", str(x), flags=re.I)
                x = re.sub(r"\[\s*(?:Chinese|English|Japanese|Korean)\s*\]", "", x, flags=re.I)
                return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", x).lower()

            repeated_d = sum(1 for b in d_blocks
                             if len(_norm(b)) >= 4 and _norm(ss).find(_norm(b)) >= 0)
            # 为什么：声音段只写环境声、动作声与非语言人声，重复台词会让模型把台词念两遍。
            if repeated_d:
                add("error", "sound-dialogue-repeat", "overall_soundscape 里重复了 %d 句台词" % repeated_d,
                    "声音段只写环境声、动作声与非语言人声；台词只出现在正文的 <d> 里（官方 4.6）。")

    # ── 11. 打斗连贯性（JS L303-L310 ｜ 核心打斗逻辑：不许发呆、要有因果链与反击）──
    #   JS 依赖 require("./sim3d/combat-logic.js") 的 FIGHT_LOGIC.checkPrompt；
    #   本移植版内置同源实现（_combat_logic.py），默认启用；用 opts["combatLogic"]=False
    #   或环境变量 H3LINT_DISABLE_COMBAT_LOGIC=1 可降级（此时不会有 fight-* 项）。
    use_combat = opts.get("combatLogic")
    if use_combat is None:
        use_combat = os.environ.get("H3LINT_DISABLE_COMBAT_LOGIC", "") not in ("1", "true", "True", "yes")
    if use_combat and combat_logic_available():
        fl = _COMBAT.check_prompt(t, {})
        for it in fl["issues"]:
            add(it["level"], "fight-" + it["code"], it["msg"], it.get("hint", ""))
        if mode == "final" and fl["stats"]["shots"] >= 2 and fl["stats"]["filler"] == 0:
            # 为什么：间隙动作一个都没有时，模型会把每个空档渲染成静止。
            add("error", "fight-no-filler", "② 打斗没有一次脚步或换架（间隙动作），空档会被渲染成静止",
                "把「垫步逼近／绕半步改角度／换架／拖步蓄势」写进每个空档。")
    elif use_combat and not combat_logic_available():  # pragma: no cover
        add("info", "fight-logic-unavailable",
            "打斗逻辑模块不可用，已跳过 fight-* 检查（打斗连贯性维度降级）",
            "把 _combat_logic.py 放回 wushu_bridge/lint/ 目录即可恢复与 JS 版一致的打斗检查。")

    # ── 计分（JS L312-L317）─────────────────────────────────────────────
    w = {"error": 15, "warn": 5, "info": 1}
    score = 100
    for it in items:
        score -= w.get(it["level"], 0)
    score = max(0, min(100, score))
    grade = "A" if score >= 90 else "B" if score >= 75 else "C" if score >= 60 else "D"
    return {                                                    # JS L318-L327
        "version": VERSION, "mode": mode, "score": score, "grade": grade, "items": items,
        "stats": {
            "length": len(t), "shots": len(_shots_for_count(t, opts)), "sentences": len(sentences),
            "timecodes": len(_rx_timecodes(t)),
            "errors": sum(1 for i in items if i["level"] == "error"),
            "warns": sum(1 for i in items if i["level"] == "warn"),
            "infos": sum(1 for i in items if i["level"] == "info"),
        },
    }


# ── 八维诊断（JS L331-L342 ｜ 不满意时先看是哪一维出问题，只改那一维）────────
DIMENSIONS = [
    {"key": "content", "zh": "内容与结构", "ids": [
        "placeholder", "ruleblock", "field-integrated_multimodal_description", "field-overall_soundscape",
        "field-non_diegetic_music", "field-subject_definitions", "field-summary", "field-retention_analysis",
        "field-detailed_description", "no-shot", "shot-many", "shot1-timecode", "timecode-order",
        "timecode-range", "cut-continuity", "move-codebook", "move-label-leak", "move-first-use", "length"]},
    {"key": "motion", "zh": "运动与节奏", "ids": [
        "start-slow", "beat-density", "slowmo", "empty-word", "effect-unanchored", "tier-inner",
        "tier-inner-shape", "tier-cataclysm"]},
    {"key": "audio", "zh": "音频", "ids": ["sound-metal"]},
    {"key": "physics", "zh": "物理逻辑", "ids": ["air-unsupported", "teleport"]},
    {"key": "character", "zh": "人物真实感", "ids": ["both-fighters", "name-mix", "no-subject", "cut-identity"]},
    {"key": "style", "zh": "风格一致性", "ids": [
        "ref-legacy-field", "ref-main-field", "ref-subject", "ref-tags", "ref-recast", "effect-subject",
        "scenario-hint"]},
    {"key": "dialogue", "zh": "对白与说话人", "ids": [
        "d-unclosed", "d-lang", "speaker-before-d", "speaker-first", "voiceover-lips", "scenetrans-pair",
        "cutoff-hint", "sound-dialogue-repeat"]},
    {"key": "continuity", "zh": "打斗连贯性", "ids": [
        "fight-no-filler-motion", "fight-no-causal-chain", "fight-no-counter", "fight-idle-unexplained",
        "fight-filler-sparse", "fight-no-filler"]},
]
MARK = {"error": "✗", "warn": "!", "info": "·", "ok": "✓"}    # JS L343


def diagnose(res):
    """把体检项归到八维（JS L345-L360）。"""
    items = (res or {}).get("items") or []
    groups = []
    for d in DIMENSIONS:
        hit = [i for i in items if i["id"] in d["ids"]]
        level = ("error" if any(i["level"] == "error" for i in hit)
                 else "warn" if any(i["level"] == "warn" for i in hit)
                 else "info" if hit else "ok")
        groups.append({"key": d["key"], "zh": d["zh"], "level": level, "count": len(hit), "items": hit})
    bad = [g["zh"] for g in groups if g["level"] in ("error", "warn")]
    worst = ("error" if any(g["level"] == "error" for g in groups)
             else "warn" if any(g["level"] == "warn" for g in groups)
             else "info" if any(g["level"] == "info" for g in groups) else "ok")
    verdict = ("需要修改：" + "、".join(bad)) if bad else (
        "基本可用，只剩提示项" if any(g["level"] == "info" for g in groups) else "八维全过")
    return {"groups": groups, "worst": worst, "verdict": verdict}


def diagnose_report(res):
    """八维诊断报告 → 纯文本（JS L363-L370，即 diagnoseReport）。"""
    d = diagnose(res)
    head = "八维诊断（" + d["verdict"] + "）：" + "  ".join(MARK[g["level"]] + g["zh"] for g in d["groups"])
    body = []
    for g in d["groups"]:
        if g["level"] == "ok":
            continue
        body.append(MARK[g["level"]] + " " + g["zh"] + "：" +
                    "；".join(i["msg"] for i in g["items"][:4]) +
                    ("…（共 %d 项）" % len(g["items"]) if len(g["items"]) > 4 else ""))
    return "\n".join([head] + body)


def report(res):
    """体检报告 → 纯文本（JS L373-L381）。"""
    lv = {"error": "✗ 必改", "warn": "! 建议", "info": "· 提示"}
    st = res["stats"]
    lines = [
        "提示词体检 %s（%s/100）｜长度 %d 字，分镜 %d 个，时间码 %d 处"
        % (res["grade"], res["score"], st["length"], st["shots"], st["timecodes"]),
        "必改 %d｜建议 %d｜提示 %d" % (st["errors"], st["warns"], st["infos"]),
        "八维：" + "  ".join(MARK[g["level"]] + g["zh"] for g in diagnose(res)["groups"]),
    ]
    for it in res["items"]:
        lines.append("%s  %s%s%s" % (lv.get(it["level"], it["level"]), it["msg"],
                                     ("（" + it["at"] + "）") if it["at"] else "",
                                     ("\n      → " + it["hint"]) if it["hint"] else ""))
    if not res["items"]:
        lines.append("没有发现问题：结构、节奏、措辞、声音、物理、一致性与安全各项都过关。")
    return "\n".join(lines)
