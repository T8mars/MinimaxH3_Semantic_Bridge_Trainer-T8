# -*- coding: utf-8 -*-
"""build_h3lint_cases.py — 生成差分测试语料 tools/h3lint_cases.json。

为什么单独做成脚本：语料是一批长中文文本，用 Python 三引号维护比手写 JSON 的 \\n 转义
可靠得多；JSON 是最终产物（差分脚本与 pytest 都读它），本脚本只是它的可复现来源。

用法::

    python tools/build_h3lint_cases.py
"""

from __future__ import annotations

import json
import pathlib

# ── 基线：完全合格的基础 T2VA 稿（official base-en.txt 三段壳）────────────────
L1 = "integrated_multimodal_description:"
L2 = "暴雨夜的镖局前院，青石板上积水反光，暖黄灯笼从左侧打亮洪七公与杨过的轮廓。"
L3 = "洪七公在左、杨过在右，相距两米，面向彼此，各持一根木棍。"
L4 = "杨过重心压低，脚步在积水里碾半步，因此抢到中线。"
L5 = "洪七公顺势侧移，棍身压住杨过的棍梢，借这个空档垫步逼近。"
S1 = ("[Shot 1] 洪七公在左侧踏前半步，重心下沉，木棍压住杨过的棍身；"
      "杨过顺势侧移半步，格挡后立刻反打，棍梢擦过洪七公的护腕，闷响震开积水。")
S2 = ("[Shot 2] At 00:03.200 洪七公仍在左、杨过退到右侧，相距一米半，同一张脸，同一套夜行衣与木棍；"
      "杨过脚步一沉借这个空档垫步逼近，因此抢到中线，木棍扫在洪七公架起的棍身上。")
S3 = ("[Shot 3] At 00:07.500 洪七公在左、杨过在右，相距两米，还是这两人，同一套木棍；"
      "洪七公借着杨过收招的间隙拖步蓄势，木棍由上向下砸在杨过架起的棍身上，火星溅开。")
SND = "overall_soundscape: 雨声、衣料摩擦声、木棍相击的闷响与脚步踩水声。"
MUS = "non_diegetic_music: 低音鼓点从第二秒进入，随棍击的节奏收紧，结尾收在一记长音。"
OK1 = "\n".join([L1, L2, L3, L4, L5, S1, S2, S3, SND, MUS]) + "\n"

# ── 基线：完全合格的多参考 Ref2VA 稿（official ref-en.txt 六段壳）────────────
R1 = "subject_definitions:"
R2 = "<Subject 1> is 洪七公, the grey-bearded master from <Picture 1>, wearing a dark cotton robe."
R3 = "<Subject 2> is 杨过, the one-armed swordsman from <Picture 2>, holding a wooden staff."
R4 = "summary: 洪七公与杨过在镖局前院的雨夜里交手，洪七公压住中线，杨过借这个空档反打。"
R5 = ("retention_analysis: <Subject 1> and <Subject 2> are fully_preserved in every shot; "
      "faces, robes and the wooden staff are unchanged.")
R6 = "detailed_description: 洪七公在左、杨过在右，相距两米，面向彼此，各持木棍。"
RS1 = ("[Shot 1] 洪七公在左侧踏前半步，重心下沉，木棍压住杨过的棍身；"
       "杨过顺势侧移半步，格挡后立刻反打，棍梢擦过护腕。")
RS2 = ("[Shot 2] At 00:03.200 洪七公仍在左、杨过退到右侧，相距一米半，同一张脸，同一套夜行衣；"
       "杨过脚步一沉借这个空档垫步逼近，因此抢到中线，木棍扫在洪七公架起的棍身上。")
RSND = "overall_soundscape: 雨声、衣料摩擦声、木棍相击的闷响与脚步踩水声。"
RMUS = "non_diegetic_music: 低音鼓点从第二秒进入，随棍击的节奏收紧。"
OK2 = "\n".join([R1, R2, R3, R4, R5, R6, RS1, RS2, RSND, RMUS]) + "\n"

# ── 基线：对白稿（<d> 在最后一镜里，便于验证 cutoff / scenetrans / 声音段重复）──
D1 = "integrated_multimodal_description:"
D2 = "洪七公与杨过在雨里对招，木棍相击。"
DS1 = ("[Shot 1] 洪七公在左侧垫步逼近，重心下沉，木棍压住杨过的棍身；"
       "杨过顺势格挡，脚步后撤。木棍在雨里碰出短促的清响，洪七公与杨过各自压低身形，把棍端指向对方的肩线。")
DS2 = ("[Shot 2] At 00:03.200 洪七公仍在左、杨过退到右侧，相距一米半，同一张脸；"
       "the grey-bearded man with a low, raspy voice (S1) says: <d>[Chinese] 你先出手。</d> "
       "杨过借这个空档反打，因此抢到中线，木棍扫在洪七公架起的棍身上。")
DSND = "overall_soundscape: 雨声、脚步踩水声、木棍的闷响。"
DMUS = "non_diegetic_music: 低音鼓点慢慢收紧。"
OK3 = "\n".join([D1, D2, DS1, DS2, DSND, DMUS]) + "\n"

BASE = {"mode": "final", "names": ["洪七公", "杨过"], "weapons": ["bang"], "duration": 15}
REF = {"mode": "final", "names": ["洪七公", "杨过"], "weapons": ["bang"], "duration": 8}
DLG = {"mode": "final", "names": ["洪七公", "杨过"], "weapons": ["bang"], "duration": 8}

CASES = []


def C(cid, note, text, base=BASE, **opts):
    o = dict(base)
    o.update(opts)
    CASES.append({"id": cid, "note": note, "text": text, "opts": o})


# ── 1. 合格稿 ───────────────────────────────────────────────────────────────
C("ok-base-t2va", "合格文生视频稿：三段壳齐全、3 镜、时间码规范、双人同框", OK1)
C("ok-ref2va", "合格多参考图稿：六段壳齐全、<Subject N>/<Picture N> 对称", OK2, base=REF)
C("ok-dialogue", "合格对白稿：<d> 有语言标签、说话人编号在 <d> 外", OK3, base=DLG)

# ── 2. 结构字段 ─────────────────────────────────────────────────────────────
C("missing-fields", "缺 overall_soundscape / non_diegetic_music 两个基础字段",
  OK1.replace(SND + "\n", "").replace(MUS + "\n", ""))
C("missing-main-field", "缺基础模式主字段 integrated_multimodal_description",
  OK1.replace(L1 + "\n", ""))
C("design-mode", "design 模式（第1步设计稿）：不查官方三段壳，其余检查照旧",
  OK1.replace(L1 + "\n", "").replace(SND + "\n", "").replace(MUS + "\n", ""), mode="design")
C("no-shot", "通篇没有 [Shot N] 分镜", OK1.replace("[Shot 1] ", "").replace("[Shot 2] ", "").replace("[Shot 3] ", ""))
C("shot-many-error", "6 镜以上：切得太碎（error）",
  OK1.replace(S3, S3 + "\n[Shot 4] At 00:09.000 洪七公在左、杨过在右，相距两米，同一张脸，木棍相击。"
              + "\n[Shot 5] At 00:10.500 洪七公在左、杨过在右，相距一米，同一张脸，木棍相击。"
              + "\n[Shot 6] At 00:12.000 洪七公在左、杨过在右，相距一米，同一张脸，木棍相击。"))
C("shot-many-warn", "4 镜：偏多（warn）",
  OK1.replace(S3, S3 + "\n[Shot 4] At 00:09.000 洪七公在左、杨过在右，相距两米，同一张脸，木棍相击。"))
C("shot1-timecode", "[Shot 1] 带了时间码（官方规定第一镜不写）",
  OK1.replace("[Shot 1] 洪七公", "[Shot 1] At 00:00.000 洪七公"))

# ── 3. 时间码 ───────────────────────────────────────────────────────────────
C("timecode-order", "第二镜时间码大于第三镜：不严格递增",
  OK1.replace("At 00:03.200", "At 00:09.200"))
C("timecode-range-over", "最后一个切镜时间不小于片长", OK1, duration=5)
C("timecode-range-info", "最后一个切镜时间不到片长的 35%（info）", OK1, duration=30)

# ── 4. 节奏与词表 ───────────────────────────────────────────────────────────
C("slowmo", "慢动作/定格类词出现 4 种（>2）",
  OK1.replace(L4, L4.rstrip("。") + "，慢动作、慢镜、定格、子弹时间轮番上。"))
C("start-slow", "开场 0~0.5 秒写了对峙空转",
  OK1.replace("洪七公在左侧踏前半步", "洪七公与杨过在雨里对峙"))
C("beat-density", "每镜平均句数不足 1.5（一镜装不满）",
  "\n".join([L1, S1, S2, S3, SND, MUS]) + "\n")
C("empty-words", "空泛强度词：极快 / 激烈 / 爆发",
  OK1.replace(L4, "杨过重心压低，脚步极快，出手激烈，一阵爆发之后抢到中线。"))
C("neg-words", "正文出现否定式措辞（不要/禁止）",
  OK1.replace(L4, "杨过重心压低，不要站着等，禁止发呆，脚步一沉抢到中线。"))
C("blood", "血腥/致命直述：喷血", OK1.replace("闷响震开积水", "闷响震开积水，杨过喷血"))
C("length", "长度超过 maxLen 红线", OK1, maxLen=200)
C("scenario-hint", "正文没有出现情景名（追逐战）", OK1, scenarioZh="追逐战")

# ── 5. 占位符与规则段 ───────────────────────────────────────────────────────
C("placeholder-var", "未替换的 {{变量}} 占位符",
  OK1.replace(SND, "overall_soundscape: {{声音清单}}"))
C("placeholder-todo", "残留 TODO 标记", OK1.replace(L4, "TODO 补杨过的受力描写"))
C("placeholder-ruleblock", "残留【特效等级】规则段（占位符＋规则段双报）",
  OK1.replace(L3, L3.rstrip("。") + "【特效等级】。"))
C("ruleblock-only", "正文里出现【镜头规则】这类规则段（仅 ruleblock）",
  OK1.replace(L3, L3.rstrip("。") + "【镜头规则】。"))

# ── 6. 招式与特效写法 ───────────────────────────────────────────────────────
C("move-codebook", "定义清单式写法「招式A：…」",
  OK1.replace(L4, "招式A：横扫千军，反手一压。"))
C("move-label-leak", "用编号指代招式（使用招式A）",
  OK1.replace("杨过顺势侧移半步", "杨过使用招式A后侧移半步"))
C("move-first-use", "招式名被引用两次但首现没写效果",
  OK1.replace(L4, "杨过重心压低，脚步在积水里碾半步，「旋风棍」斜劈而下，因此抢到中线。")
     .replace("洪七公借着杨过收招的间隙拖步蓄势", "洪七公借着杨过收招的间隙拖步蓄势，「旋风棍」再起")
     .replace("，火星溅开", "，闷响震开积水"))
_EFFECT_TEXT = "\n".join([
    "integrated_multimodal_description:",
    "洪七公与杨过在雨里交手，杨过脚步一转，掌风横扫，洪七公顺势抬臂去接，因此两人各退半步。",
    "[Shot 1] 洪七公在左侧半步，杨过在右侧半步，相距两米，木棍相击。",
    "overall_soundscape: 雨声、脚步踩水声、木棍的闷响。",
    "non_diegetic_music: 低音鼓点慢慢收紧。",
]) + "\n"
C("effect-unanchored", "出现特效词但没有来源与落点", _EFFECT_TEXT)
C("effect-subject", "Ref2VA 里反复出现的特效没有定义成 <Subject N>",
  OK2.replace("木棍压住杨过的棍身", "剑气压住杨过的棍身").replace("棍梢擦过护腕", "剑气擦过护腕"),
  base=REF)

# ── 7. 等级（内力外放）──────────────────────────────────────────────────────
C("tier-inner", "5 级高手却看不到任何内力外放", OK1, tier=5)
C("tier-inner-shape", "7 级只有「内力一震」没有气劲的形",
  OK1.replace("棍身压住杨过的棍梢", "掌风压住杨过的棍梢"), tier=7)
C("tier-cataclysm", "9 级绝世档没有天地级异象",
  OK1.replace("棍身压住杨过的棍梢", "掌风压住杨过的棍梢，气浪成环"), tier=9)

# ── 8. 物理逻辑 ─────────────────────────────────────────────────────────────
C("air-unsupported", "写了腾空但全篇没有借力依据",
  OK1.replace("杨过重心压低", "杨过腾空而起，重心压低").replace("踏前半步", "上前半步"))
C("teleport", "出现瞬移 / 闪现",
  OK1.replace("杨过重心压低", "杨过瞬移半步，随即闪现到中线"))

# ── 9. 声音、主语与人物 ─────────────────────────────────────────────────────
C("sound-metal", "双方徒手却出现剑鸣", OK1.replace("闷响震开积水", "剑鸣震开积水"),
  weapons=["none", "none"])
C("both-fighters", "第 2 镜只提到一名角色",
  OK1.replace("洪七公仍在左、杨过退到右侧", "杨过退到右侧")
     .replace("木棍扫在洪七公架起的棍身上", "木棍扫在对方架起的棍身上"))
C("name-mix", "同时出现「角色A」与角色真名",
  OK1.replace(L3, L3.rstrip("。") + "角色A 先动。"))
C("no-subject", "有长句看不出主语",
  OK1.replace(L2, L2 + "雨水顺着屋檐落下，地面反光刺眼。"))
C("cut-continuity", "第 2/3 镜开头没有交代接续位置",
  OK1.replace("洪七公仍在左、杨过退到右侧，相距一米半，同一张脸，同一套夜行衣与木棍；",
              "杨过脚步一沉，同一张脸，同一套夜行衣与木棍；")
     .replace("洪七公在左、杨过在右，相距两米，还是这两人，同一套木棍；",
              "洪七公还是这两人，同一套木棍；"))
C("cut-identity", "切镜后没有重申「还是这两人」",
  OK1.replace("同一张脸，同一套夜行衣与木棍；", "夜行衣已经湿透；")
     .replace("还是这两人，同一套木棍；", "雨水沿着棍身往下淌；"))

# ── 10. 打斗连贯性（FIGHT_LOGIC）────────────────────────────────────────────
C("fight-no-counter", "全文没有任何格挡/闪避/反击的回应动作",
  OK1.replace("格挡后立刻反打", "接住后立刻再攻"))
# 抽掉全部因果连接词（因此／借这个／顺势／立刻／抢到），但保留间隙动作与反击
_NOCAUSE = (OK1.replace("因此抢到中线", "占住中线").replace("借这个空档", "这个空档")
               .replace("顺势", "跟着").replace("立刻", "马上"))
C("fight-no-causal", "因果连接词不足 2 个", _NOCAUSE)
C("fight-no-filler", "没有任何间隙动作（不会走步换架）",
  OK1.replace("垫步逼近", "上前压近").replace("垫步", "上前").replace("重心", "身形")
     .replace("脚步", "步子").replace("碾半步", "踩半步").replace("侧移", "挪身")
     .replace("拖步蓄势", "拖身蓄劲"))
C("fight-idle", "出现静止却没有写原因与动作",
  _NOCAUSE.replace(L3, L3.rstrip("。") + "两人纹丝不动。"))
C("fight-filler-sparse", "4 镜只有三处间隙动作（每镜不到一处）",
  OK1.replace(S3, S3 + "\n[Shot 4] At 00:09.000 洪七公在左、杨过在右，相距两米，同一张脸，木棍相击。")
     .replace(L4, "杨过身形压低，步子一沉，因此抢到中线。")
     .replace("重心下沉", "身形下沉")
     .replace("侧移半步", "挪身半步")
     .replace("脚步一沉", "步子一沉")
     .replace("与脚步踩水声", "与踩水声")
     .replace("拖步蓄势", "拖身蓄劲"))

# ── 11. 对白与说话人 ────────────────────────────────────────────────────────
C("d-unclosed", "<d> 没有闭合",
  OK3.replace("</d> ", ""), base=DLG)
C("d-lang", "<d> 内缺少语言标签",
  OK3.replace("<d>[Chinese] 你先出手。</d>", "<d>你先出手。</d>"), base=DLG)
C("speaker-before-d", "台词前面没有说话人编号 (S1)",
  OK3.replace("(S1) says:", "says:"), base=DLG)
C("speaker-first", "第一个出现的说话人是 (S2)",
  OK3.replace("(S1) says:", "(S2) says:"), base=DLG)
C("voiceover-lips", "画外音没有紧跟「嘴唇保持不动」",
  OK3.replace("says:", "says in an off-screen voiceover:"), base=DLG)
C("voiceover-ok", "画外音写法正确（应当没有对白相关 error）",
  OK3.replace("says:", "says in an off-screen voiceover:").replace(
      "杨过借这个空档反打", "while his lips remain completely closed 杨过借这个空档反打"), base=DLG)
C("scenetrans-pair", "<scenetrans> 出现奇数次",
  OK3.replace("> 杨过借这个空档反打", "> <scenetrans> 杨过借这个空档反打"), base=DLG)
C("cutoff-hint", "最后一句台词没有收尾标点，像被片尾截断",
  OK3.replace("<d>[Chinese] 你先出手。</d>", "<d>[Chinese] 你先出手</d>"), base=DLG)
C("sound-dialogue-repeat", "overall_soundscape 里重复了台词",
  OK3.replace(DSND, "overall_soundscape: 雨声里有人在说「你先出手」，脚步踩水声、木棍的闷响。"), base=DLG)

# ── 12. 多参考 Ref2VA 一致性 ────────────────────────────────────────────────
C("ref-missing-sections", "Ref2VA 缺 summary / retention_analysis",
  "\n".join([R1, R2, R3, R6, RS1, RS2, RSND, RMUS]) + "\n", base=REF)
C("ref-legacy-field", "用了旧字段名 preservation_analysis",
  OK2.replace(R5, "preservation_analysis: 全部保留。"), base=REF)
C("ref-main-field", "Ref2VA 里用了基础模式主字段",
  OK2.replace(R6, "integrated_multimodal_description: 洪七公在左、杨过在右，相距两米，面向彼此，各持木棍。"),
  base=REF)
C("ref-subject-missing", "没有 <Subject N> 标签",
  OK2.replace(R2, "洪七公, the grey-bearded master, taken from <Picture 1>.")
     .replace(R3, "杨过, the one-armed swordsman, taken from <Picture 2>.")
     .replace("retention_analysis: <Subject 1> and <Subject 2> are", "retention_analysis: Both are"),
  base=REF)
C("ref-tags-mismatch", "Subject 与 Picture 标签数量不一致",
  OK2.replace("detailed_description: 洪七公", "detailed_description: <Picture 3> 洪七公"), base=REF)
C("ref-recast", "出现「换脸」这类否定说法",
  OK2.replace(R5, R5.rstrip(".") + " 绝不换脸。"), base=REF)


def main():
    out = pathlib.Path(__file__).with_name("h3lint_cases.json")
    out.write_text(json.dumps({"version": 1, "count": len(CASES), "cases": CASES},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote %s (%d cases)" % (out, len(CASES)))


if __name__ == "__main__":
    main()
