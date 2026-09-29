# -*- coding: utf-8 -*-
"""_combat_logic.py — ``sim3d/combat-logic.js`` 中 checkPrompt 部分的 Python 移植。

移植来源：用户提供的 ``sim3d/combat-logic.js``（264 行，UMD，
VERSION=fight-logic-0.2）。h3lint.js 第 16 行 ``require("./sim3d/combat-logic.js")``
拿到的 FIGHT_LOGIC，在 h3lint.js 的 L303-L310 只用到 ``checkPrompt``，
所以这里只忠实移植 ``checkPrompt`` 及其词表（CAUSE/FILLER/IDLE/COUNTER），
以及它 return 的 ``{ok, issues, stats}`` 结构；库表（FILLERS/CORE_RULES/
WINDOWS/expandBeats/logicBlock/checkEvents）与体检无关，不移植。

对应关系（JS 行号）：
    L20          VERSION
    L150-L158    CAUSE_WORDS / FILLER_WORDS / IDLE_WORDS / COUNTER_WORDS
    L211-L225    checkPrompt

注意：JS 的 ``hits()`` 用 ``t.indexOf(w) >= 0``——**大小写敏感**，本文件保持一致。
"""

from __future__ import annotations

import re

__all__ = ["VERSION", "CAUSE_WORDS", "FILLER_WORDS", "IDLE_WORDS", "COUNTER_WORDS", "check_prompt"]

VERSION = "fight-logic-0.2"  # ← JS L20

CAUSE_WORDS = [  # JS L150-L152
    "因此", "于是", "借这个", "趁", "因为", "被压", "失衡", "露出的空档", "抢到", "反过来",
    "顺势", "接着", "随即", "立刻", "紧接", "不等", "还没收完", "反弹", "回弹", "受力",
    "so ", "therefore", "because", "as a result", "which lets", "off the rebound", "off-balance",
    "opens the gap", "takes the gap", "riding the", "following the",
]
FILLER_WORDS = [  # JS L153-L154
    "垫步", "绕", "换架", "撤步", "拖步", "碎步", "碾", "踏步", "重心", "护面", "抬手", "蓄势",
    "调整", "侧移", "绕行", "逼近", "站位", "脚步",
    "footwork", "steps in", "steps back", "pads", "circles", "sidestep", "switch stance",
    "stance", "shifts weight", "hand up", "cocks", "drags", "closes the distance", "retreats", "adjusts",
]
IDLE_WORDS = [  # JS L155-L156
    "静止", "站定", "不动", "僵持", "对峙", "发呆", "停顿", "停手", "凝立", "纹丝不动",
    "stands still", "standoff", "motionless", "frozen", "freezes", "pauses", "stares",
]
COUNTER_WORDS = [  # JS L157-L158
    "格挡", "架住", "挡开", "卸力", "闪", "侧身避", "后仰", "退半步", "反击", "反打", "回敬",
    "抢位", "压制", "反压",
    "blocks", "parries", "deflects", "slips", "dodges", "sidesteps", "counters", "answers with",
    "presses", "re-presses", "absorbs",
]

_RX_SHOT = re.compile(r"\[Shot\s*\d+\]", re.I)  # JS L217


def check_prompt(text, opt=None):
    """打斗连贯性判定（JS L211-L225）。

    为什么这五条：内核里打得很密，但**证据与模板把「动作之间的东西」丢掉了**，
    视频模型只能把空档脑补成「站着发呆」。所以要求：有间隙动作、有因果链、有反击、
    静止必须有解释、每镜至少一处间隙动作。
    """
    _ = opt or {}
    t = "" if text is None else str(text)
    issues = []

    def hits(arr):
        return [w for w in arr if t.find(w) >= 0]

    cause = hits(CAUSE_WORDS)
    filler = hits(FILLER_WORDS)
    idle = hits(IDLE_WORDS)
    counter = hits(COUNTER_WORDS)
    shots = len(_RX_SHOT.findall(t))

    if not filler:  # JS L219
        issues.append({"level": "error", "code": "no-filler-motion",
                       "msg": "正文里没有任何间隙动作（垫步／绕步／换架／撤步…）",
                       "hint": "把空档写成具体动作，否则模型会把空档渲染成静止。"})
    if len(cause) < 2:  # JS L220
        issues.append({"level": "error", "code": "no-causal-chain",
                       "msg": "因果连接词只有 %d 个，看不出上一拍的结果与下一拍的起因" % len(cause),
                       "hint": "用「借这个空档／因此／顺势」把拍与拍连起来。"})
    if not counter:  # JS L221
        issues.append({"level": "error", "code": "no-counter",
                       "msg": "全文没有任何格挡／闪避／反击的回应动作",
                       "hint": "每轮交锋都要有回应：格挡、闪避落位或硬吃，并给出受力结果。"})
    if idle and not cause:  # JS L222
        issues.append({"level": "error", "code": "idle-unexplained",
                       "msg": "出现「%s」却没有写原因与动作" % "、".join(idle[:3]),
                       "hint": "要么删掉静止，要么写明原因（受击硬直／被压住兵器／换架）并把它写成动作。"})
    if shots >= 3 and filler and len(filler) < shots:  # JS L223
        issues.append({"level": "warn", "code": "filler-sparse",
                       "msg": "间隙动作 %d 处、镜头 %d 个，平均每镜不到一处" % (len(filler), shots),
                       "hint": "每个切镜后的第一句都可以带上脚步或换架。"})
    return {"ok": not any(i["level"] == "error" for i in issues), "issues": issues,
            "stats": {"cause": len(cause), "filler": len(filler), "idle": len(idle),
                      "counter": len(counter), "shots": shots}}
