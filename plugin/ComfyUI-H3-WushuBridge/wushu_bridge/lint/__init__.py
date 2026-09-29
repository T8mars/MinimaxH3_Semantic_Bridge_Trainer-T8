# -*- coding: utf-8 -*-
"""wushu_bridge.lint — H3 提示词体检（h3lint.js 的 Python 移植）。

用法::

    from wushu_bridge.lint import check, lint_score_01, diagnose_report

    res = check(text, {"mode": "final", "names": ["洪七公", "杨过"], "duration": 15})
    res["score"], res["grade"], res["items"]
    print(diagnose_report(res))
    s01 = lint_score_01(text, {...})      # 0.0 ~ 1.0，给训练数据打标签用

模块级导出与 JS 版（h3lint.js L383-L384）一一对应：
``VERSION / check / report / diagnose / diagnose_report(JS: diagnoseReport) /
DIMENSIONS / BASE_SECTIONS / REF_SECTIONS / LEGACY_FIELDS / EMPTY_WORDS / SLOWMO /
BLOOD / METAL_SOUND / IDLE_OPEN / AIR_WORDS / TELEPORT``。
"""

from __future__ import annotations

from .h3lint import (  # noqa: F401
    VERSION,
    check,
    report,
    diagnose,
    diagnose_report,
    DIMENSIONS,
    BASE_SECTIONS,
    REF_SECTIONS,
    LEGACY_FIELDS,
    EMPTY_WORDS,
    SLOWMO,
    BLOOD,
    METAL_SOUND,
    IDLE_OPEN,
    AIR_WORDS,
    TELEPORT,
    AIR_SUPPORT,
    NEG_WORDS,
    PLACEHOLDER,
    MARK,
    combat_logic_available,
)

__all__ = [
    "VERSION", "check", "report", "diagnose", "diagnose_report", "lint_score_01",
    "DIMENSIONS", "BASE_SECTIONS", "REF_SECTIONS", "LEGACY_FIELDS",
    "EMPTY_WORDS", "SLOWMO", "BLOOD", "METAL_SOUND", "IDLE_OPEN", "AIR_WORDS", "TELEPORT",
    "AIR_SUPPORT", "NEG_WORDS", "PLACEHOLDER", "MARK", "combat_logic_available",
]


def lint_score_01(text, opts=None):
    """把 0-100 的 score 归一化到 0.0-1.0（训练数据打标签用）。"""
    return check(text, opts)["score"] / 100.0
