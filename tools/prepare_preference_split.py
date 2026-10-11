"""Generate/reuse a source-bound preference split; no H3 encoding or training."""
from __future__ import annotations

import json
from pathlib import Path

from wushu_bridge.trainer_data import (
    pair_supervision, partition_text_pairs, read_pairs, sha256,
)


def prepare_split(pairs, output, *, reuse_existing=True):
    source, target = Path(pairs).resolve(), Path(output).resolve()
    if source == target:
        raise ValueError("分组文件不能与源训练集同名，原训练集不会覆盖。")
    source_sha = sha256(source)
    rows = read_pairs(source)
    if pair_supervision(rows) != "preference_only" or any(
        row.get("label_type") != "authoring_rubric" for row in rows
    ):
        raise ValueError("自动分组仅用于自行编写的无评分偏好数据；数值评分或真实视频反馈请使用对应流程。")
    for index, row in enumerate(rows, 1):
        if not isinstance(row.get("group_id"), str) or not row["group_id"].strip():
            raise ValueError(
                f"第 {index} 条训练对缺少场景编号 group_id。请让制作数据的 Agent 补齐；"
                "同源改写/翻译使用同一编号，程序不会猜测场景来源。"
            )
    if len(rows) < 8:
        raise ValueError("自动分组至少需要 8 对训练文本和 8 个可分开的场景关联组；请先补充独立场景。")
    try:
        original = partition_text_pairs(rows, None)
    except ValueError as error:
        if "8 independent groups" not in str(error):
            raise
        raise ValueError(
            "自动分组至少需要 8 个可分开的场景关联组；同源或共享文本会合并，"
            "不能靠修改编号拆开。请补充独立场景。"
        ) from None
    indices = {"train": original["train"] + original["calibration"],
               "validation": original["val"], "test": original["test"]}
    manifest = {
        "schema": 1, "split_unit": "group_id",
        "splits": {name: sorted({rows[i]["group_id"] for i in members})
                   for name, members in indices.items()},
        "generation": {"method": "authored_preference_connected_groups_v1",
                       "seed": 1234, "pairs_sha256": source_sha},
    }
    checked = partition_text_pairs(rows, manifest)
    if sha256(source) != source_sha:
        raise ValueError("生成分组时训练集发生变化，请保存训练集后重试；本次未写入分组。")
    status = "split_created"
    if target.exists():
        if not reuse_existing:
            raise FileExistsError("分组文件已存在，请选择新文件；已有文件不会覆盖。")
        existing = json.loads(target.read_text(encoding="utf-8-sig"))
        if existing != manifest:
            raise ValueError("已有自动分组与当前数据不一致，请选择新缓存路径或手动提供分组；已有文件不会覆盖。")
        status = "split_reused"
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation: concurrent launches cannot overwrite an existing split.
        with target.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    return {
        "status": status, "split_manifest": str(target), "pairs_sha256": source_sha,
        "pairs": len(rows),
        "split_pair_counts": {key: len(value) for key, value in checked.items()},
        "split_group_counts": {key: len(value) for key, value in manifest["splits"].items()},
        "notice": "按场景编号和共享文本自动划分；不能识别未标注的同源场景。仅数据检查，不代表视频效果。",
    }
