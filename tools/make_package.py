"""把插件打包成一个可以直接丢进 custom_nodes 的 zip。

用法::

    python tools\\make_package.py
    python tools\\make_package.py --out E:\\Wushu\\ComfyUI-H3-WushuBridge-0.1.0.zip

排除：__pycache__、测试临时目录、权重与数据集、以及差分验证时复制过来的
h3lint.js 临时副本（那是用户自有代码，不随插件分发）。
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG_NAME = "ComfyUI-H3-WushuBridge"

EXCLUDE_DIRS = {
    "__pycache__", ".pytest_cache", ".git", "models", ".h3lint_js_tmp", "_work", "local", "runs", "dist", "tests",
}
EXCLUDE_EXT = {".pyc", ".pyo", ".safetensors", ".npz", ".mp4"}
EXCLUDE_FILES = {"h3lint_js_results.json", "_pytest_out.txt", "roadmap.md", "AGENTS.md"}
VIDEO_RESEARCH_PREFIXES = ("tools/h3_video_", "tools/v2_", "tools/test_h3_video_", "tools/test_v2_")
VIDEO_RESEARCH_DOCS = {"H3_VIDEO_LORA_BACKEND.md", "H3_VIDEO_TRANSPORT.md", "H3_VIDEO_DATA_PREPARATION.md"}
PRIVATE_RESEARCH_FILES = {"docs/训练流程.md", "tools/h3lint_diff.mjs", "tools/h3lint_diff_report.md"}


def should_skip(rel: str) -> bool:
    normalized = rel.replace("\\", "/")
    parts = normalized.split("/")
    if any(p in EXCLUDE_DIRS for p in parts):
        return True
    if os.path.basename(rel) in EXCLUDE_FILES:
        return True
    if (normalized.startswith(VIDEO_RESEARCH_PREFIXES) or normalized in PRIVATE_RESEARCH_FILES
            or (parts[0] == "docs" and parts[-1] in VIDEO_RESEARCH_DOCS)):
        return True
    return os.path.splitext(rel)[1].lower() in EXCLUDE_EXT


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(ROOT), f"{PKG_NAME}-0.1.0.zip"))
    args = ap.parse_args()

    count = 0
    total = 0
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for root, dirs, files in os.walk(ROOT):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for name in files:
                full = os.path.join(root, name)
                rel = os.path.relpath(full, ROOT)
                if should_skip(rel):
                    continue
                z.write(full, os.path.join(PKG_NAME, rel))
                count += 1
                total += os.path.getsize(full)

    size = os.path.getsize(args.out)
    print(f"打包完成：{args.out}")
    print(f"  文件数 {count}｜原始 {total/1024:.0f} KB｜压缩后 {size/1024:.0f} KB")
    print(f"  解压到：ComfyUI/custom_nodes/{PKG_NAME}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
