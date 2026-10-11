"""Synthetic data/software regressions; not model or video-effect evidence."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.prepare_preference_split import prepare_split
from wushu_bridge.trainer_data import partition_text_pairs, sha256


class AutoSplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="auto-split-中文 space-")
        self.folder = Path(self.temp.name)
        self.source = self.folder / "训练对.jsonl"
        self.target = self.folder / "分组.json"
        self.rows = [{"id": f"test_{i}", "group_id": f"scene_{i}",
                      "bad": f"测试坏描述 {i}", "good": f"测试好描述 {i}",
                      "pair_preference": "good", "label_type": "authoring_rubric"}
                     for i in range(20)]
        self.write()

    def tearDown(self):
        self.temp.cleanup()

    def write(self):
        self.source.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in self.rows) + "\n", encoding="utf-8")

    def test_create_cover_all_nonempty_and_bind_source(self):
        report = prepare_split(self.source, self.target)
        manifest = json.loads(self.target.read_text(encoding="utf-8"))
        self.assertEqual(report["split_group_counts"], {"train": 16, "validation": 2, "test": 2})
        self.assertEqual(manifest["generation"]["pairs_sha256"], sha256(self.source))
        checked = partition_text_pairs(self.rows, manifest)
        self.assertEqual(sorted(i for part in checked.values() for i in part), list(range(20)))

    def test_reuse_leaves_original_bytes_and_timestamp(self):
        prepare_split(self.source, self.target)
        original, timestamp = self.target.read_bytes(), self.target.stat().st_mtime_ns
        report = prepare_split(self.source, self.target)
        self.assertEqual(report["status"], "split_reused")
        self.assertEqual(self.target.read_bytes(), original)
        self.assertEqual(self.target.stat().st_mtime_ns, timestamp)

    def test_changed_source_cannot_overwrite_previous_split(self):
        prepare_split(self.source, self.target)
        original = self.target.read_bytes()
        self.rows[0]["good"] += " 新内容"
        self.write()
        with self.assertRaisesRegex(ValueError, "已有自动分组"):
            prepare_split(self.source, self.target)
        self.assertEqual(self.target.read_bytes(), original)

    def test_manual_split_is_not_overwritten(self):
        self.target.write_text('{"user_file":true}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "不会覆盖"):
            prepare_split(self.source, self.target)
        self.assertEqual(self.target.read_text(encoding="utf-8"), '{"user_file":true}')

    def test_related_group_and_shared_text_stay_together(self):
        self.rows[1]["group_id"] = self.rows[0]["group_id"]
        self.rows[2]["bad"] = self.rows[0]["bad"]
        self.write()
        prepare_split(self.source, self.target)
        parts = partition_text_pairs(self.rows, self.target)
        self.assertTrue(any({0, 1, 2} <= set(indices) for indices in parts.values()))

    def test_missing_group_has_actionable_error_and_no_output(self):
        del self.rows[2]["group_id"]
        self.write()
        with self.assertRaisesRegex(ValueError, "第 3 条.*group_id.*Agent"):
            prepare_split(self.source, self.target)
        self.assertFalse(self.target.exists())

    def test_too_few_independent_groups_cannot_be_faked_by_row_count(self):
        for row in self.rows:
            row["group_id"] = "same_source"
        self.write()
        with self.assertRaisesRegex(ValueError, "8 个可分开的"):
            prepare_split(self.source, self.target)
        self.assertFalse(self.target.exists())

    def test_numeric_data_rejected_without_output(self):
        for row in self.rows:
            row.update(bad_score=.2, good_score=.8)
        self.write()
        with self.assertRaisesRegex(ValueError, "无评分"):
            prepare_split(self.source, self.target)
        self.assertFalse(self.target.exists())

    def test_cannot_write_over_source(self):
        original = self.source.read_bytes()
        with self.assertRaisesRegex(ValueError, "源训练集同名"):
            prepare_split(self.source, self.source)
        self.assertEqual(self.source.read_bytes(), original)

    def test_cli_create_reuse_and_inspect_in_chinese_path(self):
        base = [sys.executable, "-B", "-X", "utf8", str(ROOT / "tools/trainer.py")]
        for status in ("split_created", "split_reused"):
            result = subprocess.run(base + ["auto-split", str(self.source), "--output", str(self.target)],
                                    cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["status"], status)
        result = subprocess.run(base + ["inspect", str(self.source), "--require-supervision", "preference_only",
                                       "--split-manifest", str(self.target)], cwd=ROOT,
                                capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["split_pair_counts"], {"train": 16, "val": 2, "test": 2})

    def test_cli_reports_actionable_failure_without_traceback_or_output(self):
        del self.rows[0]["group_id"]
        self.write()
        result = subprocess.run([sys.executable, "-B", "-X", "utf8", str(ROOT / "tools/trainer.py"),
                                 "auto-split", str(self.source), "--output", str(self.target)],
                                cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 2)
        self.assertIn("group_id", result.stderr)
        self.assertIn("Agent", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse(self.target.exists())


if __name__ == "__main__":
    unittest.main()
