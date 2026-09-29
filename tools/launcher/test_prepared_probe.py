"""CPU-only synthetic tests for the launcher's prepared-data reuse guard."""
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).with_name("Launcher.cs").read_text(encoding="utf-8-sig")
PROBE = re.search(r'const string PreparedProbe = @"(.*?)";', SOURCE, re.S).group(1)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PreparedProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prepared = self.root / "prepared"
        self.prepared.mkdir()
        self.sources = {
            "pairs": self.root / "source.jsonl", "dataset": self.root / "source.npz",
            "metadata": self.root / "source.json", "encoding_receipt": self.root / "source.npz.receipt.json",
            "encoder": self.root / "encoder.test", "split": self.root / "split.json",
        }
        for name, path in self.sources.items():
            path.write_text(name, encoding="utf-8")
        self.sources["metadata"].write_text(json.dumps({"encoder": json.dumps({"synthetic": True})}), encoding="utf-8")
        self.module = types.ModuleType("tools.prepare_aligned_dataset")
        self.module.__file__ = str(self.root / "tools" / "prepare.synthetic.py")
        Path(self.module.__file__).parent.mkdir()
        Path(self.module.__file__).write_text("# synthetic test fixture", encoding="utf-8")
        self.module.sha256 = sha256
        self.module.METHOD = "synthetic-test-only"
        self.module.TRANSPORT_METHOD = "synthetic-transport-test-only"
        self.module.tokenizer_provenance = lambda *args: {"synthetic": True}
        self.report = {
            "status": "completed", "method": self.module.METHOD,
            "source_sha256": {name: sha256(path) for name, path in self.sources.items()},
            "script_sha256": sha256(self.module.__file__), "tokenizer_provenance": {"synthetic": True}, "artifacts": {},
        }
        for name in ("pairs.npz", "pairs.json", "pairs.jsonl", "pairs.npz.receipt.json", "split_manifest.json"):
            artifact = self.prepared / name
            artifact.write_text(name, encoding="utf-8")
            self.report["artifacts"][name] = {"sha256": sha256(artifact), "bytes": artifact.stat().st_size}

    def run_probe(self, expected=None):
        (self.prepared / "preparation_receipt.json").write_text(json.dumps(self.report), encoding="utf-8")
        parent = types.ModuleType("tools")
        parent.prepare_aligned_dataset = self.module
        argv = ["probe", str(self.prepared)] + [str(self.sources[k]) for k in ("dataset", "pairs", "encoder", "split")] + [str(self.root), expected or self.module.METHOD]
        with patch.dict(sys.modules, {"tools": parent, "tools.prepare_aligned_dataset": self.module}), patch.object(sys, "argv", argv):
            exec(compile(PROBE, "Launcher.PreparedProbe", "exec"), {})

    def test_unchanged_receipt_passes(self):
        self.run_probe()

    def test_changed_source_fails(self):
        self.sources["split"].write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Prepared source changed: split"):
            self.run_probe()

    def test_changed_artifact_fails(self):
        (self.prepared / "pairs.npz").write_text("tampered", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Prepared artifact changed"):
            self.run_probe()

    def test_missing_artifact_manifest_entry_fails(self):
        del self.report["artifacts"]["split_manifest.json"]
        with self.assertRaisesRegex(ValueError, "artifact manifest"):
            self.run_probe()

    def test_changed_preparation_code_fails(self):
        Path(self.module.__file__).write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Preparation code changed"):
            self.run_probe()

    def test_changed_tokenizer_fails(self):
        self.module.tokenizer_provenance = lambda *args: {"changed": True}
        with self.assertRaisesRegex(ValueError, "Tokenizer provenance changed"):
            self.run_probe()

    def test_prepared_method_cannot_be_reused_in_the_other_gui_mode(self):
        with self.assertRaisesRegex(ValueError, "selected mode"):
            self.run_probe(self.module.TRANSPORT_METHOD)

    def test_transport_preparation_checks_alignment_source_hash(self):
        source = self.root / "wushu_bridge" / "token_alignment.py"
        source.parent.mkdir()
        source.write_text("# synthetic transport", encoding="utf-8")
        self.report["method"] = self.module.TRANSPORT_METHOD
        self.report["transport_policy"] = {"token_alignment_source_sha256": sha256(source)}
        self.run_probe(self.module.TRANSPORT_METHOD)
        source.write_text("# changed transport", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Transport code changed"):
            self.run_probe(self.module.TRANSPORT_METHOD)


if __name__ == "__main__":
    unittest.main()
