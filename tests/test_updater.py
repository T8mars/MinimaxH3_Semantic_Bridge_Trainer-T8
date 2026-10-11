"""Synthetic software fixtures only; no H3 training or video evidence."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from tools import updater as u


class UpdaterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "便携目录 with spaces"
        self.root.mkdir()
        u.write_json(self.root / "app_version.json", {"version": "0.1.0", "runtime_epoch": 1})

    def package(self, extra=None):
        files = {"WushuBridge-Trainer.exe": b"MZ-new-fixture", "tools/updater.py": b"new updater fixture",
                 "configs/preference_v1.json": b"new default", "datasets/example.jsonl": b"example",
                 "app_version.json": json.dumps({"version": "0.2.1", "runtime_epoch": 1}).encode()}
        files.update(extra or {})
        manifest = {"schema": 1, "version": "0.2.1", "runtime_epoch": 1,
                    "files": {n: {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()} for n, b in files.items()}}
        raw = json.dumps(manifest).encode()
        zpath = self.base / "update.zip"
        with zipfile.ZipFile(zpath, "w") as z:
            for name, data in files.items():
                z.writestr(name, data)
            z.writestr("UPDATE_MANIFEST.json", raw)
        meta = {"schema": 1, "repository": u.REPOSITORY, "version": "0.2.1", "runtime_epoch": 1,
                "asset": "T8-SemanticBridge-Trainer-v0.2.1-Windows-NoPython.zip", "bytes": zpath.stat().st_size,
                "sha256": u.digest(zpath), "manifest_sha256": hashlib.sha256(raw).hexdigest()}
        return zpath, meta

    def test_install_preserves_user_data_runtime_and_seed_configs(self):
        pins = {"python/python.exe": b"runtime", "models/model.safetensors": b"weight", "runs/user.pt": b"checkpoint",
                "local/launcher.settings.xml": b"settings", "configs/preference_v1.json": b"custom config",
                "datasets/example.jsonl": b"custom dataset"}
        for name, data in pins.items():
            p = self.root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        u.write_json(self.root / "PORTABLE_MANIFEST.json", {"files": {}, "launcher_provenance": {"source_sha256": {}}})
        z, m = self.package({"tools/launcher/Launcher.cs": b"source"})
        stage = u.stage_archive(self.root, z, m)
        with patch.object(u, "running_bundle_processes", return_value=[]):
            result = u.apply_update(self.root, stage["staging"])
        self.assertEqual(result["status"], "installed")
        self.assertEqual(u.installed(self.root)["version"], "0.2.1")
        for name, data in pins.items():
            self.assertEqual((self.root / name).read_bytes(), data)
        inventory = u.read_json(self.root / "PORTABLE_MANIFEST.json")
        self.assertEqual(inventory["files"]["configs/preference_v1.json"]["sha256"], hashlib.sha256(b"custom config").hexdigest())

    def test_archive_corruption_rejected(self):
        z, m = self.package()
        z.write_bytes(z.read_bytes() + b"damage")
        with self.assertRaises(ValueError):
            u.stage_archive(self.root, z, m)

    def test_traversal_private_and_case_collisions_rejected(self):
        for name in ["../evil.py", "tools/../../evil.py", "local/secret", "python/python.exe", "models/m.safetensors", "tools/roadmap.md", "tools/CON.txt", "tools/x.", "tools\\x.py"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                u.safe_name(name)
        z, m = self.package({"TOOLS/updater.py": b"collision"})
        with self.assertRaises(ValueError):
            u.verify_archive(z, m)

    def test_staged_tampering_rejected(self):
        z, m = self.package()
        staged = u.stage_archive(self.root, z, m)
        (Path(staged["staging"]) / "payload/tools/updater.py").write_bytes(b"tampered")
        with self.assertRaises(ValueError):
            u.apply_update(self.root, staged["staging"])
        self.assertEqual(u.installed(self.root)["version"], "0.1.0")

    def test_active_task_rejected(self):
        z, m = self.package()
        staged = u.stage_archive(self.root, z, m)
        with patch.object(u, "running_bundle_processes", return_value=[123]), self.assertRaises(RuntimeError):
            u.apply_update(self.root, staged["staging"])

    def test_failed_replace_rolls_back(self):
        (self.root / "WushuBridge-Trainer.exe").write_bytes(b"MZ-old-fixture")
        z, m = self.package()
        staged = u.stage_archive(self.root, z, m)
        real = u.os.replace
        def fail_once(src, dst):
            if str(src).endswith("app_version.json.t8-update-tmp"):
                raise OSError("Synthetic file lock")
            return real(src, dst)
        with patch.object(u, "running_bundle_processes", return_value=[]), patch.object(u.os, "replace", side_effect=fail_once), self.assertRaises(OSError):
            u.apply_update(self.root, staged["staging"])
        self.assertEqual((self.root / "WushuBridge-Trainer.exe").read_bytes(), b"MZ-old-fixture")
        self.assertEqual(u.installed(self.root)["version"], "0.1.0")
        self.assertFalse((self.root / "tools/updater.py").exists())
        self.assertEqual(u.read_json(self.root / "local/updater/latest_result.json")["status"], "rolled_back")

    def test_downgrade_and_runtime_change_rejected(self):
        z, m = self.package()
        u.write_json(self.root / "app_version.json", {"version": "0.9.0", "runtime_epoch": 1})
        with self.assertRaises(ValueError):
            u.stage_archive(self.root, z, m)
        m["runtime_epoch"] = 2
        with self.assertRaises(ValueError):
            u.validate_metadata(m)

    def test_github_asset_url_scoped(self):
        u.asset_url("https://github.com/" + u.REPOSITORY + "/releases/download/v0.2.1/update.json")
        for url in ["http://github.com/" + u.REPOSITORY + "/releases/download/x", "https://example.com/update", "https://github.com/other/repo/releases/download/x"]:
            with self.assertRaises(ValueError):
                u.asset_url(url)


if __name__ == "__main__":
    unittest.main()
