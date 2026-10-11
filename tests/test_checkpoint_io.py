"""Software regressions, including real Windows file-sharing locks."""
import ctypes
import io as streams
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import torch
from wushu_bridge import checkpoint_io as io


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "bridge_last.pt"

    def test_150_successive_saves_reload_each_epoch(self):
        for epoch in range(1, 151):
            io.save_checkpoint(self.path, {"epoch": epoch, "tensor": torch.arange(32)})
            state = torch.load(self.path, weights_only=True)
            self.assertEqual(state["epoch"], epoch)
            self.assertTrue(torch.equal(state["tensor"], torch.arange(32)))
        self.assertFalse(list(self.root.glob("*.pending")))

    def test_transient_access_denied_retries_same_complete_file(self):
        real = os.replace
        errors = []
        for code in (5, 32, 33):
            exc = PermissionError("simulated Windows sharing lock")
            exc.winerror = code
            errors.append(exc)
        attempts = []
        def replace(src, dst):
            attempts.append(src)
            if errors:
                raise errors.pop(0)
            return real(src, dst)
        with patch.object(io.os, "replace", side_effect=replace), patch.object(io.time, "sleep"):
            io.save_checkpoint(self.path, {"epoch": 124})
        self.assertEqual(len(attempts), 4)
        self.assertEqual(len(set(attempts)), 1)
        self.assertEqual(torch.load(self.path, weights_only=True)["epoch"], 124)

    def test_permanent_error_preserves_old_and_complete_pending(self):
        io.save_checkpoint(self.path, {"epoch": 123})
        original = self.path.read_bytes()
        with patch.object(io.os, "replace", side_effect=OSError("permanent filesystem failure")):
            with self.assertRaisesRegex(OSError, "完整状态保留"):
                io.save_checkpoint(self.path, {"epoch": 124})
        self.assertEqual(self.path.read_bytes(), original)
        pending = list(self.root.glob("*.pending"))
        self.assertEqual(len(pending), 1)
        self.assertEqual(torch.load(pending[0], weights_only=True)["epoch"], 124)

    def test_partial_serialization_never_replaces_or_leaves_bad_pending(self):
        io.save_checkpoint(self.path, {"epoch": 3})
        original = self.path.read_bytes()
        with patch.object(torch, "save", side_effect=OSError("disk write failure")):
            with self.assertRaises(OSError):
                io.save_checkpoint(self.path, {"epoch": 4})
        self.assertEqual(self.path.read_bytes(), original)
        self.assertFalse(list(self.root.glob("*.pending")))

    def test_english_windows_console_cannot_interrupt_retry(self):
        raw = streams.BytesIO()
        console = streams.TextIOWrapper(raw, encoding='cp1252', errors='strict')
        real = os.replace
        attempts = []
        def replace(a, b):
            attempts.append(a)
            if len(attempts) == 1:
                exc = PermissionError('sharing lock'); exc.winerror = 5
                raise exc
            return real(a, b)
        with patch('sys.stdout', console), patch.object(io.os, 'replace', side_effect=replace), patch.object(io.time, 'sleep'):
            io.save_checkpoint(self.path, {'epoch': 124})
        console.flush()
        self.assertIn(b'Checkpoint temporarily busy', raw.getvalue())
        console.detach()
        self.assertEqual(torch.load(self.path, weights_only=True)['epoch'], 124)

    def test_run_lock_rejects_second_writer_then_releases(self):
        with io.training_lock(self.root):
            with self.assertRaisesRegex(RuntimeError, "另一个训练进程"):
                with io.training_lock(self.root):
                    self.fail("Concurrent writer entered")
        with io.training_lock(self.root):
            pass

    def test_run_lock_released_by_os_after_process_termination(self):
        code = "import sys,time;sys.path.insert(0,sys.argv[1]);from wushu_bridge.checkpoint_io import training_lock;\nwith training_lock(sys.argv[2]):\n print('locked',flush=True);time.sleep(30)"
        child = subprocess.Popen([sys.executable, '-B', '-c', code, str(Path(io.__file__).resolve().parents[1]), str(self.root)],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            self.assertEqual(child.stdout.readline().strip(), 'locked')
            with self.assertRaisesRegex(RuntimeError, "另一个训练进程"):
                with io.training_lock(self.root):
                    self.fail("Concurrent process entered")
            child.terminate()
            child.communicate(timeout=15)
            with io.training_lock(self.root):
                pass
        finally:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=15)

    @unittest.skipUnless(os.name == "nt", "Real Windows API sharing test")
    def test_real_windows_lock_then_release_succeeds(self):
        io.save_checkpoint(self.path, {"epoch": 123})
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                                      ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
        kernel.CreateFileW.restype = ctypes.c_void_p
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.CreateFileW(str(self.path), 0x80000000, 3, None, 3, 0, None)
        self.assertNotEqual(handle, ctypes.c_void_p(-1).value)
        released = threading.Event()
        def release():
            kernel.CloseHandle(handle)
            released.set()
        timer = threading.Timer(.6, release)
        timer.start()
        started = time.monotonic()
        try:
            io.save_checkpoint(self.path, {"epoch": 124})
        finally:
            timer.join()
        self.assertTrue(released.is_set())
        self.assertGreater(time.monotonic() - started, .5)
        self.assertEqual(torch.load(self.path, weights_only=True)["epoch"], 124)

    @unittest.skipUnless(os.name == "nt", "Real Windows read-only test")
    def test_readonly_denial_is_bounded_and_old_checkpoint_preserved(self):
        io.save_checkpoint(self.path, {"epoch": 123})
        original = self.path.read_bytes()
        self.path.chmod(0o444)
        real = io.replace_with_retry
        try:
            with patch.object(io, "replace_with_retry", side_effect=lambda a, b: real(a, b, timeout=.1)):
                with self.assertRaisesRegex(OSError, "上一份断点未改动"):
                    io.save_checkpoint(self.path, {"epoch": 124})
            self.assertEqual(self.path.read_bytes(), original)
        finally:
            self.path.chmod(0o666)


if __name__ == "__main__":
    unittest.main()
