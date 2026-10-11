"""Crash-safe checkpoint writes and an OS-held, per-run training lock."""
from contextlib import contextmanager
import errno
import os
from pathlib import Path
import tempfile
import time
import warnings


def _busy_notice():
    # A diagnostic must never interrupt recovery on an English Windows console
    # or when the launcher has closed its log pipe.
    try:
        print("断点文件暂被占用，正在重试保存；上一份断点保持完整。", flush=True)
    except UnicodeError:
        try:
            print("Checkpoint temporarily busy; retrying. Previous checkpoint is intact.", flush=True)
        except (OSError, ValueError):
            pass
    except (OSError, ValueError):
        pass


def replace_with_retry(source, destination, timeout=10.0):
    """Keep the previous file intact while Windows readers release their handles."""
    deadline = time.monotonic() + timeout
    announced = False
    while True:
        try:
            os.replace(source, destination)
            return
        except OSError as exc:
            transient = (getattr(exc, "winerror", None) in (5, 32, 33)
                         or (os.name == "nt" and exc.errno in (errno.EACCES, errno.EBUSY)))
            if not transient or time.monotonic() >= deadline:
                raise
            if not announced:
                _busy_notice()
                announced = True
            time.sleep(min(.25, max(0, deadline - time.monotonic())))


def save_checkpoint(path, state):
    import torch
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".pending", dir=path.parent)
    pending = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # A partial serialization is not a usable checkpoint.
        pending.unlink(missing_ok=True)
        raise
    try:
        replace_with_retry(pending, path)
    except OSError as exc:
        # Never remove the previous checkpoint or the fully serialized new state.
        raise OSError(
            f"无法替换断点 {path}。上一份断点未改动；本轮完整状态保留在 {pending}。"
            "请关闭占用文件的程序并使用断点续训。") from exc


@contextmanager
def training_lock(folder):
    """The OS releases this lock on process exit, including a forced termination."""
    path = Path(folder) / ".training.lock"
    stream = path.open("a+b")
    acquired = False
    try:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError as exc:
            raise RuntimeError(f"该输出目录已有另一个训练进程：{folder}。请等待其结束，或选择新输出目录。") from exc
        yield
    finally:
        if acquired:
            try:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            except OSError as exc:
                warnings.warn(f"Training lock release: {exc}")
        stream.close()
