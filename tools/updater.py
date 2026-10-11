"""Standard-library app updater. Never installs Python, models or user training data."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import uuid
import zipfile

REPOSITORY = "T8mars/MinimaxH3_Semantic_Bridge_Trainer-T8"
API = "https://api.github.com/repos/" + REPOSITORY + "/releases/latest"
RUNTIME_EPOCH = 1
MAX_ZIP = 200 * 1024 * 1024
TOP_FILES = {"WushuBridge-Trainer.exe", "app_version.json", "README.md", "README.en.md",
             "LICENSE", "requirements.txt", "requirements-training.txt", "AGENTS.md",
             "AGENT交接.md", "给AGENT的任务.txt", "如何训练.md", "训练集制作.md",
             "先读说明.txt", "启动训练器.cmd"}
MANAGED_DIRS = {"wushu_bridge", "tools", "docs", "skills", "defaults"}
SEED_DIRS = {"configs", "datasets", "examples"}
FORBIDDEN = {"local", "runs", "python", "comfyui", "models", "对照视频", ".git",
             "__pycache__", ".pytest_cache", "roadmap.md"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def version(value):
    if not isinstance(value, str) or not re.fullmatch(r"v?\d+\.\d+\.\d+", value):
        raise ValueError("Invalid release version")
    return tuple(map(int, value.lstrip("v").split(".")))


def installed(root):
    p = Path(root) / "app_version.json"
    return read_json(p) if p.exists() else {"version": "0.0.0", "runtime_epoch": RUNTIME_EPOCH}


def safe_name(name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ValueError("Unsafe update path: " + str(name))
    parts = name.split("/")
    if any(p in {"", ".", ".."} or p.endswith((".", " ")) for p in parts):
        raise ValueError("Unsafe update path: " + name)
    if any(p.lower() in FORBIDDEN for p in parts):
        raise ValueError("Private/runtime path in update: " + name)
    if any(re.fullmatch(r"(con|prn|aux|nul|com[1-9]|lpt[1-9])(\..*)?", p, re.I) for p in parts):
        raise ValueError("Reserved Windows filename: " + name)
    if name not in TOP_FILES and (len(parts) < 2 or parts[0] not in MANAGED_DIRS | SEED_DIRS):
        raise ValueError("Unmanaged update path: " + name)
    if PurePosixPath(name).suffix.lower() in {".pyc", ".pyo", ".safetensors", ".gguf", ".npz", ".ckpt"}:
        raise ValueError("Runtime/cache/model in update: " + name)
    return name


def safe_target(root, name):
    safe_name(name)
    root = Path(root).resolve()
    target = root.joinpath(*name.split("/"))
    # Resolve existing ancestors too; refuse symlinks and Windows junctions.
    for part in [target, *target.parents]:
        if part == root:
            break
        if part.is_symlink() or (part.exists() and getattr(part.lstat(), "st_file_attributes", 0) & 0x400):
            raise ValueError("Reparse point in update destination: " + name)
    if not target.resolve().is_relative_to(root):
        raise ValueError("Update escapes application directory")
    return target


def asset_url(url):
    parsed = urlparse(url)
    prefix = "/" + REPOSITORY + "/releases/download/"
    if parsed.scheme != "https" or parsed.netloc != "github.com" or not parsed.path.startswith(prefix):
        raise ValueError("Update URL must belong to the trainer GitHub releases")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Unexpected update URL fields")
    return url


def fetch(url, limit):
    req = Request(url, headers={"User-Agent": "T8-SemanticBridge-Updater", "Accept": "application/json"})
    with urlopen(req, timeout=25) as response:
        parsed = urlparse(response.url)
        if parsed.scheme != "https" or parsed.hostname not in {"api.github.com", "github.com", "release-assets.githubusercontent.com", "objects.githubusercontent.com"}:
            raise ValueError("Unexpected download redirect")
        data = response.read(limit + 1)
        if len(data) > limit:
            raise ValueError("Update response exceeds size limit")
        return data


def check(root):
    current = installed(root)
    release = json.loads(fetch(API, 2 * 1024 * 1024))
    assets = {a["name"]: a for a in release.get("assets", [])}
    if "update.json" not in assets:
        return {"current": current["version"], "available": False, "message": "此 Release 尚无轻量更新包"}
    metadata = json.loads(fetch(asset_url(assets["update.json"]["browser_download_url"]), 1024 * 1024))
    validate_metadata(metadata)
    if release["tag_name"] != "v" + metadata["version"] or release.get("draft") or release.get("prerelease"):
        raise ValueError("Release/version mismatch")
    asset = assets.get(metadata["asset"])
    if not asset or asset["size"] != metadata["bytes"]:
        raise ValueError("Release asset mismatch")
    metadata["download_url"] = asset_url(asset["browser_download_url"])
    newer = version(metadata["version"]) > version(current["version"])
    compatible = metadata["runtime_epoch"] == current.get("runtime_epoch", RUNTIME_EPOCH)
    return {"current": current["version"], "latest": metadata["version"], "available": newer and compatible,
            "runtime_compatible": compatible, "metadata": metadata,
            "message": "需要更新 Python 环境，请重新获取整合包" if not compatible else "发现新版本" if newer else "已是最新版本"}


def validate_metadata(meta):
    if meta.get("schema") != 1 or meta.get("repository") != REPOSITORY:
        raise ValueError("Unknown update manifest")
    version(meta["version"])
    if meta.get("runtime_epoch") != RUNTIME_EPOCH:
        raise ValueError("此版本需要新的运行环境，请重新获取完整整合包")
    if not re.fullmatch(r"T8-SemanticBridge-Trainer-v\d+\.\d+\.\d+-Windows-NoPython\.zip", meta.get("asset", "")):
        raise ValueError("Unexpected update asset name")
    if not isinstance(meta.get("bytes"), int) or not 0 < meta["bytes"] <= MAX_ZIP:
        raise ValueError("Invalid update size")
    for key in ("sha256", "manifest_sha256"):
        if not re.fullmatch("[0-9a-f]{64}", meta.get(key, "")):
            raise ValueError("Invalid update digest")


def verify_archive(archive, metadata):
    validate_metadata(metadata)
    if Path(archive).stat().st_size != metadata["bytes"] or digest(archive) != metadata["sha256"]:
        raise ValueError("更新包 SHA-256 校验失败，未安装")
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        names = [i.filename for i in infos]
        if len(names) > 5000 or len(names) != len(set(n.casefold() for n in names)):
            raise ValueError("Duplicate/oversized update archive")
        if "UPDATE_MANIFEST.json" not in names or sum(i.file_size for i in infos) > 400 * 1024 * 1024:
            raise ValueError("Invalid update archive")
        raw = z.read("UPDATE_MANIFEST.json")
        if hashlib.sha256(raw).hexdigest() != metadata["manifest_sha256"]:
            raise ValueError("Update manifest checksum mismatch")
        manifest = json.loads(raw)
        if manifest.get("schema") != 1 or manifest.get("version") != metadata["version"] or manifest.get("runtime_epoch") != RUNTIME_EPOCH:
            raise ValueError("Update manifest version mismatch")
        if set(names) != set(manifest["files"]) | {"UPDATE_MANIFEST.json"}:
            raise ValueError("Unexpected update archive members")
        for info in infos:
            if info.filename == "UPDATE_MANIFEST.json":
                continue
            safe_name(info.filename)
            if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("Link/directory member in update")
            expected = manifest["files"][info.filename]
            if expected["bytes"] != info.file_size or hashlib.sha256(z.read(info)).hexdigest() != expected["sha256"]:
                raise ValueError("Damaged update member: " + info.filename)
        if "app_version.json" not in manifest["files"] or "WushuBridge-Trainer.exe" not in manifest["files"]:
            raise ValueError("Incomplete app update")
        app = json.loads(z.read("app_version.json"))
        if app.get("version") != metadata["version"] or app.get("runtime_epoch") != RUNTIME_EPOCH:
            raise ValueError("Installed version metadata mismatch")
    return manifest


def stage_archive(root, archive, metadata):
    root = Path(root).resolve()
    manifest = verify_archive(archive, metadata)
    if version(metadata["version"]) <= version(installed(root)["version"]):
        raise ValueError("拒绝降级或重复安装")
    staging_base = root / "local/updater/staging"
    if not staging_base.resolve().is_relative_to(root):
        raise ValueError("Staging directory escapes application")
    folder = staging_base / uuid.uuid4().hex
    folder.mkdir(parents=True)
    with zipfile.ZipFile(archive) as z:
        for name in manifest["files"]:
            target = safe_target(folder / "payload", name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(name))
    write_json(folder / "transaction.json", {"metadata": metadata, "manifest": manifest})
    # Standalone helper survives replacement of tools/updater.py.
    shutil.copyfile(__file__, folder / "apply_helper.py")
    return {"staging": str(folder), "helper": str(folder / "apply_helper.py"), "version": metadata["version"]}


def stage(root):
    result = check(root)
    if not result["available"]:
        raise ValueError(result["message"])
    metadata = result["metadata"]
    data = fetch(metadata["download_url"], MAX_ZIP)
    cache = Path(root).resolve() / "local/updater/downloads"
    if not cache.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Download directory escapes application")
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / (uuid.uuid4().hex + ".zip")
    archive.write_bytes(data)
    return stage_archive(root, archive, metadata)


def wait_process(pid, timeout=60):
    if pid <= 0:
        return
    if os.name == "nt":
        from ctypes import wintypes
        api = ctypes.WinDLL("kernel32", use_last_error=True)
        api.OpenProcess.restype = wintypes.HANDLE
        api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        api.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        api.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = api.OpenProcess(0x100000, False, pid)
        if not handle:
            if ctypes.get_last_error() == 87:
                return
            raise RuntimeError("无法确认启动器已退出")
        try:
            if api.WaitForSingleObject(handle, int(timeout * 1000)) != 0:
                raise RuntimeError("启动器尚未退出，未安装更新")
        finally:
            api.CloseHandle(handle)
    else:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(.2)
        raise RuntimeError("Launcher did not exit")


def running_bundle_processes(root):
    """Refuse a second GUI or a same-bundle Python job. Never terminate processes."""
    if os.name != "nt":
        return []
    from ctypes import wintypes
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.OpenProcess.restype = wintypes.HANDLE
    api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    api.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    psapi = ctypes.WinDLL("psapi")
    pids = (wintypes.DWORD * 32768)()
    needed = wintypes.DWORD()
    if not psapi.EnumProcesses(pids, ctypes.sizeof(pids), ctypes.byref(needed)):
        raise RuntimeError("无法检查运行中的训练任务")
    found = []
    prefix = str(Path(root).resolve()).lower() + os.sep
    for pid in list(pids)[:needed.value // ctypes.sizeof(wintypes.DWORD)]:
        if pid in (0, os.getpid()):
            continue
        handle = api.OpenProcess(0x1000, False, pid)
        if not handle:
            continue
        try:
            size = wintypes.DWORD(32768)
            buf = ctypes.create_unicode_buffer(size.value)
            if api.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                path = buf.value.lower()
                if path.startswith(prefix) and Path(path).name in {"wushubridge-trainer.exe", "python.exe", "pythonw.exe"}:
                    found.append(pid)
        finally:
            api.CloseHandle(handle)
    return found


def apply_update(root, staging, wait_pid=0):
    root = Path(root).resolve()
    folder = Path(staging).resolve()
    if not folder.is_relative_to((root / "local/updater/staging").resolve()):
        raise ValueError("Invalid staging directory")
    tx = read_json(folder / "transaction.json")
    meta, manifest = tx["metadata"], tx["manifest"]
    validate_metadata(meta)
    if manifest["version"] != meta["version"] or version(meta["version"]) <= version(installed(root)["version"]):
        raise ValueError("Version mismatch or downgrade")
    if installed(root).get("runtime_epoch", RUNTIME_EPOCH) != meta["runtime_epoch"]:
        raise ValueError("Runtime migration required")
    # Recheck staged bytes after download; no trust in a previously checked cache.
    for name, expected in manifest["files"].items():
        p = safe_target(folder / "payload", name)
        safe_target(root, name)
        if p.stat().st_size != expected["bytes"] or digest(p) != expected["sha256"]:
            raise ValueError("Staged file changed: " + name)
    wait_process(wait_pid)
    if running_bundle_processes(root):
        raise RuntimeError("本目录还有启动器或 Python 任务，请全部关闭后重试更新")
    state = root / "local/updater"
    lock = state / "install.lock"
    # Exclusive creation; stale lock is kept for inspection instead of unsafe takeover.
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode("ascii"))
    os.close(fd)
    backup = state / "backups" / uuid.uuid4().hex
    backup.mkdir(parents=True)
    changed, preserved = [], []
    journal = {"version": meta["version"], "backup": str(backup), "changed": changed, "preserved": preserved, "status": "installing"}
    result_path = state / "latest_result.json"
    try:
        names = sorted(manifest["files"], key=lambda n: (n == "app_version.json", n))
        for name in names:
            target = safe_target(root, name)
            if name.split("/")[0] in SEED_DIRS and target.exists():
                preserved.append(name)
                continue
            exists = target.exists()
            if exists and digest(target) == manifest["files"][name]["sha256"]:
                continue
            if exists:
                old = safe_target(backup, name)
                old.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, old)
            changed.append({"name": name, "existed": exists})
            write_json(backup / "journal.json", journal)
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_name(target.name + ".t8-update-tmp")
            shutil.copyfile(folder / "payload" / name, temp)
            os.replace(temp, target)
        # Keep the full-bundle integrity inventory consistent with the installed app.
        portable = root / "PORTABLE_MANIFEST.json"
        if portable.exists():
            shutil.copy2(portable, backup / "PORTABLE_MANIFEST.json")
            changed.append({"name": "PORTABLE_MANIFEST.json", "existed": True})
            write_json(backup / "journal.json", journal)
            inventory = read_json(portable)
            for name in manifest["files"]:
                p = root / name
                inventory["files"][name] = {"bytes": p.stat().st_size, "sha256": digest(p)}
            inventory["app_release"] = {"version": meta["version"], "source_commit": manifest.get("source_commit")}
            if "launcher_provenance" in inventory:
                inventory["launcher_provenance"]["exe_sha256"] = digest(root / "WushuBridge-Trainer.exe")
                inventory["launcher_provenance"]["source_sha256"]["tools/launcher/Launcher.cs"] = digest(root / "tools/launcher/Launcher.cs")
            write_json(portable, inventory)
        journal["status"] = "installed"
        write_json(backup / "journal.json", journal)
        write_json(result_path, journal)
        (state / "last_error.json").unlink(missing_ok=True)
    except BaseException as exc:
        errors = []
        for item in reversed(changed):
            target = root / item["name"]
            try:
                if item["existed"]:
                    os.replace(backup / item["name"], target)
                elif target.exists():
                    target.unlink()
            except OSError as rollback_error:
                errors.append(str(rollback_error))
        journal.update(status="rollback_failed" if errors else "rolled_back", error=str(exc), rollback_errors=errors)
        write_json(backup / "journal.json", journal)
        write_json(result_path, journal)
        raise
    finally:
        lock.unlink(missing_ok=True)
    return journal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "stage", "apply"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--staging", type=Path)
    parser.add_argument("--wait-pid", type=int, default=0)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    try:
        if sys.version_info < (3, 10):
            raise RuntimeError("Python 3.10+ required")
        result = check(args.root) if args.command == "check" else stage(args.root) if args.command == "stage" else apply_update(args.root, args.staging, args.wait_pid)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        if args.command == "apply":
            write_json(args.root / "local/updater/last_error.json", {"error": str(exc), "time": time.time()})
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), flush=True)
        return 1
    finally:
        if args.command == "apply" and args.restart:
            subprocess.Popen([str(args.root.resolve() / "WushuBridge-Trainer.exe")], cwd=args.root)


if __name__ == "__main__":
    raise SystemExit(main())
