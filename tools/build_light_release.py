"""Build a reproducible, allowlisted Windows NoPython app/update ZIP (stdlib only)."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from updater import REPOSITORY, RUNTIME_EPOCH, TOP_FILES, MANAGED_DIRS, SEED_DIRS, safe_name, digest, version, write_json, verify_archive
from wushu_bridge.source_revision import valid_commit


def build(root, output, release_version, commit):
    version(release_version)
    if not valid_commit(commit):
        raise ValueError("Release source commit must be one full Git SHA")
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in sorted(TOP_FILES):
        p = root / name
        if p.is_file():
            files[name] = p.read_bytes()
    for directory in sorted(MANAGED_DIRS | SEED_DIRS):
        for p in sorted((root / directory).rglob("*")):
            if not p.is_file():
                continue
            name = p.relative_to(root).as_posix()
            if "__pycache__" in p.parts:
                continue
            safe_name(name)
            files[name] = p.read_bytes()
    for name in files:
        safe_name(name)
    if "WushuBridge-Trainer.exe" not in files or not files["WushuBridge-Trainer.exe"].startswith(b"MZ"):
        raise ValueError("Build the Windows launcher first")
    files["app_version.json"] = (json.dumps({"version": release_version, "runtime_epoch": RUNTIME_EPOCH,
        "minimum_python": "3.10", "repository": REPOSITORY, "source_commit": commit}, indent=2) + "\n").encode()
    manifest = {"schema": 1, "version": release_version, "runtime_epoch": RUNTIME_EPOCH,
        "source_commit": commit, "files": {n: {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()} for n, b in sorted(files.items())}}
    raw = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    asset = "T8-SemanticBridge-Trainer-v" + release_version + "-Windows-NoPython.zip"
    archive = output / asset
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name, data in sorted({**files, "UPDATE_MANIFEST.json": raw}.items()):
            info = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, data)
    metadata = {"schema": 1, "repository": REPOSITORY, "version": release_version,
        "runtime_epoch": RUNTIME_EPOCH, "asset": asset, "bytes": archive.stat().st_size,
        "sha256": digest(archive), "manifest_sha256": hashlib.sha256(raw).hexdigest()}
    write_json(output / "update.json", metadata)
    (output / "SHA256SUMS.txt").write_text(metadata["sha256"] + "  " + asset + "\n" + digest(output / "update.json") + "  update.json\n", encoding="ascii")
    verify_archive(archive, metadata)
    print(json.dumps({"asset": asset, "bytes": metadata["bytes"], "sha256": metadata["sha256"], "files": len(files)}, ensure_ascii=False))
    return metadata


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--commit", required=True)
    a = p.parse_args()
    build(a.root, a.output, a.version, a.commit)
