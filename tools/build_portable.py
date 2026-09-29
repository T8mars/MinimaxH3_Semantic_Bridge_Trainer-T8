"""Build a relocatable Windows trainer using an existing, working Python runtime.

Copies only the dependency closure recorded by installed wheel metadata, never
the whole personal environment. Private research, caches, settings and runs are
excluded. Run with the source Python; no installers or network are required.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # Direct `python tools/build_portable.py` needs the project package.
DEPENDENCIES = [
    "pip", "setuptools", "wheel", "torch", "torchvision", "torchaudio", "torchsde",
    "numpy", "einops", "transformers", "tokenizers", "sentencepiece", "safetensors",
    "aiohttp", "yarl", "pyyaml", "Pillow", "scipy", "tqdm", "psutil", "filelock",
    "av", "comfy-kitchen", "comfy-aimdo", "requests", "simpleeval", "blake3",
    "kornia", "spandrel", "pydantic", "pydantic-settings", "PyOpenGL", "gguf",
]
PRIVATE_PARTS = {"local", "runs", ".git", "__pycache__", ".pytest_cache", "dist"}
MODEL_EXTENSIONS = {".safetensors", ".npz", ".ckpt", ".gguf"}
OUT_OF_SCOPE_VIDEO_PREFIXES = ("h3_video_", "v2_", "test_h3_video_", "test_v2_")
PRIVATE_RESEARCH_FILES = {"h3lint_diff.mjs", "h3lint_diff_report.md", "run_h3lint_diff.py"}
ALLOWED_RUNTIME_NPZ = "python/Lib/site-packages/scipy/stats/_sobol_direction_numbers.npz"
PLUGIN_NAME = "ComfyUI-H3-WushuBridge"
MANIFEST_SCHEMA = 5
RELEASE_WEIGHT_REL = "models/wushu_bridge/semantic_bridge_release.safetensors"
RELEASE_CARD_REL = "models/wushu_bridge/MODEL_CARD.json"
RELEASE_CARD_SCHEMA = 1
KNOWN_AUTHOR_WEIGHT_SHA256 = {
    "4b7a459aac43e066b9cdd8d3f84f52b7d6c2777e7de05f35a3cb042abab889ebd",
    "a053296d161d3a8a9d544543e1604d63471d0640f9a7e8aad79734a57ec17809",
}
LAUNCHER_BUILD_SOURCES = (
    "tools/launcher/Launcher.cs",
    "tools/launcher/app.manifest",
    "tools/build_launcher.ps1",
)
DISTRIBUTED_CONFIGS = (
    "author_v1.json",
    "preference_v1.json",
    "experimental_aligned_token_v1.json",
    "experimental_monotone_token_v1.json",
)


def dependency_closure():
    pending = [(name, "") for name in DEPENDENCIES]
    seen = set()
    result = {}
    while pending:
        name, extra = pending.pop()
        key = (canonicalize_name(name), extra)
        if key in seen:
            continue
        seen.add(key)
        dist = metadata.distribution(name)
        result[canonicalize_name(name)] = dist
        for value in dist.requires or []:
            req = Requirement(value)
            if req.marker and not req.marker.evaluate({"extra": extra}):
                continue
            installed = metadata.version(req.name)
            if req.specifier and installed not in req.specifier:
                print(f"Existing runtime constraint: {req.name} {installed}, declared {req.specifier}", flush=True)
            pending.append((req.name, ""))
            pending.extend((req.name, e) for e in req.extras)
    return result


def copy_file(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == src.stat().st_size and dst.stat().st_mtime_ns == src.stat().st_mtime_ns:
        return
    shutil.copy2(src, dst)


def sha256_file(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def _sha256_value(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_release_card(path, weight_sha256):
    """Accept only a small, path-free card bound to one reviewed bridge file."""
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16_384:
        raise ValueError("Release model card must be a regular JSON file of at most 16 KiB")
    try:
        card = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Release model card is not valid UTF-8 JSON") from exc
    expected = {
        "schema", "model_id", "weight_sha256", "encoder_sha256",
        "training_report_sha256", "video_ab_report_sha256",
        "video_ab_protocol", "video_ab_outcome", "inference",
    }
    if not isinstance(card, dict) or type(card.get("schema")) is not int or card["schema"] != RELEASE_CARD_SCHEMA:
        raise ValueError("Release model card has an unsupported schema or fields")
    outcome = card.get("video_ab_outcome")
    if outcome == "limited_evidence":
        expected.add("video_ab_observation")
    if set(card) != expected:
        raise ValueError("Release model card has an unsupported schema or fields")
    if not isinstance(card["model_id"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{1,63}", card["model_id"]):
        raise ValueError("Release model_id must be a short path-free identifier")
    if card["weight_sha256"] != weight_sha256 or any(
        not _sha256_value(card[key]) for key in
        ("weight_sha256", "encoder_sha256", "training_report_sha256", "video_ab_report_sha256")
    ):
        raise ValueError("Release model card hashes are missing or weight SHA-256 differs")
    if card["video_ab_protocol"] != "same_prompt_same_seed_h3_video_ab" or outcome not in {"passed", "limited_evidence"}:
        raise ValueError("Release model card requires a reviewed same-prompt, same-seed H3 video A/B report")
    if outcome == "limited_evidence":
        observation = card["video_ab_observation"]
        required = {"pairs", "relative_wins", "ties", "material_regressions", "prior_target_wins", "cohorts", "review_limit"}
        if (not isinstance(observation, dict) or set(observation) != required
                or any(type(observation[key]) is not int or observation[key] < 0 for key in
                       ("pairs", "relative_wins", "ties", "material_regressions", "prior_target_wins"))
                or observation["pairs"] == 0
                or observation["relative_wins"] + observation["ties"] + observation["material_regressions"] != observation["pairs"]
                or observation["relative_wins"] >= observation["prior_target_wins"]
                or not isinstance(observation["cohorts"], str) or not 1 <= len(observation["cohorts"]) <= 256
                or not isinstance(observation["review_limit"], str) or not 1 <= len(observation["review_limit"]) <= 256):
            raise ValueError("Limited-evidence card must disclose the below-target A/B tally and review limits")
    inference = card["inference"]
    if not isinstance(inference, dict) or set(inference) != {
        "alpha", "magnitude_match", "token_span", "tail_ratio", "chunk_tokens",
        "allow_dim_mismatch", "auto_alpha", "guard",
    }:
        raise ValueError("Release model card needs the validated fixed-alpha H3 inference preset")
    alpha = inference["alpha"]
    if (isinstance(alpha, bool) or not isinstance(alpha, (int, float))
            or not math.isfinite(alpha) or not 0 < alpha <= 1
            or type(inference["magnitude_match"]) is not str
            or inference["magnitude_match"] not in {"per_token", "global", "none"}
            or inference["token_span"] != "all"
            or type(inference["tail_ratio"]) is not float or inference["tail_ratio"] != 1.0
            or type(inference["chunk_tokens"]) is not int or inference["chunk_tokens"] != 0
            or inference["allow_dim_mismatch"] is not False
            or inference["auto_alpha"] is not False or inference["guard"] is not False):
        raise ValueError("Release model card needs the validated fixed-alpha H3 inference preset (chunk_tokens=0)")
    return card


def validate_release_weight(path, expected_sha256=None):
    """Validate exact bytes and that the asset loads as an H3 Semantic Bridge."""
    path = Path(path)
    if path.suffix.lower() != ".safetensors" or path.is_symlink() or not path.is_file():
        raise ValueError("Release weight must be a regular .safetensors file")
    size = path.stat().st_size
    if not 8 < size <= 512 * 1024 * 1024:
        raise ValueError("Release weight has an invalid size")
    digest = sha256_file(path)
    if digest in KNOWN_AUTHOR_WEIGHT_SHA256:
        raise ValueError("Known author weight cannot be bundled as the new release model")
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Release weight SHA-256 differs from the selected model card or manifest")
    from wushu_bridge.bridge_model import load_bridge
    try:
        model = load_bridge(str(path), device="cpu")
    except Exception as exc:
        raise ValueError("Release weight is not a loadable Semantic Bridge safetensors model") from exc
    if model.cfg.dim != 5120 or model.cfg.mode not in {"t2v", "any"}:
        raise ValueError("Release weight is not a 5120-dimensional H3 t2v Semantic Bridge")
    del model
    if sha256_file(path) != digest:
        raise ValueError("Release weight changed during validation")
    return {"path": RELEASE_WEIGHT_REL, "sha256": digest, "bytes": size}


def validate_release_application_contract(weight, inference):
    """Bind a new fixed-alpha weight's runtime contract to its release card.

    Legacy weights have no contract, so the existing validated card preset
    remains their release policy.
    """
    from wushu_bridge.apply import application_contract
    from wushu_bridge.bridge_model import load_bridge

    model = load_bridge(str(weight), device="cpu")
    contract = application_contract(model)
    if contract is None:
        return
    expected = {
        "alpha": contract["alpha_min"],
        "magnitude_match": contract["magnitude_match"],
        "token_span": contract["token_span"],
        "tail_ratio": contract["tail_ratio"],
        "chunk_tokens": contract["chunk_tokens"],
        "allow_dim_mismatch": contract["allow_dim_mismatch"],
        "auto_alpha": contract["auto_alpha"],
        "guard": contract["guard"],
    }
    if inference != expected:
        raise ValueError("Release model card inference differs from weight application contract")


def _validate_release_report(path, expected_sha256, label):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 64 * 1024 * 1024:
        raise ValueError(f"{label} must be a nonempty regular file of at most 64 MiB")
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} SHA-256 differs from the release model card")


def release_asset_record(weight, card, training_report, video_ab_report):
    weight_record = validate_release_weight(weight)
    card_data = validate_release_card(card, weight_record["sha256"])
    validate_release_application_contract(weight, card_data["inference"])
    _validate_release_report(training_report, card_data["training_report_sha256"], "Training report")
    _validate_release_report(video_ab_report, card_data["video_ab_report_sha256"], "H3 video A/B report")
    return {
        "weight": weight_record,
        "model_card": {"path": RELEASE_CARD_REL, "sha256": sha256_file(Path(card)), "bytes": Path(card).stat().st_size},
    }


def copy_release_asset(weight, card, training_report, video_ab_report, asset, output):
    """Copy selected bytes, then recheck inputs and outputs for source-copy races."""
    targets = ((Path(weight), RELEASE_WEIGHT_REL, asset["weight"]),
               (Path(card), RELEASE_CARD_REL, asset["model_card"]))
    for source, rel, expected in targets:
        copy_file(source, output / rel)
        if sha256_file(source) != expected["sha256"] or sha256_file(output / rel) != expected["sha256"]:
            raise ValueError(f"Release asset changed while being copied: {rel}")
    card_data = validate_release_card(output / RELEASE_CARD_REL, asset["weight"]["sha256"])
    validate_release_application_contract(output / RELEASE_WEIGHT_REL, card_data["inference"])
    _validate_release_report(training_report, card_data["training_report_sha256"], "Training report")
    _validate_release_report(video_ab_report, card_data["video_ab_report_sha256"], "H3 video A/B report")


def launcher_provenance(launcher, source_root=ROOT):
    """Bind the chosen EXE to the current build inputs before copying it."""
    launcher = Path(launcher)
    if not launcher.is_file():
        raise FileNotFoundError(launcher)
    sources = {name: source_root / name for name in LAUNCHER_BUILD_SOURCES}
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if launcher.stat().st_mtime_ns < max(path.stat().st_mtime_ns for path in sources.values()):
        raise ValueError("Launcher EXE predates its build inputs; rebuild it with tools/build_launcher.ps1")
    return {
        "exe_sha256": sha256_file(launcher),
        "source_sha256": {name: sha256_file(path) for name, path in sources.items()},
        "freshness_check": "EXE mtime >= every launcher build input at package build",
    }


def copy_tree(src, dst, ignore=(), source_only=False, skip_video_research=False):
    for current, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in {"__pycache__", ".git", *ignore} and (not source_only or d.lower() not in PRIVATE_PARTS)]
        for name in dirs:
            if (Path(current) / name).is_symlink():
                raise ValueError(f"Source directory symlink requires review: {Path(current) / name}")
        for name in files:
            p = Path(current) / name
            if p.is_symlink():
                raise ValueError(f"Source file symlink requires review: {p}")
            if skip_video_research and p.parent == src and (
                name.startswith(OUT_OF_SCOPE_VIDEO_PREFIXES) or name in PRIVATE_RESEARCH_FILES
            ):
                continue
            if p.suffix in {".pyc", ".pyo"} or name in {"sitecustomize.py", "usercustomize.py"}:
                continue
            if name.lower() == "roadmap.md" or (source_only and p.suffix.lower() in MODEL_EXTENSIONS):
                continue
            copy_file(p, dst / p.relative_to(src))


def copy_plugin_snapshot(source_root, output):
    """Copy only the working ComfyUI plugin's Python source and MIT license."""
    target = output / "plugin" / PLUGIN_NAME
    for name in ("__init__.py", "LICENSE"):
        copy_file(source_root / name, target / name)
    modules = source_root / "wushu_bridge"
    for p in sorted(modules.rglob("*.py")):
        rel = p.relative_to(modules)
        if p.is_symlink():
            raise ValueError(f"Plugin source symlink requires review: {rel}")
        if any(part.lower() in PRIVATE_PARTS for part in rel.parts):
            continue
        copy_file(p, target / "wushu_bridge" / rel)


def wheel_file_allowed(rel):
    """Exclude wheel test archives while retaining reviewed runtime data."""
    if rel.suffix.lower() == ".npz":
        return (Path("python/Lib/site-packages") / rel).as_posix() == ALLOWED_RUNTIME_NPZ
    return True


def revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def package_files(output, release_asset=None):
    """List exact shipped bytes; fail closed on private or unrelated content."""
    if not output.is_dir():
        raise FileNotFoundError(output)
    result = {}
    for p in sorted(output.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(output)
        # A recipient may smoke-test the bundled Python before archiving.
        # Bytecode generated by that test is never part of the sealed ZIP.
        if "__pycache__" in rel.parts and p.suffix.lower() == ".pyc":
            continue
        if p.is_symlink():
            raise ValueError(f"Symlink in portable folder: {rel}")
        rel_name = rel.as_posix()
        if any(part.lower() in PRIVATE_PARTS for part in rel.parts):
            raise ValueError(f"Private content in portable folder: {rel}")
        if rel.name.lower() == "roadmap.md" or (
            rel.suffix.lower() in MODEL_EXTENSIONS | {".pyc", ".pyo"}
            and rel_name not in {ALLOWED_RUNTIME_NPZ, RELEASE_WEIGHT_REL if release_asset else ""}
        ):
            raise ValueError(f"Forbidden file in portable folder: {rel}")
        if rel.parts[0] == "models" and rel.suffix.lower() not in {".txt", ".md"} and rel_name not in (
            {RELEASE_WEIGHT_REL, RELEASE_CARD_REL} if release_asset else set()
        ):
            raise ValueError(f"Portable models directory must contain instructions only: {rel}")
        if rel.parts[0] == "datasets":
            raise ValueError(f"User datasets must not be bundled: {rel}")
        if rel.parts[0] == "configs" and rel_name not in {
                f"configs/{name}" for name in DISTRIBUTED_CONFIGS}:
            raise ValueError(f"Unreleased experiment config in portable folder: {rel}")
        if rel.parts[0] in {"tools", "docs", "wushu_bridge", "plugin"} and (
            rel.name.lower().startswith(OUT_OF_SCOPE_VIDEO_PREFIXES)
            or "video_lora" in rel.name.lower()
        ):
            raise ValueError(f"Video LoRA research file in portable folder: {rel}")
        if rel.parts[0] == "tools" and rel.name in PRIVATE_RESEARCH_FILES:
            raise ValueError(f"Private research file in portable folder: {rel}")
        if rel.parts[0] == "plugin":
            if len(rel.parts) < 3 or rel.parts[1] != PLUGIN_NAME:
                raise ValueError(f"Unexpected plugin content in portable folder: {rel}")
            plugin_rel = rel.parts[2:]
            if plugin_rel not in {("__init__.py",), ("LICENSE",)} and not (
                len(plugin_rel) >= 2 and plugin_rel[0] == "wushu_bridge" and rel.suffix == ".py"
            ):
                raise ValueError(f"Non-source plugin content in portable folder: {rel}")
        if rel_name == "PORTABLE_MANIFEST.json":
            continue
        result[rel_name] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    if release_asset is not None:
        if not isinstance(release_asset, dict) or set(release_asset) != {"weight", "model_card"}:
            raise ValueError("Invalid release asset record")
        for key, expected_path in (("weight", RELEASE_WEIGHT_REL), ("model_card", RELEASE_CARD_REL)):
            record = release_asset[key]
            if not isinstance(record, dict) or set(record) != {"path", "sha256", "bytes"} or record["path"] != expected_path or not _sha256_value(record["sha256"]) or type(record["bytes"]) is not int or record["bytes"] <= 0:
                raise ValueError("Invalid release asset path or digest record")
            if result.get(expected_path) != {"sha256": record["sha256"], "bytes": record["bytes"]}:
                raise ValueError(f"Release asset differs from its manifest: {expected_path}")
        validate_release_weight(output / RELEASE_WEIGHT_REL, release_asset["weight"]["sha256"])
        card_data = validate_release_card(output / RELEASE_CARD_REL, release_asset["weight"]["sha256"])
        validate_release_application_contract(output / RELEASE_WEIGHT_REL, card_data["inference"])
    required = {"WushuBridge-Trainer.exe", "python/python.exe",
                "skills/minimax-h3-anime-battle-prompts/SKILL.md",
                "handoff/README.md", "handoff/PROMPT_SKILL_AGENT_BRIEF.md",
                "handoff/MODEL_EVIDENCE.md",
                "tools/trainer.py", "tools/prepare_aligned_dataset.py",
                "tools/build_sentence_targets.py", "tools/fit_explicit_targets.py",
                "docs/精确目标训练.md",
                "configs/author_v1.json", "configs/preference_v1.json",
                "configs/experimental_aligned_token_v1.json",
                "configs/experimental_monotone_token_v1.json",
                "wushu_bridge/trainer_engine.py", "wushu_bridge/token_alignment.py",
                "wushu_bridge/dataset.py", "wushu_bridge/trainer_data.py",
                "wushu_bridge/video_feedback.py",
                *LAUNCHER_BUILD_SOURCES,
                f"plugin/{PLUGIN_NAME}/__init__.py",
                f"plugin/{PLUGIN_NAME}/LICENSE",
                f"plugin/{PLUGIN_NAME}/wushu_bridge/__init__.py",
                f"plugin/{PLUGIN_NAME}/wushu_bridge/nodes.py",
                f"plugin/{PLUGIN_NAME}/wushu_bridge/bridge_model.py",
                f"plugin/{PLUGIN_NAME}/wushu_bridge/apply.py"}
    missing = required - result.keys()
    if missing:
        raise ValueError(f"Portable folder is missing required files: {sorted(missing)}")
    trainer_modules = {p.removeprefix("wushu_bridge/"): record
                       for p, record in result.items() if p.startswith("wushu_bridge/")}
    plugin_prefix = f"plugin/{PLUGIN_NAME}/wushu_bridge/"
    plugin_modules = {p.removeprefix(plugin_prefix): record
                      for p, record in result.items() if p.startswith(plugin_prefix)}
    if trainer_modules != plugin_modules:
        raise ValueError("Plugin runtime source differs from bundled trainer source")
    return result


def verify_package(output):
    manifest_path = output / "PORTABLE_MANIFEST.json"
    if not manifest_path.is_file():
        raise ValueError("Portable manifest missing; rebuild from current source")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA or not isinstance(manifest.get("files"), dict):
        raise ValueError("Old portable manifest; rebuild from current source")
    actual = package_files(output, manifest.get("release_asset"))
    provenance = manifest.get("launcher_provenance")
    if not isinstance(provenance, dict) or provenance.get("exe_sha256") != actual["WushuBridge-Trainer.exe"]["sha256"] or provenance.get("source_sha256") != {
        name: actual[name]["sha256"] for name in LAUNCHER_BUILD_SOURCES
    } or provenance.get("freshness_check") != "EXE mtime >= every launcher build input at package build":
        raise ValueError("Launcher provenance differs from portable content; rebuild from current source")
    if actual != manifest["files"]:
        missing = sorted(manifest["files"].keys() - actual.keys())
        added = sorted(actual.keys() - manifest["files"].keys())
        changed = sorted(k for k in actual.keys() & manifest["files"].keys()
                         if actual[k] != manifest["files"][k])
        raise ValueError(f"Portable content differs from manifest: missing={missing[:5]}, added={added[:5]}, changed={changed[:5]}")
    return actual


def build(output, comfy, launcher=None, release_weight=None, release_model_card=None,
          release_training_report=None, release_video_ab_report=None):
    # Never update an already-used package in place: stale private files and old
    # bundled author weights must not survive an otherwise clean rebuild.
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new empty portable output directory; existing packages are preserved")
    release_inputs = (release_weight, release_model_card, release_training_report, release_video_ab_report)
    if any(value is not None for value in release_inputs) and not all(value is not None for value in release_inputs):
        raise ValueError("Specify all four --release-* inputs for a model release, or none")
    release_asset = release_asset_record(*release_inputs) if release_weight is not None else None
    launcher = Path(launcher) if launcher is not None else ROOT / "WushuBridge-Trainer.exe"
    launcher_receipt = launcher_provenance(launcher, ROOT)
    source_python = Path(sys.executable).parent
    source_site = source_python / "Lib/site-packages"
    runtime = output / "python"
    output.mkdir(parents=True, exist_ok=True)
    print("Collecting installed dependency closure", flush=True)
    dependencies = dependency_closure()
    print(f"Copying Python standard library and {len(dependencies)} distributions", flush=True)
    copy_tree(source_python / "Lib", runtime / "Lib", ignore={"site-packages", "test", "idlelib"})
    copy_tree(source_python / "DLLs", runtime / "DLLs")
    for p in source_python.iterdir():
        if p.is_file() and (p.suffix.lower() == ".dll" or p.name in {"python.exe", "pythonw.exe", "LICENSE.txt"}):
            copy_file(p, runtime / p.name)
    copied = set()
    for name, dist in sorted(dependencies.items()):
        print(f"  {name}=={dist.version}", flush=True)
        if dist.files is None:
            raise RuntimeError(f"No wheel file manifest: {name}")
        for item in dist.files:
            src = Path(dist.locate_file(item)).resolve()
            if not src.is_relative_to(source_site.resolve()):
                continue  # Do not copy absolute-path console-script wrappers.
            rel = src.relative_to(source_site.resolve())
            if "__pycache__" in rel.parts or src.suffix in {".pyc", ".pyo"} or src.name == "direct_url.json":
                continue
            if not wheel_file_allowed(rel):
                continue  # Drop wheel test fixtures; retain the one reviewed SciPy runtime resource.
            if src.suffix == ".pth" and src.name != "distutils-precedence.pth":
                raise RuntimeError(f"Review required before bundling startup hook: {rel}")
            if not src.is_file():
                raise FileNotFoundError(src)
            if rel not in copied:
                copy_file(src, runtime / "Lib/site-packages" / rel)
                copied.add(rel)
    # The _pth file isolates imports from user-site, registry and PYTHONPATH.
    (runtime / f"python{sys.version_info.major}{sys.version_info.minor}._pth").write_text(".\nDLLs\nLib\nLib/site-packages\n..\n../tools\nimport site\n", encoding="utf-8")

    print("Copying trainer and ComfyUI encoding source", flush=True)
    for name in ["wushu_bridge", "skills", "handoff"]:
        copy_tree(ROOT / name, output / name, source_only=True)
    for name in DISTRIBUTED_CONFIGS:
        copy_file(ROOT / "configs" / name, output / "configs" / name)
    copy_plugin_snapshot(ROOT, output)
    copy_tree(ROOT / "tools", output / "tools", source_only=True, skip_video_research=True)
    for name in ["便携整合包.md", "可复用训练器.md", "preference-training.md", "精确目标训练.md"]:
        copy_file(ROOT / "docs" / name, output / "docs" / name)
    copy_file(launcher, output / "WushuBridge-Trainer.exe")
    for name in ["requirements-training.txt", "LICENSE"]:
        if (ROOT / name).is_file():
            copy_file(ROOT / name, output / name)
    copy_file(ROOT / "docs/便携整合包.md", output / "先读说明.txt")
    # The upstream plugin README describes assets shipped in that repository,
    # not this deliberately weight/data-free trainer distribution.
    copy_file(ROOT / "docs/便携整合包.md", output / "README.md")
    for name in ["comfy", "comfy_extras", "comfy_execution", "comfy_config", "comfy_api", "utils"]:
        if (comfy / name).is_dir():
            copy_tree(comfy / name, output / "ComfyUI" / name, source_only=True)
    for name in ["folder_paths.py", "node_helpers.py", "comfyui_version.py", "latent_preview.py", "cuda_malloc.py", "LICENSE", "requirements.txt"]:
        copy_file(comfy / name, output / "ComfyUI" / name)
    # Recipients supply datasets and encoder weights. A separately selected and
    # validated new Semantic Bridge may be included in a final release only.
    (output / "models/text_encoders").mkdir(parents=True, exist_ok=True)
    (output / "models/text_encoders/放入H3编码器.txt").write_text("将兼容的 H3 文本编码器 safetensors 放在这里，或在启动器中浏览选择。\n本包不附带大型编码器权重。\n", encoding="utf-8")
    if release_asset is not None:
        copy_release_asset(release_weight, release_model_card, release_training_report,
                           release_video_ab_report, release_asset, output)
    # The offline trainer reads this as one exact revision value. The manifest
    # separately records every shipped file hash, including uncommitted edits.
    (output / "SOURCE_REVISION.txt").write_text(revision(ROOT) + "\n", encoding="utf-8")
    (output / "ComfyUI/SOURCE_REVISION.txt").write_text(revision(comfy) + "\n", encoding="utf-8")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "python": sys.version.split()[0], "platform": "Windows x64", "torch": metadata.version("torch"),
        "comfy_upstream_revision": revision(comfy), "trainer_upstream_revision": revision(ROOT),
        "trainer_source_note": "Git revision is the base commit; the manifest hashes exact shipped bytes, including any uncommitted changes.",
        "comfy_source_note": "Snapshot of the working source tree; may include local upstream patches. Distributed source is authoritative.",
        "packages": {n: d.version for n, d in sorted(dependencies.items())},
        "launcher_provenance": launcher_receipt,
        "launcher_capabilities": ["legacy numeric bridge+judge", "preference_only bridge with explicit group splits", "validation-only preference diagnostics", "experimental equal-length aligned-token and variable-length monotone-token preparation, bridge-only training and fixed-alpha validation", "opt-in video_feedback_v1 external evidence verification at inspect, encode, preparation and train", "explicit_sentence_target_v1: built train-target manifest, CPU dry-run then fixed-step fit, optional strict CLI-mapped parameters, no resume or video-quality claim"],
        "scope": "Semantic Bridge text-conditioning trainer, ComfyUI inference plugin and MiniMax H3 prompt skill; no H3 video LoRA training backend",
        "prompt_skill": "skills/minimax-h3-anime-battle-prompts/SKILL.md",
        "agent_handoff": "handoff/README.md",
        "comfyui_plugin": f"plugin/{PLUGIN_NAME}",
        "excluded": ["roadmap.md", "local/", "runs/", "personal settings", "all user datasets", "author weights", "author 676 pairs", "training outputs other than an explicitly selected release weight", "large H3 encoder weights"],
        "release_asset": release_asset,
        "licenses": "Python/LICENSE.txt, LICENSE, ComfyUI/LICENSE, python/Lib/site-packages/*dist-info/ license files",
        "files": package_files(output, release_asset),
    }
    (output / "PORTABLE_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Portable folder ready: " + str(output), flush=True)


def archive(output, destination):
    # Refuse stale folders and any mutation after the clean build. In particular,
    # --archive-only must never turn an old research bundle into a new release.
    files = verify_package(output)
    if destination.exists():
        raise FileExistsError("Existing archive is preserved; choose a new destination")
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for rel in [*files, "PORTABLE_MANIFEST.json"]:
            z.write(output / rel, Path(output.name) / rel)
    print("ZIP ready: " + str(destination), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--comfy-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=ROOT / "dist/WushuBridge-Portable-Preference")
    ap.add_argument("--zip", type=Path)
    ap.add_argument("--archive-only", action="store_true")
    ap.add_argument("--launcher", type=Path, help="Freshly compiled launcher; copied under the standard EXE name")
    ap.add_argument("--release-weight", type=Path, help="Validated new H3 Semantic Bridge safetensors to include in the final package")
    ap.add_argument("--release-model-card", type=Path, help="JSON card binding the selected weight, H3 encoder and passed video A/B report hashes")
    ap.add_argument("--release-training-report", type=Path, help="Local training report whose SHA-256 must match the card; not bundled")
    ap.add_argument("--release-video-ab-report", type=Path, help="Local passed H3 video A/B report whose SHA-256 must match the card; not bundled")
    args = ap.parse_args()
    if args.archive_only and any(value is not None for value in (
        args.release_weight, args.release_model_card, args.release_training_report, args.release_video_ab_report
    )):
        ap.error("--release-* are build-only options; --archive-only reads the sealed manifest")
    if not args.archive_only:
        build(args.output.resolve(), args.comfy_root.resolve(), args.launcher,
              args.release_weight, args.release_model_card,
              args.release_training_report, args.release_video_ab_report)
    if args.zip:
        archive(args.output.resolve(), args.zip.resolve())
