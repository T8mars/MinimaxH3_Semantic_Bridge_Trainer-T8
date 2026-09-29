"""Reusable WushuBridge trainer: inspect / pairs / encode / train / verify."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from wushu_bridge.trainer_data import encode_pairs, inspect_pairs, partition_text_pairs, read_pairs, sha256, validate_dataset, write_json


def revision(path):
    # A portable distribution contains source and a revision receipt, not Git.
    receipt = Path(path) / "SOURCE_REVISION.txt"
    if receipt.is_file():
        value = receipt.read_text(encoding="utf-8").strip()
        if len(value) == 40 and all(c in "0123456789abcdef" for c in value):
            return value
        if value == "unversioned":
            return value
        raise ValueError("Invalid SOURCE_REVISION.txt; expected one Git SHA or unversioned")
    try:
        p = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else "unversioned"
    except FileNotFoundError:
        return "unversioned"


def code_hash():
    h = hashlib.sha256()
    for path in sorted((ROOT / "wushu_bridge").rglob("*.py")):
        h.update(path.relative_to(ROOT).as_posix().encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def validate_provenance(ds):
    try:
        identity = json.loads(ds.meta.encoder)
    except (ValueError, TypeError):
        raise ValueError("Expected structured H3 encoder provenance; encode the JSONL with this CLI") from None
    if not isinstance(identity, dict) or identity.get("tokenizer") != "MiniMaxH3Tokenizer" or identity.get("mode") != "t2v" or identity.get("dim") != 5120:
        raise ValueError("Dataset does not identify a real H3 T2V encoder")
    digest = identity.get("encoder_sha256", "")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Encoder SHA256 missing or invalid")
    if not identity.get("encoder_file") or not identity.get("encoder_sources"):
        raise ValueError("Encoder source fingerprint missing")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="Validate JSONL pairs without loading H3")
    inspect.add_argument("pairs")
    inspect.add_argument("--report")
    inspect.add_argument("--split-manifest", help="Preflight explicit group/text split coverage without loading H3")
    inspect.add_argument("--require-supervision", choices=("numeric_scores", "preference_only"),
                         help="Reject data of another supervision type before encoding")
    inspect.add_argument("--video-feedback-root", help="Explicit private root of bound two-seed full-AV evidence")
    pairs = sub.add_parser("pairs", help="Build fresh text pairs using the author's degradation rules")
    pairs.add_argument("--corpus", nargs="+", required=True)
    pairs.add_argument("--output", required=True)
    pairs.add_argument("--max-pairs", type=int, default=2000)
    pairs.add_argument("--seed", type=int, default=1234)
    encode = sub.add_parser("encode", help="Encode the exact JSONL using a local ComfyUI H3 encoder")
    encode.add_argument("pairs")
    encode.add_argument("--comfy-root", required=True)
    encode.add_argument("--encoder", required=True)
    encode.add_argument("--output", required=True)
    encode.add_argument("--cache", default=str(ROOT / "local" / "encoder_cache"))
    encode.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    encode.add_argument("--max-tokens", type=int, default=0)
    encode.add_argument("--video-feedback-root", help="Required for video_feedback_v1 pairs")
    train = sub.add_parser("train", help="Train, or resume an identical run from its last completed epoch")
    train.add_argument("dataset")
    train.add_argument("--output", required=True)
    train.add_argument("--config", default=str(ROOT / "configs" / "author_v1.json"))
    train.add_argument("--stage", choices=("bridge", "judge", "both"), default="both")
    train.add_argument("--split-manifest", help="JSON schema=1, split_unit=group_id, splits mapping names to group IDs")
    train.add_argument("--device", choices=("auto", "cuda", "cpu"))
    train.add_argument("--resume", action="store_true")
    train.add_argument("--init-bridge", help="Initialize a new bridge-only run from exact compatible safetensors weights")
    train.add_argument("--init-bridge-sha256", help="Required SHA256 of --init-bridge; not an optimizer checkpoint")
    train.add_argument("--stop-after-epochs", type=int)
    train.add_argument("--threads", type=int, default=8)
    train.add_argument("--video-feedback-root", help="Reverify external full-AV receipts before training video-feedback data")
    verify = sub.add_parser("verify", help="Hash, load and check distributed or trained weights")
    verify.add_argument("--bridge", default=str(ROOT / "models/wushu_bridge/wushu_bridge_wushu_v1.safetensors"))
    verify.add_argument("--judge", default=str(ROOT / "models/wushu_bridge/wushu_jev_wushu_v1.safetensors"))
    verify.add_argument("--report")
    verify.add_argument("--bridge-only", action="store_true", help="Verify only the bridge; do not load or require a judge")
    evaluate = sub.add_parser("evaluate", help="Detailed holdout diagnostics for a saved bridge")
    evaluate.add_argument("dataset")
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--bridge")
    evaluate.add_argument("--output")
    evaluate.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    evaluate.add_argument("--split", choices=("validation", "test"), default="test",
                          help="Use validation for alpha/magnitude selection; reserve test for final diagnostics")
    evaluate.add_argument("--alpha", type=float, nargs="+", default=[0., .12, .2])
    evaluate.add_argument("--magnitude-mode", choices=("none", "per_token"), default="per_token")
    check = sub.add_parser("check-dataset", help="Verify that an existing dataset matches the supplied text pairs and encoder")
    check.add_argument("dataset")
    check.add_argument("--pairs", required=True)
    check.add_argument("--encoder", required=True)
    aligned = sub.add_parser("prepare-aligned", help="Experimental: prepare native aligned-token pairs without loading H3 weights")
    aligned.add_argument("dataset")
    aligned.add_argument("--pairs", required=True)
    aligned.add_argument("--comfy-root", required=True)
    aligned.add_argument("--encoder", required=True)
    aligned.add_argument("--split-manifest", required=True)
    aligned.add_argument("--output-dir", required=True)
    aligned.add_argument("--max-tokens", type=int, default=2048)
    aligned.add_argument("--method", choices=("equal_length_edit_blocks_v1", "token_id_monotone_transport_v1"),
                         default="equal_length_edit_blocks_v1",
                         help="Opt-in variable-length native token transport; does not train or evaluate video")
    aligned.add_argument("--video-feedback-root", help="Required for video_feedback_v1 pairs")
    aligned.add_argument("--selection-policy", choices=("one_per_group", "all_rows"),
                         default="one_per_group",
                         help="Authored preferences: keep one deterministic row per group (default), or every eligible row; video-feedback always keeps unique admitted scenes")
    args = parser.parse_args(argv)
    if args.command == "train":
        if bool(args.init_bridge) != bool(args.init_bridge_sha256):
            parser.error("--init-bridge and --init-bridge-sha256 must be supplied together")
        if args.init_bridge and (args.resume or args.stage != "bridge"):
            parser.error("--init-bridge requires --stage bridge and cannot be combined with --resume")
        if args.init_bridge and Path(args.output).exists():
            parser.error("--init-bridge requires a new, nonexistent output directory")
    if args.command == "evaluate" and any(not math.isfinite(a) or not 0 <= a <= 1 for a in args.alpha):
        parser.error("--alpha values must be finite and between 0 and 1")
    if args.command == "prepare-aligned":
        from tools.prepare_aligned_dataset import prepare
        result = prepare(args.dataset, args.pairs, args.comfy_root, args.encoder,
                         args.split_manifest, args.output_dir, args.max_tokens, args.method,
                         video_feedback_root=args.video_feedback_root,
                         selection_policy=args.selection_policy)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "inspect":
        rows = read_pairs(args.pairs, video_feedback_root=args.video_feedback_root)
        report = {**inspect_pairs(rows), "sha256":sha256(args.pairs)}
        if args.require_supervision and report["supervision_type"] != args.require_supervision:
            raise ValueError(f"Expected {args.require_supervision} data, got {report['supervision_type']}")
        if args.split_manifest:
            report["split_pair_counts"] = {k:len(v) for k,v in partition_text_pairs(rows, args.split_manifest).items()}
            report["embedding_leakage_check"] = "deferred until encoded training dataset is loaded"
        if args.report:
            write_json(args.report, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "pairs":
        from wushu_bridge.pairs import build_pairs, dump_pairs, load_corpus
        if Path(args.output).exists():
            raise FileExistsError(args.output)
        rows = build_pairs(sources=load_corpus(args.corpus, mode="t2v"), mode="t2v", max_pairs=args.max_pairs, seed=args.seed)
        if not rows:
            raise ValueError("No qualifying pairs")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        dump_pairs(rows, args.output)
        print(json.dumps(inspect_pairs(read_pairs(args.output)), indent=2))
    elif args.command == "encode":
        rows = read_pairs(args.pairs, video_feedback_root=args.video_feedback_root)
        comfy = Path(args.comfy_root).resolve()
        encoder = Path(args.encoder).resolve()
        if not (comfy / "comfy/sd.py").is_file() or not encoder.is_file():
            raise FileNotFoundError("Supply an existing ComfyUI root and H3 encoder file")
        if args.max_tokens < 0 or not args.output.endswith(".npz"):
            raise ValueError("max-tokens must be >=0 and output must end with .npz")
        if Path(args.output).exists() or Path(args.output).with_suffix(".json").exists():
            raise FileExistsError(args.output)
        print("[encode] hashing encoder and tokenizer sources", flush=True)
        source_paths = list((comfy / "comfy/text_encoders").glob("*.py"))
        source_paths += list((comfy / "comfy/text_encoders/qwen25_tokenizer").glob("*"))
        source_paths += [comfy / "comfy" / name for name in ("ops.py", "quant_ops.py", "sd.py", "sd1_clip.py")]
        sources = {str(p.relative_to(comfy)):sha256(p) for p in sorted(source_paths) if p.is_file()}
        identity = {"encoder_file":encoder.name, "encoder_sha256":sha256(encoder), "comfy_commit":revision(comfy),
                    "encoder_sources":sources, "max_tokens":args.max_tokens, "mode":"t2v", "dim":5120,
                    "quantization_note":"Exact local file; equivalence to author's encoder is not established"}
        # ComfyUI may parse sys.argv at import. Keep our CLI arguments out.
        sys.path.insert(0, str(comfy))
        sys.argv = [sys.argv[0]] + (["--cpu"] if args.device == "cpu" else [])
        import comfy.options
        comfy.options.enable_args_parsing()
        import torch
        import comfy.sd
        identity["cache_schema"] = 2
        identity["device"] = args.device
        identity["cuda"] = torch.version.cuda
        identity["gpu"] = torch.cuda.get_device_name() if args.device == "cuda" else None
        identity["compute_capability"] = list(torch.cuda.get_device_capability()) if args.device == "cuda" else None
        identity["packages"] = {}
        for package in ("comfy-kitchen", "transformers", "tokenizers", "safetensors"):
            try:
                identity["packages"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                identity["packages"][package] = "unavailable"
        options = {"load_device":torch.device("cpu"), "offload_device":torch.device("cpu")} if args.device == "cpu" else {}
        clip = comfy.sd.load_clip(ckpt_paths=[str(encoder)], clip_type=comfy.sd.CLIPType.MINIMAX, model_options=options)
        tokenizer_name = type(clip.tokenizer).__name__
        if "MiniMaxH3" not in tokenizer_name:
            raise ValueError(f"Not a MiniMax H3 encoder: {tokenizer_name}")
        identity["tokenizer"] = tokenizer_name
        identity["torch"] = str(torch.__version__)
        ds = encode_pairs(rows, clip, args.cache, args.output, identity, args.max_tokens,
                          video_feedback_root=args.video_feedback_root)
        write_json(str(args.output) + ".receipt.json", {"pairs_sha256":sha256(args.pairs), "dataset_sha256":sha256(args.output),
                                                       "identity":identity, "stats":ds.stats()})
    elif args.command == "train":
        import torch
        from wushu_bridge.dataset import PairDataset
        from wushu_bridge.trainer_engine import RunConfig, train_run
        torch.set_num_threads(args.threads)
        values = json.loads(Path(args.config).read_text(encoding="utf-8"))
        if args.device:
            values["device"] = args.device
        cfg = RunConfig(**values)
        cfg.validate()
        path = Path(args.dataset).resolve()
        if not path.with_suffix(".json").is_file():
            raise ValueError("Dataset metadata .json is required")
        ds = PairDataset.load(str(path))
        validate_dataset(ds)
        if ds.meta.mode != "t2v":
            raise ValueError("This CLI run requires a T2V dataset")
        validate_provenance(ds)
        identity = {"dataset_sha256":sha256(path), "metadata_sha256":sha256(path.with_suffix(".json")),
                    "code_sha256":code_hash(), "upstream_commit":revision(ROOT), "torch":str(torch.__version__),
                    "cuda":torch.version.cuda, "device":torch.cuda.get_device_name() if cfg.device != "cpu" and torch.cuda.is_available() else "cpu"}
        with torch.inference_mode(False), torch.enable_grad():
            print(json.dumps(train_run(ds, cfg, args.output, identity, args.resume, args.stage,
                                       args.stop_after_epochs, args.split_manifest,
                                       video_feedback_root=args.video_feedback_root,
                                       init_bridge=args.init_bridge,
                                       init_bridge_sha256=args.init_bridge_sha256), indent=2))
    elif args.command == "check-dataset":
        receipt = json.loads(Path(args.dataset + ".receipt.json").read_text(encoding="utf-8"))
        print("Verifying dataset and encoder hashes", flush=True)
        for actual, expected in ((sha256(args.pairs), receipt["pairs_sha256"]),
                                 (sha256(args.encoder), receipt["identity"]["encoder_sha256"]),
                                 (sha256(args.dataset), receipt["dataset_sha256"])):
            if actual != expected:
                raise ValueError("Existing dataset does not match these pairs/encoder; choose a new dataset output path")
        print("Dataset matches the supplied pairs and encoder.")
    elif args.command == "evaluate":
        import torch
        from wushu_bridge.dataset import PairDataset
        from wushu_bridge.bridge_model import load_bridge
        from wushu_bridge.trainer_engine import RunConfig, run_signature
        from wushu_bridge.trainer_data import partition
        from wushu_bridge.trainer_eval import bridge_diagnostics
        torch.set_num_threads(8)
        run = Path(args.run)
        manifest = json.loads((run / "run.json").read_text(encoding="utf-8"))
        if manifest["identity"]["dataset_sha256"] != sha256(args.dataset):
            raise ValueError("Dataset does not match training run")
        if manifest["identity"]["metadata_sha256"] != sha256(Path(args.dataset).with_suffix(".json")):
            raise ValueError("Metadata does not match training run")
        ds = PairDataset.load(args.dataset)
        validate_dataset(ds)
        validate_provenance(ds)
        cfg = RunConfig(**manifest["config"])
        if run_signature(cfg, manifest["identity"]) != manifest["signature"]:
            raise ValueError("Run signature does not match its configuration and identity")
        splits = json.loads((run / "splits.json").read_text(encoding="utf-8"))
        source_splits = None
        if "split_manifest_sha256" in manifest["identity"]:
            source_splits = json.loads((run / "source_split_manifest.json").read_text(encoding="utf-8"))
            digest = hashlib.sha256(json.dumps(source_splits, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if digest != manifest["identity"]["split_manifest_sha256"]:
                raise ValueError("Source split manifest does not match the training identity")
        if splits != partition(ds, cfg.seed, source_splits):
            raise ValueError("Split manifest differs from the deterministic data split")
        ids = splits["val" if args.split == "validation" else "test"]
        path = args.bridge or str(run / "bridge.safetensors")
        model = load_bridge(path, device=args.device)
        if not args.bridge and model.cfg.extra.get("signature") != manifest["signature"]:
            raise ValueError("Saved bridge does not match this run")
        report = {"bridge_sha256":sha256(path), "dataset_sha256":sha256(args.dataset),
                  "evaluated_split":args.split, "pair_ids":ids,
                  "magnitude_mode":args.magnitude_mode,
                  "metrics":[bridge_diagnostics(model, ds, ids, cfg, a, args.magnitude_mode) for a in args.alpha],
                  "comparison_note":"An external author model may have seen these pairs; this is not an independent head-to-head ranking"}
        if args.split == "test":
            report["test_pair_ids"] = ids  # Preserve the existing report contract.
        prefix = "external_" + sha256(path)[:12] if args.bridge else "bridge"
        suffix = "_validation_diagnostics.json" if args.split == "validation" else "_diagnostics.json"
        output = args.output or str(run / (prefix + suffix))
        write_json(output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        import torch
        from wushu_bridge.bridge_model import load_bridge
        from wushu_bridge.judge import load_judge
        from wushu_bridge.apply import application_contract, apply_bridge_to_conditioning
        torch.set_num_threads(4)
        model = load_bridge(args.bridge)
        contract = application_contract(model)
        apply_settings = ({
            "alpha": contract["alpha_min"],
            "magnitude_match_mode": contract["magnitude_match"],
            "token_span": contract["token_span"],
            "tail_ratio": contract["tail_ratio"],
            "max_tokens_per_chunk": contract["chunk_tokens"],
            "allow_dim_mismatch": contract["allow_dim_mismatch"],
        } if contract is not None else {"alpha": .12})
        judge = None if args.bridge_only else load_judge(args.judge)
        torch.manual_seed(0)
        cond = [[torch.randn(1, 16, 5120), {"keep":"metadata", "minimax_frame_count":124}]]
        with torch.no_grad():
            result, _ = apply_bridge_to_conditioning(cond, model, **apply_settings)
            prob = judge.probability(result[0][0]) if judge is not None else None
        assert result[0][1]["keep"] == "metadata" and torch.isfinite(result[0][0]).all() and (prob is None or torch.isfinite(prob).all())
        report = {"bridge_sha256":sha256(args.bridge), "judge_sha256":None if args.bridge_only else sha256(args.judge),
                  "bridge_bytes":Path(args.bridge).stat().st_size, "judge_bytes":None if args.bridge_only else Path(args.judge).stat().st_size,
                  "stage":"bridge" if args.bridge_only else "both",
                   "load_and_forward_passed":True, "application_settings":apply_settings,
                   "synthetic_check_only":True, "video_quality_validated":False}
        if args.report:
            write_json(args.report, report)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
