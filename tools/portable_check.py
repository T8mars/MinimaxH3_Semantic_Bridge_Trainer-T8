"""Check a distributed trainer without loading H3 weights or starting training."""
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-files', action='store_true', help='Also hash every packaged file')
    args = parser.parse_args()
    required = ('WushuBridge-Trainer.exe', 'python/python.exe', 'ComfyUI/comfy/sd.py',
                'configs/preference_v1.json', 'AGENT交接.md', '训练集制作.md', '如何训练.md',
                'datasets/wushu_pairs_v1_pairs.jsonl', 'datasets/preference_example.jsonl',
                'datasets/preference_example_split.json')
    for name in required:
        if not (ROOT / name).is_file():
            raise FileNotFoundError(name)
    runtime = ROOT / 'python'
    if not Path(sys.executable).resolve().is_relative_to(runtime.resolve()):
        raise ValueError('Run this check with the bundled python/python.exe.')
    versions = {}
    for name in ('torch', 'numpy', 'safetensors', 'transformers', 'tokenizers', 'sentencepiece', 'av'):
        module = importlib.import_module(name)
        if not Path(module.__file__).resolve().is_relative_to(runtime.resolve()):
            raise ValueError('Dependency resolved outside the package: ' + name)
        versions[name] = str(getattr(module, '__version__', 'available'))
    import torch
    from wushu_bridge.trainer_data import read_pairs, pair_supervision, partition_text_pairs
    numeric = read_pairs(ROOT / 'datasets/wushu_pairs_v1_pairs.jsonl')
    if len(numeric) != 676 or pair_supervision(numeric) != 'numeric_scores':
        raise ValueError('Author dataset is missing or changed.')
    sample = read_pairs(ROOT / 'datasets/preference_example.jsonl')
    if len(sample) != 8 or pair_supervision(sample) != 'preference_only':
        raise ValueError('Preference format example is missing or changed.')
    splits = partition_text_pairs(sample, ROOT / 'datasets/preference_example_split.json')
    from wushu_bridge.trainer_engine import RunConfig
    config = RunConfig(**json.loads((ROOT / 'configs/preference_v1.json').read_text(encoding='utf-8')))
    config.validate()
    from wushu_bridge.inference_guidance import make_inference_guidance
    from types import SimpleNamespace
    fixture = SimpleNamespace(cfg=SimpleNamespace(extra={}))
    guidance = make_inference_guidance(fixture, {'alpha_min':.05,'alpha_max':.35,'magnitude_mode':'none'})
    if guidance['status'] != 'unvalidated_starting_point' or guidance['settings']['alpha'] != .12:
        raise ValueError('Deployment guidance must distinguish a starting point from a fixed contract.')
    count = 0
    if args.verify_files:
        manifest = json.loads((ROOT / 'PORTABLE_MANIFEST.json').read_text(encoding='utf-8'))
        for relative, expected in manifest['files'].items():
            path = ROOT / relative
            if not path.is_file() or path.stat().st_size != expected['bytes']:
                raise ValueError('Missing file or different size: ' + relative)
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != expected['sha256']:
                raise ValueError('Different file hash: ' + relative)
            count += 1
    print(json.dumps({'status': 'passed', 'python': sys.version.split()[0],
                      'bundled_dependency_versions': versions, 'cuda_available': torch.cuda.is_available(),
                      'author_numeric_pairs': len(numeric), 'format_example_pairs': len(sample),
                      'format_example_splits': {key: len(value) for key, value in splits.items()},
                      'verified_packaged_files': count,
                      'scope': 'Software, dependency and data-format checks only. No H3 weight was loaded; no training or video quality was tested.'},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
