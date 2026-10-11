"""Prepare a preference split manifest; CPU software/data checks only, no encoding or training."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from wushu_bridge.trainer_data import read_pairs, pair_supervision, partition_text_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pairs', help='Your UTF-8 preference JSONL')
    parser.add_argument('--output', required=True, help='A new split manifest JSON file')
    args = parser.parse_args()
    source = Path(args.pairs).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError('Output already exists; choose a new split file. Nothing was overwritten.')
    rows = read_pairs(source)
    if pair_supervision(rows) != 'preference_only':
        raise ValueError('Use authored preference-only pairs without numeric scores.')
    if len(rows) < 8:
        raise ValueError('At least 8 text pairs are required by the trainer.')
    if any(r.get('label_type') != 'authoring_rubric' for r in rows):
        raise ValueError('This starter tool accepts authoring_rubric text preferences only.')
    if any(not isinstance(r.get('group_id'), str) or not r['group_id'].strip() for r in rows):
        raise ValueError('Every row needs a nonempty string group_id.')
    original = partition_text_pairs(rows, None)
    indices = {'train': original['train'] + original['calibration'],
               'validation': original['val'], 'test': original['test']}
    manifest = {'schema': 1, 'split_unit': 'group_id', 'splits': {
        name: sorted({rows[i]['group_id'] for i in members}) for name, members in indices.items()
    }}
    checked = partition_text_pairs(rows, manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': 'split_created', 'pairs': len(rows),
                      'split_pair_counts': {key: len(value) for key, value in checked.items()},
                      'split_group_counts': {key: len(value) for key, value in manifest['splits'].items()},
                      'notice': 'Text/group software checks only. No H3 encoding, training or video quality claim.'},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
