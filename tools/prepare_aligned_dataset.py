"""Prepare experimental exact-position H3 token pairs without loading H3 weights."""
from __future__ import annotations

import copy
import builtins
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile
from difflib import SequenceMatcher
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from wushu_bridge.dataset import PairDataset, PairMeta
from wushu_bridge.token_alignment import METHOD, TRANSPORT_METHOD, transport_edges, validate_token_alignment
from wushu_bridge.trainer_data import (partition, partition_text_pairs, read_pairs,
                                      sha256, validate_dataset, write_json)
from wushu_bridge.video_feedback import LABEL as VIDEO_FEEDBACK_LABEL

TOKENIZER_FILES = (
    'comfy/sd1_clip.py', 'comfy/text_encoders/minimax.py',
    'comfy/text_encoders/qwen3vl.py',
    'comfy/text_encoders/qwen25_tokenizer/vocab.json',
    'comfy/text_encoders/qwen25_tokenizer/merges.txt',
    'comfy/text_encoders/qwen25_tokenizer/tokenizer_config.json',
)


def text_sha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def tokenizer_provenance(comfy, identity):
    sources = {}
    for name, digest in identity['encoder_sources'].items():
        name = name.replace('\\', '/')
        parts = name.split('/')
        if name.startswith('/') or ':' in name or '..' in parts:
            raise ValueError('Unsafe tokenizer provenance path')
        normalized = '/'.join(p for p in parts if p)
        if normalized in sources and sources[normalized] != digest:
            raise ValueError('Conflicting tokenizer provenance paths')
        sources[normalized] = digest
    verified = {}
    for name in TOKENIZER_FILES:
        candidate = (comfy / name).resolve()
        if not candidate.is_relative_to(comfy) or not candidate.is_file():
            raise ValueError(f'Missing tokenizer source: {name}')
        digest = sha256(candidate)
        if digest != sources.get(name):
            raise ValueError(f'Tokenizer source fingerprint mismatch: {name}')
        verified[name] = digest
    versions = {}
    for name in ('transformers', 'tokenizers'):
        versions[name] = importlib.metadata.version(name)
        if versions[name] != identity.get('packages', {}).get(name):
            raise ValueError(f'Tokenizer package version mismatch: {name}; use the encoding environment')
    return {'sources': verified, 'packages': versions}


def load_tokenizer(comfy):
    # This process is CLI-owned: parse ComfyUI with CPU flags, never our args.
    comfy_root = Path(comfy).resolve()
    original_path = list(sys.path)
    sys.path.insert(0, str(comfy_root))
    original = sys.argv
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    try:
        # Some ComfyUI optional kernels probe CUDA even with --cpu. Hide GPU
        # devices before importing its modules in this dedicated CLI process.
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        sys.argv = [sys.argv[0], '--cpu']
        import torch
        if torch.cuda.is_initialized():
            raise RuntimeError('Run preparation in a fresh CPU CLI process, not a GPU-initialized host')
        # Optional attention backends in some ComfyUI versions probe CUDA
        # unconditionally at import. Tokenization never uses those kernels.
        original_import = builtins.__import__
        def cpu_import(name, *args, **kwargs):
            if name.split('.')[0] in {'triton', 'sageattention', 'sageattn3', 'flash_attn', 'xformers'}:
                raise ModuleNotFoundError(f'{name} disabled during CPU-only token preparation', name=name.split('.')[0])
            return original_import(name, *args, **kwargs)
        with (patch.object(torch.cuda, 'is_available', return_value=False), patch.object(
                torch.cuda, '_lazy_init', side_effect=RuntimeError('GPU initialization is forbidden during token preparation')),
                patch.object(builtins, '__import__', new=cpu_import)):
            import comfy.options
            comfy.options.enable_args_parsing()
            from comfy.text_encoders.minimax import MiniMaxH3Tokenizer
            for name in ('comfy.sd1_clip', 'comfy.text_encoders.minimax', 'comfy.text_encoders.qwen3vl'):
                expected_path = comfy_root / (name.replace('.', '/') + '.py')
                if Path(sys.modules[name].__file__).resolve() != expected_path:
                    raise RuntimeError('Tokenizer was imported from a different ComfyUI root')
            return MiniMaxH3Tokenizer()
    finally:
        sys.argv = original
        sys.path[:] = original_path
        if visible is None:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = visible


def native_rows(path, prefix, lengths):
    with zipfile.ZipFile(path) as archive, archive.open(prefix + '_flat.npy') as stream:
        version = np.lib.format.read_magic(stream)
        shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
        if fortran or dtype != np.dtype('float16') or shape != (int(lengths.sum()), 5120):
            raise ValueError(f'{prefix}: expected native float16 [tokens,5120] array')
        for n in lengths:
            n = int(n)
            content = stream.read(n * 5120 * 2)
            if len(content) != n * 5120 * 2:
                raise ValueError('Truncated native array')
            row = np.frombuffer(content, dtype=dtype).reshape(n, 5120)
            if not np.isfinite(row).all():
                raise ValueError('Nonfinite native embedding')
            yield row
        if stream.read(1):
            raise ValueError('Trailing native array bytes')


def prepare(dataset, pairs, comfy_root, encoder, split_manifest, output_dir, max_tokens=2048,
            method=METHOD, video_feedback_root=None, selection_policy='one_per_group'):
    dataset, pairs, comfy, encoder, split_path, output = map(
        lambda p: Path(p).resolve(), (dataset, pairs, comfy_root, encoder, split_manifest, output_dir))
    if output.exists():
        raise FileExistsError(f'Refusing existing output directory: {output}')
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError('max_tokens must be a positive integer')
    if method not in (METHOD, TRANSPORT_METHOD):
        raise ValueError('Unknown token preparation method')
    if selection_policy not in ('one_per_group', 'all_rows'):
        raise ValueError('Unknown token selection policy')
    rows = read_pairs(pairs, video_feedback_root=video_feedback_root)
    video_feedback = all(r.get('label_type') == VIDEO_FEEDBACK_LABEL for r in rows)
    ids = [r.get('id') for r in rows]
    if any(not isinstance(i, str) or not i.strip() for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Stable unique nonempty record IDs are required')
    source_split = json.loads(split_path.read_text(encoding='utf-8-sig'))
    source_partitions = partition_text_pairs(rows, source_split)
    side = {i:s for s,indices in source_partitions.items() for i in indices}
    meta_path = dataset.with_suffix('.json')
    raw = json.loads(meta_path.read_text(encoding='utf-8-sig'))
    receipt_path = Path(str(dataset) + '.receipt.json')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8-sig'))
    identity = json.loads(raw['encoder'])
    if (raw.get('kind') != 'wushu_bridge_pairs' or type(raw.get('schema')) is not int or raw['schema'] != 1
            or raw.get('supervision_type') != 'preference_only' or raw.get('score_label_type') != 'none'
            or raw.get('dim') != 5120 or raw.get('mode') != 't2v'
            or identity.get('tokenizer') != 'MiniMaxH3Tokenizer' or identity.get('dim') != 5120
            or identity.get('mode') != 't2v' or identity != receipt.get('identity')):
        raise ValueError('A real H3 T2V preference encoding identity/receipt is required')
    if raw.get('records') != rows or raw.get('count') != len(rows):
        raise ValueError('Encoded metadata records differ from original JSONL')
    hashes = {'pairs':sha256(pairs), 'dataset':sha256(dataset), 'metadata':sha256(meta_path),
              'split':sha256(split_path), 'encoding_receipt':sha256(receipt_path), 'encoder':sha256(encoder)}
    if (hashes['pairs'] != receipt.get('pairs_sha256') or hashes['dataset'] != receipt.get('dataset_sha256')
            or hashes['encoder'] != identity.get('encoder_sha256')):
        raise ValueError('Source pairs, dataset or encoder hash mismatch')
    provenance = tokenizer_provenance(comfy, identity)
    tokenizer = load_tokenizer(comfy)
    with np.load(dataset, allow_pickle=False) as z:
        xl, yl = z['x_len'], z['y_len']
        for prefix, lengths in (('x',xl),('y',yl)):
            if lengths.ndim != 1 or lengths.dtype.kind not in 'iu' or len(lengths) != len(rows) or (lengths <= 0).any():
                raise ValueError('Invalid native row lengths')
            score = z[prefix + '_score']
            if score.shape != (len(rows),) or score.dtype.kind != 'f' or not np.isnan(score).all():
                raise ValueError('Preference encoding needs NaN score sentinels')
        if 'e_len' in z and len(z['e_len']):
            raise ValueError('Unpaired embeddings are unsupported')
    eligible, alignment, rejected = {}, {}, []
    for i,row in enumerate(rows):
        token_ids = {s:[v[0] for v in tokenizer.tokenize_with_weights(row[s])['qwen3vl_32b'][0]]
                     for s in ('bad','good')}
        if len(token_ids['bad']) != xl[i] or len(token_ids['good']) != yl[i]:
            raise ValueError(f'Native tokenizer length mismatch: {row["id"]}')
        blocks = SequenceMatcher(None,token_ids['bad'],token_ids['good'],autojunk=False).get_opcodes()
        edits = [b for b in blocks if b[0] != 'equal']
        if max(xl[i],yl[i]) > max_tokens:
            reason = 'over_max_tokens_no_truncation'
        elif not edits or (method == METHOD and any(t != 'replace' or b-a != d-c for t,a,b,c,d in edits)):
            reason = 'not_equal_length_edit_blocks'
        else:
            reason = None
        if reason:
            rejected.append({'source_index':i,'id':row['id'],'reason':reason})
            continue
        if method == TRANSPORT_METHOD:
            transport_edges(token_ids['bad'], token_ids['good'])
        # A video-feedback group_id is the whole work. Selecting one per group
        # would silently discard independently admitted scenes of that work.
        selection_key = row['scene_family_id'] if video_feedback else row['group_id']
        eligible.setdefault(selection_key,[]).append(i)
        alignment[i] = {'method':method,'encoder_identity_sha256':text_sha(raw['encoder']),
                        'bad_ids':token_ids['bad'],'good_ids':token_ids['good'],
                        'bad_text_sha256':text_sha(row['bad']),'good_text_sha256':text_sha(row['good']),
                        'opcodes':[list(b) for b in blocks],'source_index':i}
    if video_feedback and any(len(v) != 1 for v in eligible.values()):
        raise ValueError('video_feedback_v1 refuses duplicate scene_family_id; keep every admitted scene unique')
    selected = ([i for members in eligible.values() for i in members]
                if video_feedback or selection_policy == 'all_rows' else
                [min(v,key=lambda i:(text_sha(rows[i]['id']),rows[i]['id'])) for v in eligible.values()])
    selected.sort(key=lambda i:(rows[i]['group_id'],rows[i]['id']))
    wanted = set(selected)
    native, seen, row_hashes = {}, {}, {}
    for prefix,lengths in (('x',xl),('y',yl)):
        native[prefix] = {}
        for i,row in enumerate(native_rows(dataset,prefix,lengths)):
            digest = hashlib.sha256(row.tobytes()).hexdigest()
            if digest in seen and seen[digest] != side[i]:
                raise ValueError('Original embedding leakage across splits')
            seen[digest] = side[i]
            if i in wanted:
                native[prefix][i] = row.copy()
                row_hashes.setdefault(i,{})[prefix] = digest
                if method == TRANSPORT_METHOD:
                    alignment[i][('bad' if prefix == 'x' else 'good') + '_embedding_sha256'] = digest
    ds = PairDataset(PairMeta(**{k:v for k,v in raw.items() if k != 'records'}))
    for i in selected:
        record = copy.deepcopy(rows[i]); record['token_alignment'] = alignment[i]
        ds.add(native['x'][i],native['y'][i],float('nan'),float('nan'),record)
    derived_split = copy.deepcopy(source_split)
    selected_groups = {rows[i]['group_id'] for i in selected}
    derived_split['splits'] = {s:[g for g in groups if g in selected_groups] for s,groups in source_split['splits'].items()}
    derived_split['selection'] = ('all unique admitted scene families; whole-work group_id retained'
                                  if video_feedback else
                                  'all eligible rows, output ordered by group_id/id' if selection_policy == 'all_rows' else
                                  'one per group, lowest SHA256(id UTF8), output ordered by group_id/id')
    validate_dataset(ds)
    validate_token_alignment(ds,max_tokens,method=method)
    splits = partition(ds,1234,derived_split)
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.'+output.name+'-',dir=output.parent)).resolve()
    try:
        ds.save(str(temporary/'pairs.npz'))
        (temporary/'pairs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in ds.records),encoding='utf-8')
        write_json(temporary/'split_manifest.json',derived_split)
        standard = {'pairs_sha256':sha256(temporary/'pairs.jsonl'),'dataset_sha256':sha256(temporary/'pairs.npz'),
                    'identity':identity,'preparation_method':method}
        write_json(temporary/'pairs.npz.receipt.json',standard)
        report = {'schema':1,'status':'completed','method':method,'source_sha256':hashes,
                  'script_sha256':sha256(__file__),'tokenizer_provenance':provenance,
                  'selection':derived_split['selection'],
                  'selection_policy':('all_unique_scene_families' if video_feedback else selection_policy),
                  'eligible_count':sum(map(len,eligible.values())),
                  'selected_count':len(selected),
                  'counts':{s:len(v) for s,v in splits.items()},
                  'native_reencoded':False,'test_metrics_computed':False,'training_run':False,
                  'supervision':('externally verified video-feedback receipt hashes; actual human AV perception is not machine-proven'
                                 if video_feedback else 'authored text preference; not video-quality evidence'),
                  'selected':[{'source_index':i,'id':rows[i]['id'],'group_id':rows[i]['group_id'],
                               'split':side[i],'native_sha256':row_hashes[i]} for i in selected],
                  'rejected':rejected,
                  'eligible_not_selected':[rows[i]['id'] for v in eligible.values() for i in v if i not in wanted],
                  'artifacts':{p.name:{'sha256':sha256(p),'bytes':p.stat().st_size} for p in temporary.iterdir()}}
        if method == TRANSPORT_METHOD:
            report['transport_policy'] = {
                'implementation':'difflib.SequenceMatcher','autojunk':False,
                'python_version':list(sys.version_info[:3]),
                'token_alignment_source_sha256':sha256(ROOT/'wushu_bridge/token_alignment.py'),
                'replacement':'uniform_cell_overlap','insertion_deletion_boundary':'left_else_right',
                'row_normalization':'sum_to_one'}
        write_json(temporary/'preparation_receipt.json',report)
        temporary.rename(output)
    finally:
        if temporary.exists() and temporary.parent == output.parent and temporary.name.startswith('.'+output.name+'-'):
            shutil.rmtree(temporary)
    return {'status':'completed','output':str(output),'counts':report['counts'],
            'eligible_count':report['eligible_count'],'selected_count':len(selected),
            'test_metrics_computed':False,'experimental':True}
