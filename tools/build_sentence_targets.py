"""CPU-only explicit-span sentence targets; paired-y supervision, never a model.

Preserves native x/y dtype and rebuilds monotone transport T outside sentence blocks.
No tokenizer inference, automatic sentence segmentation, encoding, or training.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile
from types import SimpleNamespace
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from wushu_bridge.apply import rms_normalize, magnitude_match
from wushu_bridge.token_alignment import transport_edges
from wushu_bridge.trainer_engine import transported_target

METHOD = 'sentence_uniform_cell_over_monotone_v1'
SINGLE_PAIR_NPZ_KEYS = frozenset({'x','y','target','old_target','monotone_target',
    'sentence_mask','outside_sentence_mask','prefix_mask'})
EMBEDDING_NPZ_KEYS = frozenset({'x','y','target','old_target','monotone_target'})

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def rawsha(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def textsha(s): return hashlib.sha256(s.encode('utf-8')).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def bound_path(ref, base, bindings):
    if not isinstance(ref, dict) or not isinstance(ref.get('path'), str):
        raise ValueError('Source needs explicit path and sha256')
    path = Path(ref['path'])
    if not path.is_absolute(): path = base / path
    path = path.resolve()
    digest = ref.get('sha256')
    if not isinstance(digest, str) or len(digest) != 64 or sha(path) != digest:
        raise ValueError(f'Source SHA256 mismatch: {path}')
    bindings[str(path)] = digest
    return path

def load_array(ref, base, bindings):
    path = bound_path(ref, base, bindings)
    if path.suffix == '.npy':
        if 'key' in ref: raise ValueError('NPY has no named key')
        a = np.load(path, allow_pickle=False)
    elif path.suffix == '.npz':
        if not isinstance(ref.get('key'), str): raise ValueError('NPZ requires explicit key')
        with np.load(path, allow_pickle=False) as z:
            if not set(z.files) <= SINGLE_PAIR_NPZ_KEYS or ref['key'] not in EMBEDDING_NPZ_KEYS:
                raise ValueError('Refusing aggregate or unknown NPZ schema before array materialization')
            # A mislabeled x=[pairs,tokens,dim] must also be rejected before loading.
            with zipfile.ZipFile(path) as archive, archive.open(ref['key']+'.npy') as stream:
                version=np.lib.format.read_magic(stream)
                shape,fortran,dtype=np.lib.format._read_array_header(stream,version)
                if (dtype not in (np.dtype('float16'),np.dtype('float32'))
                    or not (len(shape)==2 or (len(shape)==3 and shape[0]==1))):
                    raise ValueError('NPZ embedding header is not a single native pair side')
            a = z[ref['key']].copy()
    elif path.suffix == '.safetensors':
        if not isinstance(ref.get('key'), str): raise ValueError('Safetensors requires explicit key')
        from safetensors import safe_open
        with safe_open(str(path), framework='pt', device='cpu') as f:
            t = f.get_tensor(ref['key'])
        if t.dtype not in (torch.float16, torch.float32):
            raise ValueError('Only actual FP16/FP32 source tensors are supported')
        a = t.numpy()
    else: raise ValueError('Expected explicit NPY, single-pair NPZ, or safetensors source')
    if a.ndim == 3 and a.shape[0] == 1: a = a[0]
    if a.ndim != 2 or not len(a) or a.dtype not in (np.float16, np.float32):
        raise ValueError('Expected native FP16/FP32 [tokens,dim] (or leading batch=1)')
    if not np.isfinite(a).all(): raise ValueError('Nonfinite native source')
    if 'raw_sha256' in ref and rawsha(a) != ref['raw_sha256']:
        raise ValueError('Source raw payload SHA mismatch')
    return np.ascontiguousarray(a)

def span(value, length, name):
    if (not isinstance(value, list) or len(value) != 2
        or any(type(v) is not int for v in value)
        or not 0 <= value[0] < value[1] <= length):
        raise ValueError(f'{name}: nonempty half-open integer span required')
    return tuple(value)

def uniform_edges(p, q):
    if type(p) is not int or type(q) is not int or p < 1 or q < 1:
        raise ValueError('Uniform-cell blocks must be nonempty on both sides')
    result = []
    # Traverse only intersecting cells; ascending (i,j) matches the reference index_add_ accumulation order.
    covered = set()
    for i in range(p):
        total = 0
        for j in range((i*q)//p, min(q, ((i+1)*q+p-1)//p)):
            overlap = min((i+1)*q,(j+1)*p)-max(i*q,j*p)
            if overlap > 0:
                result.append((i,j,overlap/q))
                total += overlap; covered.add(j)
        if total != q: raise ValueError('Uniform-cell row sum failed')
    if covered != set(range(q)): raise ValueError('Uniform-cell column coverage failed')
    return result

def validate_blocks(blocks, nx, ny, edges):
    if not isinstance(blocks,list) or not blocks: raise ValueError('At least one explicit block required')
    spans = []
    for i,b in enumerate(blocks):
        a,z = span(b.get('x_token_span'),nx,f'block{i}.x')
        c,d = span(b.get('y_token_span'),ny,f'block{i}.y')
        if spans and (a < spans[-1][1] or c < spans[-1][3]):
            raise ValueError('Blocks must be ordered and nonoverlapping on both axes')
        for ix,iy,_ in edges:
            if (a <= ix < z) != (c <= iy < d):
                raise ValueError('Original monotone edge crosses sentence boundary; explicitly expand span')
        spans.append((a,z,c,d))
    return spans

def construct(x_native,y_native,alignment,blocks,prefix_token_span=None,max_tokens=2048):
    """Pure CPU construction; caller must verify native token/text/span provenance."""
    if type(max_tokens) is not int or max_tokens < 1: raise ValueError('Positive integer max_tokens required')
    for a in (x_native,y_native):
        if a.ndim!=2 or a.dtype not in (np.float16,np.float32) or not np.isfinite(a).all():
            raise ValueError('Expected finite native float16/float32 matrix')
    nx,dim=x_native.shape; ny,dy=y_native.shape
    if not nx or not ny or dim != dy or not dim or max(nx,ny)>max_tokens:
        raise ValueError('Invalid source dimensions or cap exceeded; truncation is forbidden')
    for key,n in [('bad_ids',nx),('good_ids',ny)]:
        ids=alignment.get(key)
        if not isinstance(ids,list) or len(ids)!=n or any(type(v)is not int or v<0 for v in ids):
            raise ValueError('Native token IDs/embedding length mismatch')
    old_edges=transport_edges(alignment['bad_ids'],alignment['good_ids'])
    spans=validate_blocks(blocks,nx,ny,old_edges)
    x=torch.from_numpy(np.array(x_native,copy=True)).float()
    y=torch.from_numpy(np.array(y_native,copy=True)).float()
    ds=SimpleNamespace(x=[x],y=[y],records=[{'token_alignment':alignment}])
    with torch.no_grad():
        t=transported_target(ds,[0],x[None],y[None],torch.ones((1,nx),dtype=torch.bool))[0]
        s=t.clone()
        for a,b,c,d in spans:
            e=uniform_edges(b-a,d-c)
            ii=torch.tensor([i for i,j,w in e],dtype=torch.long)
            jj=torch.tensor([j+c for i,j,w in e],dtype=torch.long)
            weights=torch.tensor([w for i,j,w in e],dtype=torch.float32)
            bary=torch.zeros((b-a,dim),dtype=torch.float32)
            bary.index_add_(0,ii,rms_normalize(y[jj])*weights[:,None])
            if not bool(torch.isfinite(bary).all()) or not bool((bary.square().mean(-1)>1e-8).all()):
                raise ValueError('Zero/nonfinite sentence barycenter')
            s[a:b]=magnitude_match(bary,x[a:b],'per_token')
    sentence=np.zeros(nx,dtype=bool)
    for a,b,c,d in spans: sentence[a:b]=True
    masks={'sentence_mask':sentence,'outside_sentence_mask':~sentence}
    if prefix_token_span is not None:
        a,b=span(prefix_token_span,nx,'prefix')
        if sentence[a:b].any(): raise ValueError('Protected prefix must not overlap target sentence')
        prefix=np.zeros(nx,dtype=bool);prefix[a:b]=True;masks['prefix_mask']=prefix
    target=s.numpy(); monotone=t.numpy()
    if not np.isfinite(target).all(): raise ValueError('Nonfinite target')
    if target[~sentence].tobytes()!=monotone[~sentence].tobytes():
        raise AssertionError('Outside sentence must stay original T, not x')
    return {'x':np.array(x_native,copy=True),'y':np.array(y_native,copy=True),
            'target':target,'monotone_target':monotone,**masks}

def bpe_decode_bytes(ids,vocab):
    inv={i:s for s,i in vocab.items()}
    bs=list(range(33,127))+list(range(161,173))+list(range(174,256));cs=bs[:];n=0
    for b in range(256):
        if b not in bs: bs.append(b);cs.append(256+n);n+=1
    back=dict(zip(map(chr,cs),bs))
    try: return b''.join(bytes(back[c] for c in inv[i]) for i in ids)
    except (KeyError,TypeError) as e: raise ValueError('Native IDs are not exactly decodable with bound vocab') from e

def validate_text_spans(alignment,texts,blocks,vocab):
    proof=[]
    for side,key in [('x','bad'),('y','good')]:
        text=texts[side];ids=alignment[key+'_ids']
        if textsha(text)!=alignment.get(key+'_text_sha256'):
            raise ValueError('Native alignment text hash mismatch')
        if bpe_decode_bytes(ids,vocab)!=text.encode('utf-8'):
            raise ValueError('Native tokens do not reconstruct exact original UTF8 text')
    for i,block in enumerate(blocks):
        p={'block_index':i}
        for side,key in [('x','bad'),('y','good')]:
            a,b=span(block.get(side+'_token_span'),len(alignment[key+'_ids']),side)
            prefix=bpe_decode_bytes(alignment[key+'_ids'][:a],vocab)
            content=bpe_decode_bytes(alignment[key+'_ids'][a:b],vocab)
            try: decoded=content.decode('utf-8'); cp=prefix.decode('utf-8')
            except UnicodeDecodeError as e: raise ValueError('Span splits a UTF8 character') from e
            if not isinstance(block.get(side+'_text'),str) or decoded!=block[side+'_text']:
                raise ValueError('Explicit span text differs from native token slice')
            p[side]={'token_span':[a,b],'byte_span':[len(prefix),len(prefix)+len(content)],
                     'character_span':[len(cp),len(cp)+len(decoded)],'text_sha256':textsha(decoded)}
        proof.append(p)
    return proof

def build(manifest_path,output_dir):
    manifest_path=Path(manifest_path).resolve();m=read(manifest_path);base=manifest_path.parent
    if m.get('schema')!=1 or m.get('kind')!='sentence_target_input_v1': raise ValueError('Unknown input schema')
    if m.get('purpose') not in ('research_candidate','synthetic_test','regression_test'):
        raise ValueError('Explicit experimental research/test purpose required; building targets does not certify their quality')
    if m.get('source_split') not in ('train','synthetic_test'):
        raise ValueError('This train-only experimental builder refuses val/test source data')
    if not isinstance(m.get('pair_id'),str) or not m['pair_id']: raise ValueError('pair_id required')
    bindings={str(manifest_path):sha(manifest_path)}
    for p in [Path(__file__),ROOT/'wushu_bridge/apply.py',ROOT/'wushu_bridge/trainer_engine.py',ROOT/'wushu_bridge/token_alignment.py']:
        bindings[str(p.resolve())]=sha(p)
    x=load_array(m['x'],base,bindings);y=load_array(m['y'],base,bindings)
    if x.shape[1]!=m.get('expected_dim',5120) or y.shape[1]!=m.get('expected_dim',5120): raise ValueError('Dimension mismatch')
    ap=bound_path(m['alignment'],base,bindings);ar=read(ap)
    if ar.get('source_split',m['source_split'])!=m['source_split']:
        raise ValueError('Alignment source split differs from explicit input split')
    if m['alignment'].get('layout')=='native_pair_records_v1':
        records=ar['records']
        ar={'token_alignment':{
            'bad_ids':records['x']['native_ids'],'good_ids':records['y']['native_ids'],
            'bad_text_sha256':records['x']['prompt_sha256'],'good_text_sha256':records['y']['prompt_sha256']}}
    elif m['alignment'].get('layout') not in (None,'token_alignment_record_v1'):
        raise ValueError('Unknown explicit native alignment layout')
    if 'record_id' in m['alignment']:
        matches=[r for r in ar['records'] if r.get('id')==m['alignment']['record_id']]
        if len(matches)!=1: raise ValueError('Native record selector not unique')
        ar=matches[0]
    alignment=ar.get('token_alignment',ar)
    texts={'x':ar.get('bad'),'y':ar.get('good')}
    if 'texts' in m:
        for side in ('x','y'):
            tp=bound_path(m['texts'][side],base,bindings)
            text=tp.read_bytes().decode('utf-8')
            if texts[side] is not None and texts[side]!=text: raise ValueError('Text file/native record conflict')
            texts[side]=text
    if any(not isinstance(t,str) for t in texts.values()): raise ValueError('Bound native prompt text required')
    for side in ('x','y'):
        if m[side].get('prompt_sha256')!=textsha(texts[side]):
            raise ValueError('Embedding source must explicitly bind its exact prompt SHA')
    vp=bound_path(m['tokenizer_vocab'],base,bindings)
    proof=validate_text_spans(alignment,texts,m['blocks'],read(vp))
    arrays=construct(x,y,alignment,m['blocks'],m.get('prefix_token_span'),m.get('max_tokens',2048))
    def recheck_sources():
        for path,digest in bindings.items():
            if sha(path)!=digest: raise ValueError(f'Bound source changed during construction: {path}')
    recheck_sources()
    out=Path(output_dir).resolve()
    if out.exists(): raise FileExistsError(f'Exclusive output directory required: {out}')
    out.mkdir(parents=True)
    artifact=out/'sentence_target.npz'
    with artifact.open('xb') as f: np.savez(f,**arrays)
    recheck_sources()
    receipt={'schema':1,'kind':'sentence_target_cpu_preparation','method':METHOD,'purpose':m['purpose'],
      'pair_id':m['pair_id'],'source_split':m['source_split'],'uses_paired_y':True,
      'training_admitted':False,'video_quality_evidence':False,'gpu_used':False,'trained':False,
      'prompt_x_sha256':textsha(texts['x']),'prompt_y_sha256':textsha(texts['y']),
      'x_raw_sha256':rawsha(x),'y_raw_sha256':rawsha(y),
      'blocks':proof,'prefix_token_span':m.get('prefix_token_span'),'source_precision':{'x':str(x.dtype),'y':str(y.dtype)},
      'arithmetic_precision':'CPU FP32; no conversion through PairDataset.add',
      'outside_sentence':'original recomputed monotone T, never replaced with x','truncated_sequences':0,
      'source_bindings':bindings,'output':{'path':str(artifact),'sha256':sha(artifact)},
      'artifact':{'path':str(artifact),'sha256':sha(artifact)},
      'arrays':{k:{'shape':list(v.shape),'dtype':str(v.dtype),'raw_sha256':rawsha(v)} for k,v in arrays.items()},
      'limitations':['Uniform-cell weights are a supervision hypothesis, not semantic/video evidence.',
       'Explicit spans are verified against exact native text/tokens; sentence semantics remain caller-reviewed.',
       'Previously rejected or ambiguous targets are not automatically accepted as training positives.']}
    with (out/'TARGET_RECEIPT.json').open('x',encoding='utf-8') as f: json.dump(receipt,f,ensure_ascii=False,indent=2)
    return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--output-dir',required=True)
    args=p.parse_args();torch.set_num_threads(2)
    if torch.cuda.is_initialized(): raise RuntimeError('Use a fresh CPU process')
    result=build(args.manifest,args.output_dir)
    print(json.dumps({'artifact':result['artifact'],'training_admitted':False,'gpu_used':False},ensure_ascii=False))

if __name__=='__main__': main()
