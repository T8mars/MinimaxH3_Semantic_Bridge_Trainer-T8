"""Exercise packaged CLI train/resume/export/diagnostics. Software test, not quality evidence."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile


def dump(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def smoke(root,work,dataset=None,split=None,legacy_receipt=False,epochs=None,device='cpu',amp=False):
    root,work=Path(root).resolve(),Path(work).resolve()
    if work.exists():raise ValueError('Smoke test needs a fresh work directory')
    work.mkdir(parents=True)
    sys.path.insert(0,str(root))
    from wushu_bridge.source_revision import revision
    expected=json.loads((root/'app_version.json').read_text(encoding='utf-8'))['source_commit']
    if legacy_receipt:
        (root/'SOURCE_REVISION.txt').write_text('Trainer release 0.2.2\nPublic source commit '+('b'*40)+'\nExact distributed contents are bound by PORTABLE_MANIFEST.json.\n',encoding='utf-8')
    assert revision(root)==expected
    kind='Existing H3 encoded format examples; software pipeline test, not an effectiveness dataset'
    if dataset is None:
        import numpy as np
        from wushu_bridge.dataset import PairDataset,PairMeta
        rows=[json.loads(line) for line in (root/'datasets/preference_example.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        # Parser-shaped identity only. These random arrays are explicitly synthetic,
        # have not been produced by H3, and must never be distributed as a model.
        identity={'tokenizer':'MiniMaxH3Tokenizer','mode':'t2v','dim':5120,
            'encoder_sha256':'0'*64,'encoder_file':'synthetic-software-fixture-only',
            'encoder_sources':{'synthetic_fixture':True},'synthetic_fixture':True}
        ds=PairDataset(PairMeta(encoder=json.dumps(identity),supervision_type='preference_only',score_label_type='none'))
        rng=np.random.default_rng(42)
        for row in rows:
            x=rng.normal(size=(5,5120)).astype(np.float16)
            ds.add(x,(x.astype(np.float32)+rng.normal(scale=.1,size=x.shape)).astype(np.float16),float('nan'),float('nan'),row)
        dataset=work/'synthetic_only.npz';ds.save(str(dataset))
        split=root/'datasets/preference_example_split.json'
        kind='Synthetic random 5120-dimensional software fixture; not H3 tokens or video-quality evidence'
    dataset,split=Path(dataset).resolve(),Path(split).resolve()
    config=json.loads((root/'configs/preference_v1.json').read_text(encoding='utf-8'))
    epoch_limit=config['bridge_epochs'] if epochs is None else epochs
    config.update(device=device,amp=amp)
    config_path=work/'software_check.json';dump(config_path,config)
    run=work/'cli_run'
    cli=root/'tools/trainer.py'
    common=['train',str(dataset),'--output',str(run),'--config',str(config_path),'--stage','bridge','--split-manifest',str(split),'--epochs',str(epoch_limit)]
    commands=[
        ['inspect',str(root/'datasets/preference_example.jsonl'),'--require-supervision','preference_only','--split-manifest',str(split)],
        common+['--stop-after-epochs','1'],common+['--resume'],
        ['verify','--bridge',str(run/'bridge.safetensors'),'--bridge-only','--report',str(work/'verification.json')],
    ]
    for mode in ['none','per_token']:
        commands.append(['evaluate',str(dataset),'--run',str(run),'--split','validation','--alpha','0','.05','.08','.12','.2','--magnitude-mode',mode,'--output',str(work/('validation_'+mode+'.json'))])
    commands.append(['usage','--bridge',str(run/'bridge.safetensors'),'--output-dir',str(work/'regenerated_usage')])
    env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1'}
    started=time.monotonic()
    for index,command in enumerate(commands):
        result=subprocess.run([sys.executable,'-B','-X','utf8',str(cli),*command],cwd=root,env=env,capture_output=True,text=True,encoding='utf-8',timeout=600)
        (work/('step_'+str(index)+'_'+command[0]+'.log')).write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        if result.returncode:raise RuntimeError(command[0]+' failed:\n'+result.stdout+'\n'+result.stderr)
        print(json.dumps({'phase':'cli_'+command[0],'step':index+1,'total':len(commands),'seconds':round(time.monotonic()-started)},ensure_ascii=False),flush=True)
    for name in ['bridge.safetensors','模型使用说明.md','inference_settings.json','bridge_report.json','run.json']:
        assert (run/name).is_file(),name
    guidance=json.loads((run/'inference_settings.json').read_text(encoding='utf-8'))
    weight_hash=hashlib.sha256((run/'bridge.safetensors').read_bytes()).hexdigest()
    assert guidance['model_sha256']==weight_hash
    assert not guidance['video_calibrated_by_this_export']
    provenance=json.loads((run/'run.json').read_text(encoding='utf-8'))
    assert provenance['identity']['upstream_commit']==expected
    assert provenance['status']=='completed'
    assert provenance['config']['bridge_epochs']==epoch_limit
    report=json.loads((run/'bridge_report.json').read_text(encoding='utf-8'))
    import torch
    checkpoint=torch.load(run/'bridge_last.pt',map_location='cpu',weights_only=True)
    assert checkpoint['epoch']==report['epochs_completed']
    assert checkpoint['epoch']==epoch_limit or checkpoint['bad_epochs']>=config.get('patience',12)
    assert not list(run.glob('*.pending'))
    assert not (run/'judge.safetensors').exists()
    receipt={'status':'passed','steps':len(commands),'dimension':5120,'config_hidden':config['hidden'],
        'source_commit':expected,'resume':True,'weight_reload_verify':True,'deployment_guide_sha_binding':True,
        'epoch_limit':epoch_limit,'epochs_completed':checkpoint['epoch'],'device':device,'amp':amp,
        'last_checkpoint_reloaded':True,'early_stopping_verified':checkpoint['epoch']<epoch_limit,
        'both_magnitude_validation_diagnostics':True,'scope':kind,'seconds':round(time.monotonic()-started)}
    dump(work/'SMOKE_RESULT.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False),flush=True)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path)
    parser.add_argument('--archive',type=Path)
    parser.add_argument('--work-dir',type=Path,required=True)
    parser.add_argument('--dataset',type=Path)
    parser.add_argument('--split-manifest',type=Path)
    parser.add_argument('--legacy-receipt',action='store_true')
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--device',choices=['cpu','cuda','auto'],default='cpu')
    parser.add_argument('--amp',action='store_true')
    args=parser.parse_args()
    if args.archive:
        extracted=args.work_dir.resolve()/'extracted_no_git'
        extracted.mkdir(parents=True,exist_ok=False)
        with zipfile.ZipFile(args.archive) as z:z.extractall(extracted)
        assert not (extracted/'.git').exists()
        smoke(extracted,args.work_dir/'test',args.dataset,args.split_manifest,legacy_receipt=True,epochs=args.epochs,device=args.device,amp=args.amp)
    else:
        smoke(args.root,args.work_dir,args.dataset,args.split_manifest,args.legacy_receipt,args.epochs,args.device,args.amp)
