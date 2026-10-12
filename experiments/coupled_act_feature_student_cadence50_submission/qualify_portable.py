"""Actual relocated all10 FILE/STATS/raw checkpoint transport, no models."""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def child(root,output):
    sys.path.insert(0,str(root))
    from experiments.coupled_act_feature_student_cadence50_submission.contracts import materialize,load_manifest,validate_saved,STATS,borrowed
    from experiments.coupled_goal_act_all14_submission.prepare import ref,sha,write
    from experiments.coupled_act_feature_student_cadence50 import contracts as original
    spec=json.loads((root/'submission.json').read_text());started=time.monotonic()
    entry=importlib.import_module('policy.compact_current_ACT_student50_v1')
    for name in ('get_model','eval','reset_model','close_model'):assert callable(getattr(entry,name))
    try:entry.get_model({})
    except KeyError:pass
    else:raise AssertionError('Missing admission must reject before model creation')
    assert 'torch' not in sys.modules
    import torch
    assert not torch.cuda.is_initialized()
    def forbidden(*args,**kwargs):raise AssertionError('No model construction in portable CPU qualification')
    torch.nn.Module.__init__=forbidden
    from experiments.coupled_act_feature_student_cadence50_submission.runtime import Runtime,OriginalRuntime,OriginalPrecisionRuntime
    assert Runtime.__init__.__code__ is OriginalRuntime.__init__.__code__
    assert Runtime.select_action is OriginalPrecisionRuntime.select_action
    assert borrowed(original.load_manifest,{}).__code__ is original.load_manifest.__code__
    weights={};rows=[];proofs={}
    for row in spec['tasks']:
        folder=output/(row['mode']+'-'+row['task']);folder.mkdir()
        path,cfg=materialize(root,root/row['template'],folder)
        checked=load_manifest(path,sha(path));assert checked['mode']==row['mode'] and checked['task']==row['task']
        stats={};saved_validated=False
        if row['mode']=='student':
            teacher=json.loads(Path(cfg['assets']['teacher_contract']['path']).read_text())
            for key,width in STATS.items():
                value=torch.tensor(teacher['statistics'][key],dtype=torch.float32,device='cpu')
                assert value.shape==(width,) and bool(torch.isfinite(value).all())
                if key.endswith('scale'):assert bool((value>0).all())
                stats[key]=list(value.shape)
            wr=cfg['assets']['student_weights']
            if wr['sha256'] not in weights:
                weights[wr['sha256']]=torch.load(wr['path'],map_location='cpu',weights_only=False)
            training=json.loads(Path(cfg['assets']['student_training_receipt']['path']).read_text())
            validate_saved(weights[wr['sha256']],cfg,training);saved_validated=True
        for group in ('assets','act_files'):
            for key,value in cfg[group].items():
                assert Path(value['path']).resolve().is_relative_to(root),(key,value)
                if group=='assets' and Path(value['path']).suffix=='.json':proofs[value['path']]=value['sha256']
        rows.append(dict(task=row['task'],mode=row['mode'],official_horizon=row['official_horizon'],operational=ref(path),
            actual_original_evidence_predicates_passed=True,STATS=stats,actual_saved_checkpoint_metadata_validated=saved_validated,
            generated=50,executed=50,no_historical_asset_paths_required=True))
    for name,digest in proofs.items():assert sha(name)==digest
    paths={}
    for name,value in tuple(sys.modules.items()):
        if name.startswith(('experiments.','policy.')) and getattr(value,'__file__',None):
            p=Path(value.__file__).resolve();assert p.is_relative_to(root),(name,p);paths[name]=str(p.relative_to(root))
    assert not torch.cuda.is_initialized()
    write(output/'actual-binding-STATS-checkpoint.json',dict(schema='compact_student_portable_actual_all10_CPU_binding_v1',
        status='all10_FILE_original_predicates_STATS_actual_checkpoint_metadata_PASS_no_model',package=ref(root/'submission.json'),
        rows=rows,source_imports=paths,raw_evidence_items_byte_identical=len(proofs),original_native_init_code_object_identical=True,
        original_native_methods_inherited=True,original_validator_code_object_identical=True,CUDA_initialized=False,models_constructed=0,
        native_world_count=0,GPU_queries=0,GPU_launches=0,seconds=time.monotonic()-started,submission_ready=False,
        installed_native_qualified=False,RTX5090_qualified=False,cold_installation_qualified=False))
    print(json.dumps(dict(status='actual_all10_portable_FILE_STATS_saved_checkpoint_CPU_PASS',rows=len(rows),CUDA_initialized=False,models_constructed=0)))

def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--child',action='store_true');a=p.parse_args()
    root=Path(a.root).resolve(strict=True);out=Path(a.output).absolute()
    if a.child:child(root,out);return
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from experiments.coupled_goal_act_all14_submission.prepare import ref,write
    from experiments.coupled_goal_act_all14_submission.smoke import identity,family
    assert not out.exists();out.mkdir(parents=True)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    for key in ('PYTHONPATH','PYTHONHOME','PYTHONSTARTUP'):env.pop(key,None)
    argv=[str(Path(sys.executable).absolute()),'-I','-B',str(Path(__file__).resolve()),'--root',str(root),'--output',str(out),'--child']
    with (out/'original.stdout.log').open('xb') as stdout,(out/'original.stderr.log').open('xb') as stderr:
        proc=subprocess.Popen(argv,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
        owner=identity(proc.pid);code=proc.wait(timeout=600)
    absent=identity(proc.pid) is None;live=family(owner['group'],owner['session'])
    write(out/'original-wait.json',dict(schema='compact_student_portable_CPU_original_child_wait_v1',owner=owner,argv=argv,
        original_child_exit_code=code,child_reaped=True,exact_identity_absent=absent,live_owned_groups_sessions=live,
        signals_sent=[],stdout=ref(out/'original.stdout.log'),stderr=ref(out/'original.stderr.log'),GPU_queries=0,native_world_count=0))
    assert code==0 and absent and not live,(code,(out/'original.stderr.log').read_text()[-5000:])
    write(out/'receipt.json',dict(schema='compact_student_portable_CPU_qualified_v1',
        status='CPU_portable_binding_qualified_no_native_or_model_or_5090_promotion',package=ref(root/'submission.json'),
        original_wait=ref(out/'original-wait.json'),actual_bindings=ref(out/'actual-binding-STATS-checkpoint.json'),
        models_constructed=0,native_world_count=0,GPU_queries=0,GPU_launches=0,submission_ready=False))
    print(json.dumps(dict(receipt=ref(out/'receipt.json'))))

if __name__=='__main__':main()
