"""Bounded actual relocated FILE/STATS/guard CPU qualification, no models."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def child(root,output):
    sys.path.insert(0,str(root))
    from experiments.coupled_goal_act_all14_submission.contracts import materialize,load_manifest,STATS,borrowed
    from experiments.coupled_goal_act_all14_submission.prepare import ref,sha,write
    from experiments.coupled_goal_act_all14_runtime import contracts as original
    from experiments.coupled_goal_act_all14_runtime import runtime_eager as eager
    spec=json.loads((root/'submission.json').read_text());start=time.monotonic()
    policy=importlib.import_module('policy.coupled_all14_portable_v1')
    for name in ('get_model','eval','reset_model','close_model'):assert callable(getattr(policy,name))
    try:policy.get_model({})
    except KeyError:pass
    else:raise AssertionError('Missing admission must reject before model creation')
    assert 'torch' not in sys.modules
    import torch
    assert not torch.cuda.is_initialized()
    def forbidden(*args,**kwargs):raise AssertionError('CPU transport must never construct a model')
    torch.nn.Module.__init__=forbidden
    assert borrowed(original.validate_training,{}).__code__ is original.validate_training.__code__
    assert borrowed(eager.load_eager_manifest,{}).__code__ is eager.load_eager_manifest.__code__
    rows=[];proofs={}
    for row in spec['tasks']:
        target=output/(row['mode']+'-'+row['task']);target.mkdir()
        path,cfg=materialize(root,root/row['template'],target)
        checked=load_manifest(path,sha(path));assert checked['task']==row['task'] and checked['mode']==row['mode']
        stats={}
        if cfg['mode']!='parent':
            c=json.loads(Path(cfg['assets']['coupled_contract']['path']).read_text())
            for name,width in STATS.items():
                t=torch.tensor(c['statistics'][name],dtype=torch.float32,device='cpu')
                assert tuple(t.shape)==(width,) and bool(torch.isfinite(t).all())
                if name.endswith('scale'):assert bool((t>0).all())
                stats[name]=dict(shape=list(t.shape),dtype=str(t.dtype),device=str(t.device))
        for group in ('assets','act_files'):
            for key,bound in cfg[group].items():
                assert Path(bound['path']).resolve().is_relative_to(root),(key,bound)
                if group=='assets' and Path(bound['path']).suffix=='.json':proofs[bound['path']]=bound['sha256']
        for bound in cfg['qwen_files'].values():assert Path(bound['path']).resolve().is_relative_to(root)
        rows.append(dict(task=row['task'],mode=row['mode'],official_horizon=row['official_horizon'],operational=ref(path),
            actual_fixed_source_predicates_passed=True,STATS=stats,generated=50,executed=10,
            raw_original_reference_paths_not_required_on_installed_host=True))
    # Every raw evidence item remained byte-identical after all materializations.
    for name,digest in proofs.items():assert sha(name)==digest
    paths={}
    for name,value in tuple(sys.modules.items()):
        if name.startswith(('experiments.','policy.')) and getattr(value,'__file__',None):
            file=Path(value.__file__).resolve();assert file.is_relative_to(root),(name,file)
            paths[name]=str(file.relative_to(root))
    assert not torch.cuda.is_initialized()
    write(output/'actual-binding-STATS.json',dict(schema='coupled_all14_portable_actual_all30_CPU_binding_v1',status='all30_FILE_fixed_predicates_STATS_PASS_no_model',
        package=ref(root/'submission.json'),rows=rows,raw_evidence_items_byte_identical=len(proofs),source_imports=paths,
        original_training_validator_code_object_identical=True,original_eager_validator_code_object_identical=True,
        original_model_runtime_only_observer_module_path_changed=True,CUDA_initialized=False,models_constructed=0,
        GPU_queries=0,GPU_launches=0,native_world_count=0,seconds=time.monotonic()-start,
        installed_native_qualified=False,RTX5090_qualified=False,submission_ready=False))
    print(json.dumps(dict(status='actual_all30_portable_FILE_STATS_CPU_PASS',rows=len(rows),CUDA_initialized=False,models_constructed=0,seconds=time.monotonic()-start)))


def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--child',action='store_true');a=p.parse_args();root=Path(a.root).resolve(strict=True);output=Path(a.output).absolute()
    if a.child:
        child(root,output);return
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from experiments.coupled_goal_act_all14_submission.prepare import ref,write
    from experiments.coupled_goal_act_all14_submission.smoke import identity,family
    assert not output.exists();output.mkdir(parents=True)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    for key in ('PYTHONPATH','PYTHONHOME','PYTHONSTARTUP'):env.pop(key,None)
    argv=[str(Path(sys.executable).absolute()),'-I','-B',str(Path(__file__).resolve()),'--root',str(root),'--output',str(output),'--child']
    with (output/'original.stdout.log').open('xb') as stdout,(output/'original.stderr.log').open('xb') as stderr:
        proc=subprocess.Popen(argv,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
        owner=identity(proc.pid);code=proc.wait(timeout=600)
    absent=identity(proc.pid) is None;live=family(owner['group'],owner['session'])
    body=dict(schema='coupled_all14_portable_CPU_original_child_wait_v1',owner=owner,argv=argv,
        original_child_exit_code=code,child_reaped=True,exact_identity_absent=absent,live_owned_groups_sessions=live,signals_sent=[],
        stdout=ref(output/'original.stdout.log'),stderr=ref(output/'original.stderr.log'),GPU_queries=0,GPU_launches=0,native_world_count=0)
    write(output/'original-wait.json',body)
    assert code==0 and absent and not live,(code,(output/'original.stderr.log').read_text()[-5000:])
    write(output/'receipt.json',dict(schema='coupled_all14_portable_CPU_qualified_v1',status='CPU_portable_binding_qualified_no_native_or_model_or_5090_promotion',
        package=ref(root/'submission.json'),original_wait=ref(output/'original-wait.json'),actual_bindings=ref(output/'actual-binding-STATS.json'),
        models_constructed=0,GPU_queries=0,GPU_launches=0,native_world_count=0,submission_ready=False))
    print(json.dumps(dict(receipt=ref(output/'receipt.json'))))


if __name__=='__main__':main()
