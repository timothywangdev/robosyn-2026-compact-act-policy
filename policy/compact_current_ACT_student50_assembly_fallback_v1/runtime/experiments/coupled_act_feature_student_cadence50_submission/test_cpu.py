"""Transport negatives and real official action-loop ABI, without model/world."""
import ast
from collections import deque
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import time
import types
import numpy as np
from .contracts import Resolver,borrowed,validate_saved,STATS

def main():
    from .contracts import original
    from .runtime import Runtime,OriginalRuntime,OriginalPrecisionRuntime
    from experiments.coupled_act_feature_student_runtime.contracts import MOMENTS
    from experiments.coupled_goal_act_all14_submission.prepare import ref
    import torch
    assert not torch.cuda.is_initialized()
    cases=[]
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);asset=root/'asset';asset.write_bytes(b'current immutable asset')
        historical=dict(ref(asset),path='/unavailable/original/asset')
        r=Resolver(root,[dict(original=historical,installed='asset')]);assert r.verify_ref(historical)==asset
        assert r.operational_ref(historical)['path']==str(asset);cases.append('historical_path_absent_installed_bytes_pass')
        for label,fn in [('unbound',lambda:r.verify_ref(dict(historical,path='/other'))),
                         ('escape',lambda:Resolver(root,[dict(original=historical,installed='../asset')])),
                         ('conflict',lambda:Resolver(root,[dict(original=historical,installed='asset'),dict(original=historical,installed='other')]))]:
            try:fn()
            except ValueError:cases.append(label+'_reject')
            else:raise AssertionError(label)
        asset.write_bytes(b'changed immutable asset')
        try:r.verify_ref(historical)
        except ValueError:cases.append('changed_bytes_reject')
        else:raise AssertionError('changed bytes')
        asset.unlink();(root/'real').write_bytes(b'current immutable asset');asset.symlink_to(root/'real')
        try:r.verify_ref(historical)
        except ValueError:cases.append('symlink_reject')
        else:raise AssertionError('symlink')
    old=dict(path='/old/plan',bytes=3,sha256='1'*64);new=dict(old,path='/installed/plan')
    saved=dict(model={},statistics={k:np.full(v,.01 if k.endswith('scale') else 0.,dtype=np.float32) for k,v in MOMENTS.items()},
               pooling='spatial2x3',qpos_only=False,feature_plan=old,protocol={'fixed':'original'},fixed_epoch=30)
    cfg=dict(student_qpos_only=False,assets={'student_feature_plan':new},portable_binding={'historical_assets':{'student_feature_plan':old}})
    validate_saved(saved,cfg,{'protocol':saved['protocol']});assert saved['feature_plan']==old;cases.append('raw_saved_feature_plan_historical_view_pass')
    bad=deepcopy(saved);bad['feature_plan']=new
    try:validate_saved(bad,cfg,{'protocol':saved['protocol']})
    except (AssertionError,RuntimeError,ValueError):cases.append('rewritten_checkpoint_reference_reject')
    else:raise AssertionError('checkpoint rewrite')
    assert Runtime.__init__.__code__ is OriginalRuntime.__init__.__code__
    assert Runtime.select_action is OriginalPrecisionRuntime.select_action
    assert Runtime.reset is OriginalRuntime.reset;cases.append('native_actor_code_objects_preserved')
    # Execute the exact installed official eval function with an actual borrowed
    # queue pop. The organizer act_step argument can be 50; model.act_step=50
    # owns the real loop, and returns one [1,14] command per env.step.
    base=Path(__file__).resolve().parents[2]
    official=base/'upstream/RoboSynChallenge/policy/act/deploy_policy.py'
    if not official.exists():official=Path('/tmp/ra5/upstream/RoboSynChallenge/policy/act/deploy_policy.py')
    tree=ast.parse(official.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='eval')
    ns=dict(torch=torch,np=np,start_inference=lambda d:None,finish_inference=lambda s,t,d:t.append(0.))
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(official),'exec'),ns)
    actor=object.__new__(Runtime);actor.torch=torch
    actor.cfg={'episode_wall_seconds':60,'native_ACT_precision':dict(matmul_allow_tf32=False,cudnn_allow_tf32=True),'pairing_initial_current_buffers_logged':True}
    actor.episode=0;actor.executed=8;actor.started=time.monotonic()
    actor.act_step=50;actor.config=types.SimpleNamespace(temporal_ensemble_coeff=None);actor.act_device=torch.device('cpu')
    actor.state_obs_path='robot/qpos';actor.act_image_keys=[];actor.strict_action_dim=True
    actor._action_queue=deque(torch.arange(700,dtype=torch.float32).reshape(50,14).unbind(0))
    expected=torch.stack(tuple(actor._action_queue))
    class Env:
        def __init__(self):self.unwrapped=self;self.device='cpu';self.single_action_space=types.SimpleNamespace(shape=(14,));self.rows=[]
        def step(self,a):self.rows.append(a.clone());return {'robot':{'qpos':torch.zeros(1,14)}},0.,False,False,{'elapsed_steps':torch.tensor(len(self.rows))}
        def get_wrapper_attr(self,name):assert name=='is_task_success';return lambda:False
    env=Env();obs={'robot':{'qpos':torch.zeros(1,14)}}
    returned=ns['eval'](env,actor,obs)
    assert len(returned)==4 and len(env.rows)==50 and actor.executed==58 and not actor._action_queue
    assert torch.equal(torch.cat(env.rows),expected) and all(a.shape==(1,14) for a in env.rows)
    cases.append('exact_official_eval_fifty_single_native14_actions_with_queue50')
    # The exact positive native50 queue implementation must consume the whole
    # generated sequence, replan on command51 and discard it on episode reset.
    native=torch.arange(700,dtype=torch.float32).reshape(1,50,14)/1000;calls=[0]
    class Base:
        config=types.SimpleNamespace(image_features=[])
        def normalize_inputs(self,b):return b
        def model(self,b):calls[0]+=1;return native,None
        def unnormalize_outputs(self,b):return b
        def reset(self):pass
    queued=object.__new__(Runtime);queued.torch=torch;queued.base=Base();queued.sync=lambda:None
    queued.cfg={'episode_wall_seconds':60,'mode':'student'};queued.episode=0;queued.executed=0;queued.started=time.monotonic()
    queued.capture=None;queued.student=None;queued._action_queue=deque()
    actual=torch.cat([OriginalRuntime.select_action(queued,{}) for _ in range(50)])
    assert torch.equal(actual,native[0]) and calls[0]==1 and queued.executed==50 and not queued._action_queue
    OriginalRuntime.select_action(queued,{})
    assert calls[0]==2 and queued.executed==51 and len(queued._action_queue)==49
    queued.reset();assert not queued._action_queue and queued.executed==0 and queued.episode==1
    cases.append('positive_native50_queue_command51_and_episode_reset_preserved')
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(schema='compact_student_portable_meaningful_CPU_tests_v1',cases=cases,count=len(cases),CUDA_initialized=False,models_constructed=0,worlds=0,GPU_queries=0)))

if __name__=='__main__':main()
