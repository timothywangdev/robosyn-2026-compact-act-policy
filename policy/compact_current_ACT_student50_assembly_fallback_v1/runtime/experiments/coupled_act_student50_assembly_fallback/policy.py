"""External candidate adapter; the immutable Student50 payload is unchanged."""
import hashlib
import importlib
import inspect
import json
import os
import sys
from pathlib import Path
import tempfile

TASKS=('click_bell','drawer_open_place','handle_basket','item_assembly','items_handover',
       'manipulate_pipette','mixer_operating','sample_loading','table_rearrangement','water_pouring')
PINS={
    'experiments.coupled_act_student50_release.policy':'557d559e252deff8531cc98632b1e57e50d966c72e380b0d893af39006fdded2',
    'experiments.coupled_act_student50_release.binder':'d723bb36539b97b72e6ef8307c8bbf6b5c4815fef341c4a5f4f74deeba579810',
    'experiments.coupled_act_feature_student_cadence50_submission.contracts':'11fb28e0e65761ee2bf145c1630e257a9f0dd0672fa3ba3654ab789f2de94ef3',
}

def require(value,message):
    if not value:raise ValueError(message)

def pinned_module(name):
    module=importlib.import_module(name)
    require(hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()==PINS[name], 'Unchanged public/package module required: '+name)
    return module

def route(task):
    require(task in TASKS,'Only the ten explicit official tasks are supported')
    return 'plain_ACT50_Assembly' if task=='item_assembly' else 'unchanged_public_Student50'

def verified_configuration(root,scratch):
    """Validate exact existing template, retaining mode/student provenance."""
    root=Path(root).resolve(strict=True);scratch=Path(scratch).resolve(strict=True)
    binder=pinned_module('experiments.coupled_act_student50_release.binder')
    contracts=pinned_module('experiments.coupled_act_feature_student_cadence50_submission.contracts')
    bindings=binder.bind_all(root,scratch/'unchanged-artifact-bindings')
    original=bindings['tasks']['item_assembly'];actor=scratch/'verified-original-view';actor.mkdir()
    operational,cfg=contracts.materialize(root,original['template']['path'],actor)
    checked=contracts.load_manifest(operational,contracts.sha(operational))
    require(checked['task']=='item_assembly' and checked['mode']=='student'
        and checked['native_chunk_generated']==checked['prefix_executed']==50 and checked['motor_seed']==17
        and checked['gripper_codec']=='physical_meters_identity_v1','Original immutable Assembly provenance changed')
    return checked,dict(unchanged_package=bindings['verification']['package'],
        unchanged_registered_template=original['template'],original_operational_manifest=contracts.ref(operational),
        original_template_mode_preserved='student',new_candidate_route='plain_ACT50_Assembly',
        original_payload_or_admission_modified=False)

class NativeAssemblyACT:
    """Raw official ACT Tensor/lifecycle delegation, with no correction hook."""
    def __init__(self,base,official,provenance):
        self._base,self._official,self._provenance=base,official,provenance
    def __getattr__(self,name):return getattr(self._base,name)
    def select_action(self,batch):return self._base.select_action(batch)
    def reset(self):return self._base.reset()

def construct_raw_ACT(cfg,args,torch,official):
    torch.manual_seed(17)
    base=official.get_model(dict(checkpoint_path=cfg['act_root'],device=args.get('pytorch_device','cuda'),
        pytorch_device=args.get('pytorch_device','cuda'),act_step=10,state_obs_path='robot/qpos',strict_action_dim=True))
    require(base.config.chunk_size==base.config.n_action_steps==50 and base.config.temporal_ensemble_coeff is None
        and not base.training and next(base.parameters()).dtype==torch.float32,'Exact released eager FP32/native50 ACT')
    require(list(base.config.image_features)==cfg['camera_order'],'Exact high/right/left camera loop')
    return base

def _get_model(args):
    selected=route(args['task_name'])
    if selected=='unchanged_public_Student50':
        return pinned_module('experiments.coupled_act_student50_release.policy').get_model(args)
    root=args.get('compact_student_submission_root',os.environ.get('ROBOSYN_STUDENT50_PACKAGE_ROOT'))
    require(root is not None,'Explicit unchanged installed Student50 package root required')
    scratch=Path(tempfile.mkdtemp(prefix='rsc50-asm-',dir='/tmp'))
    cfg,provenance=verified_configuration(root,scratch)
    from experiments.coupled_goal_act_all14_runtime.contracts import load_official,sha
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy
    require(sha(inspect.getfile(ACTPolicy))==cfg['assets']['ACT_model_source']['sha256'],'Imported native ACT implementation changed')
    official=load_official(cfg);base=construct_raw_ACT(cfg,args,torch,official)
    print('[Assembly conservative ACT-only candidate] '+json.dumps(dict(provenance,
        task='item_assembly',native_generated=50,native_executed=50,official_callback_actions=10,
        Student_models_loaded=0,correction_models_loaded=0,Qwen_models_loaded=0,ACT_models_loaded=1,
        action_transform=False,terminal_retention_wrapper=False,
        route_uses_only_declared_task_name=True,original28_29_results_reused_as_candidate_results=False)),flush=True)
    return NativeAssemblyACT(base,official,provenance)

def guarded(name, function, *args):
    try:return function(*args)
    except BaseException as error:
        print('COUPLED_ALL14_POLICY_EXCEPTION_CAPTURED '+name+' '+type(error).__name__,file=sys.stderr,flush=True)
        raise

def get_model(args):return guarded('get_model',_get_model,args)

def _eval(env,model,obs):
    if isinstance(model,NativeAssemblyACT):return model._official.eval(env,model,obs)
    return pinned_module('experiments.coupled_act_student50_release.policy').eval(env,model,obs)

def _reset_model(model):
    if isinstance(model,NativeAssemblyACT):return model._official.reset_model(model)
    return pinned_module('experiments.coupled_act_student50_release.policy').reset_model(model)

def _close_model(model):
    if isinstance(model,NativeAssemblyACT):return None
    return pinned_module('experiments.coupled_act_student50_release.policy').close_model(model)

def eval(env,model,obs):return guarded('eval',_eval,env,model,obs)
def reset_model(model):return guarded('reset_model',_reset_model,model)
def close_model(model):return guarded('close_model',_close_model,model)
