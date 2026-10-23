"""Native50 ACT with compiled coupled correction and atomic seven-channel hands."""
import atexit
from contextlib import contextmanager
from collections import deque
import inspect
import json
import os
from pathlib import Path
import tempfile
import time
from .contracts import STATS, load_manifest, load_official, module, require, sha, verify_ref
from experiments.goal_latent_act_arm_prefix10.runtime import Runtime as FrozenPrefixRuntime


class Runtime:
    # These functions keep the already qualified camera/state snapshot, role
    # RPC identity, current FK, queue reset, and zero-latent equations verbatim.
    _latent = FrozenPrefixRuntime._latent
    reset = FrozenPrefixRuntime.reset
    goal = FrozenPrefixRuntime.goal
    close = FrozenPrefixRuntime.close

    def __init__(self,args):
        import torch
        from safetensors.torch import load_model
        from policy.relational_hierarchy.runtime import RpcWorker
        self.torch = torch
        self.cfg = cfg = load_manifest(args['coupled_all14_manifest'],args['coupled_all14_manifest_sha256'])
        require(args['task_name']==cfg['task'], 'Actual native task differs from bound model')
        self.official = load_official(cfg)
        torch.manual_seed(17)
        self.base = self.official.get_model(dict(checkpoint_path=cfg['act_root'],device='cuda',pytorch_device='cuda',act_step=50))
        load_model(self.base,str(verify_ref(cfg['act_files']['model.safetensors'])),strict=True,device='cuda:0')
        self.base.eval().requires_grad_(False)
        from lerobot.policies.act.modeling_act import ACTPolicy
        require(sha(inspect.getfile(ACTPolicy))==cfg['assets']['ACT_model_source']['sha256'], 'Imported original ACT source differs')
        require(self.base.config.chunk_size==self.base.config.n_action_steps==50
                and self.base.config.temporal_ensemble_coeff is None, 'Original native ACT50 configuration required')
        require(next(self.base.parameters()).dtype==torch.float32, 'Original ACT FP32 precision required')
        self.act_device = self.base.act_device
        self.act_step,self.state_obs_path,self.strict_action_dim = 10,'robot/qpos',True
        self.config = self.base.config
        self.act_image_keys,self.image_key_map = self.base.act_image_keys,self.base.image_key_map
        self._action_queue = deque()
        self.episode,self.executed,self.observer = -1,0,None
        self.correction,self.correction_contract,self.compiler_report = None,None,None
        self.temp = tempfile.TemporaryDirectory(prefix='coupled-all14-',dir=cfg['scratch_root'])
        self.inputs,self.goals = Path(self.temp.name)/'inputs',Path(self.temp.name)/'goals'
        self.inputs.mkdir(); self.goals.mkdir()
        self.latent_hook = self.base.model.encoder_latent_input_proj.register_forward_pre_hook(self._latent)
        self.act_execution = dict(mode='eager',compiled=False,dtype='FP32',native_generated=50,native_executed=10,replanning_commands=10)
        print('[ACT execution] '+json.dumps(self.act_execution),flush=True)
        try:
            if cfg['mode']!='parent':
                from experiments.coupled_goal_act_all14.contracts import load_checkpoint
                contract_path = verify_ref(cfg['assets']['coupled_contract'])
                self.correction,contract = load_checkpoint(contract_path.parent,cfg['assets']['coupled_weights']['sha256'],device=self.act_device)
                require(contract==json.loads(contract_path.read_text()), 'Strict all14 loader contract differs')
                self.correction_contract = contract
                self.stats = {key:torch.tensor(contract['statistics'][key],dtype=torch.float32,device=self.act_device) for key in STATS}
                if cfg['mode']=='goal_free':
                    self.geometry = module('coupled_all14_current_geometry',verify_ref(cfg['assets']['geometry']))
                    self.chain = self.geometry.read_chain(verify_ref(cfg['assets']['urdf']))
                else:
                    env = os.environ.copy(); env.pop('PYTHONHOME',None)
                    env.update(PYTHONPATH=os.pathsep.join(cfg['qwen_python_path']),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                        PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
                    self.observer = RpcWorker([cfg['qwen_python'],'-B','-m','experiments.coupled_goal_act_all14_submission.observer_worker',
                        '--manifest',str(cfg['_manifest']),'--manifest-sha256',args['coupled_all14_manifest_sha256'],
                        '--input-root',str(self.inputs),'--output-root',str(self.goals)],env,cfg['worker_timeout_seconds'])
                    require(self.observer.ready['kind']=='coupled_all14_current_frozen_Qwen'
                        and self.observer.ready['highlevel_sha256']==cfg['assets']['highlevel_weights']['sha256'], 'Current observer identity differs')
            atexit.register(self.close)
        except BaseException:
            self.close(); raise

    def qualify_compiler(self,native,current):
        """Real first-current kwargs captured from the scientific inference path."""
        from experiments.coupled_goal_act_all14.inference import correct_native50
        from experiments.coupled_goal_act_all14.compiler import compile_and_compare
        captured = []
        def remember(_model,_args,kwargs): captured.append(dict(kwargs))
        hook = self.correction.register_forward_pre_hook(remember,with_kwargs=True)
        try:
            correct_native50(self.correction,native,current,self.stats,task=self.cfg['task'],
                codec=self.cfg['gripper_codec'],state_action_only=self.cfg['mode']=='goal_free')
        finally: hook.remove()
        require(len(captured)==1, 'Exactly one original current-context all14 call required')
        inputs = captured[0]
        require(inputs.pop('state_action_only')==(self.cfg['mode']=='goal_free'), 'Compiled intervention differs')
        compiled,report = compile_and_compare(self.correction,inputs,state_action_only=self.cfg['mode']=='goal_free',
            backend=self.cfg['correction_compiler_backend'],warm_calls=5)
        require(not self.cfg['require_compile_speedup'] or report['acceleration_qualified'], 'Declared compiled acceleration gate failed; no eager fallback')
        self.correction,self.compiler_report = compiled,report
        print('[coupled all14 compiler] '+json.dumps(report),flush=True)

    @contextmanager
    def correction_precision(self):
        """Training FP32 contract only for the new model; native ACT flags retained."""
        previous_matmul = self.torch.backends.cuda.matmul.allow_tf32
        previous_cudnn = self.torch.backends.cudnn.allow_tf32
        try:
            self.torch.backends.cuda.matmul.allow_tf32 = False
            self.torch.backends.cudnn.allow_tf32 = False
            yield
        finally:
            self.torch.backends.cuda.matmul.allow_tf32 = previous_matmul
            self.torch.backends.cudnn.allow_tf32 = previous_cudnn

    def sync(self):
        if str(self.act_device).startswith('cuda'): self.torch.cuda.synchronize()

    def select_action(self,batch):
        require(self.episode>=0 and time.monotonic()-self.started<=self.cfg['episode_wall_seconds'], 'Reset and bounded native episode required')
        if not self._action_queue:
            self.sync()
            started = time.monotonic()
            with self.torch.inference_mode():
                normalized = dict(self.base.normalize_inputs(batch))
                normalized['observation.images'] = [normalized[k] for k in self.base.config.image_features]
                prediction = self.base.model(normalized)[0]
                native = self.base.unnormalize_outputs({'action':prediction})['action']
                require(native.shape==(1,50,14) and native.dtype==self.torch.float32
                    and bool(self.torch.isfinite(native).all()), 'Finite original native50 ACT required')
                self.sync()
                core_done = time.monotonic()
                timings = dict(native_ACT_core_wall_seconds=core_done-started,current_observer_wall_seconds=0.,
                    coupled_correction_wall_seconds=0.,cold_compiler_gate_wall_seconds=0.)
                action = native
                if self.correction is not None:
                    from experiments.coupled_goal_act_all14.inference import correct_native50
                    snapshot = self.goal(batch)
                    current = {key:self.torch.from_numpy(value[None]).to(self.act_device) for key,value in snapshot.items()}
                    self.sync()
                    observer_done = time.monotonic()
                    timings['current_observer_wall_seconds']=observer_done-core_done
                    with self.correction_precision():
                        if self.compiler_report is None:
                            compile_started=time.monotonic(); self.qualify_compiler(native,current); self.sync()
                            timings['cold_compiler_gate_wall_seconds']=time.monotonic()-compile_started
                        correction_started=time.monotonic()
                        action,result = correct_native50(self.correction,native,current,self.stats,task=self.cfg['task'],
                            codec=self.cfg['gripper_codec'],state_action_only=self.cfg['mode']=='goal_free')
                        self.sync()
                        timings['coupled_correction_wall_seconds']=time.monotonic()-correction_started
                    rejected = (~result['accepted'])[:,None,:,None].expand(1,50,2,7).reshape_as(native)
                    require(self.torch.equal(action[rejected],native[rejected]), 'Rejected whole-hand native fallback changed')
                    print('[coupled all14] '+json.dumps(dict(episode=self.episode,executed=self.executed,
                        accepted=result['accepted'][0].tolist(),reliability=result['reliability_logits'][0].sigmoid().tolist(),
                        delta_arm_abs_max_rad=float(result['delta_arm_rad'].abs().max()),
                        delta_opening_abs_max_m=float(result['delta_opening_m'].abs().max()),
                        rejected_hand_all7_bitexact=True,native_grippers_bitexact=False,output_native_codec=self.cfg['gripper_codec'])),flush=True)
                self._action_queue.extend(action[0,:10].unbind(0))
                print('[coupled prefix10 replan] '+json.dumps(dict(episode=self.episode,executed=self.executed,mode=self.cfg['mode'],
                    native_generated=50,native_executed=10,current_observation_action=self.executed,
                    total_replan_wall_seconds=time.monotonic()-started,timings=timings)),flush=True)
        action = self._action_queue.popleft()[None]
        self.executed += 1
        require(action.shape==(1,14) and bool(self.torch.isfinite(action).all()), 'Finite native14 output required')
        return action


def get_model(args): return Runtime(args)


def eval(env,model,obs):
    from experiments.goal_latent_act_sequence_runtime_full.terminal import TerminalEnvironment
    from policy.robosyn_sprint.act_drawer_adapter import _ActionBoundary
    view = _ActionBoundary(model,env) if model.cfg['gripper_codec']=='drawer_normalized_opening_inverse_v1' else model
    return model.official.eval(TerminalEnvironment(env),view,obs)


def reset_model(model): return model.reset()


def close_model(model): return model.close()
