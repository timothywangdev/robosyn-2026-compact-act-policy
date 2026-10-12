"""Generated50/executed10 closed-loop cycle; frozen weights and gates unchanged."""
import atexit
from collections import deque
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np
from .contracts import CAMERAS, load_manifest, load_official, module, require, sha, verify_ref


class Runtime:
    def __init__(self, args):
        import torch
        from safetensors.torch import load_model
        from experiments.goal_latent_act_arm_runtime_full.semantics import ARM_SCHEMA, STATS, validate_pair
        from policy.relational_hierarchy.runtime import RpcWorker
        self.torch = torch
        self.cfg = cfg = load_manifest(args['goal_latent_manifest'], args['goal_latent_manifest_sha256'])
        require(args['task_name'] == cfg['task'], 'Actual task differs from bound model')
        self.official = load_official(cfg)
        require(cfg['motor_seed']==17, 'Predeclared Torch motor seed17 required')
        torch.manual_seed(17)
        self.base = self.official.get_model(dict(checkpoint_path=cfg['act_root'],device='cuda',pytorch_device='cuda',act_step=50))
        load_model(self.base, str(verify_ref(cfg['act_files']['model.safetensors'])), strict=True, device='cuda:0')
        self.base.eval().requires_grad_(False)
        import inspect
        from lerobot.policies.act.modeling_act import ACTPolicy
        require(sha(inspect.getfile(ACTPolicy)) == cfg['assets']['ACT_model_source']['sha256'], 'Imported ACT implementation differs from qualified cache source')
        require(self.base.config.chunk_size == self.base.config.n_action_steps == 50 and self.base.config.temporal_ensemble_coeff is None, 'Released FP32 native50 ACT required')
        require(next(self.base.parameters()).dtype == torch.float32, 'Unchanged frozen ACT precision required')
        self.act_device = self.base.act_device
        self.act_step, self.state_obs_path, self.strict_action_dim = 10, 'robot/qpos', True
        self.config = self.base.config
        self.act_image_keys, self.image_key_map = self.base.act_image_keys, self.base.image_key_map
        self._action_queue = deque()
        self.episode, self.executed, self.observer = -1, 0, None
        self.correction, self.correction_contract = None, None
        self.temp = tempfile.TemporaryDirectory(prefix='goal-latent-arm12-', dir=cfg['scratch_root'])
        self.inputs, self.goals = Path(self.temp.name)/'inputs', Path(self.temp.name)/'goals'
        self.inputs.mkdir(); self.goals.mkdir()
        self.latent_hook = self.base.model.encoder_latent_input_proj.register_forward_pre_hook(self._latent)
        require(cfg['act_core_mode']=='eager' and cfg['act_preprocess_mode']=='native_every_action', 'Unchanged eager/original preprocessing required')
        self.act_execution=dict(mode='eager',compiled=False,dtype='FP32',native_generated=50,native_executed=10,replanning_commands=10)
        print('[ACT execution] '+json.dumps(self.act_execution),flush=True)
        if cfg['mode'] != 'parent':
            admission = json.loads(verify_ref(cfg['assets']['robot_admission']).read_text())
            require(admission['schema']=='goal-latent-act-arm12-full63000-offline-admission-v1'
                    and admission['admit_robot_smoke'] is True
                    and admission['checkpoint_sha256']==cfg['assets']['arm_weights']['sha256']
                    and admission['training_pair_sha256']==cfg['assets']['training_pair']['sha256']
                    and admission['purpose']=='exploratory_native_comparison_after_failed_offline_promotion'
                    and admission['offline_promotion_criterion_passed'] is False,
                    'Measured offline admission for exact sequence required')
            contract=json.loads(verify_ref(cfg['assets']['arm_contract']).read_text())
            head=json.loads(verify_ref(cfg['assets']['highlevel_contract']).read_text())
            validate_pair(head,contract,cfg)
            from experiments.goal_latent_act_arm_sequence_full.contracts import load_checkpoint
            self.correction, loaded_contract = load_checkpoint(
                verify_ref(cfg['assets']['arm_contract']).parent, cfg['assets']['arm_weights']['sha256'])
            require(loaded_contract == contract, 'Full63000 strict loader contract differs')
            self.correction.eval().requires_grad_(False).to(self.act_device)
            self.correction_contract=contract
            self.stats={k:torch.tensor(contract['statistics'][k],dtype=torch.float32,device=self.act_device) for k in STATS}
            env = os.environ.copy()
            env.pop('PYTHONHOME',None)
            env.update(PYTHONPATH=os.pathsep.join(cfg['qwen_python_path']),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                       PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            if cfg['mode']=='goal_free':
                # The trainer removes both projected roles and goal_delta.
                # Measured qpos/current FK remain exactly the same inputs.
                self.geometry=module('arm_control_current_geometry',verify_ref(cfg['assets']['geometry']))
                self.chain=self.geometry.read_chain(verify_ref(cfg['assets']['urdf']))
            try:
                if cfg['mode']=='goal_free':
                    self.observer=None
                else:
                    self.observer = RpcWorker([cfg['qwen_python'],'-B','-m','experiments.goal_latent_act_arm_prefix10.observer_worker',
                        '--manifest',str(cfg['_manifest']),'--manifest-sha256',args['goal_latent_manifest_sha256'],
                        '--input-root',str(self.inputs),'--output-root',str(self.goals)],env,cfg['worker_timeout_seconds'])
                    require(self.observer.ready['kind']=='goal_latent_arm_prefix10_frozen_Qwen' and self.observer.ready['highlevel_sha256']==cfg['assets']['highlevel_weights']['sha256'], 'Observer worker identity differs')
            except BaseException:
                self.close(); raise
        atexit.register(self.close)

    def _latent(self, module, args):
        require(not bool(self.torch.count_nonzero(args[0])), 'Frozen ACT inference zero latent changed')

    def reset(self):
        self.base.reset(); self._action_queue = deque()
        self.episode += 1; self.executed = 0; self.started = time.monotonic()

    def goal(self, batch):
        from experiments.goal_latent_act_sequence_runtime_full.current_snapshot import current_inputs
        from policy.object_relative_hierarchy.canonical_context import STATE_SCHEMA
        evidence = json.loads(verify_ref(self.cfg['assets']['state_evidence']).read_text())
        observer, snapshot_proof = current_inputs(batch, evidence)
        if self.cfg['mode']=='goal_free':
            q=observer['qpos']
            now=np.stack([self.geometry.fk_link(self.chain,q[axes].astype(np.float64),side)@self.geometry.tcp()
                for side,axes in (('left',slice(0,6)),('right',slice(7,13)))]).astype(np.float32)
            return dict(qpos=q.copy(),current_eef=now,goal=now.copy(),current_roles=np.zeros((3,4096),dtype=np.float32))
        key = f'episode-{self.episode}-action-{self.executed}'
        path = self.inputs/(key+'.npz')
        with path.open('xb') as stream:
            np.savez(stream, **observer)
        original = sha(path)
        request = dict(request_id=key,path=path.name,instruction=self.cfg['instruction'],frame_ids={c:key+'-'+c for c in CAMERAS},
                       state_schema=STATE_SCHEMA,state_evidence_sha256=self.cfg['assets']['state_evidence']['sha256'])
        reply = self.observer.rpc(dict(cmd='extract',request=request))
        require(reply['request_id']==key and reply['input_sha256']==original and reply['generated_tokens']==0, 'Stale observer reply')
        result = (self.goals/reply['path']).resolve(strict=True)
        require(result.is_relative_to(self.goals.resolve()) and sha(result)==reply['sha256'], 'Bound current goal output differs')
        with np.load(result,allow_pickle=False) as z:
            require(set(z.files)=={'goal','current_eef','qpos','current_roles','input_sha256'}, 'Only physical prediction/current FK permitted')
            require(str(z['input_sha256'])==original and np.array_equal(z['qpos'],observer['qpos']), 'Goal and motor current observations differ')
            values = {k:z[k].copy() for k in ('goal','current_eef','qpos','current_roles')}
        require(values['current_roles'].shape==(3,4096) and values['current_roles'].dtype==np.float32
                and values['goal'].shape==values['current_eef'].shape==(2,4,4) and all(np.isfinite(v).all() for v in values.values()), 'Finite physical goals/current state required')
        path.unlink(); result.unlink()
        return values

    def select_action(self, batch):
        require(self.episode>=0 and time.monotonic()-self.started<=self.cfg['episode_wall_seconds'], 'Reset and bounded episode required')
        if not self._action_queue:
            with self.torch.inference_mode():
                normalized = dict(self.base.normalize_inputs(batch))
                normalized['observation.images'] = [normalized[k] for k in self.base.config.image_features]
                prediction = self.base.model(normalized)[0]
                physical = self.base.unnormalize_outputs({'action':prediction})['action']
                require(physical.shape==(1,50,14) and bool(self.torch.isfinite(physical).all()), 'Finite frozen native50 physical ACT commands required')
                if self.correction is not None:
                    goal = self.goal(batch)
                    from experiments.goal_latent_act_arm_runtime_full.inference import correct_native50
                    current={k:self.torch.from_numpy(v[None]).to(self.act_device) for k,v in goal.items()}
                    physical,result=correct_native50(self.correction,physical,current,self.stats,
                        state_action_only=self.cfg['mode']=='goal_free')
                    print('[temporal arm12] '+json.dumps(dict(episode=self.episode,executed=self.executed,
                        accepted=result['accepted'][0].tolist(),reliability=result['reliability_logits'][0].sigmoid().tolist(),
                        delta_arm_abs_max_rad=float(result['delta_arm_rad'].abs().max()),
                        native_grippers_bitexact=True,output_native_codec=self.cfg['gripper_codec'])),flush=True)
                self._action_queue.extend(physical[0,:10].unbind(0))
                print('[prefix10 replan] '+json.dumps(dict(episode=self.episode,executed=self.executed,mode=self.cfg['mode'],native_generated=50,native_executed=10,replanning_commands=10,current_observation_action=self.executed,native_grippers_bitexact=True)),flush=True)
        action = self._action_queue.popleft()[None]
        self.executed += 1
        require(action.shape==(1,14) and bool(self.torch.isfinite(action).all()),'Finite generated50/executed10 command required')
        return action

    def close(self):
        observer = getattr(self,'observer',None)
        if observer is not None:
            observer.close(); self.observer=None
        hook = getattr(self,'latent_hook',None)
        if hook is not None:
            hook.remove(); self.latent_hook=None
        temp = getattr(self,'temp',None)
        if temp is not None:
            temp.cleanup(); self.temp=None


def get_model(args):
    return Runtime(args)


def eval(env,model,obs):
    from experiments.goal_latent_act_sequence_runtime_full.terminal import TerminalEnvironment
    from policy.robosyn_sprint.act_drawer_adapter import _ActionBoundary
    view = _ActionBoundary(model,env) if model.cfg['gripper_codec']=='drawer_normalized_opening_inverse_v1' else model
    return model.official.eval(TerminalEnvironment(env),view,obs)


def reset_model(model):
    return model.reset()


def close_model(model):
    return model.close()
