"""One current ACT backbone forward supplies both native50 and student maps."""
import atexit
from collections import deque
import hashlib
import inspect
import json
import time
from .contracts import STATS,MOMENTS,load_manifest,load_official,module,require,verify_ref,sha,validate_saved
from experiments.coupled_goal_act_all14_runtime.runtime import Runtime as TeacherRuntime
from experiments.goal_latent_act_arm_prefix10.runtime import Runtime as PrefixRuntime


class Runtime:
    _latent=PrefixRuntime._latent
    sync=TeacherRuntime.sync
    correction_precision=TeacherRuntime.correction_precision

    def __init__(self,args):
        import torch
        from safetensors.torch import load_model
        self.torch=torch;self.cfg=cfg=load_manifest(args['coupled_student_manifest'],args['coupled_student_manifest_sha256'])
        require(args['task_name']==cfg['task'], 'Actual current task differs from student binding')
        self.official=load_official(cfg);torch.manual_seed(17)
        self.base=self.official.get_model(dict(checkpoint_path=cfg['act_root'],device='cuda',pytorch_device='cuda',act_step=50))
        load_model(self.base,str(verify_ref(cfg['act_files']['model.safetensors'])),strict=True,device='cuda:0')
        self.base.eval().requires_grad_(False)
        from lerobot.policies.act.modeling_act import ACTPolicy
        require(sha(inspect.getfile(ACTPolicy))==cfg['assets']['ACT_model_source']['sha256'], 'Actual released ACT implementation differs')
        require(self.base.config.chunk_size==self.base.config.n_action_steps==50
                and self.base.config.temporal_ensemble_coeff is None and next(self.base.parameters()).dtype==torch.float32, 'Native eager FP32 ACT50')
        require(list(self.base.config.image_features)==cfg['camera_order'], 'Actual ACT feature loop differs from trained camera order')
        self.act_device=self.base.act_device;self.config=self.base.config
        self.act_step,self.state_obs_path,self.strict_action_dim=10,'robot/qpos',True
        self.act_image_keys,self.image_key_map=self.base.act_image_keys,self.base.image_key_map
        self.episode,self.executed=-1,0;self._action_queue=deque()
        self.student,self.correction,self.capture,self.compiler_report=None,None,None,None
        self.student_call,self.projected_call=None,None
        self.latent_hook=self.base.model.encoder_latent_input_proj.register_forward_pre_hook(self._latent)
        try:
            if cfg['mode']=='student':
                from experiments.coupled_act_feature_student.model import Student
                from experiments.coupled_act_feature_student.hook import CurrentACTFeatures
                from experiments.coupled_goal_act_all14.contracts import load_checkpoint
                saved=torch.load(verify_ref(cfg['assets']['student_weights']),map_location='cpu',weights_only=False)
                training=json.loads(verify_ref(cfg['assets']['student_training_receipt']).read_text())
                validate_saved(saved,cfg,training)
                self.student=Student('spatial2x3',qpos_only=saved['qpos_only'])
                self.student.load_state_dict(saved['model'],strict=True);self.student.eval().requires_grad_(False).to(self.act_device)
                self.visual_stats={k:torch.as_tensor(saved['statistics'][k],dtype=torch.float32,device=self.act_device) for k in MOMENTS}
                for key,shape in MOMENTS.items():
                    v=self.visual_stats[key];require(v.shape==shape and bool(torch.isfinite(v).all()),'Exact optimizer800 visual moments: '+key)
                    if key.endswith('scale'):require(bool((v>=.01).all()), 'Frozen optimizer moment floor .01')
                contract_path=verify_ref(cfg['assets']['teacher_contract'])
                self.correction,self.correction_contract=load_checkpoint(contract_path.parent,cfg['assets']['teacher_weights']['sha256'],device=self.act_device)
                require(self.correction_contract==json.loads(contract_path.read_text()), 'Actual strict teacher contract differs')
                # projected_forward never calls the teacher-only roles module.
                # Keep all trained tensors for exact strict checkpoint identity.
                require(sum(p.numel() for p in self.correction.parameters())==180528, 'Exact frozen coupled teacher parameter count')
                self.stats={k:torch.tensor(self.correction_contract['statistics'][k],dtype=torch.float32,device=self.act_device) for k in STATS}
                self.geometry=module('student_current_geometry',verify_ref(cfg['assets']['geometry']))
                self.chain=self.geometry.read_chain(verify_ref(cfg['assets']['urdf']))
                self.capture=CurrentACTFeatures(self.base.model.backbone,cfg['camera_order'])
            self.act_execution=dict(mode='eager',compiled=False,dtype='FP32',native_generated=50,native_executed=10,
                                    Qwen_models_loaded=0,ACT_models_loaded=1,second_CNN_forward=False)
            print('[student ACT execution] '+json.dumps(self.act_execution),flush=True)
            atexit.register(self.close)
        except BaseException:self.close();raise

    def reset(self):
        self.base.reset();self._action_queue=deque();self.episode+=1;self.executed=0;self.started=time.monotonic()
        if self.capture is not None:self.capture.reset()

    def current(self,batch):
        import numpy as np
        from experiments.goal_latent_act_sequence_runtime_full.current_snapshot import current_inputs
        observer,proof=current_inputs(batch,json.loads(verify_ref(self.cfg['assets']['state_evidence']).read_text()))
        q=observer['qpos'];now=np.stack([self.geometry.fk_link(self.chain,q[axes].astype(np.float64),side)@self.geometry.tcp()
            for side,axes in (('left',slice(0,6)),('right',slice(7,13)))]).astype(np.float32)
        digest=hashlib.sha256(q.tobytes()+json.dumps(proof['images'],sort_keys=True).encode()).hexdigest()
        # Episode/action are capture identity only, never student/motor features.
        identity=dict(task=self.cfg['task'],episode=self.episode,action=self.executed,input_sha256=digest,batch_size=1)
        return dict(qpos=self.torch.from_numpy(q[None]).to(self.act_device),current_eef=self.torch.from_numpy(now[None]).to(self.act_device)),identity

    def qualify_compiler(self,inputs,motor):
        from .compiler import qualify_current
        self.student_call,self.projected_call,self.compiler_report=qualify_current(self.student,self.correction,inputs,motor)
        print('[current student compiler] '+json.dumps(self.compiler_report),flush=True)

    def select_action(self,batch):
        require(self.episode>=0 and time.monotonic()-self.started<=self.cfg['episode_wall_seconds'], 'Reset and bounded episode required')
        if not self._action_queue:
            self.sync();start=time.monotonic()
            with self.torch.inference_mode():
                current,identity=(self.current(batch) if self.capture is not None else (None,None))
                if self.capture is not None:self.capture.begin(**identity)
                try:
                    normalized=dict(self.base.normalize_inputs(batch))
                    normalized['observation.images']=[normalized[k] for k in self.base.config.image_features]
                    prediction=self.base.model(normalized)[0]
                    visual=(self.capture.finish(identity) if self.capture is not None else None)
                except BaseException:
                    if self.capture is not None:self.capture.reset()
                    raise
                native=self.base.unnormalize_outputs({'action':prediction})['action']
                require(native.shape==(1,50,14) and native.dtype==self.torch.float32 and bool(self.torch.isfinite(native).all()), 'Finite original native ACT50')
                self.sync();core_done=time.monotonic();action=native
                timings=dict(current_and_ACT_wall_seconds=core_done-start,student_motor_wall_seconds=0.,cold_compiler_wall_seconds=0.)
                if self.student is not None:
                    from .inference import student_inputs,motor_inputs,correct_with_student
                    inputs=student_inputs(*visual,current['qpos'],self.visual_stats,self.stats)
                    motor=motor_inputs(native,current,self.stats,task=self.cfg['task'],codec=self.cfg['gripper_codec'])
                    with self.correction_precision():
                        if self.compiler_report is None:
                            begin=time.monotonic();self.qualify_compiler(inputs,motor);self.sync();timings['cold_compiler_wall_seconds']=time.monotonic()-begin
                        begin=time.monotonic();action,result,_=correct_with_student(self.student,self.correction,inputs,motor,
                            student_call=self.student_call,projected_call=self.projected_call)
                        self.sync();timings['student_motor_wall_seconds']=time.monotonic()-begin
                    print('[current ACT student] '+json.dumps(dict(episode=self.episode,executed=self.executed,
                        accepted=result['accepted'][0].tolist(),reliability=result['reliability_logits'][0].sigmoid().tolist(),
                        rejected_hand_all7_bitexact=True,Qwen_models_loaded=0,second_CNN_forward=False)),flush=True)
                self._action_queue.extend(action[0,:10].unbind(0))
                print('[current student prefix10 replan] '+json.dumps(dict(episode=self.episode,executed=self.executed,
                    mode=self.cfg['mode'],native_generated=50,native_executed=10,current_observation_action=self.executed,
                    total_replan_wall_seconds=time.monotonic()-start,timings=timings)),flush=True)
        action=self._action_queue.popleft()[None];self.executed+=1
        require(action.shape==(1,14) and bool(self.torch.isfinite(action).all()), 'Finite native14 command')
        return action

    def close(self):
        capture=getattr(self,'capture',None)
        if capture is not None:capture.close();self.capture=None
        hook=getattr(self,'latent_hook',None)
        if hook is not None:hook.remove();self.latent_hook=None


def get_model(args):return Runtime(args)


def eval(env,model,obs):
    from experiments.goal_latent_act_sequence_runtime_full.terminal import TerminalEnvironment
    from policy.robosyn_sprint.act_drawer_adapter import _ActionBoundary
    view=_ActionBoundary(model,env) if model.cfg['gripper_codec']=='drawer_normalized_opening_inverse_v1' else model
    return model.official.eval(TerminalEnvironment(env),view,obs)


def reset_model(model):return model.reset()
def close_model(model):return model.close()
