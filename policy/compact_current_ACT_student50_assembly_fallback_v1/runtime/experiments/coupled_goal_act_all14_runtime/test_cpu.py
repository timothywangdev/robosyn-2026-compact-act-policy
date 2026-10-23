"""Actual small-model CPU tests; no ACT/Qwen construction or CUDA context."""
from collections import deque
import contextlib
import io
import os
if __name__=='__main__' and os.environ.get('CUDA_VISIBLE_DEVICES')!='':
    raise RuntimeError('Standalone CPU compiler fixtures require CUDA_VISIBLE_DEVICES empty before Torch import')
import time
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from experiments.coupled_goal_act_all14.model import CoupledGoalACT
from experiments.coupled_goal_act_all14.inference import correct_native50
from .contracts import STATS
from .runtime import Runtime


def native(frame):
    one = torch.tensor([0.,1.,-1.,0.,0.,0.,.02,0.,1.,-1.,0.,0.,0.,.03])
    value = one.view(1,1,14).repeat(1,50,1)
    value[...,0] += float(frame)/10000
    value[...,7] += torch.arange(50)/10000
    return value


class FakeACT:
    def __init__(self):
        self.config = SimpleNamespace(image_features=['observation.images.cam_high'])
        self.calls = []
    def normalize_inputs(self,batch): return dict(batch)
    def unnormalize_outputs(self,batch): return batch
    def model(self,batch):
        frame = float(batch['observation.images'][0].mean())
        self.calls.append(frame)
        return native(frame),None
    def reset(self): pass


def context(frame=0):
    now = np.tile(np.eye(4,dtype=np.float32),(2,1,1))
    goal = now.copy(); goal[:,0,3] = .05
    qpos = native(frame)[0,0].numpy().copy(); qpos[[6,13]] = [.4,.6]
    return dict(current_roles=np.full((3,4096),frame,np.float32),qpos=qpos,goal=goal,current_eef=now)


def batch(frame): return {'observation.images.cam_high':torch.full((1,3,4,4),float(frame))}


def fixture(mode='goal',task='item_assembly'):
    actor = Runtime.__new__(Runtime)
    actor.torch = torch
    actor.cfg = dict(mode=mode,task=task,episode_wall_seconds=1800,gripper_codec=
        'drawer_normalized_opening_inverse_v1' if task=='drawer_open_place' else 'physical_meters_identity_v1')
    actor.base = FakeACT(); actor.act_device = 'cpu'; actor._action_queue = deque()
    actor.episode,actor.executed,actor.started = 0,0,time.monotonic()
    actor.compiler_report = dict(CPU_fixture_not_compile_qualification=True)
    actor.correction = None
    actor.stats = {key:torch.ones(width) if key.endswith('scale') else torch.zeros(width) for key,width in STATS.items()}
    if mode!='parent':
        torch.manual_seed(1701); actor.correction=CoupledGoalACT().eval().requires_grad_(False)
        with torch.no_grad():
            actor.correction.output.bias.fill_(.2)
            actor.correction.reliability[-1].bias.copy_(torch.tensor([4.,-4.]))
    actor.goal_calls = []
    def goal(current):
        frame = float(current['observation.images.cam_high'].mean())
        actor.goal_calls.append(frame)
        return context(frame)
    actor.goal = goal
    return actor


class RuntimeTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(2)

    def test_parent_prefix_and_replanning_use_actual_current_frame(self):
        actor = fixture('parent')
        with contextlib.redirect_stdout(io.StringIO()):
            for step in range(31):
                action = actor.select_action(batch(step))
                self.assertTrue(torch.equal(action,native((step//10)*10)[:,step%10]))
        self.assertEqual(actor.base.calls,[0.,10.,20.,30.])
        self.assertEqual(actor.goal_calls,[])

    def test_real_all14_inference_matches_queued_prefix_for_both_modes_and_codecs(self):
        for mode in ('goal','goal_free'):
            for task in ('item_assembly','drawer_open_place'):
                actor = fixture(mode,task)
                current = {key:torch.from_numpy(value[None]) for key,value in context().items()}
                with torch.inference_mode():
                    expected,result=correct_native50(actor.correction,native(0),current,actor.stats,task=task,
                        codec=actor.cfg['gripper_codec'],state_action_only=mode=='goal_free')
                    with contextlib.redirect_stdout(io.StringIO()): actual=torch.cat([actor.select_action(batch(0)) for _ in range(10)])[None]
                self.assertTrue(torch.equal(actual,expected[:,:10]))
                self.assertEqual(result['accepted'].tolist(),[[True,False]])
                self.assertTrue(torch.equal(actual[...,7:14],native(0)[:,:10,7:14]))
                self.assertFalse(torch.equal(actual[...,6],native(0)[:,:10,6]))
                self.assertEqual(actor.goal_calls,[0.]); self.assertEqual(len(actor._action_queue),0)

    def test_drawer_decode_and_inverse_are_once_in_unit_conversion(self):
        from experiments.coupled_goal_act_all14.units import native_to_physical,physical_to_native
        original=native(0); original[...,6]=.4; original[...,13]=.6
        physical=native_to_physical(original,torch.tensor([.05]))
        self.assertTrue(torch.equal(physical[...,6],original[...,6]*.05))
        restored=physical_to_native(physical,torch.tensor([.05]))
        self.assertTrue(torch.allclose(restored[...,6],original[...,6],atol=1e-7,rtol=0))
        self.assertTrue(torch.equal(physical[...,0:6],original[...,0:6]))

    def test_raw_rejected_hand_is_exact_even_when_native_opening_outside_physical_range(self):
        actor=fixture(task='drawer_open_place')
        original=native(0); original[...,13]=1.17
        current={key:torch.from_numpy(value[None]) for key,value in context().items()}
        with torch.inference_mode():
            corrected,result=correct_native50(actor.correction,original,current,actor.stats,
                task='drawer_open_place',codec=actor.cfg['gripper_codec'],state_action_only=False)
        self.assertFalse(bool(result['accepted'][0,1]))
        self.assertTrue(torch.equal(corrected[...,7:14],original[...,7:14]))

    def test_role_and_state_replan_current_context_every10_and_reset_clears_queue(self):
        for mode in ('goal','goal_free'):
            actor=fixture(mode)
            with contextlib.redirect_stdout(io.StringIO()):
                for step in range(21): actor.select_action(batch(step))
                actor.reset(); actor.select_action(batch(91))
            self.assertEqual(actor.base.calls,[0.,10.,20.,91.])
            self.assertEqual(actor.goal_calls,[0.,10.,20.,91.])
            self.assertEqual((actor.episode,actor.executed,len(actor._action_queue)),(1,1,9))

    def test_future_feature_packet_rejected_by_real_inference(self):
        actor=fixture(); current={key:torch.from_numpy(value[None]) for key,value in context().items()}
        current['future_qpos']=current['qpos']
        with self.assertRaises(ValueError),torch.inference_mode():
            correct_native50(actor.correction,native(0),current,actor.stats,task='item_assembly',
                codec=actor.cfg['gripper_codec'],state_action_only=False)

    def test_real_current_camera_snapshot_preserves_each_sensor_and_normalized_state(self):
        from experiments.goal_latent_act_sequence_runtime_full.current_snapshot import current_inputs
        cameras=('cam_high','cam_left_wrist','cam_right_wrist')
        current={'observation.state':torch.from_numpy(context()['qpos'][None])}
        for index,camera in enumerate(cameras):
            current['observation.images.'+camera]=torch.full((1,3,480,640),index*51,dtype=torch.uint8).float()/255.
        values,proof=current_inputs(current,dict(conversion='identity; source-declared state grippers already normalized'))
        self.assertTrue(np.array_equal(values['qpos'],context()['qpos']))
        for index,camera in enumerate(cameras): self.assertTrue(np.all(values[camera]==index*51))
        self.assertEqual(list(proof['images']),list(cameras))
        current['future_frame']=torch.zeros(1)
        with self.assertRaises(ValueError):current_inputs(current,dict(conversion='identity; source-declared state grippers already normalized'))

    def test_compiler_capture_uses_actual_inference_kwargs_and_four_real_AOT_paths(self):
        actor=fixture();actor.compiler_report=None
        actor.cfg.update(correction_compiler_backend='aot_eager',require_compile_speedup=False)
        current={key:torch.from_numpy(value[None]) for key,value in context().items()}
        with torch.inference_mode(),contextlib.redirect_stdout(io.StringIO()):actor.qualify_compiler(native(0),current)
        report=actor.compiler_report
        self.assertEqual(report['backend'],'aot_eager')
        self.assertFalse(report['full_GRU_compiled']);self.assertFalse(report['acceleration_qualified'])
        self.assertEqual(set(report['graph_and_kernel_calls']),{'roles','initial','output','reliability'})
        self.assertTrue(all(row['graphs']>0 and row['calls']>=6 for row in report['graph_and_kernel_calls'].values()))
        self.assertTrue(all(row['max_abs']==0 and row['RMSE']==0 for row in report['parity'].values()))

    def test_failed_declared_compiler_speed_gate_does_not_select_eager_fallback(self):
        actor=fixture();actor.compiler_report=None
        original=actor.correction
        actor.cfg.update(correction_compiler_backend='aot_eager',require_compile_speedup=True)
        current={key:torch.from_numpy(value[None]) for key,value in context().items()}
        with self.assertRaises(ValueError),torch.inference_mode():actor.qualify_compiler(native(0),current)
        self.assertIs(actor.correction,original);self.assertIsNone(actor.compiler_report)

    def test_no_CUDA_initialized(self): self.assertFalse(torch.cuda.is_initialized())

    def test_new_correction_precision_scope_restores_original_ACT_backend_flags(self):
        actor=fixture('parent')
        before=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        with actor.correction_precision():
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32);self.assertFalse(torch.backends.cudnn.allow_tf32)
        self.assertEqual((torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32),before)

    def test_typed_command_all30_flags_use_actual_official_horizons(self):
        from unittest.mock import patch
        from experiments.goal_latent_act_campaign.common import ref
        from .prepare import protocol,NATIVE
        from .supervisor import validate_command,UUID
        from .native import COMMAND,GENERATOR
        p=protocol();binding=ref(Path(__file__))
        for task,horizon in p['full_official_horizons'].items():
            for mode in ('parent','goal','goal_free'):
                command=dict(schema=COMMAND,GPU_uuid=UUID,mode=mode,task=task,official_horizon=horizon,
                    bindings=dict(deployment=binding,gym=ref(NATIVE/'upstream/RoboSynChallenge/configs'/task/'random/gym_config.json')),
                    inner_argv=['evaluator','--seed',str(GENERATOR),'--act_step','10','--max_episodes','1',
                        '--max_steps',str(horizon),'--eval_expert_check','True','--eval_max_seed_attempts_per_episodes','100',
                        '--eval_video_log','True','--coupled_all14_manifest',binding['path'],
                        '--coupled_all14_manifest_sha256',binding['sha256']])
                with patch('experiments.coupled_goal_act_all14_runtime.supervisor.load_manifest',return_value=dict(task=task,mode=mode)):
                    validate_command(command)
                    command['inner_argv'][command['inner_argv'].index('--act_step')+1]='50'
                    with self.assertRaises(ValueError):validate_command(command)

    def test_typed_plan_rejects_wrong_memory_cap_before_loading_any_actor(self):
        from .supervisor import validate_plan,SCHEMA,UUID
        plan=dict(schema=SCHEMA,gpu_uuid=UUID,admission_memory_floor_gib=32,
            whole_GPU_cap_bytes=24*(1<<30),owned_tree_RSS_cap_bytes=32*(1<<30))
        with self.assertRaises(ValueError):validate_plan(plan)


if __name__=='__main__': unittest.main()
