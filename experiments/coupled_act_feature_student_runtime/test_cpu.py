"""Actual causal/queue/unit tests with fake native models; no pretrained loads."""
import os
if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Hide CUDA before CPU fixtures')
from collections import deque
import contextlib
import io
import time
import unittest
import numpy as np
import torch
from experiments.coupled_act_feature_student.contracts import CAMERAS
from experiments.coupled_act_feature_student.hook import CurrentACTFeatures
from .inference import student_inputs,motor_inputs,correct_with_student
from .runtime import Runtime


class Backbone(torch.nn.Module):
    def __init__(self):super().__init__();self.calls=0
    def forward(self,x):self.calls+=1;return {'feature_map':x}


class Native(torch.nn.Module):
    def __init__(self):super().__init__();self.backbone=Backbone();self.calls=0
    def forward(self,batch):
        self.calls+=1
        for value in batch['observation.images']:self.backbone(value)
        native=torch.tensor([0.,1.,-1.,0.,0.,0.,.02,0.,1.,-1.,0.,0.,0.,1.17]).reshape(1,1,14).repeat(1,50,1)
        return native,None


class Base:
    def __init__(self):
        self.model=Native();self.config=type('Config',(),{'image_features':['observation.images.'+c for c in CAMERAS]})();self.resets=0
    def normalize_inputs(self,batch):return batch
    def unnormalize_outputs(self,batch):return batch
    def reset(self):self.resets+=1


class StudentFixture:
    def __init__(self):self.seen=[]
    def __call__(self,**inputs):
        self.seen.append({k:v.clone() for k,v in inputs.items()})
        return dict(projected_roles=torch.zeros(1,96),predicted_goal_delta=torch.zeros(1,18))


def projection(**kwargs):
    native=kwargs['native_base'];out=native.clone();out[...,:7]+=.005
    return dict(native_actions=out,accepted=torch.tensor([[True,False]]),reliability_logits=torch.tensor([[4.,-4.]]))


def stats():
    return {k:torch.full((width,), 2. if k.endswith('scale') else 1.) for k,width in
        {'qpos_mean':14,'qpos_scale':14,'base_mean':14,'base_scale':14,'current_eef_mean':18,'current_eef_scale':18,'eef_scale':18}.items()}


def visual_stats():return {k:torch.full((3,512),2. if k.endswith('scale') else 1.) for k in ('visual_mean','visual_scale','global_mean','global_scale')}


def fixture():
    value=Runtime.__new__(Runtime);value.torch=torch;value.base=Base();value.act_device='cpu'
    value.cfg=dict(task='item_assembly',mode='student',episode_wall_seconds=100,gripper_codec='physical_meters_identity_v1')
    value.episode=-1;value.executed=0;value._action_queue=deque();value.latent_hook=None
    value.student=StudentFixture();value.correction=object();value.capture=CurrentACTFeatures(value.base.model.backbone,value.base.config.image_features)
    value.stats=stats();value.visual_stats=visual_stats();value.compiler_report=None;value.student_call=None;value.projected_call=None
    value.compiler_calls=0
    def current(batch):
        q=batch['observation.state'];now=torch.eye(4).reshape(1,1,4,4).repeat(1,2,1,1)
        return dict(qpos=q,current_eef=now),dict(task='item_assembly',episode=value.episode,action=value.executed,input_sha256=str(float(q[0,0])),batch_size=1)
    value.current=current
    def qualify(inputs,motor):
        value.compiler_calls+=1;value.compiler_report=dict(fixture=True);value.projected_call=projection
    value.qualify_compiler=qualify
    return value


def batch(number):
    return {'observation.state':torch.full((1,14),float(number)),**{'observation.images.'+cam:torch.full((1,512,4,6),float(number+100*i)) for i,cam in enumerate(CAMERAS)}}


class Tests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_same_forward_three_camera_capture_and_prefix10_no_second_CNN(self):
        model=fixture();model.reset()
        with contextlib.redirect_stdout(io.StringIO()):
            actions=[model.select_action(batch(i)) for i in range(21)]
        self.assertEqual(model.base.model.calls,3);self.assertEqual(model.base.model.backbone.calls,9)
        self.assertEqual(model.compiler_calls,1);self.assertEqual(len(model.student.seen),3)
        for index,inputs in zip((0,10,20),model.student.seen):
            for cam in range(3):self.assertTrue(torch.equal(inputs['visual'][0,cam],torch.full((512,2,3),(index+100*cam-1)/2)))
            self.assertTrue(torch.equal(inputs['qpos_features'],torch.full((1,14),(index-1)/2)))
        self.assertTrue(torch.equal(actions[0],actions[9]));self.assertEqual(model.executed,21);model.close()

    def test_reset_clears_native_queue_and_fresh_capture_identity(self):
        model=fixture();model.reset()
        with contextlib.redirect_stdout(io.StringIO()):model.select_action(batch(3))
        self.assertEqual(len(model._action_queue),9);model.reset()
        self.assertEqual(len(model._action_queue),0);self.assertEqual((model.episode,model.executed),(1,0));self.assertIsNone(model.capture.active)
        with contextlib.redirect_stdout(io.StringIO()):model.select_action(batch(7))
        self.assertEqual(model.base.model.calls,2);model.close()

    def test_future_and_inconsistent_current_state_are_rejected(self):
        native=torch.zeros(1,50,14);current=dict(qpos=torch.zeros(1,14),current_eef=torch.eye(4).reshape(1,1,4,4).repeat(1,2,1,1))
        with self.assertRaises(ValueError):motor_inputs(native,dict(current,future_qpos=current['qpos']),stats(),task='item_assembly',codec='physical_meters_identity_v1')
        inputs=student_inputs(torch.zeros(1,3,512,2,3),torch.zeros(1,3,512),current['qpos'],visual_stats(),stats())
        motor=motor_inputs(native,current,stats(),task='item_assembly',codec='physical_meters_identity_v1');motor['qpos_features']=motor['qpos_features']+1
        with self.assertRaises(ValueError):correct_with_student(StudentFixture(),object(),inputs,motor,projected_call=projection)

    def test_student_visual_normalization_bitexact_to_training_batch(self):
        from experiments.coupled_act_feature_student.data import batch as training_batch
        visual=torch.arange(3*512*6).reshape(1,3,512,2,3).float();global_mean=visual.mean((-2,-1));q=torch.randn(1,14)
        statistics={k:v.numpy() for k,v in visual_stats().items()};teacher=stats()
        actual=student_inputs(visual,global_mean,q,visual_stats(),teacher)
        arrays=dict(visual=visual.numpy(),visual_global=global_mean.numpy(),qpos_features=((q-teacher['qpos_mean'])/teacher['qpos_scale']).numpy())
        expected=training_batch(arrays,np.array([0]),statistics)
        for key in actual:self.assertTrue(torch.equal(actual[key],expected[key]),key)

    def test_two_codecs_full_hand_fallback_and_no_goal_renormalization(self):
        from experiments.coupled_act_feature_student.model import Student
        from experiments.coupled_goal_act_all14.model import CoupledGoalACT
        actor=CoupledGoalACT().eval().requires_grad_(False);student=Student().eval().requires_grad_(False)
        self.assertEqual(sum(p.numel() for p in actor.parameters()),180528)
        with torch.no_grad():actor.reliability[-1].bias.copy_(torch.tensor([4.,-4.]));actor.output.bias.fill_(.1)
        native=torch.tensor([0.,1.,-1.,0.,0.,0.,.02,0.,1.,-1.,0.,0.,0.,1.17]).reshape(1,1,14).repeat(1,50,1)
        current=dict(qpos=torch.zeros(1,14),current_eef=torch.eye(4).reshape(1,1,4,4).repeat(1,2,1,1))
        inputs=student_inputs(torch.zeros(1,3,512,2,3),torch.zeros(1,3,512),current['qpos'],visual_stats(),stats())
        for task,codec,scale in (('item_assembly','physical_meters_identity_v1',1.),('drawer_open_place','drawer_normalized_opening_inverse_v1',.05)):
            motor=motor_inputs(native,current,stats(),task=task,codec=codec)
            with torch.inference_mode():action,result,predicted=correct_with_student(student,actor,inputs,motor)
            self.assertTrue(torch.equal(action[...,7:],native[...,7:]));self.assertTrue(result['accepted'][0,0]);self.assertFalse(result['accepted'][0,1])
            self.assertEqual(float(motor['grip_scale'][0]),float(torch.tensor(scale)))
            seen=[]
            def capture(**kwargs):seen.append(kwargs['goal_delta'].clone());return projection(**kwargs)
            with torch.inference_mode():_,_,prediction=correct_with_student(student,actor,inputs,motor,projected_call=capture)
            self.assertTrue(torch.equal(seen[0],prediction['predicted_goal_delta']))

    def test_wrong_camera_order_stale_capture_and_fourth_callback_reject(self):
        with self.assertRaises(ValueError):CurrentACTFeatures(Backbone(),['observation.images.'+c for c in reversed(CAMERAS)])
        backbone=Backbone();capture=CurrentACTFeatures(backbone,['observation.images.'+c for c in CAMERAS])
        identity=dict(task='t',episode=0,action=0,input_sha256='a',batch_size=1);capture.begin(**identity)
        for _ in range(3):backbone(torch.zeros(1,512,2,3))
        with self.assertRaises(ValueError):capture.finish(dict(identity,action=10))
        with self.assertRaises(ValueError):backbone(torch.zeros(1,512,2,3))
        capture.close()

    def test_native_parent_is_exact_and_does_not_capture_or_infer_student(self):
        model=fixture();model.capture.close();model.capture=None;model.student=None;model.cfg['mode']='parent';model.reset()
        with contextlib.redirect_stdout(io.StringIO()):actions=[model.select_action(batch(i)) for i in range(11)]
        self.assertEqual(model.base.model.calls,2);self.assertEqual(model.compiler_calls,0);self.assertEqual(len(model.student.seen) if model.student else 0,0)
        self.assertEqual(float(actions[0][0,13]),float(torch.tensor(1.17)));self.assertTrue(torch.equal(actions[0],actions[10]));model.close()

    def test_exception_during_native_forward_clears_capture_and_propagates(self):
        model=fixture();model.reset()
        def failure(_):raise RuntimeError('fixture ACT exception')
        model.base.model.forward=failure
        with self.assertRaisesRegex(RuntimeError,'fixture ACT exception'):model.select_action(batch(0))
        self.assertIsNone(model.capture.active);self.assertEqual(len(model._action_queue),0);model.close()

    def test_CPU_fixture_never_initializes_CUDA_or_imports_Qwen(self):
        self.assertFalse(torch.cuda.is_initialized());self.assertNotIn('transformers',__import__('sys').modules)

    def test_actual_AOT_compile_attempt_includes_full_tail_and_Boolean_decisions(self):
        from experiments.coupled_act_feature_student.model import Student
        from experiments.coupled_goal_act_all14.model import CoupledGoalACT
        from .compiler import qualify_current
        student=Student().eval().requires_grad_(False);actor=CoupledGoalACT().eval().requires_grad_(False)
        with torch.no_grad():actor.reliability[-1].bias.copy_(torch.tensor([4.,-4.]))
        q=torch.zeros(1,14);native=torch.tensor([0.,1.,-1.,0.,0.,0.,.02,0.,1.,-1.,0.,0.,0.,1.17]).reshape(1,1,14).repeat(1,50,1)
        current=dict(qpos=q,current_eef=torch.eye(4).reshape(1,1,4,4).repeat(1,2,1,1))
        inputs=student_inputs(torch.zeros(1,3,512,2,3),torch.zeros(1,3,512),q,visual_stats(),stats())
        motor=motor_inputs(native,current,stats(),task='item_assembly',codec='physical_meters_identity_v1')
        call,projected,report=qualify_current(student,actor,inputs,motor,backend='aot_eager',benchmark=False)
        self.assertIsNone(projected);self.assertFalse(report['acceleration_qualified']);self.assertEqual(report['selected_inference_mode'],'eager')
        self.assertTrue(report['actual_tail_numeric_and_Boolean_parity_required']);self.assertEqual(report['numerical'][0]['actual_tail_maximum_difference']['accepted'],0)
        self.assertTrue(any(values.get('unique_graphs',0)>0 for values in report['compiler_counter_delta'].values()))
        with torch.inference_mode():one=student(**inputs);two=call(**inputs)
        for key in one:self.assertTrue(torch.equal(one[key],two[key]))

    def test_actual_checkpoint_metadata_moments_reject_future_and_wrong_scope(self):
        from copy import deepcopy
        from .contracts import validate_saved
        reference=dict(path='explicit_fixture',bytes=1,sha256='0'*64)
        cfg=dict(student_qpos_only=False,assets=dict(student_feature_plan=reference));training=dict(protocol=reference)
        saved=dict(model={},statistics={k:v.numpy() for k,v in visual_stats().items()},pooling='spatial2x3',qpos_only=False,
                   feature_plan=reference,protocol=reference,fixed_epoch=30)
        validate_saved(saved,cfg,training)
        for key,change in (('future_commands',lambda x:x.update(future_commands=np.zeros((50,14),np.float32))),
                           ('wrong_epoch',lambda x:x.update(fixed_epoch=29)),
                           ('wrong_qpos_control',lambda x:x.update(qpos_only=True)),
                           ('wrong_moment_shape',lambda x:x['statistics'].update(visual_mean=np.zeros((512,),np.float32))),
                           ('nonpositive_scale',lambda x:x['statistics'].update(visual_scale=np.zeros((3,512),np.float32)))):
            changed=deepcopy(saved);change(changed)
            with self.assertRaises(ValueError,msg=key):validate_saved(changed,cfg,training)

    def test_actual_source_FK_current_matches_qualified_state_runtime(self):
        import json
        from pathlib import Path
        from .contracts import module,verify_ref
        from experiments.goal_latent_act_arm_prefix10.runtime import Runtime as OldRuntime
        campaign=os.environ.get('STUDENT_RUNTIME_CPU_CAMPAIGN')
        if not campaign:self.skipTest('Actual owned qualification supplies the exact staged campaign')
        cfg=json.loads(Path(json.loads(Path(campaign).read_text())['plans'][0]['deployment']['path']).read_text())
        geometry=module('student_actual_CPU_FK_fixture',verify_ref(cfg['assets']['geometry']));chain=geometry.read_chain(verify_ref(cfg['assets']['urdf']))
        student=Runtime.__new__(Runtime);student.torch=torch;student.cfg=cfg;student.geometry=geometry;student.chain=chain;student.act_device='cpu';student.episode=0;student.executed=10
        old=OldRuntime.__new__(OldRuntime);old.cfg=dict(cfg,mode='goal_free');old.geometry=geometry;old.chain=chain
        images={'observation.images.'+cam:torch.full((1,3,480,640),float(i*51)/255.) for i,cam in enumerate(CAMERAS)}
        for shift in (0.,.1,-.1):
            q=torch.tensor([[shift,1.,-1.,0.,0.,0.,.4,-shift,1.,-1.,0.,0.,0.,.8]])
            current_batch=dict(images,**{'observation.state':q})
            before={k:v.clone() for k,v in current_batch.items()}
            actual,identity=student.current(current_batch);expected=old.goal(current_batch)
            self.assertTrue(torch.equal(actual['qpos'],torch.from_numpy(expected['qpos'][None])))
            self.assertTrue(torch.equal(actual['current_eef'],torch.from_numpy(expected['current_eef'][None])))
            self.assertEqual(identity['action'],10);self.assertEqual(set(actual),{'qpos','current_eef'})
            for key in before:self.assertTrue(torch.equal(before[key],current_batch[key]))


if __name__=='__main__':unittest.main()
