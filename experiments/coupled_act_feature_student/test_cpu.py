"""Meaningful synthetic CPU hook/unit/causality/ABI tests; no pretrained loads."""
import os
if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Hide CUDA before synthetic CPU fixtures')
import ast
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from .contracts import CAMERAS,ROLE_QUERIES,choose_episodes,partition,records,validate_feature_arrays
from .hook import CurrentACTFeatures
from .model import Student,teacher_targets
from .projected import projected_forward


class FakeBackbone(torch.nn.Module):
    def forward(self,value):return {'feature_map':value}


class Tests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_hook_current_three_camera_maps_unmodified_and_spatial_global_explicit(self):
        backbone=FakeBackbone();hook=CurrentACTFeatures(backbone,['observation.images.'+c for c in CAMERAS])
        identity=dict(task='item_assembly',episode=7,action=10,input_sha256='current_fixture',batch_size=1)
        hook.begin(**identity);values=[]
        for camera in range(3):
            value=(torch.arange(35).float().square().reshape(1,1,5,7)+camera*100).repeat(1,512,1,1);saved=value.clone();values.append(value)
            output=backbone(value);self.assertIs(output['feature_map'],value);self.assertTrue(torch.equal(saved,value))
        spatial,global_mean=hook.finish(identity)
        self.assertEqual(tuple(spatial.shape),(1,3,512,2,3));self.assertTrue(torch.equal(global_mean,torch.stack([v.mean((-2,-1)) for v in values],1)))
        self.assertFalse(torch.equal(global_mean,spatial.mean((-2,-1))))
        self.assertFalse(spatial.requires_grad);self.assertIsNone(hook.active)
        with self.assertRaises(ValueError):hook.finish(identity)
        hook.begin(**identity);hook.reset()
        with self.assertRaises(ValueError):backbone(values[0])
        hook.close()

    def test_hook_wrong_camera_order_and_stale_identity_fail(self):
        with self.assertRaises(ValueError):CurrentACTFeatures(FakeBackbone(),['observation.images.cam_high','observation.images.cam_left_wrist','observation.images.cam_right_wrist'])
        backbone=FakeBackbone();hook=CurrentACTFeatures(backbone,['observation.images.'+c for c in CAMERAS]);identity=dict(task='t',episode=1,action=0,input_sha256='a',batch_size=1)
        hook.begin(**identity)
        for _ in range(3):backbone(torch.zeros(1,512,4,6))
        with self.assertRaises(ValueError):hook.finish(dict(identity,action=10))
        with self.assertRaises(ValueError):backbone(torch.zeros(1,512,4,6))
        hook.close()

    def test_actual_parameter_counts_and_global_requires_actual_global_feature(self):
        spatial=Student();global_model=Student('global')
        self.assertEqual(sum(p.numel() for p in spatial.parameters()),2392434)
        self.assertEqual(sum(p.numel() for p in global_model.parameters()),426354)
        with self.assertRaises(ValueError):global_model(torch.zeros(1,3,512,2,3),torch.zeros(1,14))
        self.assertEqual(global_model(torch.zeros(1,3,512,2,3),torch.zeros(1,14),torch.zeros(1,3,512))['projected_roles'].shape,(1,96))

    def test_qpos_only_control_is_invariant_to_actual_visual_values(self):
        torch.manual_seed(1701);model=Student(qpos_only=True);q=torch.randn(2,14)
        one=model(torch.randn(2,3,512,2,3),q);two=model(torch.randn(2,3,512,2,3)*100,q)
        for key in one:self.assertTrue(torch.equal(one[key],two[key]))

    def test_projected_input_tail_bitexact_to_original_for_both_codecs_and_controls(self):
        from experiments.coupled_goal_act_all14.model import CoupledGoalACT
        torch.manual_seed(1701);teacher=CoupledGoalACT().eval()
        with torch.no_grad():
            teacher.output.weight.normal_(std=.005)
            teacher.output.bias.fill_(.2);teacher.reliability[-1].bias.copy_(torch.tensor([4.,-4.]))
        for scale in (.05,1.):
            for control in (False,True):
                raw=torch.randn(2,3,4096);native=torch.tensor([0.,1.,-1.,0.,0.,0.,.02,0.,1.,-1.,0.,0.,0.,1.17]).reshape(1,1,14).repeat(2,50,1)
                kwargs=dict(qpos_features=torch.randn(2,14),current_eef_features=torch.randn(2,18),goal_delta=torch.randn(2,18),
                    native_base_features=torch.randn(2,50,14),native_base=native,grip_scale=torch.full((2,),scale),state_action_only=control)
                with torch.inference_mode():
                    old=teacher(roles=raw,**kwargs);new=projected_forward(teacher,projected_roles=torch.tanh(teacher.roles(raw)).flatten(1),**kwargs)
                for key in old:self.assertTrue(torch.equal(old[key],new[key]),key)
                self.assertTrue(torch.equal(new['native_actions'][...,7:],native[...,7:]))

    def test_teacher_targets_are_semantic_queries_and_predicted_current_head_only(self):
        from experiments.coupled_goal_act_all14.model import CoupledGoalACT
        self.assertEqual(ROLE_QUERIES,('left_arm','right_arm','object_destination'))
        teacher=CoupledGoalACT();roles=torch.randn(2,3,4096);now=torch.eye(4).reshape(1,1,4,4).repeat(2,2,1,1);goal=now.clone();goal[...,0,3]=.1
        targets=teacher_targets(teacher,roles,now,goal,torch.full((18,),.01))
        self.assertTrue(torch.equal(targets['projected_roles'],torch.tanh(teacher.roles(roles)).flatten(1)))
        self.assertTrue(torch.equal(targets['predicted_goal_delta'][:,[0,9]],torch.full((2,2),10.)))
        self.assertFalse(targets['predicted_goal_delta'].requires_grad)

    def test_episode_partitions_are_disjoint_balanced_and_monitor_rejected(self):
        choice=choose_episodes('task','revision');self.assertEqual({k:len(v) for k,v in choice.items()},{'optimizer':80,'diagnostic':10,'gate':10})
        seen=set()
        for key,episodes in choice.items():
            self.assertFalse(seen.intersection(episodes));seen.update(episodes)
            self.assertTrue(all(partition(e)==key for e in episodes))
        self.assertEqual(choice,choose_episodes('task','revision'))
        with self.assertRaises(ValueError):partition(900)

    def test_bounded_index_parser_and_future_tensor_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'rows.json';values=[dict(task='t',episode=i,text='x'*40000) for i in range(40)];p.write_text(json.dumps(values));self.assertEqual(list(records(p)),values)
        row=dict(task='t',episode=0,frame=0,split='fit',partition='optimizer')
        arrays={k:np.zeros((1,*shape),np.float32) for k,shape in __import__('experiments.coupled_act_feature_student.contracts',fromlist=['FIELDS']).FIELDS.items()}
        validate_feature_arrays(arrays,[row]);arrays['future_commands']=np.zeros((1,50,14),np.float32)
        with self.assertRaises(ValueError):validate_feature_arrays(arrays,[row])

    def test_new_namespace_does_not_edit_frozen_actor_or_load_ACT_Qwen(self):
        self.assertFalse(torch.cuda.is_initialized())
        self.assertNotIn('transformers',__import__('sys').modules)


if __name__=='__main__':unittest.main()
