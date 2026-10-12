"""Actual frozen small motor checkpoint gradient tests, no ACT/Qwen/frames."""
import os
if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Hide CUDA before CPU motor-gradient checks')
import ast,hashlib,json,unittest
from pathlib import Path
import torch
from safetensors.torch import load_file
from .contracts import ref,verify
from .model import Student
from .projected import projected_forward
from .policy_aware import policy_aware_forward
from .policy_aware_objective import objective,WEIGHTS
from experiments.coupled_goal_act_all14.model import CoupledGoalACT
from experiments.goal_latent_act_arm_sequence_full.fk import CommandFK

ROOT=Path(__file__).resolve().parents[2]
PLAN=ROOT/'outputs/architecture-reassessment/coupled-act-feature-student-preparation-v1/CPU-plan-v4/plan.json'


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cls.plan=json.loads(PLAN.read_text());cls.contract=json.loads(verify(cls.plan['teacher_contract']).read_text())
        cls.weights=ref(verify(cls.plan['teacher_weights']))
        assert cls.weights['sha256']==cls.contract['weights_sha256']=='b70de5a2c62b8aa6ea85925dd9bbbbc83b65f6aa1b878b70f6a4431df7e12649'
        cls.actor=CoupledGoalACT().eval().requires_grad_(False);cls.actor.load_state_dict(load_file(cls.weights['path'],device='cpu'),strict=True)
        cls.fk=CommandFK(cls.contract['geometry'],cls.contract['urdf']).eval().requires_grad_(False)
        cls.original={k:v.clone() for k,v in cls.actor.state_dict().items()}

    def inputs(self,scale=.05):
        torch.manual_seed(1701)
        native=torch.tensor([0.,1.,-1.,0.,0.,0.,.02,0.,1.,-1.,0.,0.,0.,.02]).reshape(1,1,14).repeat(2,50,1)
        return dict(qpos_features=torch.randn(2,14),current_eef_features=torch.randn(2,18),
            native_base_features=torch.randn(2,50,14),native_base=native,grip_scale=torch.full((2,),scale))

    def test_only_two_student_detaches_removed_and_motor_outputs_bitexact(self):
        old=ast.parse((ROOT/'experiments/coupled_act_feature_student/projected.py').read_text());fn=next(x for x in old.body if isinstance(x,ast.FunctionDef))
        class Inputs(ast.NodeTransformer):
            def visit_Call(self,x):
                if isinstance(x.func,ast.Attribute) and x.func.attr=='detach' and isinstance(x.func.value,ast.Name) and x.func.value.id in ('projected_roles','goal_delta'):return x.func.value
                return self.generic_visit(x)
        expected=Inputs().visit(fn)
        new=ast.parse((ROOT/'experiments/coupled_act_feature_student/policy_aware.py').read_text());actual=next(x for x in new.body if isinstance(x,ast.FunctionDef))
        self.assertEqual([ast.dump(x) for x in expected.body],[ast.dump(x) for x in actual.body[1:]])
        for scale in (.05,1.):
            for control in (False,True):
                inputs=self.inputs(scale);roles=torch.randn(2,96);goal=torch.randn(2,18)
                with torch.no_grad():
                    a=projected_forward(self.actor,projected_roles=roles,goal_delta=goal,state_action_only=control,**inputs)
                    b=policy_aware_forward(self.actor,projected_roles=roles,goal_delta=goal,state_action_only=control,**inputs)
                for key in a:self.assertTrue(torch.equal(a[key],b[key]),key)

    def test_actual_frozen_actor_policy_loss_reaches_both_student_heads(self):
        torch.manual_seed(1701);student=Student();pred=student(torch.randn(2,3,512,2,3),torch.randn(2,14))
        inputs=self.inputs(1.);result=policy_aware_forward(self.actor,projected_roles=pred['projected_roles'],goal_delta=pred['predicted_goal_delta'],**inputs)
        target=result['parent_physical_actions'].detach().clone();target[...,0]+=.02;target[...,7]-=.02;target[...,6]=.03;target[...,13]=.01
        valid=torch.zeros(2,50,dtype=torch.bool);valid[:,:10]=True
        labels={k:v.detach().clone() for k,v in pred.items()}
        total,parts=objective(pred,result,labels,target,valid,self.fk)
        policy=sum(WEIGHTS[k]*v for k,v in parts.items() if k not in ('role_distillation','goal_distillation'))
        rolegrad,goalgrad=torch.autograd.grad(policy,(pred['projected_roles'],pred['predicted_goal_delta']),retain_graph=True)
        for g in (rolegrad,goalgrad):self.assertTrue(torch.isfinite(g).all());self.assertGreater(float(g.abs().sum()),0.)
        policy.backward()
        for parameter in (student.hidden[0].weight,student.roles.weight,student.goal.weight):
            self.assertIsNotNone(parameter.grad);self.assertTrue(torch.isfinite(parameter.grad).all());self.assertGreater(float(parameter.grad.abs().sum()),0.)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in self.actor.parameters()))
        for key,value in self.actor.state_dict().items():self.assertTrue(torch.equal(value,self.original[key]),key)
        self.assertFalse(torch.cuda.is_initialized())

    def test_state_control_has_zero_foundation_gradient_and_executed10_only(self):
        roles=torch.randn(2,96,requires_grad=True);goal=torch.randn(2,18,requires_grad=True)
        result=policy_aware_forward(self.actor,projected_roles=roles,goal_delta=goal,state_action_only=True,**self.inputs(1.))
        rolegrad,goalgrad=torch.autograd.grad(result['ungated_physical_actions'].sum(),(roles,goal))
        self.assertTrue(torch.equal(rolegrad,torch.zeros_like(rolegrad)));self.assertTrue(torch.equal(goalgrad,torch.zeros_like(goalgrad)))
        target=result['parent_physical_actions'].detach().clone();valid=torch.ones(2,50,dtype=torch.bool)
        predicted=dict(projected_roles=roles,predicted_goal_delta=goal);teacher={k:v.detach() for k,v in predicted.items()}
        old,parts=objective(predicted,result,teacher,target,valid,self.fk)
        changed=target.clone();changed[:,10:,0]+=100.;changed[:,10:,6]=.049
        new,parts_new=objective(predicted,result,teacher,changed,valid,self.fk)
        self.assertTrue(torch.equal(old,new))
        for key in parts:self.assertTrue(torch.equal(parts[key],parts_new[key]))


if __name__=='__main__':unittest.main()
