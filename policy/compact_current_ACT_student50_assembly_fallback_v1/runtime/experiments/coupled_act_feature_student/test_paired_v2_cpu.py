import os
if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Hide CUDA before paired CPU gradient tests')
import unittest
import torch
from .test_policy_aware_cpu import Tests as OriginalFixture
from .model import Student
from .policy_aware import policy_aware_forward
from .policy_aware_objective_v2 import objective,WEIGHTS


class Tests(OriginalFixture):
    def test_actual_frozen_tail_gated_primary_and_tenth_auxiliary_reach_both_heads(self):
        torch.manual_seed(1701);visual=Student();torch.manual_seed(1701);control=Student(qpos_only=True)
        self.assertTrue(all(torch.equal(x,control.state_dict()[k]) for k,x in visual.state_dict().items()))
        image=torch.randn(2,3,512,2,3);qpos=torch.randn(2,14)
        one=control(image,qpos);two=control(image*100,qpos)
        self.assertTrue(all(torch.equal(one[k],two[k]) for k in one))
        predicted=visual(image,qpos);motor=policy_aware_forward(self.actor,projected_roles=predicted['projected_roles'],goal_delta=predicted['predicted_goal_delta'],**self.inputs(1.))
        target=motor['parent_physical_actions'].detach().clone();target[...,0]+=.02;target[...,7]-=.02;target[...,6]=.03;target[...,13]=.01
        valid=torch.zeros(2,50,dtype=torch.bool);valid[:,:10]=True
        labels={k:v.detach().clone() for k,v in predicted.items()}
        # Explicit rejection boundary fixture, with the actual pretrained tail
        # proposals kept intact. This is not a claimed actual gate outcome.
        rejected=dict(motor,physical_actions=motor['parent_physical_actions'].detach())
        total,parts=objective(predicted,rejected,labels,target,valid,self.fk)
        gated=sum(WEIGHTS[k]*v for k,v in parts.items() if k.startswith('gated_'))
        self.assertFalse(gated.requires_grad)
        policy=sum(WEIGHTS[k]*v for k,v in parts.items() if k.startswith(('gated_','proposal_')))
        for g in torch.autograd.grad(policy,(predicted['projected_roles'],predicted['predicted_goal_delta']),retain_graph=True):
            self.assertTrue(torch.isfinite(g).all());self.assertGreater(float(g.abs().sum()),0.)
        changed=target.clone();changed[:,10:,0]+=100.;changed[:,10:,6]=.049
        future,_=objective(predicted,rejected,labels,changed,valid,self.fk);self.assertTrue(torch.equal(total,future))
        # Now actual model acceptance, not a forced gate, gives the primary
        # result checked against fixed gated+auxiliary weights.
        actual,actual_parts=objective(predicted,motor,labels,target,valid,self.fk)
        self.assertTrue(torch.equal(actual,sum(WEIGHTS[k]*v for k,v in actual_parts.items())))
        actual.backward()
        for p in (visual.hidden[0].weight,visual.roles.weight,visual.goal.weight):
            self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(float(p.grad.abs().sum()),0.)
        for key,value in self.actor.state_dict().items():self.assertTrue(torch.equal(value,self.original[key]))
        self.assertTrue(all(p.grad is None for p in self.actor.parameters()));self.assertFalse(torch.cuda.is_initialized())
        self.assertEqual(WEIGHTS['proposal_arm'],.1);self.assertEqual(WEIGHTS['proposal_opening'],.1)
        self.assertEqual(WEIGHTS['proposal_FK_position'],.025);self.assertAlmostEqual(WEIGHTS['proposal_FK_rotation'],.01)


if __name__=='__main__':unittest.main()
