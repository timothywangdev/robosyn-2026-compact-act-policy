import os
if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Hide CUDA before CPU policy objective fixtures')
import unittest
import torch
from .policy_aware_objective import objective as base
from .policy_aware_objective_v2 import objective,WEIGHTS


def fixture_fk(arms):
    b,t,_=arms.shape;pose=torch.eye(4).reshape(1,1,1,4,4).repeat(b,t,2,1,1)
    pose[...,:3,3]=arms.reshape(b,t,2,6)[...,:3]*.1
    return pose


class Tests(unittest.TestCase):
    def test_gated_actual_is_primary_and_rejected_proposal_retains_tenth_gradient(self):
        ungated=torch.full((1,50,14),.02,requires_grad=True);parent=torch.zeros_like(ungated)
        accepted=torch.tensor([[True,False]])
        actual=torch.where(accepted[:,None,:,None],ungated.reshape(1,50,2,7),parent.reshape(1,50,2,7)).reshape(1,50,14)
        result=dict(ungated_physical_actions=ungated,physical_actions=actual)
        target=torch.zeros_like(ungated);valid=torch.zeros(1,50,dtype=torch.bool);valid[:,:10]=True
        student=dict(projected_roles=torch.zeros(1,96),predicted_goal_delta=torch.zeros(1,18));teacher=student
        value,parts=objective(student,result,teacher,target,valid,fixture_fk)
        _,old=base(student,result,teacher,target,valid,fixture_fk)
        self.assertTrue(torch.equal(value,sum(WEIGHTS[k]*parts[k] for k in parts)))
        self.assertEqual(float(parts['gated_arm']),float(old['arm'])/2)
        self.assertEqual(float(parts['gated_opening']),float(old['opening'])/2)
        value.backward();self.assertGreater(float(ungated.grad[:,:10,:7].abs().sum()),0);self.assertGreater(float(ungated.grad[:,:10,7:].abs().sum()),0)
        self.assertTrue(torch.equal(ungated.grad[:,10:],torch.zeros_like(ungated.grad[:,10:])))
        self.assertEqual(WEIGHTS['gated_opening'],1.);self.assertEqual(WEIGHTS['proposal_opening'],.1)
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
