"""Unchanged sequence numeric semantics, full schema and control independence."""
import unittest
import torch
from experiments.goal_latent_act_sequence_contact_training_30k.test_cpu import inputs
from .model import SequencePhysicalOpening,loss
from .contracts import validate,SELECTION

class CPU(unittest.TestCase):
    def test_sequence_gradient_bounds_and_bitexact_native_arms(self):
        m=SequencePhysicalOpening();x=inputs();pred=m(**x,gated=False)
        target=torch.full((3,50,2),.05);target[:,20:]=0;valid=torch.ones(3,50,dtype=torch.bool)
        value=loss(pred,target,x['measured_opening'],valid,torch.ones(2,2));value.backward()
        self.assertTrue(torch.isfinite(value));self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None),0)
        arm=[0,1,2,3,4,5,7,8,9,10,11,12];self.assertTrue(torch.equal(pred['physical_actions'][...,arm],x['native_base'][...,arm]))
        self.assertTrue(bool(((pred['grips_m']>=0)&(pred['grips_m']<=.05)).all()));self.assertFalse(torch.cuda.is_initialized())

    def test_control_excludes_roles_and_role_derived_goal(self):
        m=SequencePhysicalOpening();x=inputs();y=dict(x,roles=x['roles']*17,goal_delta=x['goal_delta']*-8)
        with torch.no_grad():a=m(**x,state_action_only=True,gated=False);b=m(**y,state_action_only=True,gated=False)
        self.assertTrue(all(torch.equal(a[k],b[k]) for k in a))

    def test_old30k_contract_and_mismatched_full_budget_rejected(self):
        with self.assertRaises(ValueError):validate({'schema':'goal-latent-act-current-role-sequence-opening50-accepted30k-v1','selection':SELECTION})
        with self.assertRaises(ValueError):
            m=SequencePhysicalOpening();x=inputs();x['parent_grips_m'][0,0,0]=.6;m(**x)

if __name__=='__main__':torch.set_num_threads(2);unittest.main()
