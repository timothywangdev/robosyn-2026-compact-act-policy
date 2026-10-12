"""Meaningful boundary, units, padding, gradients and matched-control tests."""
import unittest
import torch
from .model import CoupledGoalACT
from .units import ARM,GRIP,native_to_physical,physical_to_native
from .objective import loss,gate_labels
from experiments.goal_latent_act_arm_sequence_full.fk import CommandFK
from .common import ROOT
import json


def fixture(scale=1.,batch_size=2):
    b=batch_size;base=torch.zeros(b,50,14)
    base[...,1]=1.;base[...,2]=-.8;base[...,8]=1.;base[...,9]=-.8
    base[...,list(GRIP)]=.025/scale
    return dict(roles=torch.randn(b,3,4096),qpos_features=torch.randn(b,14),current_eef_features=torch.randn(b,18),
                goal_delta=torch.randn(b,18),native_base_features=torch.randn(b,50,14),native_base=base,
                grip_scale=torch.full((b,),scale))


class Cases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4);torch.manual_seed(1701)
        r=json.loads((ROOT/'outputs/architecture-reassessment/goal-latent-act-full63000-arm-sequence-v1/CPU-prepared-v1/receipt.json').read_text())
        cls.fk=CommandFK(r['geometry'],r['urdf'])

    def test_zero_init_gated_literal_native_all14_and_invalid_parent(self):
        for scale in (1.,.05):
            x=fixture(scale);x['native_base'][0,0,0]=-9.;x['native_base'][0,0,6]=-3.
            y=CoupledGoalACT()(**x)
            self.assertTrue(torch.equal(y['native_actions'],x['native_base']))
            self.assertFalse(bool(y['accepted'].any()))

    def test_task_native_metres_inverse_once_and_atomic_hand(self):
        for scale in (1.,.05):
            x=fixture(scale);m=CoupledGoalACT()
            with torch.no_grad():m.output.bias.fill_(.1);m.reliability[-1].bias.copy_(torch.tensor([4.,-4.]))
            y=m(**x);a=y['native_actions'].reshape(2,50,2,7);p=x['native_base'].reshape_as(a)
            self.assertTrue(torch.equal(a[:,:,1],p[:,:,1]));self.assertFalse(torch.equal(a[:,:,0],p[:,:,0]))
            expected=y['ungated_physical_actions'][:,:,6]/scale
            self.assertTrue(torch.equal(y['native_actions'][:,:,6],expected))
            self.assertTrue(bool(((y['ungated_physical_actions'][...,list(GRIP)]>=0)&(y['ungated_physical_actions'][...,list(GRIP)]<=.05)).all()))

    def test_state_control_independent_of_both_foundation_channels(self):
        x=fixture();m=CoupledGoalACT()
        with torch.no_grad():m.output.weight.normal_(0,.05);m.reliability[-1].weight.normal_(0,.05)
        y=m(**x,state_action_only=True);z=dict(x,roles=x['roles']*100+5,goal_delta=x['goal_delta']*-8+9)
        v=m(**z,state_action_only=True)
        self.assertTrue(all(torch.equal(y[k],v[k]) for k in y))
        factual=m(**x);changed=m(**z)
        self.assertFalse(torch.equal(factual['ungated_physical_actions'],changed['ungated_physical_actions']))

    def test_padding_excluded_and_real_grip_gradient(self):
        x=fixture();m=CoupledGoalACT();r=m(**x,gated=False)
        target=r['parent_physical_actions'].clone();target[...,list(GRIP)]=.04
        valid=torch.ones(2,50,dtype=torch.bool);valid[:,30:]=False
        first=loss(r,target,valid,self.fk);modified=target.clone();modified[:,30:]=99.
        second=loss(r,modified,valid,self.fk)
        self.assertTrue(torch.equal(first,second));first.backward()
        self.assertTrue(bool(torch.isfinite(m.output.weight.grad).all()))
        self.assertGreater(float(m.output.bias.grad[list(GRIP)].abs().sum()),0.)

    def test_original_arm_criteria_and_perfect_opening_not_deadlocked(self):
        x=fixture();m=CoupledGoalACT();r=m(**x,gated=False);target=r['parent_physical_actions'].clone()
        # Desired arms are .02 radians away; proposal exactly repairs them and
        # already-perfect openings stay exact. It must get positive gate labels.
        target[...,0]+=.02;target[...,7]+=.02
        r['ungated_physical_actions']=target.clone()
        label=gate_labels(r,target,torch.ones(2,50,dtype=torch.bool),self.fk)
        self.assertTrue(torch.equal(label,torch.ones(2,2)))
        r['ungated_physical_actions'][...,6]+=.0002
        label=gate_labels(r,target,torch.ones(2,50,dtype=torch.bool),self.fk)
        self.assertTrue(torch.equal(label,torch.tensor([[0.,1.],[0.,1.]])))

    def test_URDF_outside_proposal_rejects_all7_not_clipped(self):
        x=fixture();x['native_base'][...,0]=2.60;m=CoupledGoalACT()
        with torch.no_grad():m.output.bias.fill_(2.);m.reliability[-1].bias.fill_(4.)
        y=m(**x)
        self.assertFalse(bool(y['accepted'][:,0].any()))
        self.assertTrue(torch.equal(y['native_actions'][...,:7],x['native_base'][...,:7]))

    def test_actor_rejects_future_fields_and_wrong_task_codec(self):
        from .inference import correct_native50
        x=fixture(batch_size=1);pose=torch.eye(4).repeat(1,2,1,1)
        current=dict(current_roles=x['roles'],qpos=torch.zeros(1,14),goal=pose,current_eef=pose)
        stats={k:torch.ones(width) if k.endswith('scale') else torch.zeros(width) for k,width in
               [('qpos_mean',14),('qpos_scale',14),('base_mean',14),('base_scale',14),('current_eef_mean',18),('current_eef_scale',18),('eef_scale',18)]}
        with self.assertRaises(ValueError):correct_native50(CoupledGoalACT(),x['native_base'],current,stats,task='drawer_open_place',codec='physical_meters_identity_v1',state_action_only=False)
        current['episode_fraction']=torch.ones(1)
        with self.assertRaises(ValueError):correct_native50(CoupledGoalACT(),x['native_base'],current,stats,task='click_bell',codec='physical_meters_identity_v1',state_action_only=False)

    def test_no_CUDA_initialized(self):self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
