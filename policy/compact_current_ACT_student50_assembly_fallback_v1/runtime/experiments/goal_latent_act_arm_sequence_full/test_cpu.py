"""Meaningful model, physical-chain, causal-control and scope regressions."""
import copy
import ast
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from .common import ROOT, ref, validate_protocol
from .model import GoalArmSequence, ARM, GRIP
from .fk import CommandFK, geometry, pose_errors, physical_gate_labels
from .train import objective
from .launch import make_plan
from . import supervisor

BASE=ROOT/'outputs/architecture-reassessment/goal-latent-act-commanded50-full-training-preparation-v1/actual-full63000-v1/prepared'
RECEIPT=json.loads((BASE/'receipt.json').read_text())
PROTOCOL=ROOT/'outputs/architecture-reassessment/goal-latent-act-full63000-arm-sequence-v1/protocol.json'


def inputs(b=3):
    torch.manual_seed(72)
    return dict(roles=torch.randn(b,3,4096),qpos_features=torch.randn(b,14),current_eef_features=torch.randn(b,18),
                goal_delta=torch.randn(b,18),native_base_features=torch.randn(b,50,14),native_base=torch.randn(b,50,14))


class ArmSequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_initial_parent_and_native_grips(self):
        x=inputs();m=GoalArmSequence();p=m(**x)
        self.assertTrue(torch.equal(p['native_actions'],x['native_base']))
        self.assertTrue(torch.equal(p['ungated_native_actions'],x['native_base']))
        with torch.no_grad():m.output.weight.normal_()
        p=m(**x,gated=False)
        self.assertTrue(torch.equal(p['native_actions'][...,list(GRIP)],x['native_base'][...,list(GRIP)]))
        self.assertLessEqual(float(p['delta_arm_rad'].abs().max()),.15)

    def test_control_removes_roles_and_goal(self):
        x=inputs();m=GoalArmSequence()
        with torch.no_grad():m.output.weight.normal_(std=.1)
        y=copy.deepcopy(x);y['roles']*=100;y['goal_delta']+=999
        a=m(**x,state_action_only=True,gated=False);b=m(**y,state_action_only=True,gated=False)
        self.assertTrue(torch.equal(a['ungated_native_actions'],b['ungated_native_actions']))
        self.assertFalse(torch.equal(m(**x,gated=False)['native_actions'],m(**y,gated=False)['native_actions']))

    def test_handwise_rejection_preserves_parent(self):
        x=inputs();m=GoalArmSequence()
        with torch.no_grad():m.output.bias.fill_(.4);m.reliability[-1].bias.copy_(torch.tensor([10.,-10.]))
        a=m(**x)
        self.assertTrue(torch.equal(a['native_actions'][...,list(ARM[6:])],x['native_base'][...,list(ARM[6:])]))
        self.assertFalse(torch.equal(a['native_actions'][...,list(ARM[:6])],x['native_base'][...,list(ARM[:6])]))

    def test_bad_inference_shape_and_dtype(self):
        x=inputs();x['roles']=x['roles'].half()
        with self.assertRaises(ValueError):GoalArmSequence()(**x)
        x=inputs();x['native_base']=x['native_base'][:,:10]
        with self.assertRaises(ValueError):GoalArmSequence()(**x)

    def test_actual_chain_double_parity_and_gradients(self):
        g=geometry(RECEIPT['geometry']);chain=g.read_chain(RECEIPT['urdf']['path'])
        q=np.random.default_rng(72).normal(size=(3,4,12));m=CommandFK(RECEIPT['geometry'],RECEIPT['urdf'],dtype=torch.float64)
        t=torch.from_numpy(q).requires_grad_(True);p=m(t)
        target=np.stack([np.stack([np.stack([g.fk_link(chain,z[j*6:j*6+6],side)@g.tcp() for j,side in enumerate(('left','right'))]) for z in row]) for row in q])
        self.assertLess(float(np.abs(p.detach().numpy()-target).max()),1e-12)
        p[...,:3,3].square().sum().backward();self.assertTrue(torch.isfinite(t.grad).all());self.assertGreater(float(t.grad.abs().sum()),0)

    def test_FK_objective_gradients_and_invalid_tail(self):
        x=inputs();x['native_base']*=.2;m=GoalArmSequence();fk=CommandFK(RECEIPT['geometry'],RECEIPT['urdf'])
        target=x['native_base'].clone();target[...,list(ARM)]+=.025
        valid=torch.ones(3,50,dtype=torch.bool);valid[:,49]=False
        pred=m(**x,gated=False);loss=objective(pred,target,valid,fk)
        alternate=target.clone();alternate[:,49]=999
        self.assertTrue(torch.equal(loss,objective(pred,alternate,valid,fk)))
        loss.backward();self.assertTrue(torch.isfinite(m.output.weight.grad).all());self.assertGreater(float(m.output.weight.grad.abs().sum()),0)

    def test_gate_requires_physical_improvement(self):
        fk=CommandFK(RECEIPT['geometry'],RECEIPT['urdf']);p=torch.zeros(3,50,14);target=p.clone();target[...,0]=.04
        valid=torch.ones(3,50,dtype=torch.bool)
        good=physical_gate_labels(target,p,target,valid,fk)
        self.assertTrue(torch.equal(good,torch.tensor([[1.,0.]]).expand(3,-1)))
        bad=physical_gate_labels(p,p,target,valid,fk);self.assertEqual(float(bad.sum()),0.)

    def test_fixed_split_scope_and_gate_protocol(self):
        p=json.loads(PROTOCOL.read_text());validate_protocol(p)
        for key,value in [('MONITOR_selection',True),('FK_indices',[0]),('gate_epochs',19)]:
            altered=copy.deepcopy(p);altered[key]=value
            with self.assertRaises(ValueError):validate_protocol(altered)
        altered=copy.deepcopy(p);altered['partitions']['optimizer_FIT']+=7000
        with self.assertRaises(ValueError):validate_protocol(altered)

    def test_no_CUDA_initialization(self):self.assertFalse(torch.cuda.is_initialized())

    def test_supervisor_exact_oneCUDA_two_branches(self):
        with tempfile.TemporaryDirectory() as directory:
            value=make_plan(directory,'a'*64,str(ROOT/'.venv-convert/bin/python'),str(Path(directory)/'unused-output'))
            supervisor.validate_plan(value)
            bad=copy.deepcopy(value);bad['jobs'][0]['max_owned_gpu_processes']=2
            with self.assertRaises(ValueError):supervisor.validate_plan(bad)
            bad=copy.deepcopy(value);bad['admission_memory_floor_gib']=24
            with self.assertRaises(ValueError):supervisor.validate_plan(bad)
            bad=copy.deepcopy(value);bad['jobs']=bad['jobs'][:1]
            with self.assertRaises(ValueError):supervisor.validate_plan(bad)

    def test_original_supervisor_ownership_and_drain_functions_unchanged(self):
        parent=ROOT/'experiments/goal_latent_act_campaign/supervisor_v8.py'
        old={n.name:n for n in ast.parse(parent.read_text()).body if isinstance(n,ast.FunctionDef)}
        new={n.name:n for n in ast.parse(Path(supervisor.__file__).read_text()).body if isinstance(n,ast.FunctionDef)}
        self.assertEqual(set(old),set(new))
        for name in old:
            if name not in ('validate_plan','run_plan'):
                self.assertEqual(ast.dump(old[name]),ast.dump(new[name]),name)
        normalized=parent.read_text().replace('plan.get("admission_memory_floor_gib", 60) * GIB','plan.get("admission_memory_floor_gib", 32) * GIB')
        body={n.name:n for n in ast.parse(normalized).body if isinstance(n,ast.FunctionDef)}
        self.assertEqual(ast.dump(body['run_plan']),ast.dump(new['run_plan']))


if __name__=='__main__':unittest.main()
