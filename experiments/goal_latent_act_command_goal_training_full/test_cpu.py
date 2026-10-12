"""CPU numeric, independent-split, corpus and original-child regressions."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from experiments.goal_latent_act_campaign.common import ref
from experiments.goal_latent_act.model import RolePhysicalGoal
from . import data,contracts,run

def records():
    return [dict(task=t,revision=v,episode=e,frame=a,split='fit' if e<900 else 'monitor')
            for t,v in data.aggregate.p.TASK_REVISIONS.items() for e in range(1000) for a in range(7)]

class CPU(unittest.TestCase):
    def test_exact_full_episode_partitions_and_duplicate_rejection(self):
        rows=records();a,b,c,m=data.split(rows,list(data.aggregate.p.TASK_REVISIONS))
        self.assertEqual((int(a.sum()),int(b.sum()),int(c.sum()),len(m)),data.COUNTS)
        self.assertFalse(bool((a&b|a&c|b&c).any()));self.assertFalse(bool((a|b|c)[m].any()))
        for task in data.aggregate.p.TASK_REVISIONS:
            selected=np.array([r['task']==task for r in rows]);self.assertEqual(int((selected&a).sum()),5040)
        rows[0]['frame']=1
        with self.assertRaises(ValueError):data.split(rows,list(data.aggregate.p.TASK_REVISIONS))
        with self.assertRaises(ValueError):data.split(records()[:-1],list(data.aggregate.p.TASK_REVISIONS))

    def test_normalization_excludes_diagnostic_gate_monitor_and_invalid_tail(self):
        poses=np.tile(np.eye(4,dtype=np.float32),(6,2,1,1));q=np.arange(84,dtype=np.float32).reshape(6,14)
        arrays=dict(canonical_current_qpos14=q,teacher_eef_now=poses)
        optimizer=np.array([True,True,True,False,False,False]);valid=np.array([True,True,False,True,True,True])
        before=data.optimization_statistics(arrays,optimizer,valid)
        altered=dict(canonical_current_qpos14=q.copy(),teacher_eef_now=poses.copy())
        altered['canonical_current_qpos14'][2:]=1e6;altered['teacher_eef_now'][2:]=1e6
        self.assertEqual(before,data.optimization_statistics(altered,optimizer,valid));self.assertEqual(before['rows'],2)

    def test_command_head_finite_gradients_current_only_abi(self):
        torch.manual_seed(1701);m=RolePhysicalGoal(eef_translation_m=.60,eef_rotation_rad=2.)
        pose=torch.eye(4).repeat(3,2,1,1);q=torch.zeros(3,14)
        st=dict(qpos_mean=torch.zeros(14),qpos_scale=torch.ones(14),eef_mean=torch.zeros(18),eef_scale=torch.ones(18))
        pred=m(roles=torch.randn(3,3,4096),qpos=q,eef_now=pose,**st)
        target=pose.clone();target[...,0,3]=.1
        from .train_head import loss
        value=loss(pred,target);value.backward();self.assertTrue(torch.isfinite(value))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None),0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_legacy_head_contract_and_partial_counts_rejected(self):
        with self.assertRaises(ValueError):contracts.validate_head({'schema':'goal-latent-act-commanded50-accepted30k-head-v1','selection':contracts.SELECTION})
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);campaign=root/'campaign.json';wait=root/'wait.json'
            campaign.write_text(json.dumps(dict(status='running',FIT=63000,MONITOR=7000)));wait.write_text('{}')
            with self.assertRaises(ValueError):data.verify_terminal(campaign,data.sha(campaign),wait,data.sha(wait))

    def test_outer_wait_unknown_not_fabricated_zero_and_live_owner_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);index=p/'index.json';index.write_text('[]');cp=p/'campaign.json';wp=p/'wait.json'
            c=dict(status='all63000_FIT_7000_MONITOR_frozen_ACT_records_published',coverage_exact=True,FIT=63000,MONITOR=7000,
                   controller_identity=dict(pid=155029,birth_ticks=54941326,group=155029,session=155029),index=ref(index),sources={})
            cp.write_text(json.dumps(c));w=dict(schema='goal-latent-act-full-corpus-admission-v1',campaign=ref(cp),identity=dict(pid=155029,birth=54941326,group=155029,session=155029),
                mode='complete_corpus_from_original_qualified_shards_outer_wait_unknown',coordinator_original_outer_exit_code=None,
                coordinator_original_outer_wait_captured=False,corpus_index=c['index'],source_counts={'FIT':63000,'MONITOR':7000})
            wp.write_text(json.dumps(w))
            with patch.object(data,'qualify_corpus') as qualified,patch.object(data,'identity',return_value=None),patch.object(data,'group_members',return_value=[]):
                result=data.verify_terminal(cp,data.sha(cp),wp,data.sha(wp));self.assertEqual(result,c);qualified.assert_called_once()
            w['coordinator_original_outer_exit_code']=0;wp.write_text(json.dumps(w))
            with self.assertRaises(ValueError):data.verify_terminal(cp,data.sha(cp),wp,data.sha(wp))
            w['coordinator_original_outer_exit_code']=None;wp.write_text(json.dumps(w))
            with patch.object(data,'qualify_corpus'),patch.object(data,'identity',return_value={'birth':54941326}),patch.object(data,'group_members',return_value=[]):
                with self.assertRaises(ValueError):data.verify_terminal(cp,data.sha(cp),wp,data.sha(wp))

    def test_missing_publication_corpus_cannot_qualify(self):
        with patch.object(data.corpus,'seed',return_value=({},[])):
            with self.assertRaises(ValueError):data.qualify_corpus(dict(seed={},completed_shards=[]))

    def test_request_emitter_refuses_live_partial_corpus(self):
        from . import request
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);campaign=p/'campaign.json';proof=p/'proof.json'
            campaign.write_text(json.dumps(dict(status='running',FIT=63000,MONITOR=7000)));proof.write_text('{}')
            args=['request','--snapshot','not-created','--campaign',str(campaign),'--corpus-admission',str(proof),
                  '--protocol','not-created','--source-seal','not-created','--geometry','not-created','--urdf','not-created',
                  '--python',sys.executable,'--training-output',str(p/'training'),'--output',str(p/'request.json')]
            with patch.object(sys,'argv',args):
                with self.assertRaises(ValueError):request.main()
            self.assertFalse((p/'request.json').exists());self.assertFalse((p/'training').exists())

    def test_real_cpu_original_exit_failure_and_success_are_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);report={'runs':[]}
            with self.assertRaises(ValueError):run.run_cpu_child([sys.executable,'-c','raise SystemExit(7)'],out,'failed',report,5)
            self.assertEqual(report['runs'][0]['original_exit_code'],7);self.assertTrue(report['runs'][0]['original_child_reaped'])
            self.assertEqual(report['runs'][0]['signals_sent'],[]);self.assertEqual(report['runs'][0]['independently_observed_group_after_reap'],[])
            dest=out/'success';dest.mkdir();(dest/'receipt.json').write_text('{"actual_test_child":true}')
            r=run.run_cpu_child([sys.executable,'-c','raise SystemExit(0)'],out,'success',report,5)
            self.assertEqual(r['original_exit_code'],0);self.assertTrue(r['natural_completion_no_signals_sent'])

    def test_real_cpu_timeout_drains_only_owned_child(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);report={'runs':[]}
            import subprocess
            with self.assertRaises(subprocess.TimeoutExpired):run.run_cpu_child([sys.executable,'-c','import time;time.sleep(5)'],out,'timeout',report,.05)
            row=report['runs'][0];self.assertNotEqual(row['original_exit_code'],0)
            self.assertTrue(row['original_child_reaped']);self.assertTrue(row['original_process_absent'])
            self.assertEqual(row['independently_observed_group_after_reap'],[]);self.assertEqual(row['signals_sent'],[15])
            self.assertFalse(row['natural_completion_no_signals_sent'])

if __name__=='__main__':torch.set_num_threads(2);unittest.main()
