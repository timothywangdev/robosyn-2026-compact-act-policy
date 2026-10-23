"""Frozen full63000 labels, current-only head, and optimization-FIT moments."""
import argparse
import json
from pathlib import Path
import signal
import time
import numpy as np
import torch
from experiments.goal_latent_act_command_goal_training_full import data
from experiments.goal_latent_act_command_goal_training_full.contracts import load_head, HEAD
from experiments.goal_latent_act_command_goal_training_full.train_head import predict
from experiments.goal_latent_act_contact_training.prepare import audit
from experiments.goal_latent_act_full_data.correction_prefix import eef_features
from experiments.goal_latent_act_full_data.run_training_pair import identity, group_members
from experiments.goal_latent_act.train import stats_tensors
from experiments.goal_latent_act.prepare import sha, write_json
from experiments.goal_latent_act_campaign.common import ref, verify, require
from .model import hazard_targets

DATA_SCHEMA='goal-latent-act-sequence-opening-full63000-FIT-preparation-v1'
ROOT=Path(__file__).resolve().parents[2]
COUNTS=(50400,6300,6300,7000)


def pins():
    return data.pins()


def split(records):
    optimizer,diagnostic,gate,monitor=data.split(records,sorted({r['task'] for r in records}))
    return optimizer,gate,diagnostic,monitor


def verify_original_pilot(path,expected):
    require(sha(path)==expected,'Exact captured full-data head original CPU steps required')
    original=json.loads(Path(path).read_text())
    require(original['schema']=='goal-latent-act-commanded50-full63000-original-CPU-pilot-v1' and
            original['status']=='head_completed_original0_natural_CPU' and
            [r['name'] for r in original['runs']]==['prepared','head50'],
            'Original full source preparation and head50 wait receipts required')
    for row in original['runs']:
        require(row['original_exit_code']==0 and row['original_child_reaped'] is True and
                row['natural_completion_no_signals_sent'] is True and row['signals_sent']==[] and
                row['independently_observed_group_after_reap']==[],'Original natural child0/drain required')
        current=identity(row['identity']['pid'])
        require((current is None or current['birth']!=row['identity']['birth']) and
                not group_members(row['identity']['group']),'Original exact CPU owner/group still live')
        verify(row['child_receipt'])
    return original


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for k in ('data','data-sha256','highlevel','highlevel-sha256','original-pilot','original-pilot-sha256',
              'protocol','protocol-sha256','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Full63000 preparation900s cap')));signal.alarm(900)
    report=dict(schema=DATA_SCHEMA,status='CPU_preparation_running',GPU_queries=0,GPU_launches=0)
    try:
        require(sha(a.protocol)==a.protocol_sha256,'Pre-optimization protocol changed')
        original=verify_original_pilot(a.original_pilot,a.original_pilot_sha256)
        receipt,records,arrays,tasks=data.load(a.data,a.data_sha256)
        require(original['runs'][0]['child_receipt']==ref(Path(a.data)/'receipt.json'),
                'Completed CPU source and prepared full63000 data differ')
        fit,gate,diagnostic,monitor=split(records)
        require(np.array_equal(arrays['fit'],fit|gate|diagnostic),'FIT source mask differs')
        unit=audit(arrays,records,tasks);write_json(out/'target-audit.json',unit)
        head,hc=load_head(a.highlevel,a.highlevel_sha256)
        require(original['runs'][1]['child_receipt']==ref(Path(a.highlevel)/'receipt.json'),
                'Original waited head50 and selected new head artifact differ')
        actual_head=json.loads(verify(original['runs'][1]['child_receipt']).read_text())
        require(actual_head['status']=='completed_fixed50_epoch_commanded_head' and actual_head['checkpoint']==hc,
                'Actual completed original head must bind exact full checkpoint contract')
        require(hc['schema']==HEAD and hc['selected_epoch']==50 and hc['source_FIT']==63000 and hc['source_MONITOR']==7000 and hc['data_receipt']==ref(Path(a.data)/'receipt.json')
                and hc['context_version']=='qwen35-canonical-current-role-hidden-v2','Actual same full63000 fixed50 head binding required')
        stats=stats_tensors(hc['statistics']);n=len(records);goal=np.empty((n,2,4,4),np.float32)
        with torch.no_grad():
            for start in range(0,n,256):
                ids=np.arange(start,min(start+256,n));goal[ids]=predict(head,arrays,ids,stats)['eef_goal_pose'].numpy()
        shuffled=monitor.copy();rng=np.random.default_rng(9917)
        for j in range(len(tasks)):
            selected=np.flatnonzero(arrays['task_index'][monitor]==j);shuffled[selected]=rng.permutation(monitor[selected])
        altered=np.empty((len(monitor),2,4,4),np.float32)
        with torch.no_grad():
            for start in range(0,len(monitor),256):
                ids=monitor[start:start+256];roleids=shuffled[start:start+256]
                altered[start:start+len(ids)]=predict(head,arrays,ids,stats,role_ids=roleids)['eef_goal_pose'].numpy()
        physical=arrays['physical_actions'][...,[6,13]]
        require(np.isfinite(physical).all() and physical.min()>=0 and physical.max()<=.050001,
                'Literal physical target opening bounds required')
        target=torch.from_numpy(physical[fit]);valid=torch.from_numpy(arrays['valid'][fit])
        current=torch.from_numpy(arrays['canonical_current_qpos14'][fit][:,[6,13]]).clamp(0,1)
        event=hazard_targets(target,current,valid);mask=valid[...,None,None].expand_as(event)
        positive=(event*mask).sum((0,1));negative=mask.sum((0,1))-positive
        weights=torch.minimum(torch.full_like(positive,8.),((negative+1)/(positive+1)).sqrt())
        base=arrays['physical_base'][fit];q=arrays['canonical_current_qpos14'][fit]
        eef=eef_features(torch.from_numpy(arrays['teacher_eef_now'][fit])).numpy()
        moments=dict(qpos_mean=q.mean(0).tolist(),qpos_scale=np.maximum(q.std(0),.01).tolist(),
            base_mean=base.mean((0,1)).tolist(),base_scale=np.maximum(base.std((0,1)),.01).tolist(),
            current_eef_mean=eef.mean(0).tolist(),current_eef_scale=np.maximum(eef.std(0),.01).tolist(),
            eef_scale=hc['statistics']['eef_scale'],scope='50400 sequence-optimization FIT current/native moments only',
            hazard_positive_weights=weights.tolist(),hazard_positive_counts=positive.long().tolist(),
            hazard_negative_counts=negative.long().tolist(),hazard_weights_scope='50400 optimization FIT only; frozen BEFORE optimizer child',
            action_features_units='native ACT task-specific release convention; current state grip normalized opening',
            coordinate_frame='robot_root_meters')
        np.savez(out/'features.npz',predicted_goal=goal,monitor_ids=monitor,shuffled_role_ids=shuffled,
            shuffled_predicted_goal=altered,residual_fit=fit,gate_fit=gate,diagnostic_fit=diagnostic)
        write_json(out/'statistics.json',moments)
        report.update(status='CPU_source_labels_and_current_only_head_verified',source_data=ref(Path(a.data)/'receipt.json'),
            original_source_CPU_pilot=ref(a.original_pilot),target_audit=ref(out/'target-audit.json'),
            highlevel_contract=ref(Path(a.highlevel)/'checkpoint.json'),highlevel_weights=ref(Path(a.highlevel)/'highlevel.safetensors'),
            protocol=ref(a.protocol),statistics=moments,tasks=tasks,sequence_FIT=50400,reliability_FIT=6300,
            sequence_diagnostic_FIT=6300,MONITOR=7000,source_pins=pins(),
            artifacts={name:ref(out/name) for name in ('features.npz','statistics.json')},CUDA_initialized=torch.cuda.is_initialized())
        require(not report['CUDA_initialized'],'CPU-only preparation required')
        print(json.dumps(dict(status=report['status'],FIT=50400,gate=6300,diagnostic=6300,MONITOR=7000,
                             audited_parquets=unit['audited_parquets'],audited_clips=unit['audited_clips'])),flush=True)
    except BaseException as error:
        report.update(status='failed_preserved_no_retry',failure=dict(type=type(error).__name__,message=str(error)));raise
    finally:
        signal.alarm(0);report['elapsed_seconds']=time.monotonic()-started;write_json(out/'receipt.json',report)


if __name__=='__main__':main()
