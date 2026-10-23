"""Independent literal source-target audit and frozen current-only head preparation."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
import pyarrow.parquet as pq
import torch
from experiments.goal_latent_act_command_goal_training import data
from experiments.goal_latent_act_command_goal_training.contracts import load_head
from experiments.goal_latent_act_command_goal_training.train_head import predict
from experiments.goal_latent_act.train import stats_tensors
from experiments.goal_latent_act.prepare import sha,write_json
from experiments.goal_latent_act_campaign.common import ref,verify,require
from policy.wam_adaptation.multitask_data import TASK_REVISIONS,physical_commands
from experiments.goal_latent_act_full_data.correction_prefix import eef_features
from .model import map_native_grips


def pins():
    root=Path(__file__).resolve().parents[2]
    names=['experiments/goal_latent_act_contact_training/'+n for n in
           ('model.py','contracts.py','prepare.py','train.py','run.py','test_cpu.py')]
    return {n:sha(root/n) for n in names}


def audit(arrays,records,tasks):
    # Every selected Assembly episode and four fixed episode identities per other task.
    # Source-qualified physical_commands is independently compared with a literal codec.
    grouped=defaultdict(list)
    for i,r in enumerate(records):
        if r['task']=='item_assembly' or r['episode'] in (0,150,900,916):grouped[r['task'],r['episode']].append((i,r))
    rows=[];root=data.ROOT/'datasets/official-cache'
    for (task,episode),selected in sorted(grouped.items()):
        path=root/task/TASK_REVISIONS[task]/f'data/chunk-000/episode_{episode:06d}.parquet'
        digest=sha(path);raw=np.array(pq.read_table(path,columns=['action'])['action'].to_pylist(),dtype=np.float64)
        expected=raw.copy()
        expected[:,[6,13]]=np.clip(raw[:,[6,13]],0,1)*.05 if task=='drawer_open_place' else np.clip(raw[:,[6,13]],0,.05)
        if task=='click_bell':require(np.all(raw[:,[6,13]]==0),'Bell zero source contract required')
        require(np.array_equal(expected,physical_commands(task,raw)),'Literal source codec disagrees with canonical implementation')
        for i,r in selected:
            require(digest==r['parquet_sha256'],'Prepared source parquet SHA differs')
            n=min(50,len(raw)-r['frame']-1);target=np.zeros((50,14),np.float32);target[:n]=expected[r['frame']+1:r['frame']+1+n]
            require(np.array_equal(target,arrays['physical_actions'][i]) and np.array_equal(np.arange(50)<n,arrays['valid'][i]),
                    'Prepared teacher next-command or validity differs from literal source codec')
        rows.append(dict(task=task,episode=episode,source=ref(path),clips=len(selected),exact_physical_targets=True))
    return dict(schema='physical-opening-prepared-target-independent-source-audit-v1',status='literal_source_target_parity_pass',
                audited_parquets=len(rows),audited_clips=sum(r['clips'] for r in rows),
                Assembly_all_selected_episodes=True,rows=rows,canonical_source=ref(data.ROOT/'policy/wam_adaptation/multitask_data.py'),
                mapping='Drawer clamp01*.05; all other native expert commands clamp0..05; Bell source exactly0',
                claim='Desired command labels verified; no observed contact or geometry/robot quality conclusion')


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for k in ('data','data-sha256','highlevel','highlevel-sha256','protocol','protocol-sha256','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    torch.set_num_threads(8);torch.set_num_interop_threads(1);signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU contact preparation300s bound')));signal.alarm(300)
    report=dict(schema='goal-latent-act-physical-opening-data-v1',status='CPU_preparation_running',GPU_queries=0,GPU_launches=0)
    try:
        require(sha(a.protocol)==a.protocol_sha256,'Predeclared protocol changed')
        receipt,records,arrays,tasks=data.load(a.data,a.data_sha256)
        qualified=verify(receipt['runtime_unit_qualification']);q=json.loads(qualified.read_text())
        for task in q['tasks']:
            for key in ('task_source','action_bank','gym_config','parquet'):verify(task[key])
        unit=audit(arrays,records,tasks);write_json(out/'target-audit.json',unit)
        head,hc=load_head(a.highlevel,a.highlevel_sha256);require(hc['command_target_frame_offset']==50,'Frozen command50 head required')
        st=stats_tensors(hc['statistics']);n=len(records);goal=np.empty((n,2,4,4),np.float32)
        with torch.no_grad():
            for start in range(0,n,256):
                ids=np.arange(start,min(start+256,n));goal[ids]=predict(head,arrays,ids,st)['eef_goal_pose'].numpy()
        monitor=np.flatnonzero(~arrays['fit']);shuffled=monitor.copy();rng=np.random.default_rng(9917)
        for j in range(len(tasks)):
            selected=np.flatnonzero(arrays['task_index'][monitor]==j);shuffled[selected]=rng.permutation(monitor[selected])
        shuffled_goal=np.empty((len(monitor),2,4,4),np.float32)
        with torch.no_grad():
            for start in range(0,len(monitor),256):
                ids=monitor[start:start+256];roleids=shuffled[start:start+256]
                shuffled_goal[start:start+len(ids)]=predict(head,arrays,ids,st,role_ids=roleids)['eef_goal_pose'].numpy()
        gate=np.array([r['split']=='fit' and r['episode']%10==9 for r in records]);fit=arrays['fit']&~gate
        require((int(fit.sum()),int(gate.sum()),len(monitor))==(9520,1050,1190),'Predeclared whole-demonstration FIT split differs')
        base=arrays['physical_base'][fit];qpos=arrays['canonical_current_qpos14'][fit]
        stats=dict(qpos_mean=qpos.mean(0).tolist(),qpos_scale=np.maximum(qpos.std(0),.01).tolist(),
                   base_mean=base.mean((0,1)).tolist(),base_scale=np.maximum(base.std((0,1)),.01).tolist(),
                   eef_scale=hc['statistics']['eef_scale'],scope='residual FIT measured current/native predictions only; no gate or MONITOR moments',
                   roles_context='qwen35-canonical-current-role-hidden-v2',state_grips_units='normalized_opening',
                   action_features_units='native per-task ACT release space; native grips NOT globally meters')
        # Numeric label sanity precedes either optimization child.
        physical=arrays['physical_actions'][...,[6,13]];require(np.isfinite(physical).all() and physical.min()>=0 and physical.max()<=.050001,'Finite physical clamp labels required')
        parent=map_native_grips(torch.from_numpy(arrays['physical_base']),torch.from_numpy(arrays['task_index']),tasks).numpy()
        per={}
        for j,t in enumerate(tasks):
            per[t]={}
            for name,split in [('FIT',arrays['fit']),('MONITOR',~arrays['fit'])]:
                ids=np.flatnonzero(split&(arrays['task_index']==j));mask=arrays['valid'][ids,...,None]
                err=np.abs(parent[ids]-physical[ids]);per[t][name]=dict(rows=len(ids),valid_commands=int(mask.sum()),
                    physical_grip_MAE_m=((err*mask).sum((0,1))/mask.sum()).tolist(),
                    target_closed_fraction=((physical[ids]<=.005)*mask).sum((0,1)).tolist(),
                    target_open_fraction=((physical[ids]>=.045)*mask).sum((0,1)).tolist())
        np.savez(out/'features.npz',predicted_goal=goal,monitor_ids=monitor,shuffled_role_ids=shuffled,shuffled_predicted_goal=shuffled_goal,residual_fit=fit,gate_fit=gate)
        write_json(out/'statistics.json',stats)
        report.update(status='CPU_source_labels_and_current_only_head_verified',source_data=ref(Path(a.data)/'receipt.json'),
            protocol=ref(a.protocol),highlevel_weights=ref(Path(a.highlevel)/'highlevel.safetensors'),highlevel_contract=ref(Path(a.highlevel)/'checkpoint.json'),
            target_audit=ref(out/'target-audit.json'),source_pins=pins(),tasks=tasks,statistics=stats,
            residual_FIT=9520,reliability_FIT=1050,MONITOR=1190,per_task_parent_diagnostic=per,
            artifacts={k:ref(out/k) for k in ('features.npz','statistics.json')},CUDA_initialized=torch.cuda.is_initialized())
        require(not report['CUDA_initialized'],'CPU only');print(json.dumps(dict(status=report['status'],audit_parquets=unit['audited_parquets'],audit_clips=unit['audited_clips'])),flush=True)
    except BaseException as e:report.update(status='failed_preserved_no_retry',failure=dict(type=type(e).__name__,message=str(e)));raise
    finally:signal.alarm(0);report['elapsed_seconds']=time.monotonic()-started;write_json(out/'receipt.json',report)


if __name__=='__main__':main()
