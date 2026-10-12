"""Strict accepted-cache preparation; terminal labels are training-only."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import signal
import numpy as np
import torch
from experiments.goal_latent_act_full_data import aggregate
from experiments.goal_latent_act_campaign.common import ref,verify,require,write
from experiments.goal_latent_act.prepare import sha
from .contracts import DATA

SNAPSHOT_SHA='594b42f94a3003b1bfad72d95ba7faf94e6850ea3be9e42c2a85b9bd8797dcf1'
ROOT=Path(__file__).resolve().parents[2]


def pins():
 names=['experiments/goal_latent_act_command_goal_training/contracts.py','experiments/goal_latent_act_command_goal_training/data.py',
 'experiments/goal_latent_act_command_goal_training/train_head.py','experiments/goal_latent_act_command_goal_training/train_motor.py',
 'experiments/goal_latent_act/model.py','experiments/goal_latent_act/train.py','experiments/goal_latent_act/train_correction.py',
 'experiments/goal_latent_act_full_data/aggregate.py','experiments/goal_latent_act_full_data/campaign.py',
 'experiments/goal_latent_act_full_data/produce.py','experiments/goal_latent_act_full_data/correction_prefix.py',
 'experiments/goal_latent_act_full_data/train_expanding.py','policy/object_relative_hierarchy/multitask_model.py']
 return {name:sha(ROOT/name) for name in names}


def pose_features(poses):
 return np.concatenate((poses[...,:3,3],poses[...,:3,:2].swapaxes(-1,-2).reshape(len(poses),2,6)),-1).reshape(len(poses),18)


def prepare(snapshot,expected,geometry,urdf,output):
 output=Path(output);require(not output.exists(),'Fresh prepared data root required')
 require(expected==SNAPSHOT_SHA,'Exact accepted10k snapshot required')
 source,records,arrays,tasks=aggregate.load(snapshot,expected)
 spec=importlib.util.spec_from_file_location('command_goal_qualified_geometry',geometry);g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
 chain=g.read_chain(urdf);tcp=g.tcp();n=len(records);valid=arrays['valid'][:,49]
 require((int(arrays['fit'].sum()),int((~arrays['fit']).sum()))==(10570,1190),'Exact actual source counts required')
 require((int((valid&arrays['fit']).sum()),int((valid&~arrays['fit']).sum()))==(8758,986),'Audited same terminal-valid head scope required')
 max_current=0.;goals={h:np.empty((n,2,4,4),np.float32) for h in (10,50)}
 for i in range(n):
  q=arrays['canonical_current_qpos14'][i]
  current=np.stack([g.fk_link(chain,q[:6],'left')@tcp,g.fk_link(chain,q[7:13],'right')@tcp])
  max_current=max(max_current,float(np.abs(current-arrays['teacher_eef_now'][i]).max()))
  for h in (10,50):
   if not valid[i]:goals[h][i]=arrays['teacher_eef_now'][i];continue
   command=arrays['physical_actions'][i,h-1]
   goals[h][i]=np.stack([g.fk_link(chain,command[:6],'left')@tcp,g.fk_link(chain,command[7:13],'right')@tcp])
 require(max_current<2e-6,'Qualified current FK frame parity differs')
 fit=valid&arrays['fit'];q=arrays['canonical_current_qpos14'][fit];eef=pose_features(arrays['teacher_eef_now'][fit])
 stats=dict(schema='command-goal-terminal-valid-FIT-current-zscore-v1',qpos_mean=q.mean(0).tolist(),qpos_scale=np.maximum(q.std(0),.01).tolist(),
  eef_mean=eef.mean(0).tolist(),eef_scale=np.maximum(eef.std(0),.01).tolist(),rows=int(fit.sum()),
  state_units='arm radians; measured grippers normalized opening; arm-major14',eef_units='robot_root_meters and rigid rotation',
  coordinate_frame='robot_root_meters',normalizer_labels_scope='terminal-valid FIT measured-current features only',historical_collector_lineage_certified=False)
 output.mkdir(parents=True);payload={k:arrays[k] for k in ('physical_base','physical_actions','valid','canonical_current_qpos14','role_features','teacher_eef_now','fit','task_index')}
 payload.update(head_valid=valid,command_goal10=goals[10],command_goal50=goals[50]);np.savez(output/'arrays.npz',**payload)
 write(output/'rows.json',records);write(output/'statistics.json',stats)
 counts={t:{s:int(sum(r['task']==t and r['split']==s and valid[i] for i,r in enumerate(records))) for s in ('fit','monitor')} for t in tasks}
 receipt=dict(schema=DATA,status='prepared_accepted_exact_command_FK_labels',snapshot=ref(snapshot),geometry=ref(geometry),urdf=ref(urdf),
  source_pins=pins(),tasks=tasks,source_FIT=10570,source_MONITOR=1190,head_FIT=8758,head_MONITOR=986,
  terminal_invalid_FIT=1812,terminal_invalid_MONITOR=204,head_per_task=counts,statistics=stats,
  normalizers=source['normalizers'],unit_clarification=source['unit_clarification'],runtime_unit_qualification=source['runtime_unit_qualification'],
  coordinate_frame='robot_root_meters',target_type='commanded FK; NOT measured future EEF',
  frame_offsets={'command_goal10':10,'command_goal50':50},maximum_current_FK_teacher_matrix_difference=max_current,
  CUDA_initialized=torch.cuda.is_initialized(),GPU_queries=0,GPU_launches=0,
  artifacts={name:ref(output/name) for name in ('arrays.npz','rows.json','statistics.json')})
 require(not receipt['CUDA_initialized'],'CPU preparation required');write(output/'receipt.json',receipt);return receipt


def load(root,expected):
 root=Path(root);require(sha(root/'receipt.json')==expected,'Exact prepared receipt required')
 receipt=json.loads((root/'receipt.json').read_text());require(receipt['schema']==DATA and receipt['status']=='prepared_accepted_exact_command_FK_labels','Exact source-qualified data required')
 for name,row in receipt['artifacts'].items():require(verify(row)==root.resolve()/name,'Prepared artifact path differs')
 for name,digest in receipt['source_pins'].items():require(sha(ROOT/name)==digest,'Frozen scientific source changed: '+name)
 for key in ('snapshot','geometry','urdf','unit_clarification','runtime_unit_qualification'):verify(receipt[key])
 records=json.loads((root/'rows.json').read_text())
 with np.load(root/'arrays.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
 require(arrays['role_features'].shape==(11760,3,4096) and arrays['role_features'].dtype==np.float16,'Exact actual current roles required')
 require(np.array_equal(arrays['head_valid'],arrays['valid'][:,49]),'Terminal target mask differs')
 require(all((r['split']=='fit' and 0<=r['episode']<900) or (r['split']=='monitor' and 900<=r['episode']<1000) for r in records),'Independent data namespaces required')
 return receipt,records,arrays,receipt['tasks']


def main():
 p=argparse.ArgumentParser(allow_abbrev=False)
 for k in ('snapshot','snapshot-sha256','geometry','urdf','output'):p.add_argument('--'+k,required=True)
 a=p.parse_args();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU preparation300s cap')));signal.alarm(300)
 try:r=prepare(a.snapshot,a.snapshot_sha256,a.geometry,a.urdf,a.output);print(json.dumps(dict(status=r['status'],head_FIT=r['head_FIT'],head_MONITOR=r['head_MONITOR'])))
 finally:signal.alarm(0)


if __name__=='__main__':main()
