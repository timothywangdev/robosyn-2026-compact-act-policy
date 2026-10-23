"""Matched final50-epoch command-FK EEF heads; MONITOR reporting only."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
import torch
from safetensors.torch import save_file
from experiments.goal_latent_act.model import RolePhysicalGoal,goal_metrics
from experiments.goal_latent_act.train import stats_tensors,birth
from experiments.goal_latent_act.prepare import sha,write_json
from experiments.goal_latent_act_campaign.common import ref,require
from . import data
from .contracts import HEAD,SELECTION,load_head


def loss(prediction,target):
 return (torch.nn.functional.smooth_l1_loss(prediction['eef_goal_pose'][...,:3,3]/.02,target[...,:3,3]/.02)
         +(prediction['eef_goal_pose'][...,:3,:3]-target[...,:3,:3]).square().mean())


def predict(model,arrays,ids,stats,*,role_ids=None):
 return model(roles=torch.from_numpy(arrays['role_features'][ids if role_ids is None else role_ids].astype(np.float32)),
  qpos=torch.from_numpy(arrays['canonical_current_qpos14'][ids]),eef_now=torch.from_numpy(arrays['teacher_eef_now'][ids]),**stats)


@torch.no_grad()
def evaluate(model,arrays,ids,stats,tasks,horizon,intervention=None):
 model.eval();sums=Counter();counts=Counter();per={t:Counter() for t in tasks};shuffled=ids.copy()
 if intervention=='within_task_role_shuffle':
  rng=np.random.default_rng(9917)
  for j in range(len(tasks)):
   selected=np.flatnonzero(arrays['task_index'][ids]==j);shuffled[selected]=rng.permutation(ids[selected])
 for start in range(0,len(ids),256):
  current=ids[start:start+256];role_ids=shuffled[start:start+256] if intervention else None
  pred=predict(model,arrays,current,stats,role_ids=role_ids);target=torch.from_numpy(arrays['command_goal'+str(horizon)][current])
  values=goal_metrics(pred,{'eef_future':target})
  values.update({'hold_'+k:v for k,v in goal_metrics({'eef_goal_pose':torch.from_numpy(arrays['teacher_eef_now'][current])},{'eef_future':target}).items()})
  for name,value in values.items():
   sums[name]+=float(value.sum())
   for j,task in enumerate(tasks):
    mask=torch.from_numpy(arrays['task_index'][current]==j);per[task][name]+=float(value[mask].sum())
  for j,task in enumerate(tasks):counts[task]+=int((arrays['task_index'][current]==j).sum())
 return dict(rows=len(ids),metrics={k:v/len(ids) for k,v in sums.items()},
  per_task={t:dict(rows=counts[t],metrics={k:v/counts[t] for k,v in per[t].items()}) for t in tasks},intervention=intervention,
  interpretation='command-derived desired pose error; NOT measured EEF tracking or robot success')


def main():
 p=argparse.ArgumentParser(allow_abbrev=False)
 for k in ('data','data-sha256','output'):p.add_argument('--'+k,required=True)
 p.add_argument('--horizon',type=int,choices=(10,50),required=True);a=p.parse_args()
 output=Path(a.output);output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(8);torch.set_num_interop_threads(1)
 torch.manual_seed(1701);np.random.seed(1701);started=time.monotonic()
 report=dict(schema=HEAD,status='actual_CPU_head_started',pid=os.getpid(),birth=birth(os.getpid()),
  command_target_frame_offset=a.horizon,epochs_requested=50,batch_size=256,seed=1701,optimizer_updates=0,examples_seen=0,
  GPU_queries=0,GPU_launches=0,selection=SELECTION)
 def expired(*_):raise TimeoutError('CPU commanded head1800s bound')
 signal.signal(signal.SIGALRM,expired);signal.alarm(1800)
 try:
  receipt,records,arrays,tasks=data.load(a.data,a.data_sha256)
  fit=np.flatnonzero(arrays['fit']&arrays['head_valid']);monitor=np.flatnonzero(~arrays['fit']&arrays['head_valid'])
  require((len(fit),len(monitor))==(8758,986),'Exact matched terminal-valid head scope required')
  stats=receipt['statistics'];st=stats_tensors(stats);model=RolePhysicalGoal(eef_translation_m=.60,eef_rotation_rad=2.)
  for module in (model.geometry,model.carrier,model.object_goal):module.requires_grad_(False)
  parameters=[p for p in model.parameters() if p.requires_grad]
  optimizer=torch.optim.AdamW(parameters,lr=3e-4,weight_decay=.001);rng=np.random.default_rng(1701)
  report.update(source_pins=data.pins(),data_receipt=ref(Path(a.data)/'receipt.json'),FIT_rows=len(fit),MONITOR_rows=len(monitor),
   FIT_demonstrations=len({(records[i]['task'],records[i]['episode']) for i in fit}),
   MONITOR_demonstrations=len({(records[i]['task'],records[i]['episode']) for i in monitor}),head_per_task=receipt['head_per_task'],
   initial_monitor=evaluate(model,arrays,monitor,st,tasks,a.horizon),trainable_parameters=sum(p.numel() for p in parameters),
   trained_outputs=['dual_eef_goal_pose'],untrained_outputs=['object_geometry','object_goal','carrier'],statistics=stats)
  for epoch in range(50):
   model.train();order=rng.permutation(fit);total=0.
   for start in range(0,len(order),256):
    ids=order[start:start+256];target=torch.from_numpy(arrays['command_goal'+str(a.horizon)][ids]);optimizer.zero_grad(set_to_none=True)
    pred=predict(model,arrays,ids,st);value=loss(pred,target)
    require(bool(torch.isfinite(value)),'Finite command-FK EEF loss required');value.backward()
    require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters),'Finite head gradients required')
    torch.nn.utils.clip_grad_norm_(parameters,5.);optimizer.step();report['optimizer_updates']+=1;report['examples_seen']+=len(ids)
    total+=float(value.detach())*len(ids)
   report.setdefault('history',[]).append(dict(epoch=epoch+1,FIT_loss=total/len(fit),FIT_order_sha256=hashlib.sha256(order.tobytes()).hexdigest()))
   report.update(epochs_completed=epoch+1,elapsed_seconds=time.monotonic()-started);write_json(output/'receipt.json',report)
  require(report['optimizer_updates']==1750 and report['examples_seen']==437900,'Exact fixed head training budget differs')
  weights=output/'highlevel.safetensors';save_file({k:v.detach().contiguous() for k,v in model.state_dict().items()},str(weights))
  with weights.open('rb') as stream:os.fsync(stream.fileno())
  contract=dict(schema=HEAD,model_config=model.config,weights_sha256=sha(weights),statistics=stats,context_version='qwen35-canonical-current-role-hidden-v2',
   feature_shape=[3,4096],current_qpos_shape=[14],current_eef_shape=[2,4,4],output_dual_eef_goal_shape=[2,4,4],
   command_target_index=a.horizon-1,command_target_frame_offset=a.horizon,target_semantics='qualified_FK_of_recorded_arm_joint_command_NOT_measured_future_EEF',
   coordinate_frame='robot_root_meters',eef_units='robot_root_meters and rigid rotation',use_future_labels_at_inference=False,
   actor_inputs=['current canonical Qwen roles','current canonical qpos','measured-current FK'],trained_outputs=['dual_eef_goal_pose'],
   untrained_outputs=['object_geometry','object_goal','carrier'],selection=SELECTION,selected_epoch=50,optimizer_updates=1750,examples_seen=437900,
   FIT_rows=8758,MONITOR_rows=986,terminal_invalid_FIT=1812,terminal_invalid_MONITOR=204,seed=1701,batch_size=256,
   data_receipt=report['data_receipt'],geometry_sha256=receipt['geometry']['sha256'],urdf_sha256=receipt['urdf']['sha256'],source_pins=report['source_pins'],
   historical_collector_lineage_certified=False,bounds_selection='predeclared FIT-only audited coverage plus rounded margin; no MONITOR choice')
  write_json(output/'checkpoint.json',contract);reloaded,_=load_head(output,sha(weights));ids=monitor[:32]
  with torch.no_grad():
   before=predict(model,arrays,ids,st);after=predict(reloaded,arrays,ids,st)
  require(all(torch.equal(before[k],after[k]) for k in before),'Saved/reloaded head predictions changed')
  report.update(status='completed_fixed50_epoch_commanded_head',checkpoint=contract,reload_prediction_bitexact=True,
   final_monitor=evaluate(reloaded,arrays,monitor,st,tasks,a.horizon),
   within_task_role_shuffle_monitor=evaluate(reloaded,arrays,monitor,st,tasks,a.horizon,'within_task_role_shuffle'),
   CUDA_initialized=torch.cuda.is_initialized(),robot_success_claim=False)
  require(not report['CUDA_initialized'],'Actual CPU-only training required')
  print(json.dumps(dict(status=report['status'],horizon=a.horizon,updates=1750,MONITOR=report['final_monitor']['metrics'],elapsed_seconds=time.monotonic()-started)),flush=True)
 except BaseException as error:report.update(status='failed_original_preserved_no_retry',failure=dict(type=type(error).__name__,message=str(error)));raise
 finally:signal.alarm(0);report['elapsed_seconds']=time.monotonic()-started;write_json(output/'receipt.json',report)


if __name__=='__main__':main()
