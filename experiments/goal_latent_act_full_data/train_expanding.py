"""Fixed-budget prefix50 correction/control on immutable accepted ACT shards.

No MONITOR checkpoint, gate-threshold, or task selection. The requested frozen
high-level was previously validation-selected, so full-hierarchy metrics are
reused validation diagnostics, not an untouched end-to-end test.
"""
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
from safetensors.torch import load_file
from experiments.goal_latent_act_full_data import aggregate
from experiments.goal_latent_act_full_data.correction_prefix import VERSION,GoalArmCorrectionPrefix,arm_loss,per_record_error
from experiments.goal_latent_act.train import load_checkpoint as load_highlevel,stats_tensors,birth
from experiments.goal_latent_act.train_correction import batch,tensors,predict,evaluate as original_evaluate,fk_metrics,save_checkpoint
from experiments.goal_latent_act.prepare import sha,write_json

HIGHLEVEL_SHA='d700006da32d600e3505e58d52ccff4c3bd8c09627ebe50c75cce451c5b5bade'
SELECTION='fixed_predeclared_final_epoch; MONITOR_reporting_only'

def split_fit(records,fit):
    residual=np.asarray([i for i in fit if records[i]['episode']%10!=9],np.int64)
    reliability=np.asarray([i for i in fit if records[i]['episode']%10==9],np.int64)
    if not len(residual) or len(reliability)<70 or set(residual)&set(reliability):raise ValueError('Independent FIT demonstration gate split required')
    if any(records[i]['split']!='fit' or records[i]['episode']>=900 for i in np.r_[residual,reliability]):raise ValueError('MONITOR in optimizer scope')
    return residual,reliability

@torch.no_grad()
def evaluate(model,data,ids,stats,tasks,goal_free):
    result,predictions=original_evaluate(model,data,ids,stats,tasks,goal_free)
    sums=Counter();counts=Counter();total=0.
    for start in range(0,len(ids),32):
        part=ids[start:start+32];x=batch(data,part);current=x['canonical_current_qpos14'][:,None].expand(-1,50,-1)
        hold=per_record_error(current,x['physical_actions'],x['valid'],first=50);total+=float(hold.sum())
        for j,task in enumerate(tasks):
            mask=torch.from_numpy(data['task_index'][part]==j);sums[task]+=float(hold[mask].sum());counts[task]+=int(mask.sum())
    result['metrics']['hold_arm_mae_first50_rad']=total/len(ids)
    for task in tasks:result['per_task'][task]['metrics']['hold_arm_mae_first50_rad']=sums[task]/counts[task]
    result.update(execution_prefix=model.execution_prefix,gate_summary_prefix=model.execution_prefix,gate_threshold=.5,
                  interpretation='offline physical-command error; not robot success',end_to_end_monitor_previously_used_for_highlevel_selection=True)
    return result,predictions

def load_checkpoint(output):
    output=Path(output);contract=json.loads((output/'checkpoint.json').read_text())
    if contract['schema']!=VERSION or sha(output/'correction.safetensors')!=contract['weights_sha256']:raise ValueError('Exact independently typed prefix checkpoint required')
    if contract['prefix_executed']!=contract['model_config']['execution_prefix'] or contract['discarded']!=50-contract['prefix_executed']:raise ValueError('Prefix checkpoint ABI mismatch')
    model=GoalArmCorrectionPrefix(**contract['model_config']);model.load_state_dict(load_file(str(output/'correction.safetensors')),strict=True)
    return model.eval(),contract

def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--snapshot-sha256',required=True)
    p.add_argument('--highlevel',type=Path,required=True);p.add_argument('--highlevel-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--goal-free',action='store_true');p.add_argument('--execution-prefix',type=int,choices=(10,50),default=50)
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--gate-epochs',type=int,default=20);p.add_argument('--threads',type=int,default=8)
    p.add_argument('--wall-seconds',type=int,default=1800);p.add_argument('--geometry',type=Path,required=True);p.add_argument('--urdf',type=Path,required=True)
    a=p.parse_args()
    if not 1<=a.epochs<=60 or not 1<=a.gate_epochs<=20 or not 1<=a.threads<=8 or not 1<=a.wall_seconds<=1800:raise ValueError('Bounded actual CPU optimization required')
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.manual_seed(1701)
    report=dict(schema=VERSION,status='started',pid=os.getpid(),birth=birth(os.getpid()),device='cpu',GPU_launched=False,
        optimizer_updates=0,gate_updates=0,goal_free=a.goal_free,execution_prefix=a.execution_prefix,
        residual_loss_prefix=a.execution_prefix,gate_labels_prefix=a.execution_prefix,selection_prefix=a.execution_prefix,
        checkpoint_selection=SELECTION,grippers_unchanged=True)
    started=time.monotonic();old=signal.getsignal(signal.SIGALRM)
    def expired(*_):raise TimeoutError('Expanding training CPU wall bound')
    signal.signal(signal.SIGALRM,expired);signal.alarm(a.wall_seconds)
    try:
        snapshot,records,data,tasks=aggregate.load(a.snapshot,a.snapshot_sha256)
        high,hc=load_highlevel(a.highlevel)
        if a.highlevel_sha256!=HIGHLEVEL_SHA or hc['weights_sha256']!=a.highlevel_sha256 or hc.get('role_zero_control'):raise ValueError('Requested unchanged factual full-FIT high-level required')
        high.requires_grad_(False).eval();hst=stats_tensors(hc['statistics']);goals=[]
        with torch.no_grad():
            for start in range(0,len(records),128):
                sl=slice(start,start+128);out=high(roles=torch.from_numpy(data['role_features'][sl].astype(np.float32)),
                    qpos=torch.from_numpy(data['canonical_current_qpos14'][sl]),eef_now=torch.from_numpy(data['teacher_eef_now'][sl]),**hst)
                goals.append(out['eef_goal_pose'].numpy())
        data['goal_pose']=np.concatenate(goals)
        fit=np.flatnonzero(data['fit']);monitor=np.flatnonzero(~data['fit']);residual_fit,reliability_fit=split_fit(records,fit)
        base=data['physical_base'][fit].reshape(-1,14);stats={k:hc['statistics'][k] for k in ('qpos_mean','qpos_scale','eef_scale')}
        stats.update(base_mean=base.mean(0).tolist(),base_scale=np.maximum(base.std(0),.01).tolist(),
            schema='goal-latent-ACT-expanding-FIT-current-and-native-parent-v2',fit_rows=len(fit),
            action_units='ACT native task release convention: arms radians; grippers task-specific, passed through bitexact',
            target_units='literal commands: arms radians; grippers physical meters; ONLY arm12 trained/scored',
            goal_units='arena/world meters and rigid rotation',
            feature_encoding='native ACT14 zscore +currentcanonical14 zscore +predicted goal-minus-current EEF18/FITcurrentEEFscale +chunkposition')
        st=tensors(stats);model=GoalArmCorrectionPrefix(execution_prefix=a.execution_prefix);x=batch(data,residual_fit[:32]);initial=predict(model,x,st,a.goal_free,gated=False)
        if not torch.equal(initial['physical_actions'],x['physical_base']):raise ValueError('Initial correction must equal original native parent')
        residual_episodes=sorted({records[i]['episode'] for i in residual_fit});gate_episodes=sorted({records[i]['episode'] for i in reliability_fit})
        report.update(fit_rows=len(fit),monitor_rows=len(monitor),fit_demonstrations=len({(records[i]['task'],records[i]['episode']) for i in fit}),
            monitor_demonstrations=len({(records[i]['task'],records[i]['episode']) for i in monitor}),residual_fit_rows=len(residual_fit),reliability_fit_rows=len(reliability_fit),
            residual_fit_episodes=residual_episodes,reliability_fit_episodes=gate_episodes,gate_split_rule='FIT episode%10==9; all other FIT episodes residual optimization',
            highlevel_weights_sha256=a.highlevel_sha256,snapshot_sha256=a.snapshot_sha256,statistics=stats,
            source_sha256={str(Path(__file__).resolve()):sha(__file__),str(Path(__file__).with_name('correction_prefix.py').resolve()):sha(Path(__file__).with_name('correction_prefix.py')),
                str(Path(aggregate.__file__).resolve()):sha(aggregate.__file__)},zero_initial_residual_parent_bitexact=True,
            highlevel_prior_selection='frozen original high-level selected using7k MONITOR goal error; these are reused full-hierarchy validation diagnostics')
        optimizer=torch.optim.AdamW(model.residual.parameters(),lr=3e-4,weight_decay=.001);rng=np.random.default_rng(1701)
        for epoch in range(a.epochs):
            model.train();order=rng.permutation(residual_fit);total=0.
            for start in range(0,len(order),32):
                ids=order[start:start+32];x=batch(data,ids);optimizer.zero_grad(set_to_none=True);pred=predict(model,x,st,a.goal_free,gated=False)
                loss=arm_loss(pred['physical_actions'],x['physical_actions'],x['valid'],first=a.execution_prefix)
                if not torch.isfinite(loss):raise ValueError('Nonfinite physical arm optimization')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.residual.parameters(),5.);optimizer.step();report['optimizer_updates']+=1;total+=float(loss.detach())*len(ids)
            report.setdefault('history',[]).append(dict(epoch=epoch+1,FIT_loss=total/len(residual_fit),FIT_order_sha256=hashlib.sha256(order.tobytes()).hexdigest()))
            report['elapsed_seconds']=time.monotonic()-started;write_json(a.output/'receipt.json',report)
        for parameter in model.residual.parameters():parameter.requires_grad_(False)
        gate=torch.optim.AdamW(model.reliability.parameters(),lr=1e-3,weight_decay=.001);positive=0
        for epoch in range(a.gate_epochs):
            order=rng.permutation(reliability_fit);total=0.
            for start in range(0,len(order),32):
                ids=order[start:start+32];x=batch(data,ids);gate.zero_grad(set_to_none=True);pred=predict(model,x,st,a.goal_free,gated=False)
                target=(per_record_error(pred['ungated_physical_actions'],x['physical_actions'],x['valid'],first=a.execution_prefix)+.001
                        <per_record_error(x['physical_base'],x['physical_actions'],x['valid'],first=a.execution_prefix)).detach().float()
                if epoch==0:positive+=int(target.sum())
                loss=torch.nn.functional.binary_cross_entropy_with_logits(pred['reliability_logits'],target)
                if not torch.isfinite(loss):raise ValueError('Nonfinite FIT-only gate optimization')
                loss.backward();gate.step();report['gate_updates']+=1;total+=float(loss.detach())*len(ids)
            report.setdefault('gate_history',[]).append(dict(epoch=epoch+1,FIT_BCE=total/len(reliability_fit),FIT_order_sha256=hashlib.sha256(order.tobytes()).hexdigest()))
        final,predictions=evaluate(model,data,monitor,st,tasks,a.goal_free)
        report.update(status='completed_fixed_budget_expanding_prefix_training',selected_epoch=a.epochs,selected_by=SELECTION,
            reliability_positive_fit_rows=positive,final_monitor=final,final_monitor_FK=fk_metrics(data,monitor,predictions,a.geometry,a.urdf,tasks),
            CUDA_initialized=torch.cuda.is_initialized(),robot_success_claim=False)
        contract=dict(schema=VERSION,model_config=model.config,statistics=stats,goal_free=a.goal_free,reliability_threshold=.5,
            highlevel_weights_sha256=a.highlevel_sha256,highlevel_checkpoint_sha256=sha(a.highlevel/'checkpoint.json'),highlevel_model_config=hc['model_config'],
            highlevel_statistics=hc['statistics'],highlevel_context_version=hc['context_version'],snapshot_sha256=a.snapshot_sha256,
            native_task_normalizers=snapshot['normalizers'],geometry_sha256=sha(a.geometry),urdf_sha256=sha(a.urdf),
            native_chunk_generated=50,prefix_executed=a.execution_prefix,discarded=50-a.execution_prefix,grippers_unchanged=True,
            residual_loss_prefix=a.execution_prefix,gate_labels_prefix=a.execution_prefix,gate_summary_prefix=a.execution_prefix,
            selected_epoch=a.epochs,selection=SELECTION,optimizer_updates=report['optimizer_updates'],gate_updates=report['gate_updates'],
            FIT_rows=len(fit),MONITOR_rows=len(monitor),source_sha256=report['source_sha256'],residual_fit_episodes=residual_episodes,reliability_fit_episodes=gate_episodes,
            use_future_labels_at_inference=False,action_units=stats['action_units'],unit_clarification_sha256=snapshot['unit_clarification']['sha256'],
            runtime_unit_qualification_sha256=snapshot['runtime_unit_qualification']['sha256'],native_gripper_indices=[6,13],native_gripper_conversion_applied=False,
            highlevel_prior_monitor_selection=True,full_hierarchy_monitor_interpretation='reused validation diagnostics; no new motor MONITOR selection')
        save_checkpoint(a.output,model,contract);reloaded,_=load_checkpoint(a.output);x=batch(data,monitor[:32])
        with torch.no_grad():before=predict(model,x,st,a.goal_free);after=predict(reloaded,x,st,a.goal_free)
        if any(not torch.equal(before[k],after[k]) for k in before):raise ValueError('Saved/reloaded prefix predictions differ')
        report.update(checkpoint=contract,original_prediction_reload_bitexact=True)
        print(json.dumps({'status':report['status'],'arm_updates':report['optimizer_updates'],'gate_updates':report['gate_updates'],'MONITOR':final['metrics']}),flush=True)
    except BaseException as error:report.update(status='failed_preserved_no_retry',failure=dict(type=type(error).__name__,message=str(error)));raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old);report['elapsed_seconds']=time.monotonic()-started;write_json(a.output/'receipt.json',report)
if __name__=='__main__':main()
