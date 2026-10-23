"""One bounded original GPU child: fixed final temporal arm branch + FIT gate."""
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
from experiments.goal_latent_act_full_data.correction_prefix import eef_features
from .common import MODEL, require, ref, write, sha, validate_protocol
from .prepare import load
from .model import GoalArmSequence, ARM, GRIP
from .fk import CommandFK, hand_errors, physical_gate_labels
from .contracts import load_checkpoint


def tensors(arrays, features, device):
    # Keep current role storage FP16; only selected actor batches become FP32.
    fields=('physical_base','physical_actions','valid','canonical_current_qpos14','role_features','teacher_eef_now','task_index')
    result={k:torch.from_numpy(arrays[k]).to(device) for k in fields}
    result['predicted_goal']=torch.from_numpy(features['predicted_goal']).to(device)
    result['shuffled_predicted_goal']=torch.from_numpy(features['shuffled_predicted_goal']).to(device)
    result['shuffled_role_ids']=torch.from_numpy(features['shuffled_role_ids']).to(device)
    result['monitor_ids']=torch.from_numpy(features['monitor_ids']).to(device)
    return result


def batch(data, ids, stats, *, shuffled=False):
    ids=torch.as_tensor(ids,dtype=torch.long,device=data['physical_base'].device)
    q=data['canonical_current_qpos14'][ids];base=data['physical_base'][ids];now=eef_features(data['teacher_eef_now'][ids])
    role_ids=ids;goal=data['predicted_goal'][ids]
    if shuffled:
        lookup=torch.searchsorted(data['monitor_ids'],ids)
        require(torch.equal(data['monitor_ids'][lookup],ids), 'Shuffling is report-only MONITOR intervention')
        role_ids=data['shuffled_role_ids'][lookup];goal=data['shuffled_predicted_goal'][lookup]
    return dict(roles=data['role_features'][role_ids].float(),qpos_features=(q-stats['qpos_mean'])/stats['qpos_scale'],
                current_eef_features=(now-stats['current_eef_mean'])/stats['current_eef_scale'],
                goal_delta=(eef_features(goal)-now)/stats['eef_scale'],
                native_base_features=(base-stats['base_mean'])/stats['base_scale'],native_base=base)


def objective(result,target,valid,fk):
    candidate=result['ungated_native_actions'];mask=valid[...,None].expand(-1,-1,12)
    value=torch.nn.functional.smooth_l1_loss((candidate[...,list(ARM)]-target[...,list(ARM)])[mask]/.1,
                                            torch.zeros_like(candidate[...,list(ARM)][mask]))
    e=hand_errors(candidate,target,valid,fk);m=e['FK_valid'].expand(e['poses'].shape[:-2])
    delta=(e['poses'][...,:3,3]-e['labels'][...,:3,3])/.02
    pos=torch.nn.functional.smooth_l1_loss(delta[m],torch.zeros_like(delta[m]))
    rot=(e['poses'][...,:3,:3]-e['labels'][...,:3,:3]).square().mean((-1,-2))
    return value+.25*pos+.10*rot[m].mean()


@torch.no_grad()
def evaluate(model,data,ids,stats,fk,tasks,control,shuffle=False):
    model.eval();per={t:{k:Counter() for k in ('parent','candidate','ungated','hold')} for t in tasks};accept=Counter();counts=Counter();delta_max=0.
    for start in range(0,len(ids),128):
        selected=np.asarray(ids[start:start+128]);i=torch.as_tensor(selected,device=data['physical_base'].device)
        x=batch(data,selected,stats,shuffled=shuffle);result=model(**x,state_action_only=control)
        target=data['physical_actions'][i];valid=data['valid'][i];hold=data['canonical_current_qpos14'][i,None].expand(-1,50,-1).clone()
        candidates=dict(parent=x['native_base'],candidate=result['native_actions'],ungated=result['ungated_native_actions'],hold=hold)
        delta_max=max(delta_max,float(result['delta_arm_rad'].abs().max()))
        errors={k:hand_errors(v,target,valid,fk) for k,v in candidates.items()}
        for j,t in enumerate(tasks):
            mask=data['task_index'][i]==j;n=int(mask.sum());counts[t]+=n
            accept[t]+=int(result['accepted'][mask].sum())
            for name,e in errors.items():
                # Per-row/per-hand averages; FK anchors are literal1/10/32/50
                # and invalid terminal50 samples are excluded before averaging.
                per[t][name].update(rows=n,hand_rows=2*n,joint_sum_rad=float(e['joint'][mask].sum()),
                    position_sum_m=float(e['position'][mask].sum()),rotation_sum_rad=float(e['rotation'][mask].sum()))
    def finish(c):
        n=max(1,c['hand_rows'])
        return dict(rows=c['rows'],hand_rows=c['hand_rows'],arm_MAE_rad=c['joint_sum_rad']/n,
                    command_FK_position_L2_m=c['position_sum_m']/n,command_FK_rotation_geodesic_rad=c['rotation_sum_rad']/n)
    aggregate={name:Counter() for name in ('parent','candidate','ungated','hold')}
    for per_task in per.values():
        for name,value in per_task.items():aggregate[name].update(value)
    return dict(rows=len(ids),metrics={k:finish(v) for k,v in aggregate.items()},
        per_task={t:dict(rows=counts[t],accepted_hand_chunks=accept[t],hand_chunks=2*counts[t],
                         metrics={k:finish(v) for k,v in per[t].items()}) for t in tasks},
        correction_max_abs_rad=delta_max,FK_indices=[0,9,31,49],
        shuffled_roles_and_recomputed_goal=shuffle,
        interpretation='Expert desired-command joint/ideal-root-FK diagnostics, not observed tracking/contact/robot success')


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for k in ('prepared','prepared-sha256','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--state-action-only',action='store_true');p.add_argument('--execute',action='store_true')
    a=p.parse_args();require(a.execute,'GPU training requires explicit admitted execute flag')
    started=time.monotonic();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    report=dict(schema=MODEL,status='original_GPU_child_preflight',pid=os.getpid(),state_action_only=a.state_action_only,
        optimizer_updates=0,gate_updates=0,selection='fixed_final_epoch_no_MONITOR_selection',robot_launches=0,robot_success_claim=False)
    def expired(*_):raise TimeoutError('Arm branch preparation/train/report1680second original cap')
    signal.signal(signal.SIGALRM,expired);signal.alarm(1680)
    try:
        torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(1701);np.random.seed(1701)
        r,arrays,features,tasks=load(a.prepared,a.prepared_sha256);protocol=validate_protocol(json.loads(Path(r['protocol']['path']).read_text()))
        fit=np.flatnonzero(arrays['optimizer_fit']);gate=np.flatnonzero(arrays['gate_fit']);diag=np.flatnonzero(arrays['diagnostic_fit']);monitor=np.flatnonzero(~arrays['fit'])
        require((len(fit),len(gate),len(diag),len(monitor))==(50400,6300,6300,7000),'Fixed literal independent full partitions required')
        require(torch.cuda.is_available() and torch.__version__=='2.7.1+cu128','Pinned actual CUDA stack required')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.cuda.set_per_process_memory_fraction(8*(1<<30)/torch.cuda.get_device_properties(0).total_memory)
        device=torch.device('cuda');d=tensors(arrays,features,device)
        del arrays,features
        stats={k:torch.tensor(r['statistics'][k],dtype=torch.float32,device=device) for k in ('qpos_mean','qpos_scale','base_mean','base_scale','current_eef_mean','current_eef_scale','eef_scale')}
        fk=CommandFK(r['geometry'],r['urdf']).to(device);model=GoalArmSequence(**protocol['model']).to(device)
        x=batch(d,fit[:32],stats)
        with torch.no_grad():initial=model(**x,state_action_only=a.state_action_only)
        require(torch.equal(initial['native_actions'],x['native_base']) and torch.equal(initial['ungated_native_actions'],x['native_base']), 'Zero-initialized correction must preserve released parent')
        report.update(status='actual_GPU_training_running',source_pins=r['source_pins'],prepared=ref(Path(a.prepared)/'receipt.json'),protocol=r['protocol'],
                      statistics=r['statistics'],tasks=tasks,parameters=sum(v.numel() for v in model.parameters()),
                      counts=protocol['partitions'],source_counts=protocol['source_counts'],CUDA_initialized=torch.cuda.is_initialized(),
                      device_name=torch.cuda.get_device_name(),precision='float32; TF32 disabled',zero_initial_residual_parent_equal=True)
        write(out/'receipt.json',report)
        parameters=[v for name,v in model.named_parameters() if not name.startswith('reliability.')]
        optimizer=torch.optim.AdamW(parameters,lr=3e-4,weight_decay=.001);rng=np.random.default_rng(1701)
        for epoch in range(30):
            model.train();order=rng.permutation(fit);total=0.
            for start in range(0,len(order),128):
                ids=order[start:start+128];i=torch.as_tensor(ids,device=device);optimizer.zero_grad(set_to_none=True)
                result=model(**batch(d,ids,stats),state_action_only=a.state_action_only,gated=False)
                loss=objective(result,d['physical_actions'][i],d['valid'][i],fk)
                require(bool(torch.isfinite(loss)),'Finite joint+physical-FK objective required');loss.backward()
                require(all(v.grad is None or bool(torch.isfinite(v.grad).all()) for v in parameters),'Finite arm gradients required')
                torch.nn.utils.clip_grad_norm_(parameters,5.);optimizer.step();report['optimizer_updates']+=1;total+=float(loss.detach())*len(ids)
                if report['optimizer_updates'] == 100:
                    report['first100_updates_elapsed_seconds']=time.monotonic()-started
                    print(json.dumps(dict(updates=100,elapsed_seconds=report['first100_updates_elapsed_seconds'])),flush=True)
            report.setdefault('history',[]).append(dict(epoch=epoch+1,FIT_loss=total/len(fit),FIT_order_sha256=hashlib.sha256(order.tobytes()).hexdigest()))
            report.update(epochs_completed=epoch+1,elapsed_seconds=time.monotonic()-started);write(out/'receipt.json',report)
            print(json.dumps(dict(epoch=epoch+1,updates=report['optimizer_updates'],loss=total/len(fit),elapsed_seconds=report['elapsed_seconds'])),flush=True)
        for value in parameters:value.requires_grad_(False)
        optimizer=torch.optim.AdamW(model.reliability.parameters(),lr=1e-3,weight_decay=.001);positive=torch.zeros(2,device=device)
        for epoch in range(20):
            order=rng.permutation(gate);total=0.
            for start in range(0,len(order),128):
                ids=order[start:start+128];i=torch.as_tensor(ids,device=device);optimizer.zero_grad(set_to_none=True)
                result=model(**batch(d,ids,stats),state_action_only=a.state_action_only,gated=False)
                with torch.no_grad():label=physical_gate_labels(result['ungated_native_actions'],d['physical_base'][i],d['physical_actions'][i],d['valid'][i],fk)
                if epoch==0:positive+=label.sum(0)
                loss=torch.nn.functional.binary_cross_entropy_with_logits(result['reliability_logits'],label)
                require(bool(torch.isfinite(loss)),'Finite physical FIT-gate loss required');loss.backward();optimizer.step()
                report['gate_updates']+=1;total+=float(loss.detach())*len(ids)
            report.setdefault('gate_history',[]).append(dict(epoch=epoch+1,FIT_BCE=total/len(gate),FIT_order_sha256=hashlib.sha256(order.tobytes()).hexdigest()))
        require((report['optimizer_updates'],report['gate_updates'])==(11820,1000),'Exact fixed optimizer/gate budgets required')
        weights=out/'arm-sequence.safetensors';save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(weights))
        with weights.open('rb') as f:os.fsync(f.fileno())
        contract=dict(schema=MODEL,model_config=model.config,weights_sha256=sha(weights),state_action_only=a.state_action_only,
            control_zero_channels=['roles','predicted_goal_delta'] if a.state_action_only else [],statistics=r['statistics'],tasks=tasks,
            prepared=report['prepared'],highlevel_contract=r['highlevel_contract'],highlevel_weights=r['highlevel_weights'],
            geometry=r['geometry'],urdf=r['urdf'],source_pins=r['source_pins'],protocol=protocol,
            coordinate_frame='robot_root_meters',command_goal_offset=50,command_goal_semantics='commanded FK; not observed future EEF',
            native_chunk_generated=50,prefix_executed=50,native_grippers_bitexact=True,selected_epoch=30,gate_epochs=20,
            optimizer_updates=11820,gate_updates=1000,batch_size=128,seed=1701,gate_threshold=.5,
            source_counts=protocol['source_counts'],partitions=protocol['partitions'],MONITOR_selection=False,use_future_labels_at_inference=False,
            actor_inputs=['current_roles','current_qpos','current_root_FK','predicted_command50_goal','native_ACT50'],
            output='native_ACT50 +bounded arm12 correction; native2 grippers bitexact',normalization_scope='optimizer FIT only')
        write(out/'checkpoint.json',contract);loaded,_=load_checkpoint(out,device=device)
        with torch.no_grad():before=model(**x,state_action_only=a.state_action_only);after=loaded(**x,state_action_only=a.state_action_only)
        require(all(torch.equal(before[k],after[k]) for k in before),'Saved/reloaded temporal arm predictions changed')
        report.update(status='completed_fixed30_temporal_arm20_physical_FIT_gate',checkpoint=contract,reload_predictions_bitexact=True,
            positive_gate_FIT_hand_rows=positive.long().cpu().tolist(),
            diagnostic_FIT=evaluate(loaded,d,diag,stats,fk,tasks,a.state_action_only),
            final_gate_FIT=evaluate(loaded,d,gate,stats,fk,tasks,a.state_action_only),
            final_monitor=evaluate(loaded,d,monitor,stats,fk,tasks,a.state_action_only),
            role_shuffle_monitor=evaluate(loaded,d,monitor,stats,fk,tasks,a.state_action_only,shuffle=True) if not a.state_action_only else None,
            peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:
        report.update(status='failed_original_preserved_no_retry',failure=dict(type=type(error).__name__,message=str(error)));raise
    finally:
        signal.alarm(0);report['elapsed_seconds']=time.monotonic()-started;write(out/'receipt.json',report)


if __name__=='__main__':main()
