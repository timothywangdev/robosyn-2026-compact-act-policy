"""Actual fixed recurrent opening pilot; source-qualified sequence FIT."""
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
from experiments.goal_latent_act_command_goal_training_full import data
from experiments.goal_latent_act_full_data.correction_prefix import eef_features
from experiments.goal_latent_act.prepare import sha,write_json
from experiments.goal_latent_act_campaign.common import ref,verify,require
from .prepare import pins, DATA_SCHEMA
from .model import SCHEMA,SequencePhysicalOpening,loss as contact_loss
from experiments.goal_latent_act_contact_training.model import map_native_grips,per_hand_mae,opening_classes
from experiments.goal_latent_act_contact_training.inspect_fit_timing_v2 import events,aggregate as event_aggregate
from .contracts import SELECTION,load_checkpoint


def batch(arrays,features,ids,stats,*,shuffled=False):
    roleids=ids;goal=features['predicted_goal'][ids]
    if shuffled:
        require(np.array_equal(features['monitor_ids'],ids),'Whole fixed MONITOR intervention required')
        roleids=features['shuffled_role_ids'];goal=features['shuffled_predicted_goal']
    tensor=lambda x:torch.from_numpy(np.asarray(x,dtype=np.float32))
    base=tensor(arrays['physical_base'][ids]);qpos=tensor(arrays['canonical_current_qpos14'][ids])
    return dict(roles=tensor(arrays['role_features'][roleids]),qpos_features=(qpos-stats['qpos_mean'])/stats['qpos_scale'],
        goal_delta=(eef_features(tensor(goal))-eef_features(tensor(arrays['teacher_eef_now'][ids])))/stats['eef_scale'],
        native_base_features=(base-stats['base_mean'])/stats['base_scale'],native_base=base,
        parent_grips_m=map_native_grips(base,torch.from_numpy(arrays['task_index'][ids]),stats['tasks']),
        current_eef_features=(eef_features(tensor(arrays['teacher_eef_now'][ids]))-stats['current_eef_mean'])/stats['current_eef_scale'],
        measured_opening=qpos[:,[6,13]].clamp(0,1))


def diagnostics(pred,target,valid):
    cls=opening_classes(target);pc=opening_classes(pred);mask=valid[...,None].expand_as(cls)
    error=(pred-target).abs();values=dict(commands=int(valid.sum()),sum_abs_error_m=float((error*mask).sum()),
        grip_components=int(mask.sum()),correct_states=int(((cls==pc)&mask).sum()))
    for label,index in [('closed',0),('intermediate',1),('open',2)]:
        selected=(cls==index)&mask;values[label+'_components']=int(selected.sum())
        values[label+'_sum_abs_error_m']=float(error[selected].sum())
        values[label+'_correct_states']=int((pc[selected]==index).sum())
    adjacent=(valid[:,1:]&valid[:,:-1])[...,None].expand(-1,-1,2)
    transition=(cls[:,1:]!=cls[:,:-1])&adjacent;predicted=(pc[:,1:]!=pc[:,:-1])&adjacent
    values.update(actual_transitions=int(transition.sum()),predicted_transitions=int(predicted.sum()),
                  exact_transition_true_positives=int((transition&predicted).sum()))
    return values


def finish_counts(values):
    r=dict(values);r['MAE_m']=r['sum_abs_error_m']/max(1,r['grip_components'])
    r['opening_state_accuracy']=r['correct_states']/max(1,r['grip_components'])
    for label in ('closed','intermediate','open'):
        r[label+'_MAE_m']=r[label+'_sum_abs_error_m']/max(1,r[label+'_components'])
        r[label+'_recall']=r[label+'_correct_states']/max(1,r[label+'_components'])
    r['transition_recall_exact_index']=r['exact_transition_true_positives']/max(1,r['actual_transitions'])
    r['transition_precision_exact_index']=r['exact_transition_true_positives']/max(1,r['predicted_transitions'])
    return r


@torch.no_grad()
def evaluate(model,arrays,features,ids,stats,control=False,shuffle=False):
    model.eval();sums={k:Counter() for k in ('parent','candidate','ungated','hold','all_open')}
    per={t:{k:Counter() for k in sums} for t in stats['tasks']};counts=Counter();accepts=Counter()
    event_counts={t:{k:Counter() for k in ('parent','candidate')} for t in per};event_lags={t:{k:[] for k in ('parent','candidate')} for t in per}
    for start in range(0,len(ids),128):
        current=ids[start:start+128]
        if shuffle:
            # Explicit whole-MONITOR role intervention prepared with unchanged measured state.
            lookup=np.searchsorted(features['monitor_ids'],current)
            altered=dict(features);altered['predicted_goal']=features['predicted_goal'].copy()
            altered['predicted_goal'][current]=features['shuffled_predicted_goal'][lookup]
            x=batch(arrays,altered,current,stats)
            x['roles']=torch.from_numpy(arrays['role_features'][features['shuffled_role_ids'][lookup]].astype(np.float32))
        else:x=batch(arrays,features,current,stats)
        result=model(**x,state_action_only=control);target=torch.from_numpy(arrays['physical_actions'][current][:,:,[6,13]])
        valid=torch.from_numpy(arrays['valid'][current]);hold=torch.from_numpy(arrays['canonical_current_qpos14'][current][:,[6,13]]).clamp(0,1)*.05
        candidates=dict(parent=x['parent_grips_m'],candidate=result['grips_m'],ungated=result['ungated_grips_m'],
                        hold=hold[:,None].expand(-1,50,-1),all_open=torch.full_like(target,.05))
        for name,pred in candidates.items():sums[name].update(diagnostics(pred,target,valid))
        for j,t in enumerate(stats['tasks']):
            mask=torch.from_numpy(arrays['task_index'][current]==j);counts[t]+=int(mask.sum())
            accepts[t]+=int(result['accepted'][mask].sum())
            for name,pred in candidates.items():per[t][name].update(diagnostics(pred[mask],target[mask],valid[mask]))
            for name in ('parent','candidate'):
                c,l=events(target[mask],candidates[name][mask],valid[mask]);event_counts[t][name].update(c);event_lags[t][name].extend(l)
    return dict(rows=len(ids),metrics={k:finish_counts(v) for k,v in sums.items()},
                per_task={t:dict(rows=counts[t],accepted_hand_chunks=accepts[t],metrics={k:finish_counts(v) for k,v in per[t].items()},
                    closed_boundary_timing={k:event_aggregate(event_counts[t][k],event_lags[t][k]) for k in ('parent','candidate')}) for t in per},
                role_intervention='within_task_current_roles_and_predicted_goal_recomputed' if shuffle else None,
                interpretation='Physical desired-command opening error/state transition proxy; NOT measured contact/robot success')


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for k in ('prepared','prepared-sha256','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--state-action-only',action='store_true');a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.manual_seed(1701);np.random.seed(1701)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU contact1800s bound')));signal.alarm(1800)
    report=dict(schema=SCHEMA,status='actual_CPU_training_running',pid=os.getpid(),state_action_only=a.state_action_only,
                seed=1701,residual_epochs=30,gate_epochs=20,batch_size=128,optimizer_updates=0,gate_updates=0,
                selection=SELECTION,GPU_queries=0,GPU_launches=0,robot_launches=0)
    try:
        prepared=Path(a.prepared);require(sha(prepared/'receipt.json')==a.prepared_sha256,'Exact physical opening preparation required')
        r=json.loads((prepared/'receipt.json').read_text());require(r['schema']==DATA_SCHEMA and r['status']=='CPU_source_labels_and_current_only_head_verified','Source target audit must pass')
        for name,digest in r['source_pins'].items():require(sha(data.ROOT/name)==digest,'Pinned contact source changed')
        for value in r['artifacts'].values():verify(value)
        verify(r['target_audit']);verify(r['protocol']);head_contract=json.loads(verify(r['highlevel_contract']).read_text());verify(r['highlevel_weights'])
        original,records,arrays,tasks=data.load(Path(r['source_data']['path']).parent,r['source_data']['sha256'])
        with np.load(prepared/'features.npz',allow_pickle=False) as z:features={k:z[k].copy() for k in z.files}
        stats={k:torch.tensor(v,dtype=torch.float32) for k,v in r['statistics'].items() if k in ('qpos_mean','qpos_scale','base_mean','base_scale','eef_scale','current_eef_mean','current_eef_scale')};stats['tasks']=tasks
        fit=np.flatnonzero(features['residual_fit']);gatefit=np.flatnonzero(features['gate_fit']);monitor=features['monitor_ids']
        require((len(fit),len(gatefit),len(monitor))==(50400,6300,7000),'Fixed independent FIT split required')
        model=SequencePhysicalOpening();rng=np.random.default_rng(1701)
        report.update(source_pins=r['source_pins'],prepared_receipt=ref(prepared/'receipt.json'),
                      residual_FIT=50400,reliability_FIT=6300,MONITOR=7000,statistics=r['statistics'],
                      initial_monitor=evaluate(model,arrays,features,monitor,stats,a.state_action_only),
                      parameters=sum(p.numel() for p in model.parameters()))
        parameters=[v for name,v in model.named_parameters() if not name.startswith('reliability.')]
        optimizer=torch.optim.AdamW(parameters,lr=3e-4,weight_decay=.001)
        for epoch in range(30):
            model.train();order=rng.permutation(fit);total=0.
            for start in range(0,len(order),128):
                ids=order[start:start+128];x=batch(arrays,features,ids,stats);optimizer.zero_grad(set_to_none=True)
                result=model(**x,state_action_only=a.state_action_only,gated=False)
                target=torch.from_numpy(arrays['physical_actions'][ids][:,:,[6,13]]);valid=torch.from_numpy(arrays['valid'][ids])
                loss=contact_loss(result,target,x['measured_opening'],valid,torch.tensor(r['statistics']['hazard_positive_weights']));require(bool(torch.isfinite(loss)),'Finite physical contact loss required')
                loss.backward();require(all(v.grad is None or bool(torch.isfinite(v.grad).all()) for v in parameters),'Finite contact gradients required')
                torch.nn.utils.clip_grad_norm_(parameters,5.);optimizer.step();total+=float(loss.detach())*len(ids);report['optimizer_updates']+=1
            report.setdefault('history',[]).append(dict(epoch=epoch+1,FIT_loss=total/len(order),FIT_order_sha256=hashlib.sha256(order.tobytes()).hexdigest()))
            report.update(epochs_completed=epoch+1,elapsed_seconds=time.monotonic()-started);write_json(out/'receipt.json',report)
        for parameter in parameters:parameter.requires_grad_(False)
        optimizer=torch.optim.AdamW(model.reliability.parameters(),lr=1e-3,weight_decay=.001);positive=np.zeros(2,np.int64)
        for epoch in range(20):
            order=rng.permutation(gatefit);total=0.
            for start in range(0,len(order),128):
                ids=order[start:start+128];x=batch(arrays,features,ids,stats);optimizer.zero_grad(set_to_none=True)
                result=model(**x,state_action_only=a.state_action_only,gated=False)
                target=torch.from_numpy(arrays['physical_actions'][ids][:,:,[6,13]]);valid=torch.from_numpy(arrays['valid'][ids])
                label=(per_hand_mae(result['ungated_grips_m'],target,valid)+.0005<per_hand_mae(x['parent_grips_m'],target,valid)).float().detach()
                if epoch==0:positive+=label.sum(0).numpy().astype(np.int64)
                loss=torch.nn.functional.binary_cross_entropy_with_logits(result['reliability_logits'],label);require(bool(torch.isfinite(loss)),'Finite independent FIT gate loss required')
                loss.backward();optimizer.step();total+=float(loss.detach())*len(ids);report['gate_updates']+=1
            report.setdefault('gate_history',[]).append(dict(epoch=epoch+1,FIT_BCE=total/len(order),FIT_order_sha256=hashlib.sha256(order.tobytes()).hexdigest()))
        require(report['optimizer_updates']==11820 and report['gate_updates']==1000,'Exact fixed matched training budgets required')
        contract=dict(schema=SCHEMA,model_config=model.config,statistics=r['statistics'],tasks=tasks,state_action_only=a.state_action_only,
            output_units='physical_joint_opening_meters',opening_bounds_m=[0,.05],arms_unchanged=True,native_chunk_generated=50,prefix_executed=50,
            highlevel_command_target_offset=50,highlevel_weights_sha256=r['highlevel_weights']['sha256'],highlevel_contract_sha256=r['highlevel_contract']['sha256'],
            highlevel_context_version=head_contract['context_version'],highlevel_schema=head_contract['schema'],source_FIT=63000,source_MONITOR=7000,optimizer_FIT=50400,diagnostic_FIT=6300,gate_FIT=6300,normalization_scope='optimizer FIT episodes%10<8 only; shared with high-level head',source_data=r['source_data'],original_source_CPU_pilot=r['original_source_CPU_pilot'],coordinate_frame='robot_root_meters',use_future_labels_at_inference=False,
            actor_inputs=['current_roles','measured_current_qpos_and_opening','measured_current_FK','predicted_command50_goal','frozen_native_ACT50'],
            recurrence='predicted opening anchored measured-current opening; no future teacherforcing',
            hazard_positive_weights=r['statistics']['hazard_positive_weights'],
            control_zero_channels=['roles','predicted_goal_delta'] if a.state_action_only else [],
            contact_semantics='binary commanded close/open hazard proxy NOT observed contact force',selected_epoch=30,gate_epochs=20,gate_threshold=.5,
            selection=SELECTION,optimizer_updates=11820,gate_updates=1000,seed=1701,batch_size=128,
            residual_fit_split='FIT episode%10 neither8 nor9',diagnostic_fit_split='FIT episode%10==8',gate_fit_split='FIT episode%10==9',MONITOR_selection=False,
            physical_parent_codec='Drawer clamp01*.05; other9 clamp0..05; no inverse applied to other7 literal experts',
            target_audit_sha256=r['target_audit']['sha256'],prepared_receipt_sha256=a.prepared_sha256,
            unit_clarification_sha256=original['unit_clarification']['sha256'],runtime_unit_qualification_sha256=original['runtime_unit_qualification']['sha256'],
            source_pins=r['source_pins'],future_runtime_requirement='physical-only grip output must NOT receive a second Drawer ratio inverse')
        weights=out/'opening.safetensors';save_file({k:v.detach().contiguous() for k,v in model.state_dict().items()},str(weights))
        with weights.open('rb') as f:os.fsync(f.fileno())
        contract['weights_sha256']=sha(weights);write_json(out/'checkpoint.json',contract);loaded,_=load_checkpoint(out)
        x=batch(arrays,features,monitor[:32],stats)
        with torch.no_grad():before=model(**x,state_action_only=a.state_action_only);after=loaded(**x,state_action_only=a.state_action_only)
        require(all(torch.equal(before[k],after[k]) for k in before),'Saved/reloaded physical opening predictions changed')
        report.update(status='completed_fixed30_full63000_sequence20_FIT_gate',checkpoint=contract,reload_prediction_bitexact=True,
                      reliability_positive_FIT_hand_rows=positive.tolist(),final_monitor=evaluate(loaded,arrays,features,monitor,stats,a.state_action_only),
                      final_residual_FIT=evaluate(loaded,arrays,features,fit,stats,a.state_action_only),
                      final_gate_FIT=evaluate(loaded,arrays,features,gatefit,stats,a.state_action_only),
                      final_sequence_diagnostic_FIT=evaluate(loaded,arrays,features,np.flatnonzero(features['diagnostic_fit']),stats,a.state_action_only),
                      role_shuffle_monitor=evaluate(loaded,arrays,features,monitor,stats,a.state_action_only,True) if not a.state_action_only else None,
                      CUDA_initialized=torch.cuda.is_initialized(),robot_success_claim=False)
        require(not report['CUDA_initialized'],'CPU-only actual training required')
        print(json.dumps(dict(status=report['status'],control=a.state_action_only,updates=11820,gate_updates=1000,MONITOR_MAE_m={k:v['MAE_m'] for k,v in report['final_monitor']['metrics'].items()})),flush=True)
    except BaseException as e:report.update(status='failed_preserved_no_retry',failure=dict(type=type(e).__name__,message=str(e)));raise
    finally:signal.alarm(0);report['elapsed_seconds']=time.monotonic()-started;write_json(out/'receipt.json',report)


if __name__=='__main__':main()
