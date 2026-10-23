"""Bounded experimental ACT correction, independent of full scientific guards."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from .correction import VERSION, ARM, GRIP, GoalArmCorrection, arm_loss, per_record_error
from .prepare import sha, write_json
from .train import load_checkpoint as load_highlevel, stats_tensors, birth


def verify_unit_clarification(path, expected, scope):
    if sha(path)!=expected:raise ValueError('Exact native ACT unit clarification required')
    clarification=json.loads(path.read_text())
    if (clarification.get('schema')!='goal-latent-act-native-action-unit-clarification-v1'
        or clarification.get('status')!='mandatory_external_clarification_preserved_source_and_cache_bytes'
        or scope not in clarification.get('applies_to_scopes',[]) or clarification.get('cache_or_model_bytes_modified') is not False):
        raise ValueError('Preserved native task-specific action units required')
    return clarification


def live_original(identity):
    pid=identity['pid']; path=Path(f'/proc/{pid}/stat')
    if not path.exists():return False
    return int(path.read_text().rsplit(')',1)[1].split()[19])==identity['birth_ticks']


def load_cache(root, expected, supervision, supervision_sha, expected_scope='full1260'):
    if sha(supervision)!=supervision_sha:raise ValueError('Producer supervision receipt changed')
    outer=json.loads(supervision.read_text())
    if (outer['status']!='child_exit0' or outer['original_child_exit_code']!=0 or outer['exit_code']!=0
        or outer.get('drain',{}).get('natural') is not True or outer.get('independent_group_after_reap')
        or outer.get('independent_GPU_after',{}).get('compute_pids')):
        raise ValueError('Original successful, naturally drained producer required')
    for key in ('controller','child'):
        if live_original(outer[key]):raise ValueError('Producer original identity still live')
    path=root/'receipt.json'
    if sha(path)!=expected:raise ValueError('Exact complete balanced cache receipt required')
    r=json.loads(path.read_text())
    if r['schema']!='goal-latent-act-balanced-frozen50-cache-v1' or r['status']!='balanced_experimental_cache_complete':
        raise ValueError('New experimental balanced cache required')
    scopes={'full1260':(1120,140,range(16),(900,901),112,14),'smoke140':(70,70,(0,),(900,),7,7)}
    if expected_scope not in scopes:raise ValueError('Explicit experimental cache scope')
    nfit,nmonitor,fit_episodes,monitor_episodes,per_fit,per_monitor=scopes[expected_scope]
    if r['scope']!=expected_scope or (r['fit'],r['monitor'])!=(nfit,nmonitor):raise ValueError('Exact declared balanced cache scope required')
    if r.get('optimizer_updates')!=0 or r.get('native_release_sources_unchanged') is not True:
        raise ValueError('Unchanged native frozen ACT production required')
    records=r['records'];seen=set();counts=Counter();arrays={};normalizers={}
    for task in r['tasks']:
        ref=task['normalizer'];p=Path(ref['path'])
        if not p.resolve().is_relative_to(root.resolve()) or sha(p)!=ref['sha256']:raise ValueError('Native normalizer changed')
        with np.load(p,allow_pickle=False) as values:normalizers[task['task']]={k:values[k].copy() for k in values.files}
    for record in records:
        clip=record['clip_id']
        if clip in seen:raise ValueError('Duplicate ACT cache row')
        seen.add(clip); ep=record['episode']; split=record['split']
        if (split=='fit' and ep not in fit_episodes) or (split=='monitor' and ep not in monitor_episodes) or split not in ('fit','monitor'):
            raise ValueError('Actual task-local FIT/MONITOR split changed')
        counts[record['task'],split]+=1
        ref=record['file'];p=Path(ref['path'])
        if not p.resolve().is_relative_to(root.resolve()) or sha(p)!=ref['sha256'] or p.stat().st_size!=ref['bytes']:
            raise ValueError('Exact cache record changed')
        with np.load(p,allow_pickle=False) as values:row={k:values[k].copy() for k in values.files}
        for k in ('physical_base','normalized_base','physical_actions','normalized_actions'):
            if row[k].shape!=(50,14) or row[k].dtype!=np.float32 or not np.isfinite(row[k]).all():raise ValueError('Native50 FP32 cache contract')
        if row['valid'].shape!=(50,) or row['valid'].dtype!=bool or not row['valid'][:32].all():raise ValueError('Exact literal future supervision mask')
        if row['canonical_current_qpos14'].shape!=(14,) or row['canonical_current_qpos14'].dtype!=np.float32 or not np.isfinite(row['canonical_current_qpos14']).all():
            raise ValueError('Finite actual canonical measured-current14 required')
        if row['role_features'].shape!=(3,4096) or row['role_features'].dtype!=np.float16:raise ValueError('Exact current roles required')
        norm=normalizers[record['task']]; mean=norm['action_mean']; scale=norm['action_std']
        if mean.shape!=(14,) or scale.shape!=(14,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or not (scale>=0).all():
            raise ValueError('Native task normalizer shape; literal constant channels may have zero std')
        if not np.allclose(row['physical_base'],row['normalized_base']*scale+mean,atol=2e-6,rtol=1e-6):raise ValueError('Parent physical action units changed')
        if not np.allclose(row['physical_actions'][row['valid']],(row['normalized_actions']*scale+mean)[row['valid']],atol=2e-6,rtol=1e-6):
            raise ValueError('Target physical action units changed')
        for key,value in row.items():arrays.setdefault(key,[]).append(value)
    tasks=sorted(normalizers)
    if len(tasks)!=10 or len(seen)!=nfit+nmonitor or any(counts[t,'fit']!=per_fit or counts[t,'monitor']!=per_monitor for t in tasks):
        raise ValueError('Allten balanced rows required')
    arrays={k:np.stack(v) for k,v in arrays.items()}; fit=np.asarray([x['split']=='fit' for x in records])
    arrays['task_index']=np.asarray([tasks.index(x['task']) for x in records]); arrays['fit']=fit
    return r,records,arrays,tasks


def tensors(stats):
    return {k:torch.tensor(v,dtype=torch.float32) for k,v in stats.items() if k in ('qpos_mean','qpos_scale','eef_scale','base_mean','base_scale')}


def batch(data,ids):
    return {k:torch.from_numpy(np.ascontiguousarray(data[k][ids])).float() for k in ('physical_base','physical_actions','canonical_current_qpos14','goal_pose','teacher_eef_now')} | {'valid':torch.from_numpy(data['valid'][ids])}


def predict(model,x,st,goal_free,gated=True):
    return model(physical_base=x['physical_base'],qpos=x['canonical_current_qpos14'],goal_pose=x['goal_pose'],
                 current_eef=x['teacher_eef_now'],**st,goal_free=goal_free,gated=gated,threshold=.5)


@torch.no_grad()
def evaluate(model,data,ids,st,tasks,goal_free):
    model.eval();sums=Counter();task_sums={t:Counter() for t in tasks};counts=Counter();accepted=0
    predictions=[];task_accepted=Counter()
    for start in range(0,len(ids),32):
        part=ids[start:start+32];x=batch(data,part);p=predict(model,x,st,goal_free);predictions.append(p['physical_actions'].numpy())
        metrics={prefix+f'_arm_mae_first{n}_rad':per_record_error(value,x['physical_actions'],x['valid'],first=n)
                 for prefix,value in (('parent',x['physical_base']),('ungated',p['ungated_physical_actions']),('candidate',p['physical_actions'])) for n in (10,50)}
        current=x['canonical_current_qpos14'][:,None].expand(-1,50,-1)
        metrics['hold_arm_mae_first10_rad']=per_record_error(current,x['physical_actions'],x['valid'],first=10)
        metrics['reliability']=p['reliability'];accepted+=int(p['accepted'].sum())
        for key,value in metrics.items():
            sums[key]+=float(value.sum())
            for j,t in enumerate(tasks):
                mask=torch.from_numpy(data['task_index'][part]==j)
                task_sums[t][key]+=float(value[mask].sum())
        for j,t in enumerate(tasks):
            mask=torch.from_numpy(data['task_index'][part]==j)
            counts[t]+=int(mask.sum());task_accepted[t]+=int(p['accepted'][mask].sum())
    return dict(rows=len(ids),accepted=accepted,metrics={k:v/len(ids) for k,v in sums.items()},
                per_task={t:dict(rows=counts[t],accepted=task_accepted[t],metrics={k:v/counts[t] for k,v in task_sums[t].items()}) for t in tasks}),np.concatenate(predictions)


def fk_metrics(data,ids,predictions,geometry,urdf,tasks):
    spec=importlib.util.spec_from_file_location('goal_correction_fk_geometry',geometry)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);chain=module.read_chain(urdf)
    values={name:[] for name in ('candidate','parent','hold')}
    for j,index in enumerate(ids):
        targets=data['physical_actions'][index]
        for name,pred in (('candidate',predictions[j]),('parent',data['physical_base'][index]),('hold',np.repeat(data['canonical_current_qpos14'][index][None],50,axis=0))):
            errors=[]
            for f in range(10):
                for side,axes in (('left',slice(0,6)),('right',slice(7,13))):
                    actual=module.fk_link(chain,targets[f,axes],side)@module.tcp()
                    proposed=module.fk_link(chain,pred[f,axes],side)@module.tcp()
                    errors.append(float(np.linalg.norm(actual[:3,3]-proposed[:3,3])))
            values[name].append(float(np.mean(errors)))
    return dict(metric='ideal_command_FK_TCP_position_L2_first10_m; not observed robot outcome',
                geometry_sha256=sha(geometry),urdf_sha256=sha(urdf),
                mean={k:float(np.mean(v)) for k,v in values.items()},
                per_task={t:{k:float(np.asarray(v)[data['task_index'][ids]==j].mean()) for k,v in values.items()} for j,t in enumerate(tasks)})


def save_checkpoint(output,model,contract):
    weights={k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()}
    save_file(weights,str(output/'correction.safetensors'))
    with (output/'correction.safetensors').open('rb') as stream:os.fsync(stream.fileno())
    loaded=load_file(str(output/'correction.safetensors'))
    if any(not torch.equal(v,loaded[k]) for k,v in weights.items()):raise ValueError('Original saved correction tensors differ on reload')
    contract.update(weights_sha256=sha(output/'correction.safetensors'),original_state_reload_bitexact=True)
    write_json(output/'checkpoint.json',contract)


def load_checkpoint(output):
    contract=json.loads((output/'checkpoint.json').read_text())
    if contract['schema']!=VERSION or sha(output/'correction.safetensors')!=contract['weights_sha256']:raise ValueError('Exact correction checkpoint')
    model=GoalArmCorrection(**contract['model_config']);model.load_state_dict(load_file(str(output/'correction.safetensors')),strict=True)
    return model.eval(),contract


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--cache-root',type=Path,required=True);p.add_argument('--cache-receipt-sha256',required=True)
    p.add_argument('--producer-supervisor',type=Path,required=True);p.add_argument('--producer-supervisor-sha256',required=True)
    p.add_argument('--unit-clarification',type=Path,required=True);p.add_argument('--unit-clarification-sha256',required=True)
    p.add_argument('--highlevel',type=Path,required=True);p.add_argument('--highlevel-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--goal-free',action='store_true')
    p.add_argument('--cache-scope',choices=('full1260','smoke140'),default='full1260')
    p.add_argument('--qualify-only',action='store_true')
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--gate-epochs',type=int,default=8)
    p.add_argument('--threads',type=int,default=8);p.add_argument('--wall-seconds',type=int,default=1800)
    p.add_argument('--geometry',type=Path,required=True);p.add_argument('--urdf',type=Path,required=True)
    a=p.parse_args()
    if not 1<=a.epochs<=60 or not 1<=a.gate_epochs<=20 or not 1<=a.threads<=8 or not 1<=a.wall_seconds<=1800:raise ValueError('Bounded CPU actual correction training')
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.manual_seed(1701)
    report=dict(schema=VERSION,status='started',pid=os.getpid(),birth=birth(os.getpid()),device='cpu',GPU_launched=False,
                optimizer_updates=0,gate_updates=0,goal_free=a.goal_free,grippers_unchanged=True)
    started=time.monotonic()
    def expired(*_):raise TimeoutError('Correction training wall budget')
    signal.signal(signal.SIGALRM,expired);signal.alarm(a.wall_seconds)
    try:
        clarification=verify_unit_clarification(a.unit_clarification,a.unit_clarification_sha256,a.cache_scope)
        report.update(unit_clarification_sha256=a.unit_clarification_sha256,action_units='native ACT task release convention; arms radians; grippers untouched')
        receipt,records,data,tasks=load_cache(a.cache_root,a.cache_receipt_sha256,a.producer_supervisor,a.producer_supervisor_sha256,a.cache_scope)
        high,hc=load_highlevel(a.highlevel)
        if hc['weights_sha256']!=a.highlevel_sha256 or hc.get('role_zero_control'):raise ValueError('Actual factual trained high-level binding required')
        hst=stats_tensors(hc['statistics']);goals=[]
        with torch.no_grad():
            for start in range(0,len(records),128):
                sl=slice(start,start+128)
                pred=high(roles=torch.from_numpy(data['role_features'][sl].astype(np.float32)),
                          qpos=torch.from_numpy(data['canonical_current_qpos14'][sl]),eef_now=torch.from_numpy(data['teacher_eef_now'][sl]),**hst)
                goals.append(pred['eef_goal_pose'].numpy())
        data['goal_pose']=np.concatenate(goals);fit=np.flatnonzero(data['fit']);monitor=np.flatnonzero(~data['fit'])
        if a.qualify_only:
            base=data['physical_base'][fit].reshape(-1,14)
            stats={k:hc['statistics'][k] for k in ('qpos_mean','qpos_scale','eef_scale')}
            stats.update(base_mean=base.mean(0).tolist(),base_scale=np.maximum(base.std(0),.01).tolist())
            st=tensors(stats);model=GoalArmCorrection();x=batch(data,fit[:32])
            pred=predict(model,x,st,a.goal_free,gated=False)
            if not torch.equal(pred['physical_actions'],x['physical_base']):raise ValueError('Zero correction parent parity failed')
            loss=arm_loss(pred['physical_actions'],x['physical_actions'],x['valid']);loss.backward()
            gradient=sum(float(p.grad.abs().sum()) for p in model.residual.parameters() if p.grad is not None)
            if not torch.isfinite(loss) or not gradient>0:raise ValueError('Actual physical action gradient failed')
            report.update(status='actual_balanced_cache_and_correction_CPU_qualified_no_candidate_trained',scope=a.cache_scope,
                          fit_rows=len(fit),monitor_rows=len(monitor),zero_correction_parent_bitexact=True,
                          physical_loss=float(loss.detach()),physical_residual_gradient_L1=gradient,
                          highlevel_weights_sha256=a.highlevel_sha256,goal_dimensions=list(data['goal_pose'].shape),
                          cache_receipt_sha256=a.cache_receipt_sha256,producer_supervision_sha256=a.producer_supervisor_sha256,
                          CUDA_initialized=torch.cuda.is_initialized(),optimizer_updates=0)
            print(json.dumps({'status':report['status'],'rows':len(records),'loss':float(loss.detach())}),flush=True)
            return
        if a.cache_scope!='full1260':raise ValueError('Scientific motor/gate fitting requires complete full1260; smoke qualification cannot become a candidate')
        residual_fit=np.asarray([i for i in fit if records[i]['episode']<14],dtype=np.int64)
        reliability_fit=np.asarray([i for i in fit if records[i]['episode'] in (14,15)],dtype=np.int64)
        if len(residual_fit)!=980 or len(reliability_fit)!=140 or set(residual_fit)&set(reliability_fit):
            raise ValueError('FIT-only residual/gate demonstration split required')
        base=data['physical_base'][fit].reshape(-1,14)
        stats={k:hc['statistics'][k] for k in ('qpos_mean','qpos_scale','eef_scale')}
        stats.update(base_mean=base.mean(0).tolist(),base_scale=np.maximum(base.std(0),.01).tolist(),
                     schema='goal-latent-ACT-correction-fit-only-current-and-parent-v1',fit_rows=1120,
                     action_units='ACT native task release convention: arms radians; grippers task-specific, passed through bitexact',
                     target_units='literal commands: arms radians; grippers physical meters; ONLY arm12 are trained/scored',
                     goal_units='arena/world meters and rigid rotation',
                     feature_encoding='native ACT14 zscore +currentcanonical14 zscore +predicted goal-minus-current EEF18/FITcurrentEEFscale +chunkposition')
        st=tensors(stats);model=GoalArmCorrection();optimizer=torch.optim.AdamW(model.residual.parameters(),lr=3e-4,weight_decay=.001)
        report.update(fit_rows=len(fit),monitor_rows=len(monitor),fit_demonstrations=160,monitor_demonstrations=20,
                      residual_fit_rows=980,residual_fit_episodes=list(range(14)),reliability_fit_rows=140,reliability_fit_episodes=[14,15],
                      highlevel_weights_sha256=a.highlevel_sha256,cache_receipt_sha256=a.cache_receipt_sha256,
                      producer_supervision_sha256=a.producer_supervisor_sha256,
                      statistics=stats,source_sha256={'correction.py':sha(Path(__file__).with_name('correction.py')),'train_correction.py':sha(__file__)},
                      initial_monitor=evaluate(model,data,monitor,st,tasks,a.goal_free)[0])
        best=float('inf');best_state=None;best_epoch=None;rng=np.random.default_rng(1701);history=[]
        for epoch in range(a.epochs):
            model.train();order=rng.permutation(residual_fit);sums=0.
            for start in range(0,len(order),32):
                ids=order[start:start+32];x=batch(data,ids);optimizer.zero_grad(set_to_none=True)
                pred=predict(model,x,st,a.goal_free,gated=False);loss=arm_loss(pred['physical_actions'],x['physical_actions'],x['valid'])
                if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite physical arm loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.residual.parameters(),5.);optimizer.step()
                report['optimizer_updates']+=1;sums+=float(loss.detach())*len(ids)
            val,_=evaluate(model,data,monitor,st,tasks,a.goal_free);score=val['metrics']['ungated_arm_mae_first10_rad']
            history.append(dict(epoch=epoch+1,fit_loss=sums/len(residual_fit),monitor=val,fit_order_sha256=hashlib.sha256(order.tobytes()).hexdigest()))
            if score<best:best=score;best_epoch=epoch+1;best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
            report.update(history=history,elapsed_seconds=time.monotonic()-started);write_json(a.output/'receipt.json',report)
        model.load_state_dict(best_state,strict=True)
        for p0 in model.residual.parameters():p0.requires_grad_(False)
        gate_opt=torch.optim.AdamW(model.reliability.parameters(),lr=1e-3,weight_decay=.001)
        positive=0
        for epoch in range(a.gate_epochs):
            order=rng.permutation(reliability_fit);total=0.
            for start in range(0,len(order),32):
                ids=order[start:start+32];x=batch(data,ids);gate_opt.zero_grad(set_to_none=True)
                pred=predict(model,x,st,a.goal_free,gated=False)
                target=(per_record_error(pred['ungated_physical_actions'],x['physical_actions'],x['valid'])+.001
                        <per_record_error(x['physical_base'],x['physical_actions'],x['valid'])).detach().float()
                positive+=int(target.sum()) if epoch==0 else 0
                loss=torch.nn.functional.binary_cross_entropy_with_logits(pred['reliability_logits'],target)
                loss.backward();gate_opt.step();report['gate_updates']+=1;total+=float(loss.detach())*len(ids)
            report.setdefault('gate_history',[]).append(dict(epoch=epoch+1,fit_BCE=total/len(reliability_fit)))
        final,predictions=evaluate(model,data,monitor,st,tasks,a.goal_free)
        report.update(status='completed_experimental_goal_ACT_correction_training',selected_epoch=best_epoch,
                      reliability_positive_fit_rows=positive,final_monitor=final,
                      final_monitor_FK=fk_metrics(data,monitor,predictions,a.geometry,a.urdf,tasks),
                      CUDA_initialized=torch.cuda.is_initialized(),robot_success_claim=False)
        contract=dict(schema=VERSION,model_config=model.config,statistics=stats,goal_free=a.goal_free,
                      reliability_threshold=.5,highlevel_weights_sha256=a.highlevel_sha256,
                      highlevel_checkpoint_sha256=sha(a.highlevel/'checkpoint.json'),highlevel_model_config=hc['model_config'],
                      highlevel_statistics=hc['statistics'],highlevel_context_version=hc['context_version'],
                      cache_receipt_sha256=a.cache_receipt_sha256,parent_ACT_cache_plan_semantic_sha256=receipt['plan_sha256'],
                      parent_ACT_production_plan_file_sha256=json.loads(a.producer_supervisor.read_text())['plan_sha256'],
                      producer_supervisor_sha256=a.producer_supervisor_sha256,
                      native_task_normalizers={t['task']:t['normalizer'] for t in receipt['tasks']},
                      geometry_sha256=sha(a.geometry),urdf_sha256=sha(a.urdf),
                      native_chunk_generated=50,prefix_executed=10,discarded=40,grippers_unchanged=True,
                      selected_epoch=best_epoch,optimizer_updates=report['optimizer_updates'],gate_updates=report['gate_updates'],
                      FIT_rows=1120,MONITOR_rows=140,source_sha256=report['source_sha256'],
                      residual_fit_episodes=list(range(14)),reliability_fit_episodes=[14,15],
                      use_future_labels_at_inference=False,action_units=stats['action_units'])
        contract.update(cache_unit_clarification='physical_base gripper fields are task-specific native release units, despite historical cache metadata; physical_actions targets grippers are meters; no gripper loss or correction',
                        unit_clarification_sha256=a.unit_clarification_sha256,
                        unit_clarification_schema=clarification['schema'],
                        native_action_unit_contract=clarification['correct_payload_contract'],
                        native_gripper_indices=[6,13],native_gripper_conversion_applied=False)
        save_checkpoint(a.output,model,contract);reloaded,_=load_checkpoint(a.output)
        ids=monitor[:32];x=batch(data,ids)
        with torch.no_grad():before=predict(model,x,st,a.goal_free);after=predict(reloaded,x,st,a.goal_free)
        if any(not torch.equal(before[k],after[k]) for k in before):raise ValueError('Original/reloaded correction predictions differ')
        report.update(checkpoint=contract,original_prediction_reload_bitexact=True)
        print(json.dumps({'status':report['status'],'arm_updates':report['optimizer_updates'],'gate_updates':report['gate_updates'],
                          'monitor':final['metrics'],'FK':report['final_monitor_FK']['mean']}),flush=True)
    except BaseException as error:report.update(status='failed_no_retry',error=repr(error));raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started;write_json(a.output/'receipt.json',report);signal.alarm(0)


if __name__=='__main__':main()
