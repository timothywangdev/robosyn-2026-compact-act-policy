"""Separate balanced experimental ACT cache; original full-science guards stay intact."""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import signal
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.goal_latent_act_data.prepare_assets import sha,QUALIFIER,LOCAL
from policy.wam_adaptation.multitask_data import CanonicalTaskData,TASK_REVISIONS,CAMERAS,valid_starts
SCHEMA='goal-latent-act-full-data-frozen50-shard-v1'
PARENT_SOURCE_SHA='3e483a69595d77581ab3119de639b94a4cbbc40e1e9ec5b3c41a4f0a638c0d21'
UNITS=ROOT/'outputs/architecture-reassessment/goal-latent-act-balanced-cache-v1/unit-clarification.json'
UNITS_SHA='8e7499ba113ac865668a25b03d381b2e43787c98e19066840cd2dc99a9f56141'
TEACHER=ROOT/'outputs/architecture-reassessment/object-relative-full-teacher-data-v1/data'
BELL=ROOT/'outputs/architecture-reassessment/wam-joint-all10-pilot-v1/payload/bell-evidence.json'
TEACHER_REPORT_SHA='220509d25d8a9073edd865d5cafd31fd1dbeff294ce75d2b924b895cd6315e72'

def require(ok,message):
    if not ok:raise ValueError(message)

def ref(path):
    original=Path(path);require(not original.is_symlink(),'No symlink pinned files');p=original.resolve(strict=True);require(p.is_file(),'Regular pinned file required')
    return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))

def check_ref(value):
    require(ref(value['path'])==value,'Pinned source/input differs '+value['path'])

def write(path,value):
    p=Path(path);temp=p.with_suffix('.tmp')
    with temp.open('w') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    temp.replace(p)
    fd=os.open(p.parent,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def load_module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def labels50(commands,frame):
    require(commands.ndim==2 and commands.shape[1]==14 and np.isfinite(commands).all(),'Finite physical14 commands required')
    n=min(50,len(commands)-frame-1);require(n>=32,'At least32 actual next commands required')
    target=np.zeros((50,14),np.float32);target[:n]=commands[frame+1:frame+1+n]
    return target,np.arange(50)<n

def same_array_bytes(left,right):
    left=np.asarray(left);right=np.asarray(right)
    return left.shape==right.shape and left.dtype==right.dtype and left.tobytes(order='C')==right.tobytes(order='C')

def validate_payload(values):
    for name in ('normalized_base','physical_base','physical_actions','normalized_actions','canonical_current_qpos14','role_features','teacher_eef_now','teacher_eef_future'):
        require(np.isfinite(values[name]).all(),'Nonfinite actor/action/EEF payload: '+name)
    for prefix,mask in (('teacher_','teacher_object_valid'),('teacher_future_','teacher_future_object_valid')):
        for suffix in ('positions','axes'):
            value=values[prefix+suffix];valid=values[mask]
            require(not np.isinf(value).any() and np.isfinite(value[valid]).all(),'Only invalid masked teacher geometry may contain NaN')

def data(task):
    return CanonicalTaskData(task,ROOT/'datasets/official-cache'/task/TASK_REVISIONS[task],bell_state_evidence=BELL if task=='click_bell' else None,allow_bell_source_convention=task=='click_bell')

def join(row,source,teacher):
    require(row['revision']==TASK_REVISIONS[row['task']] and row['parquet_sha256']==source['parquet_sha256']==teacher['parquet_sha256'],'Exact released source identity required')
    require(row['clip_id']==teacher['clip_id'] and all(row[k]==teacher[k] for k in ('task','revision','episode','frame','split')),'Exact teacher identity required')
    frame=row['frame'];require(teacher['future_label_frame']==frame+10 and teacher['command_frames']==list(range(frame+1,frame+33)),'Teacher horizon/next-command alignment differs')
    require(source['instruction']==teacher['instruction'] and np.array_equal(source['qpos'][frame],teacher['current_qpos']),'Teacher canonical current observation differs')

def prepare(packed,assets,fit_start,fit_count,monitor_start,monitor_count,output):
    require(sha(ROOT/'experiments/goal_latent_act_data/balanced_cache_v2.py')==PARENT_SOURCE_SHA,'Immutable qualified producer changed')
    require(sha(UNITS)==UNITS_SHA,'Native task-specific gripper clarification required')
    require(type(fit_start) is int and type(fit_count) is int and 0<=fit_start<900 and 1<=fit_count<=14 and fit_start+fit_count<=900,'Explicit contiguous FIT episode shard required')
    require(type(monitor_start) is int and type(monitor_count) is int and 900<=monitor_start<=1000 and 0<=monitor_count<=2 and monitor_start+monitor_count<=1000,'Independent MONITOR episode shard required')
    require((fit_count+monitor_count)*70<=1000,'At most1000 allten seven-anchor records per bounded shard')
    scope='full-data-shard'
    packed=Path(packed).resolve(strict=True);receipt=json.loads((packed/'receipt.json').read_text())
    require(receipt['status']=='all70000_actual_roles_teacher_bound' and receipt['verified_feature_rows']==70000,'Actual completed all70k current role audit required')
    for name,value in receipt['artifacts'].items():require(ref(packed/name)['sha256']==value['sha256'] and (packed/name).stat().st_size==value['bytes'],'Packed role/teacher artifact changed')
    require(sha(TEACHER/'report.json')==TEACHER_REPORT_SHA,'Original full physics teacher changed')
    tr=json.loads((TEACHER/'report.json').read_text())
    for name in ('samples.jsonl','training-only-labels.npz'):require(sha(TEACHER/name)==tr['artifact_sha256'][name],'Teacher bytes differ')
    episodes={'fit':list(range(fit_start,fit_start+fit_count)),'monitor':list(range(monitor_start,monitor_start+monitor_count))}
    all_rows=json.loads((packed/'rows.json').read_text());rows=[dict(r,packed_row=i) for i,r in enumerate(all_rows) if r['episode'] in episodes[r['split']]]
    require(len(rows)==70*(fit_count+monitor_count) and len({r['clip_id'] for r in rows})==len(rows),'Exact allten bounded shard required')
    selected={r['clip_id'] for r in rows};teacher={}
    with (TEACHER/'samples.jsonl').open() as f:
        for line in f:
            r=json.loads(line)
            if r['clip_id'] in selected:teacher[r['clip_id']]=r
    require(set(teacher)==selected,'All selected teacher rows required')
    releases=json.loads(Path(assets).read_text());require(releases['status']=='all10_exact_ACT70_release_files_verified','Exact native ACT70 release assets required')
    models={r['task']:r for r in releases['models']};require(set(models)==set(TASK_REVISIONS),'Allten native motors required')
    qualifier=json.loads(QUALIFIER.read_text());pins={}
    for remote,digest in qualifier['sources_sha256'].items():
        if '/lerobot-sim/' in remote:local=LOCAL/'upstream/lerobot-sim'/remote.split('/lerobot-sim/',1)[1]
        elif remote.endswith('/common.py'):local=ROOT/'outputs/architecture-reassessment/act-goal-cached-gate-v2/common.py'
        elif remote.endswith('/released_deploy_policy.py'):local=ROOT/'outputs/architecture-reassessment/act-goal-cached-gate-v2/payload/released_deploy_policy.py'
        elif remote.endswith('/released_deploy_policy.yml'):local=ROOT/'outputs/architecture-reassessment/act-goal-cached-gate-v2/payload/released_deploy_policy.yml'
        elif remote.endswith('/policy/inference_timing.py'):local=ROOT/'outputs/architecture-reassessment/act-goal-cached-gate-v2/payload/policy/inference_timing.py'
        elif '/.cache/torch/' in remote:local=Path.home()/'.cache/torch/hub/checkpoints'/Path(remote).name
        elif remote.endswith('/multitask_data.py') or remote.endswith('/multitask_cache.py'):local=ROOT/'policy/wam_adaptation'/Path(remote).name
        else:continue
        require(sha(local)==digest,'Qualified ACT70 scientific source changed '+str(local));pins[str(local.resolve())]=digest
    pins[str(Path(__file__).resolve())]=sha(__file__);pins[str((ROOT/'experiments/goal_latent_act_data/balanced_cache_v2.py').resolve())]=PARENT_SOURCE_SHA
    inputs={};dataset={};grouped={}
    for r in rows:grouped.setdefault((r['task'],r['episode']),[]).append(r)
    with np.load(packed/'labels.npz',allow_pickle=False) as z:packed_q=z['qpos'];packed_ids=z['clip_id']
    for (task,ep),values in grouped.items():
        d=data(task);source=d.read_episode(ep);frames=valid_starts(len(source['qpos']))
        require([r['frame'] for r in values]==frames and len(frames)==7,'Seven exact existing anchors required')
        parquet=d.root/f'data/chunk-000/episode_{ep:06d}.parquet';inputs[str(parquet)]=ref(parquet)
        for r in values:
            t=teacher[r['clip_id']];join(r,source,t);r.update(instruction=source['instruction'],teacher_row=t['labels_row'],episode_length=len(source['qpos']))
            require(str(packed_ids[r['packed_row']])==r['clip_id'] and np.array_equal(packed_q[r['packed_row']],source['qpos'][r['frame']].astype(np.float32)),'Actual packed current state differs')
        videos={cam:ref(d.video_path(ep,cam)) for cam in CAMERAS}
        for cam,value in videos.items():require(all(teacher[r['clip_id']]['current_rgb_sources'][cam]['sha256']==value['sha256'] for r in values),'Original camera source bytes differ')
        inputs.update({v['path']:v for v in videos.values()});dataset[task+':'+str(ep)]=dict(parquet=ref(parquet),videos=videos)
    for task,m in models.items():
        for name,v in m['files'].items():p=Path(m['root'])/name;require(p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],'Native model bytes differ');inputs[str(p)]=ref(p)
    rows.sort(key=lambda r:(list(TASK_REVISIONS).index(r['task']),r['episode'],r['frame']))
    require(not Path(output).exists() and Path(output).is_absolute(),'Fresh experiment output required')
    return dict(schema=SCHEMA,status='CPU_verified_full_data_shard_production_pending',scope=scope,output=str(output),rows=rows,episodes=episodes,models=models,dataset=dataset,input_files=list(inputs.values()),source_sha256=pins,
        packed_root=str(packed),packed_receipt=ref(packed/'receipt.json'),packed_artifacts={name:ref(packed/name) for name in receipt['artifacts']},assets_receipt=ref(assets),bell_source_convention=ref(BELL),teacher_report=ref(TEACHER/'report.json'),unit_clarification=ref(UNITS),ancestor_producer_sha256=PARENT_SOURCE_SHA,
        contract=dict(precision='float32',batch_size=1,native_chunk=50,anchors=7,teacher_goal_horizon=10,optimizer_updates=0,seed=1701,source_RGB=[480,640,3],output_cap_bytes=256*1024**2,work_wall_seconds=1800,initial_host_reserve_gib=16,live_host_reserve_gib=8,gpu_free_gib=18,allocator_gib=12),
        interpretation='Immutable bounded shard toward all63000FIT/7000MON; not full coverage until aggregate proves exact70000 unique rows; no robot improvement claim',Bell_historical_unit_lineage_certified=False)

def execute(plan):
    require(plan['schema']==SCHEMA and plan['contract']['native_chunk']==50 and plan['contract']['optimizer_updates']==0,'Experimental native50 contract required')
    for p,h in plan['source_sha256'].items():require(sha(p)==h,'Scientific source/self changed')
    for v in [*plan['input_files'],plan['packed_receipt'],*plan['packed_artifacts'].values(),plan['assets_receipt'],plan['bell_source_convention'],plan['teacher_report'],plan['unit_clarification']]:check_ref(v)
    out=Path(plan['output']);out.mkdir(parents=True,exist_ok=False);(out/'records').mkdir();(out/'normalizers').mkdir();(out/'task-receipts').mkdir()
    report=dict(schema=SCHEMA,status='running',scope=plan['scope'],ancestor_producer_sha256=PARENT_SOURCE_SHA,unit_clarification_sha256=UNITS_SHA,plan_sha256=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest(),optimizer_updates=0,records=[],tasks=[],contract=plan['contract'],Bell_historical_unit_lineage_certified=False)
    started=time.monotonic()
    def expired(*_):raise TimeoutError('Balanced ACT production bounded wall/signal')
    previous={s:signal.getsignal(s) for s in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
    for s in previous:signal.signal(s,expired)
    signal.alarm(plan['contract']['work_wall_seconds'])
    try:
        sys.path.insert(0,str(LOCAL/'upstream/lerobot-sim/src'))
        import torch
        import yaml
        from safetensors.torch import load_model
        from lerobot.policies.act.modeling_act import ACTPolicy
        common=load_module('balanced_act_common',ROOT/'outputs/architecture-reassessment/act-goal-cached-gate-v2/common.py')
        load_module('policy.inference_timing',ROOT/'outputs/architecture-reassessment/act-goal-cached-gate-v2/payload/policy/inference_timing.py')
        official=load_module('balanced_released_act',ROOT/'outputs/architecture-reassessment/act-goal-cached-gate-v2/payload/released_deploy_policy.py')
        from policy.wam_adaptation.multitask_cache import decode_frames
        torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.manual_seed(1701)
        require(torch.__version__=='2.7.1+cu128','Declared proven local Torch runtime required')
        require(sha(inspect.getfile(ACTPolicy))=='48ba3a2620198154b7b63948d14ebf76fa50da53efba194aedd76d9392c955ae','Exact ACT70 native implementation required')
        torch.cuda.set_per_process_memory_fraction(12*1024**3/torch.cuda.get_device_properties(0).total_memory)
        settings=yaml.safe_load((ROOT/'outputs/architecture-reassessment/act-goal-cached-gate-v2/payload/released_deploy_policy.yml').read_text())
        roles=np.load(Path(plan['packed_root'])/'roles.npy',mmap_mode='r',allow_pickle=False)
        with np.load(Path(plan['packed_root'])/'labels.npz',allow_pickle=False) as z:teacher={k:z[k] for k in z.files}
        for task in TASK_REVISIONS:
            task_started=time.monotonic();model=plan['models'][task];base=official.get_model({**settings,'checkpoint_path':model['root'],'device':'cuda','pytorch_device':'cuda'})
            load_model(base,str(Path(model['root'])/'model.safetensors'),strict=True,device='cuda:0');base.requires_grad_(False).eval()
            require(base.config.chunk_size==base.config.n_action_steps==50 and next(base.parameters()).dtype==torch.float32,'Unchanged released FP32 native50 required')
            frozen=common.state_sha(base);norm={k:common.state_sha(getattr(base,k)) for k in ('normalize_inputs','normalize_targets','unnormalize_outputs')}
            nd=base.normalize_targets.state_dict();mean=nd['buffer_action.mean'].cpu().numpy();std=nd['buffer_action.std'].cpu().numpy()
            np.savez(out/'normalizers'/(task+'.npz'),action_mean=mean,action_std=std)
            latent=[0]
            def zero_latent(module,args):require(torch.count_nonzero(args[0]).item()==0,'Released inference zero latent required');latent[0]+=1
            hook=base.model.encoder_latent_input_proj.register_forward_pre_hook(zero_latent)
            task_rows=[r for r in plan['rows'] if r['task']==task]
            for ep in sorted({r['episode'] for r in task_rows}):
                d=data(task);source=d.read_episode(ep);selected=[r for r in task_rows if r['episode']==ep];frames=[r['frame'] for r in selected]
                videos=plan['dataset'][task+':'+str(ep)]['videos'];images={cam:decode_frames(videos[cam]['path'],frames,timeout=60) for cam in CAMERAS}
                require(all(images[cam][f].shape==(480,640,3) and images[cam][f].dtype==np.uint8 for cam in CAMERAS for f in frames),'Original three-camera640×480 required')
                for r in selected:
                    f=r['frame'];tick=time.monotonic();batch={'observation.state':torch.from_numpy(source['qpos'][f].astype(np.float32))[None].cuda()}
                    for key in base.act_image_keys:
                        value=torch.from_numpy(images[base.image_key_map[key]][f].copy()).float().cuda()[None].permute(0,3,1,2).contiguous()
                        batch[key]=value/255. if value.max()>1.5 else value
                    require(set(batch)=={'observation.state',*base.act_image_keys},'Only current RGB/proprio can enter frozen ACT')
                    with torch.no_grad():
                        normalized=dict(base.normalize_inputs(batch));normalized['observation.images']=[normalized[k] for k in base.config.image_features];prediction=base.model(normalized)[0]
                        require(prediction.shape==(1,50,14) and prediction.dtype==torch.float32 and torch.isfinite(prediction).all(),'Finite FP32 native50 prediction required')
                        physical=base.unnormalize_outputs({'action':prediction})['action'];target,valid=labels50(source['commands'],f)
                        actual=torch.from_numpy(target[valid].copy()).cuda()[None];nt=base.normalize_targets({'action':actual})['action'];roundtrip=base.unnormalize_outputs({'action':nt})['action']
                        require(torch.allclose(roundtrip,actual,atol=2e-6,rtol=1e-6),'Actual physical-target normalization roundtrip failed')
                        padded=torch.zeros((1,50,14),device='cuda',dtype=torch.float32);padded[:,torch.from_numpy(valid).cuda()]=nt
                    arrays=dict(normalized_base=prediction[0].cpu().numpy(),physical_base=physical[0].cpu().numpy(),physical_actions=target,normalized_actions=padded[0].cpu().numpy(),valid=valid,canonical_current_qpos14=source['qpos'][f].astype(np.float32),role_features=np.array(roles[r['packed_row']],copy=True))
                    arrays.update({'teacher_'+k:v[r['packed_row']] for k,v in teacher.items() if k not in ('clip_id','qpos','fit','task_index')})
                    require(np.isfinite(arrays['physical_base']).all() and arrays['role_features'].shape==(3,4096),'Finite physical parent and current role features required')
                    validate_payload(arrays)
                    name=task+'-'+str(ep)+'-'+str(f)+'.npz';p=out/'records'/name
                    with p.open('xb') as stream:np.savez(stream,**arrays);stream.flush();os.fsync(stream.fileno())
                    with np.load(p,allow_pickle=False) as z:require(set(z.files)==set(arrays) and all(same_array_bytes(z[k],v) for k,v in arrays.items()),'Fsynced cache byte/shape/dtype reload differs')
                    report['records'].append(dict(r,file=ref(p),valid_rows=int(valid.sum()),source_video_sha256={c:v['sha256'] for c,v in videos.items()},pixels_sha256={c:hashlib.sha256(images[c][f].tobytes()).hexdigest() for c in CAMERAS},normalizers_sha256=norm,camera_order=list(base.config.image_features),forward_seconds=time.monotonic()-tick,action_units='native ACT arm12 rad and task-release grip2; targets grip2 physicalm; grip normalized labels UNQUALIFIED; arm12 loss only',teacher_horizon=10))
                    require(sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<=plan['contract']['output_cap_bytes'],'Bounded experiment output cap')
                    write(out/'receipt.json',report)
                del images
            hook.remove();require(latent[0]==len(task_rows) and common.state_sha(base)==frozen,'Frozen ACT parameters/buffers or exact forward count changed')
            task_receipt=dict(schema=SCHEMA,status='task_frozen_records_complete_pending_original_exit0_publication',task=task,records=[r for r in report['records'] if r['task']==task],latent_calls=latent[0],frozen_unchanged=True,normalizer=ref(out/'normalizers'/(task+'.npz')),elapsed_seconds=time.monotonic()-task_started,unit_clarification_sha256=UNITS_SHA)
            write(out/'task-receipts'/(task+'.json'),task_receipt)
            report['tasks'].append(dict(task=task,records=len(task_rows),latent_calls=latent[0],frozen_unchanged=True,normalizer=ref(out/'normalizers'/(task+'.npz')),task_receipt=ref(out/'task-receipts'/(task+'.json')),elapsed_seconds=time.monotonic()-task_started))
            del base,batch,normalized,prediction,physical,padded;gc.collect();torch.cuda.empty_cache()
        require(len(report['records'])==len(plan['rows']) and len(report['tasks'])==10,'Exact balanced allten cache required')
        report.update(status='full_data_shard_complete_pending_original_exit0_publication',fit=sum(r['split']=='fit' for r in report['records']),monitor=sum(r['split']=='monitor' for r in report['records']),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),device=torch.cuda.get_device_name(0),native_release_sources_unchanged=True,robot_success_claim=False)
    except BaseException as error:report.update(status='failed_preserved_no_retry',failure=dict(type=type(error).__name__,message=str(error)));raise
    finally:
        signal.alarm(0)
        for s,h in previous.items():signal.signal(s,h)
        report['elapsed_seconds']=time.monotonic()-started;write(out/'receipt.json',report)
    return report

def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--packed',type=Path);p.add_argument('--assets',type=Path);p.add_argument('--fit-start',type=int);p.add_argument('--fit-count',type=int);p.add_argument('--monitor-start',type=int);p.add_argument('--monitor-count',type=int);p.add_argument('--output',type=Path);p.add_argument('--write-plan',type=Path);p.add_argument('--plan',type=Path);p.add_argument('--plan-sha256');p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.write_plan:
        require(not a.execute and a.packed and a.assets and a.output and all(x is not None for x in (a.fit_start,a.fit_count,a.monitor_start,a.monitor_count)),'Explicit CPU plan inputs required');plan=prepare(a.packed,a.assets,a.fit_start,a.fit_count,a.monitor_start,a.monitor_count,a.output.resolve());write(a.write_plan,plan);print(json.dumps(dict(status=plan['status'],rows=len(plan['rows']),plan_sha256=sha(a.write_plan),CUDA_initialized=False)));return
    require(a.plan and sha(a.plan)==a.plan_sha256,'Exact bound experimental plan required');plan=json.loads(a.plan.read_text())
    if a.execute:result=execute(plan);print(json.dumps(dict(status=result['status'],records=len(result['records']),seconds=result['elapsed_seconds'])));
    else:print(json.dumps(dict(status='CPU_balanced_plan_bound_no_GPU',rows=len(plan['rows']))))

if __name__=='__main__':main()
