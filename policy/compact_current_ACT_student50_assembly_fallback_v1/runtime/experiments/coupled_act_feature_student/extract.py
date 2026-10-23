"""Root-admitted CPU-only backbone extraction; no full ACT/Qwen/world loads."""
import argparse,gc,hashlib,json,os,subprocess,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .contracts import PLAN,CAMERAS,require,ref,verify,validate_feature_arrays


def resource_gate(start,wall,cap):
    require(time.monotonic()-start<wall,'Prospective CPU wall cap exceeded')
    parents={};rss={}
    for p in Path('/proc').iterdir():
        if not p.name.isdecimal():continue
        try:
            raw=(p/'stat').read_text();fields=raw[raw.rfind(')')+2:].split();parents[int(p.name)]=int(fields[1])
            rss[int(p.name)]=int((p/'statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    owned={os.getpid()}
    while True:
        more={pid for pid,parent in parents.items() if parent in owned}
        if more<=owned:break
        owned|=more
    total=sum(rss.get(pid,0) for pid in owned);require(total<=cap,'Aggregate CPU tree RSS cap exceeded')
    return total


def backbone_and_normalizers(binding):
    import torch,torchvision
    from safetensors import safe_open
    from torchvision.models._utils import IntermediateLayerGetter
    from torchvision.ops.misc import FrozenBatchNorm2d
    c=binding['config'];require(c['vision_backbone']=='resnet18' and c['normalization_mapping']['VISUAL']=='MEAN_STD','Actual ResNet18/meanstd')
    original=torchvision.models.resnet18(weights=None,replace_stride_with_dilation=[False,False,c['replace_final_stride_with_dilation']],norm_layer=FrozenBatchNorm2d)
    backbone=IntermediateLayerGetter(original,return_layers={'layer4':'feature_map'}).eval().requires_grad_(False)
    state={};normalizers={}
    with safe_open(verify(binding['files']['model.safetensors']),framework='pt',device='cpu') as archive:
        for key in archive.keys():
            if key.startswith('model.backbone.'):state[key.removeprefix('model.backbone.')]=archive.get_tensor(key)
        for cam in CAMERAS:
            prefix='normalize_inputs.buffer_observation_images_'+cam
            normalizers[cam]={name:archive.get_tensor(prefix+'.'+name) for name in ('mean','std')}
            require(normalizers[cam]['mean'].shape==normalizers[cam]['std'].shape==(3,1,1)
                and bool((normalizers[cam]['std']>0).all()),'Actual native image mean/std shape')
    backbone.load_state_dict(state,strict=True);del state,original
    return backbone,normalizers


def decode_current(row):
    import numpy as np
    import pyarrow.parquet as pq
    cache=verify(row['file'])
    with np.load(cache,allow_pickle=False) as z:
        values={k:z[k].copy() for k in ('canonical_current_qpos14','role_features','teacher_eef_now')}
    require(ref(row['parquet']['path'])['sha256']==row['parquet']['sha256'],'Original parquet changed')
    table=pq.read_table(row['parquet']['path'],columns=['frame_index','timestamp'])
    frames=np.array(table['frame_index'].to_pylist());times=np.array(table['timestamp'].to_pylist())
    require(np.array_equal(frames,np.arange(len(frames))) and np.allclose(times,frames/25,atol=1e-5)
        and row['frame']<len(frames),'Actual canonical frame/timestamp25Hz alignment')
    images={}
    for cam in CAMERAS:
        video=row['videos'][cam];require(ref(video['path'])['sha256']==video['sha256'],'Exact source video changed')
        argv=['ffmpeg','-v','error','-threads','1','-filter_threads','1','-i',video['path'],'-vf',f"select=eq(n\\,{row['frame']})",'-frames:v','1','-vsync','0','-f','rawvideo','-pix_fmt','rgb24','-']
        raw=subprocess.check_output(argv,timeout=60)
        require(len(raw)==480*640*3 and hashlib.sha256(raw).hexdigest()==row['pixels_sha256'][cam],
            'Decoded current RGB differs from accepted original pixel hash')
        images[cam]=np.frombuffer(raw,np.uint8).reshape(480,640,3).copy()
    return values,images


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for key in ('plan','plan-sha256','admission','admission-sha256','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--check-only',action='store_true');a=p.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only extraction must hide CUDA before imports')
    plan_ref=ref(a.plan);require(plan_ref['sha256']==a.plan_sha256,'Exact declared1000 plan')
    plan=json.loads(Path(a.plan).read_text());require(plan['schema']==PLAN and len(plan['rows'])==1000,'Fixed balanced1000 only')
    admission_ref=ref(a.admission);require(admission_ref['sha256']==a.admission_sha256,'Exact root CPU extraction admission')
    admission=json.loads(Path(a.admission).read_text())
    require(admission['schema']=='root_authorized_conditional_ACT_feature_student_CPU_extraction_v1'
        and admission['execution_authorized'] is True and admission['native_teacher_benefit_qualified'] is True
        and admission['plan']==plan_ref,'Separate actual native teacher benefit/root CPU admission required')
    verify(admission['matched_native_teacher_benefit'])
    require(admission['CPU_tree_cap_bytes']==3*(1<<30) and admission['threads']==4
        and admission['output_cap_bytes']==1<<30 and admission['wall_seconds']==1800,'Fixed CPU3GiB/thread4/disk1GiB/wall30min cap')
    root=Path(__file__).resolve().parents[2]
    for relative,digest in plan['source_sha256'].items():require(ref(root/relative)['sha256']==digest,'Exact prepared extraction source changed')
    if a.check_only:print(json.dumps(dict(status='CPU_source_plan_and_root_extraction_admission_valid',GPU_queries=0,model_loads=0,images_decoded=0)));return
    import numpy as np,torch
    from safetensors import safe_open
    from .hook import CurrentACTFeatures
    from experiments.goal_latent_act_full_data.correction_prefix import eef_features
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    start=time.monotonic();peak=resource_gate(start,1800,3*(1<<30));out=Path(a.output);require(not out.exists(),'Fresh feature output');out.mkdir(parents=True)
    with np.load(verify(plan['source_current_predictions']),allow_pickle=False) as z:predicted=z['predicted_goal'].copy()
    require(predicted.shape==(70000,2,4,4) and predicted.dtype==np.float32,'Existing exact current-only predicted-goal50')
    with safe_open(verify(plan['teacher_weights']),framework='pt',device='cpu') as archive:
        role_weight=archive.get_tensor('roles.weight');role_bias=archive.get_tensor('roles.bias')
    arrays={};completed=[];teacher_stats=plan['teacher_statistics'];model_count=0
    for task in plan['tasks']:
        binding=plan['act'][task];backbone,normalizers=backbone_and_normalizers(binding);model_count+=1
        hook=CurrentACTFeatures(backbone,binding['camera_order']);ids=[i for i,r in enumerate(plan['rows']) if r['task']==task]
        with ThreadPoolExecutor(max_workers=2) as pool:
            for first in range(0,len(ids),7):
                selected=ids[first:first+7];rows=[plan['rows'][i] for i in selected];decoded=list(pool.map(decode_current,rows))
                identity=dict(task=task,episode='declared_episode_batch',action=first,input_sha256=hashlib.sha256(json.dumps([dict(clip_id=r['clip_id'],pixels=r['pixels_sha256']) for r in rows],sort_keys=True).encode()).hexdigest(),batch_size=len(rows));hook.begin(**identity)
                with torch.inference_mode():
                    for cam in CAMERAS:
                        value=torch.from_numpy(np.stack([images[cam] for _,images in decoded])).permute(0,3,1,2).contiguous().float()
                        divisor=torch.where(value.amax((1,2,3),keepdim=True)>1.5,255.,1.)
                        value=value/divisor
                        mean,std=normalizers[cam]['mean'],normalizers[cam]['std']
                        for name,tensor in (('mean',mean),('std',std)):
                            digest=hashlib.sha256(tensor.numpy().tobytes()).hexdigest();key='buffer_observation_images_'+cam+'.'+name
                            require(all(r['normalizers_sha256']['normalize_inputs'][key]==digest for r in rows),'Native normalization buffer changed')
                        backbone((value-mean)/(std+1e-8))
                    spatial,global_visual=hook.finish(identity)
                    roles=torch.from_numpy(np.stack([v['role_features'] for v,_ in decoded])).float()
                    target_roles=torch.tanh(torch.nn.functional.linear(roles,role_weight,role_bias)).flatten(1)
                    now=torch.from_numpy(np.stack([v['teacher_eef_now'] for v,_ in decoded])).float()
                    goal=torch.from_numpy(predicted[[r['source_row_index'] for r in rows]])
                    delta=(eef_features(goal)-eef_features(now))/torch.tensor(teacher_stats['eef_scale'])
                    qpos=(torch.from_numpy(np.stack([v['canonical_current_qpos14'] for v,_ in decoded])).float()-torch.tensor(teacher_stats['qpos_mean']))/torch.tensor(teacher_stats['qpos_scale'])
                for local,index in enumerate(selected):
                    arrays[index]=dict(visual=spatial[local].numpy(),visual_global=global_visual[local].numpy(),
                        qpos_features=qpos[local].numpy(),projected_roles=target_roles[local].numpy(),predicted_goal_delta=delta[local].numpy())
                    completed.append(dict(source_row_index=rows[local]['source_row_index'],clip_id=rows[local]['clip_id'],pixels_sha256=rows[local]['pixels_sha256'],normalizers_sha256=rows[local]['normalizers_sha256']))
                peak=max(peak,resource_gate(start,1800,3*(1<<30)))
        hook.close();del backbone,normalizers;gc.collect()
    require(len(arrays)==1000,'Only complete fixed1000 cache can publish')
    arrays={key:np.stack([arrays[i][key] for i in range(1000)]).astype(np.float32) for key in arrays[0]}
    validate_feature_arrays(arrays,plan['rows']);np.savez(out/'arrays.npz',**arrays)
    require((out/'arrays.npz').stat().st_size<=1<<30,'Owned feature output cap exceeded')
    require(not torch.cuda.is_initialized(),'CPU extractor initialized CUDA')
    report=dict(schema='actual_balanced1000_current_ACT_feature_CPU_cache_v1',status='all1000_declared_current_features_complete',plan=plan_ref,
        admission=admission_ref,arrays=ref(out/'arrays.npz'),records=completed,elapsed_seconds=time.monotonic()-start,CPU_tree_peak_bytes=peak,
        backbone_models_loaded=model_count,full_ACT_models_loaded=0,Qwen_models_loaded=0,teacher_full_models_loaded=0,
        teacher_projection_weights_loaded=True,CPU_vs_GPU_feature_parity_qualified=False,parent_native_action_equivalence_pending=True,
        GPU_queries=0,GPU_launches=0,worlds_constructed=0,CUDA_initialized=False,robot_quality_promoted=False)
    (out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(ref(out/'receipt.json')))


if __name__=='__main__':main()
