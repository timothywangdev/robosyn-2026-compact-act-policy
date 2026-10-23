"""Root-admitted fixed10 CPU constructor/current-pixel pilot; no full models."""
import argparse,gc,hashlib,json,os,time
from pathlib import Path
from .contracts import CAMERAS,PLAN,require,ref,verify


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for key in ('plan','plan-sha256','admission','admission-sha256','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--check-only',action='store_true');a=p.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU constructor pilot must hide CUDA before imports')
    plan_ref=ref(a.plan);require(plan_ref['sha256']==a.plan_sha256,'Fixed1000 source plan differs')
    plan=json.loads(Path(a.plan).read_text());require(plan['schema']==PLAN and len(plan['rows'])==1000,'Declared1000 source plan')
    admission_ref=ref(a.admission);require(admission_ref['sha256']==a.admission_sha256,'Exact CPU constructor admission')
    admission=json.loads(Path(a.admission).read_text())
    require(admission['schema']=='root_authorized_ACT_feature_student_CPU_constructor10_v1'
        and admission['execution_authorized'] is True and admission['plan']==plan_ref,'Independent root constructor10 admission')
    verify(admission['matched_native_teacher_benefit'])
    require(admission['CPU_tree_cap_bytes']==3*(1<<30) and admission['threads']==4
        and admission['output_cap_bytes']==1<<30 and admission['wall_seconds']==300,'CPU3GiB/thread4/disk1GiB/wall5min')
    root=Path(__file__).resolve().parents[2]
    for relative,digest in plan['source_sha256'].items():require(ref(root/relative)['sha256']==digest,'Frozen student source differs')
    selected=[next(r for r in plan['rows'] if r['task']==task and r['partition']=='optimizer') for task in plan['tasks']]
    require(len(selected)==10,'Exactly one prospective first optimizer anchor per task')
    if a.check_only:print(json.dumps(dict(status='constructor10_source_and_root_CPU_admission_valid',model_loads=0,images_decoded=0)));return
    import numpy as np,torch
    from .extract import backbone_and_normalizers,decode_current,resource_gate
    from .hook import CurrentACTFeatures
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    start=time.monotonic();peak=resource_gate(start,300,3*(1<<30));out=Path(a.output);require(not out.exists(),'Fresh constructor qualification output');out.mkdir(parents=True)
    records=[];spatial=[];global_visual=[]
    for row in selected:
        backbone,normalizers=backbone_and_normalizers(plan['act'][row['task']])
        hook=CurrentACTFeatures(backbone,plan['act'][row['task']]['camera_order'])
        values,images=decode_current(row)
        identity=dict(task=row['task'],episode=row['episode'],action=row['frame'],input_sha256=hashlib.sha256(json.dumps(row['pixels_sha256'],sort_keys=True).encode()).hexdigest(),batch_size=1)
        hook.begin(**identity);input_hashes={};sample_start=time.monotonic()
        with torch.inference_mode():
            for cam in CAMERAS:
                value=torch.from_numpy(images[cam]).permute(2,0,1).contiguous()[None].float()
                value=value/torch.where(value.amax((1,2,3),keepdim=True)>1.5,255.,1.)
                mean,std=normalizers[cam]['mean'],normalizers[cam]['std']
                for name,tensor in (('mean',mean),('std',std)):
                    key='buffer_observation_images_'+cam+'.'+name
                    require(hashlib.sha256(tensor.numpy().tobytes()).hexdigest()==row['normalizers_sha256']['normalize_inputs'][key],'Exact native image normalization differs')
                normalized=(value-mean)/(std+1e-8)
                input_hashes[cam]=hashlib.sha256(normalized.numpy().tobytes()).hexdigest()
                backbone(normalized);peak=max(peak,resource_gate(start,300,3*(1<<30)))
            pooled,average=hook.finish(identity)
        require(pooled.shape==(1,3,512,2,3) and average.shape==(1,3,512),'Actual current native feature shape')
        spatial.append(pooled[0].numpy());global_visual.append(average[0].numpy())
        records.append(dict(clip_id=row['clip_id'],source_row_index=row['source_row_index'],task=row['task'],episode=row['episode'],frame=row['frame'],
            current_timestamp_seconds=row['current_timestamp_seconds'],pixel_SHA_reverified=True,parquet_video_SHA_reverified=True,
            normalized_current_input_sha256=input_hashes,three_backbone_forward_seconds=time.monotonic()-sample_start))
        hook.close();del backbone,normalizers,images,values,normalized,value,pooled,average;gc.collect()
    np.savez(out/'features.npz',visual=np.stack(spatial),visual_global=np.stack(global_visual))
    require(not torch.cuda.is_initialized(),'CPU constructor initialized CUDA')
    report=dict(schema='actual_ACT_feature_student_CPU_constructor10_v1',status='ten_actual_task_backbones_and_exact_current_pixels_PASS',
        plan=plan_ref,admission=admission_ref,records=records,features=ref(out/'features.npz'),elapsed_seconds=time.monotonic()-start,
        sampled_CPU_tree_peak_bytes=peak,CPU_peak_between_samples_unproven=True,backbone_models_loaded=10,
        full_ACT_models_loaded=0,Qwen_models_loaded=0,RGB_frames_decoded=30,teacher_targets_materialized=0,
        optimizer_updates=0,GPU_queries=0,GPU_launches=0,worlds_constructed=0,CUDA_initialized=False,
        CPU_vs_GPU_feature_parity_qualified=False,B7_extraction_peak_not_qualified=True,robot_quality_promoted=False)
    target=out/'receipt.json';target.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(ref(target)))


if __name__=='__main__':main()
