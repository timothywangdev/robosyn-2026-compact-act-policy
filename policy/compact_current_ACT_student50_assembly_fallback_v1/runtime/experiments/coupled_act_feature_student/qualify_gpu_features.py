"""Root-owned prospective ten-task ACT CPU/GPU feature parity, no training."""
import argparse,json,os,time
from pathlib import Path
from .contracts import require,ref,verify,CAMERAS

GPU_UUID='GPU-8e45cdd0-bbd6-9b4f-de12-e7125a11bed9'


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for key in ('plan','plan-sha256','constructor-receipt','constructor-receipt-sha256','cache','cache-receipt-sha256','admission','admission-sha256','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--check-only',action='store_true');a=p.parse_args()
    plan_ref=ref(a.plan);require(plan_ref['sha256']==a.plan_sha256,'Exact declared feature plan')
    plan=json.loads(Path(a.plan).read_text());constructor_ref=ref(a.constructor_receipt)
    require(constructor_ref['sha256']==a.constructor_receipt_sha256,'Exact actual constructor receipt')
    constructor=json.loads(Path(a.constructor_receipt).read_text());require(constructor['status']=='ten_actual_task_backbones_and_exact_current_pixels_PASS' and constructor['plan']==plan_ref,'Actual same-source ten CPU backbones needed')
    admission_ref=ref(a.admission);require(admission_ref['sha256']==a.admission_sha256,'Root GPU parity admission')
    admission=json.loads(Path(a.admission).read_text())
    require(admission['schema']=='root_authorized_ACT_feature_student_ten_task_GPU_feature_parity_v1' and admission['execution_authorized'] is True
        and admission['plan']==plan_ref and admission['constructor']==constructor_ref and admission['GPU_UUID']==GPU_UUID,'Exact independent root GPU parity authority')
    require(admission['GPU_Torch_cap_bytes']==2*(1<<30) and admission['whole_GPU_cap_bytes']==4*(1<<30)
        and admission['CPU_tree_cap_bytes']==3*(1<<30) and admission['wall_seconds']==180,'Fixed Torch2/GPU4/host3GiB/180sec')
    verify(admission['genuine_previous_GPU_wait_and_drain'])
    for relative,digest in admission['GPU_source_sha256'].items():require(ref(Path(__file__).resolve().parents[2]/relative)['sha256']==digest,'Pinned GPU parity source differs')
    cache_ref=ref(Path(a.cache)/'receipt.json');require(cache_ref['sha256']==a.cache_receipt_sha256 and admission['cache']==cache_ref,'Exact actual complete1000 CPU cache')
    if a.check_only:print(json.dumps(dict(status='GPU_parity_source_actual_CPU_cache_and_root_admission_valid',models_loaded=0,GPU_queries=0,GPU_launches=0)));return
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==GPU_UUID,'Only explicitly admitted local device may initialize CUDA')
    import gc,numpy as np,torch
    from .data import load,moments
    from .extract import backbone_and_normalizers,decode_current,resource_gate
    from .hook import CurrentACTFeatures
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction((2*(1<<30))/torch.cuda.get_device_properties(0).total_memory,0)
    # The prior native ACT source never altered its process-default flags.
    # Record this literal default; root must bind its expected ACT flag pair.
    flags=dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
    require(flags==admission['native_ACT_precision_flags'],'Observed native ACT precision defaults differ from root binding')
    full_plan,arrays,masks=load(a.cache,cache_ref);require(full_plan==plan,'Complete CPU cache uses exact selected plan')
    statistics=moments(arrays,masks['optimizer'])
    with np.load(verify(constructor['features']),allow_pickle=False) as z:reference={k:z[k].copy() for k in z.files}
    start=time.monotonic();out=Path(a.output);require(not out.exists(),'Fresh GPU parity output');out.mkdir(parents=True)
    metrics=[];GPU_features=[];peak=0
    for index,task in enumerate(plan['tasks']):
        row=next(r for r in plan['rows'] if r['task']==task and r['partition']=='optimizer')
        require(row['clip_id']==constructor['records'][index]['clip_id'],'Fixed constructor current identity differs')
        backbone,normalizers=backbone_and_normalizers(plan['act'][task]);backbone=backbone.to('cuda:0')
        values,images=decode_current(row);hook=CurrentACTFeatures(backbone,plan['act'][task]['camera_order'])
        identity=dict(task=task,episode=row['episode'],action=row['frame'],input_sha256=row['clip_id'],batch_size=1);hook.begin(**identity)
        with torch.inference_mode():
            for cam in CAMERAS:
                value=torch.from_numpy(images[cam]).permute(2,0,1).contiguous()[None].float();value=value/torch.where(value.amax((1,2,3),keepdim=True)>1.5,255.,1.)
                normalized=(value-normalizers[cam]['mean'])/(normalizers[cam]['std']+1e-8)
                import hashlib
                require(hashlib.sha256(normalized.numpy().tobytes()).hexdigest()==constructor['records'][index]['normalized_current_input_sha256'][cam],'Exact normalized ACT input differs before GPU')
                backbone(normalized.to('cuda:0'))
            spatial,average=hook.finish(identity)
        torch.cuda.synchronize();spatial=spatial[0].cpu().numpy();average=average[0].cpu().numpy();CPU=reference['visual'][index]
        delta=spatial-CPU;relative=float(np.linalg.norm(delta)/max(np.linalg.norm(CPU),1e-12))
        normalized_delta=delta/statistics['visual_scale'][:,:,None,None]
        rms=float(np.sqrt(np.mean(normalized_delta**2)));maximum=float(np.abs(normalized_delta).max())
        global_delta=(average-reference['visual_global'][index])/statistics['global_scale']
        accepted=relative<=.002 and rms<=.01 and maximum<=.1 and float(np.sqrt(np.mean(global_delta**2)))<=.01
        metrics.append(dict(task=task,clip_id=row['clip_id'],raw_relative_L2=relative,normalized_spatial_RMS=rms,normalized_spatial_maxabs=maximum,
            normalized_global_RMS=float(np.sqrt(np.mean(global_delta**2))),numeric_feature_gate_PASS=accepted))
        GPU_features.append(dict(visual=spatial,visual_global=average));peak=max(peak,resource_gate(start,180,3*(1<<30)))
        hook.close();del backbone,normalizers,value,images,values,normalized;gc.collect();torch.cuda.empty_cache()
    np.savez(out/'GPU-features.npz',**{key:np.stack([x[key] for x in GPU_features]) for key in GPU_features[0]})
    good=all(x['numeric_feature_gate_PASS'] for x in metrics)
    report=dict(schema='actual_ten_task_CPU_GPU_ACT_feature_student_numeric_parity_v1',status='all10_exact_current_CPU_GPU_ACT_feature_numeric_parity_PASS' if good else 'CPU_GPU_feature_numeric_parity_FAILED',
        plan=plan_ref,constructor=constructor_ref,cache=cache_ref,admission=admission_ref,GPU_features=ref(out/'GPU-features.npz'),metrics=metrics,
        native_ACT_precision_flags=flags,fixed_thresholds=dict(raw_relative_L2=.002,normalized_spatial_RMS=.01,normalized_spatial_maxabs=.1,normalized_global_RMS=.01),
        CPU_GPU_features_bitexact=False,trained_student_and_native_hand_gate_parity_still_required=True,torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        sampled_CPU_tree_peak_bytes=peak,elapsed_seconds=time.monotonic()-start,GPU_queries=1,GPU_launches=1,optimizer_updates=0,full_ACT_models_loaded=0,Qwen_models_loaded=0,worlds_constructed=0,robot_quality_promoted=False)
    (out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');require(good,'Declared actual CPU/GPU numeric feature screen failed')


if __name__=='__main__':main()
