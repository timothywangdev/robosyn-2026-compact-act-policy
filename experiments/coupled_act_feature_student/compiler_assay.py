"""Bounded root-owned all10 actual Student inference and Inductor assay."""
import argparse
import contextlib
import json
import os
import time
from pathlib import Path
from .contracts import require,ref,verify

GPU_UUID='GPU-8e45cdd0-bbd6-9b4f-de12-e7125a11bed9'
CANDIDATES=('distill_visual','distill_qpos_control','policy_aware_visual','policy_aware_qpos_control')
FLAGS=dict(matmul_allow_tf32=False,cudnn_allow_tf32=True,cudnn_enabled=True)
CORRECTION_FLAGS=dict(matmul_allow_tf32=False,cudnn_allow_tf32=False,cudnn_enabled=True)


@contextlib.contextmanager
def correction_precision(torch):
    previous=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=previous


def fixed_indices(plan,constructor,parity):
    require(constructor['plan']==parity['plan'] and len(constructor['records'])==len(plan['tasks'])==10,'Same ten current constructor anchors')
    ids=[]
    for task,record,metric in zip(plan['tasks'],constructor['records'],parity['metrics']):
        row=next((i for i,r in enumerate(plan['rows']) if r['task']==task and r['partition']=='optimizer'),None)
        require(row is not None and record['clip_id']==metric['clip_id']==plan['rows'][row]['clip_id'] and metric['task']==task,
            'Exact constructor task/current frame ordering differs')
        ids.append(row)
    require(len(set(ids))==10,'One distinct fixed current input per task')
    return ids


def numeric_output_difference(left,right):
    import torch
    require(set(left)==set(right),'Complete actual output fields differ')
    result={}
    for key,value in left.items():
        other=right[key];require(value.shape==other.shape and value.dtype==other.dtype,'Output dtype/shape differs')
        result[key]=dict(bitexact=torch.equal(value,other),maxabs=(0 if value.dtype==torch.bool else float((value-other).abs().max())))
    return result


def validate_training(training,parity):
    require(training['schema']=='actual_paired_ACT_feature_student_distill_and_policy_aware_GPU_v2'
        and training['status']=='fixed30_four_candidates_complete_exploratory_no_robot_promotion'
        and training['robot_quality_promoted'] is False and training['cache']==parity['cache'],'Literal complete actual four-candidate training')
    require(tuple(training['results'])==CANDIDATES,'Predeclared four-candidate order')
    require(all(training['results'][name]['optimizer_updates']==210 for name in CANDIDATES),'Fixed210 updates per candidate')


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for key in ('training-receipt','training-receipt-sha256','parity-receipt','parity-receipt-sha256','admission','admission-sha256','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--check-only',action='store_true');a=p.parse_args()
    training_ref=ref(a.training_receipt);parity_ref=ref(a.parity_receipt);admission_ref=ref(a.admission)
    require(training_ref['sha256']==a.training_receipt_sha256 and parity_ref['sha256']==a.parity_receipt_sha256
        and admission_ref['sha256']==a.admission_sha256,'Exact actual assay inputs')
    training=json.loads(verify(training_ref).read_text());parity=json.loads(verify(parity_ref).read_text());admission=json.loads(verify(admission_ref).read_text())
    require(admission['schema']=='root_authorized_ACT_feature_student_all10_actual_inference_compile_v1'
        and admission['execution_authorized'] is True and admission['training_receipt']==training_ref
        and admission['parity_receipt']==parity_ref and admission['GPU_UUID']==GPU_UUID,'Separate root actual inference/compile admission')
    require(admission['GPU_Torch_cap_bytes']==2*(1<<30) and admission['whole_GPU_cap_bytes']==4*(1<<30)
        and admission['CPU_tree_cap_bytes']==3*(1<<30) and admission['wall_seconds']==180,'Fixed inference2/GPU4/host3GiB/wall180')
    require(admission['native_inference_flags']==FLAGS,'Actual native default cuDNN inference contract')
    verify(admission['genuine_previous_GPU_wait_and_drain'])
    for relative,digest in admission['source_sha256'].items():require(ref(Path(__file__).resolve().parents[2]/relative)['sha256']==digest,'Bound assay source differs')
    require(parity['status']=='all10_exact_current_CPU_GPU_ACT_feature_numeric_parity_PASS','Genuine ten-task feature parity required')
    validate_training(training,parity)
    cache=json.loads(verify(training['cache']).read_text());plan=json.loads(verify(cache['plan']).read_text())
    protocol=json.loads(verify(training['protocol']).read_text())
    require(cache['status']=='all1000_declared_current_features_complete' and protocol['feature_plan']==cache['plan']==parity['plan'],'Same complete trained current cache')
    constructor=json.loads(verify(parity['constructor']).read_text());ids=fixed_indices(plan,constructor,parity)
    for candidate in CANDIDATES:verify(training['results'][candidate]['weights'])
    if a.check_only:
        require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Hide CUDA before actual metadata check')
        print(json.dumps(dict(status='actual_four_weights_all10_current_source_and_default_inference_contract_valid',models_loaded=0,GPU_queries=0,GPU_launches=0,optimizer_updates=0)));return
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==GPU_UUID,'Exact local GPU authority required')
    import numpy as np,torch
    from safetensors.torch import load_file
    from torch._dynamo.utils import counters
    from .data import load,batch,moments
    from .model import Student
    from .compiler import qualify
    from .projected import projected_forward
    from .policy_aware_data import fields
    from .extract import resource_gate
    from experiments.coupled_goal_act_all14.model import CoupledGoalACT
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction((2*(1<<30))/torch.cuda.get_device_properties(0).total_memory,0)
    flags=dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,cudnn_enabled=torch.backends.cudnn.enabled)
    require(flags==FLAGS,'Inference default flags must remain unchanged; fit-only cuDNN disable is not inherited')
    start=time.monotonic();out=Path(a.output);require(not out.exists(),'Fresh original assay output');out.mkdir(parents=True)
    actual_plan,arrays,masks=load(Path(training['cache']['path']).parent,training['cache']);require(actual_plan==plan,'Actual complete cache source differs')
    selected_rows=[plan['rows'][i] for i in ids]
    with np.load(verify(constructor['features']),allow_pickle=False) as z:cpu_maps={k:z[k].copy() for k in z.files}
    with np.load(verify(parity['GPU_features']),allow_pickle=False) as z:gpu_maps={k:z[k].copy() for k in z.files}
    actor=CoupledGoalACT().to('cuda:0').eval().requires_grad_(False);actor.load_state_dict(load_file(verify(plan['teacher_weights']),device='cpu'),strict=True)
    require(sum(x.numel() for x in actor.parameters())==180528,'Exact full frozen teacher tensors')
    original={k:v.clone() for k,v in actor.state_dict().items()}
    motor_fields={k:v.to('cuda:0') for k,v in fields(selected_rows,plan['teacher_statistics']).items() if k not in ('expert_target','expert_valid')}
    all_metrics={}
    with correction_precision(torch):
        for candidate in CANDIDATES:
            saved=torch.load(verify(training['results'][candidate]['weights']),map_location='cpu',weights_only=False)
            require(set(saved)=={'model','statistics','pooling','qpos_only','feature_plan','protocol','fixed_epoch'} and saved['fixed_epoch']==30
                and saved['feature_plan']==parity['plan'] and saved['protocol']==training['protocol'] and saved['pooling']=='spatial2x3'
                and saved['qpos_only']==candidate.endswith('_qpos_control'),'Actual fixed trained checkpoint contract')
            exact_moments=moments(arrays,masks['optimizer']);require(set(saved['statistics'])==set(exact_moments),'Exact optimizer800 moment keys')
            require(all(np.array_equal(saved['statistics'][k],exact_moments[k]) for k in exact_moments),'Saved normalization must be exactly optimizer800 moments')
            values=batch(arrays,ids,saved['statistics'],device='cuda:0')
            def inputs_for(maps):
                raw=torch.from_numpy(maps['visual']).to('cuda:0');global_raw=torch.from_numpy(maps['visual_global']).to('cuda:0')
                stats={k:torch.as_tensor(v,device='cuda:0') for k,v in saved['statistics'].items()}
                normal=(raw-stats['visual_mean'][None,:,:,None,None])/stats['visual_scale'][None,:,:,None,None]
                global_normal=(global_raw-stats['global_mean'])/stats['global_scale']
                return [dict(visual=normal[i:i+1],visual_global=global_normal[i:i+1],qpos_features=values['qpos_features'][i:i+1]) for i in range(10)]
            CPU_inputs=inputs_for(cpu_maps);GPU_inputs=inputs_for(gpu_maps)
            student=Student('spatial2x3',qpos_only=saved['qpos_only']).to('cuda:0').eval().requires_grad_(False);student.load_state_dict(saved['model'],strict=True)
            def tail(pred,index):
                motor={k:v[index:index+1] for k,v in motor_fields.items()}
                return projected_forward(actor,projected_roles=pred['projected_roles'],goal_delta=pred['predicted_goal_delta'],qpos_features=values['qpos_features'][index:index+1],**motor)
            feature_outputs=[]
            with torch.inference_mode():
                for i,(cpu,gpu) in enumerate(zip(CPU_inputs,GPU_inputs)):
                    left=student(**cpu);right=student(**gpu);a_tail=tail(left,i);b_tail=tail(right,i)
                    feature_outputs.append(dict(task=plan['tasks'][i],clip_id=selected_rows[i]['clip_id'],student=numeric_output_difference(left,right),actual_tail=numeric_output_difference(a_tail,b_tail),
                        all_Boolean_outputs_bitexact=all(torch.equal(a_tail[k],b_tail[k]) for k in a_tail if a_tail[k].dtype==torch.bool),
                        normalized_spatial_difference_RMS=float((cpu['visual']-gpu['visual']).square().mean().sqrt())))
            before={g:dict(v) for g,v in counters.items()}
            try:
                _,report=qualify(student,GPU_inputs,backend='inductor',benchmark=True,postprocess=tail)
            except BaseException as error:
                import traceback
                (out/'compiler-failure.json').write_text(json.dumps(dict(status='actual_compile_or_numeric_qualification_FAILED_not_promoted',candidate=candidate,error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),counters={g:dict(v) for g,v in counters.items()},CPU_GPU_feature_output_diagnostics=feature_outputs,completed_candidates=all_metrics,training_receipt=training_ref,admission=admission_ref),indent=2)+'\n')
                raise
            report['compiler_counter_delta']={g:{k:v-before.get(g,{}).get(k,0) for k,v in values.items() if v-before.get(g,{}).get(k,0)} for g,values in counters.items()}
            report['scope']='all10_fixed_current_constructor_inputs_validation_MLP_frozen_actual_tail_not_whole_ACT_or_world'
            all_metrics[candidate]=dict(weights=training['results'][candidate]['weights'],CPU_GPU_different_feature_output_diagnostics=feature_outputs,
                CPU_GPU_Boolean_gate_PASS=all(row['all_Boolean_outputs_bitexact'] for row in feature_outputs),compile=report)
            (out/'partial-progress.json').write_text(json.dumps(dict(status='partial_actual_candidates_not_complete',candidates=all_metrics,training_receipt=training_ref),indent=2)+'\n')
            resource_gate(start,180,3*(1<<30));del student
    require(all(torch.equal(v,original[k]) for k,v in actor.state_dict().items()) and all(p.grad is None for p in actor.parameters()),'Frozen teacher inference changed')
    torch.cuda.synchronize();require(torch.cuda.max_memory_reserved()<=2*(1<<30),'Declared Torch2GiB cap exceeded')
    result=dict(schema='actual_all10_trained_ACT_feature_student_inference_compile_v1',status='all_four_actual_candidates_compile_numeric_checked_measured_modes_preserved',
        admission=admission_ref,training_receipt=training_ref,parity_receipt=parity_ref,candidates=all_metrics,native_inference_flags=flags,correction_inference_flags=CORRECTION_FLAGS,correction_flags_scoped_and_original_restored=True,
        optimizer800_moments_bitexact=True,frozen_teacher_tensor_parameters=180528,used_projected_tail_parameters=49424,
        CPU_GPU_CNN_outputs_not_bitexact=True,CPU_GPU_output_diagnostics_not_same_input_compile_test=True,
        whole_ACT_neural_cost_measured=False,full_policy_latency_inferred=False,torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        elapsed_seconds=time.monotonic()-start,models_loaded=5,full_ACT_models_loaded=0,Qwen_models_loaded=0,optimizer_updates=0,worlds_constructed=0,robot_quality_promoted=False)
    (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status=result['status'],receipt=ref(out/'receipt.json'))))


if __name__=='__main__':main()
