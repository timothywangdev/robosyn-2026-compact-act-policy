"""Post-training actual all10 FIT-current small-model compiler gate, no worlds."""
import argparse
import json
import os
from pathlib import Path
import signal
import time
from .contracts import STATS,require,verify_ref,sha
from experiments.goal_latent_act_campaign.common import ref,write


def build_statistics(statistics,device):
    """Only numeric FIT moments enter the actor; provenance remains metadata."""
    import torch
    result={key:torch.tensor(statistics[key],dtype=torch.float32,device=device) for key in STATS}
    for key,width in STATS.items():
        require(result[key].shape==(width,) and bool(torch.isfinite(result[key]).all()),'Invalid numeric statistic: '+key)
        if key.endswith('scale'):require(bool((result[key]>0).all()),'Nonpositive FIT scale: '+key)
    return result


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    for key in ('prepared','prepared-sha256','training-root','output'):parser.add_argument('--'+key)
    parser.add_argument('--check-statistics-only',action='store_true')
    parser.add_argument('--execute',action='store_true');parser.add_argument('--admission');parser.add_argument('--admission-sha256')
    args=parser.parse_args()
    require(args.prepared and args.prepared_sha256,'Exact prepared FIT receipt required')
    if args.check_statistics_only:
        require(not args.execute and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU statistics check must hide CUDA')
        prepared_ref=ref(Path(args.prepared)/'receipt.json')
        require(prepared_ref['sha256']==args.prepared_sha256,'Prepared metadata bytes changed')
        prepared=json.loads(verify_ref(prepared_ref).read_text())
        stats=build_statistics(prepared['statistics'],'cpu')
        import torch
        require(not torch.cuda.is_initialized(),'CPU statistics check initialized CUDA')
        print(json.dumps(dict(status='actual_prepared_numeric_statistics_CPU_PASS',prepared=prepared_ref,
            numeric_statistics={key:list(value.shape) for key,value in stats.items()},CUDA_initialized=False,
            GPU_queries=0,models_constructed=0,worlds_constructed=0)))
        return
    require(args.training_root and args.output,'Actual trained root and fresh evidence directory required')
    require(args.execute and args.admission and args.admission_sha256,'Separate reviewed root compiler assay admission required')
    admission_ref=ref(args.admission);require(admission_ref['sha256']==args.admission_sha256,'Exact compiler admission bytes required')
    admission=json.loads(verify_ref(admission_ref).read_text())
    require(admission['schema']=='root_authorized_coupled_all14_compiler_assay_v1' and admission['execution_authorized'] is True
        and admission['runner']==ref(__file__) and admission['prepared']==ref(Path(args.prepared)/'receipt.json'), 'Root must bind exact source/current FIT packet')
    require(admission['predecessor_genuine_wait_consumed'] is True and admission['predecessor_processes_drained'] is True
        and admission['exclusive_GPU_owner'] is True and admission['GPU_uuid']=='GPU-8e45cdd0-bbd6-9b4f-de12-e7125a11bed9',
        'Genuine pair wait/drain and separate exclusive owner required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==admission['GPU_uuid'],'Exact root-owned CUDA device binding required')
    for binding in admission['predecessor_proofs']:verify_ref(binding)
    training=Path(args.training_root).resolve(strict=True)
    for mode in ('role_goal','state_action_control'):
        require(admission['checkpoints'][mode]==ref(training/mode/'checkpoint.json')
            and admission['weights'][mode]==ref(training/mode/'coupled-all14.safetensors'), 'Actual new trained mode differs')
    output=Path(args.output).absolute();require(not output.exists(),'Fresh original compiler evidence required');output.mkdir(parents=True)
    started=time.monotonic();report=dict(schema='coupled_all14_actual_all10_FIT_current_GPU_compiler_assay_v1',
        status='running_original',admission=admission_ref,branches={},robot_worlds=0,optimizer_updates=0,
        model_scope='only small all14 actor; original cached current-role/nativeACT50 FIT input; ACT/Qwen not loaded',quality_promoted=False)
    def expired(*_):raise TimeoutError('Small compiler assay180-second inclusive wall cap')
    signal.signal(signal.SIGALRM,expired);signal.alarm(180)
    try:
        import numpy as np
        import torch
        from experiments.coupled_goal_act_all14.data import load,tensors,batch
        from experiments.coupled_goal_act_all14.contracts import load_checkpoint
        from experiments.coupled_goal_act_all14.compiler import compile_and_compare
        require(torch.__version__=='2.7.1+cu128' and torch.cuda.is_available(),'Pinned local CUDA/Torch required')
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.manual_seed(17);torch.cuda.set_per_process_memory_fraction(8*(1<<30)/torch.cuda.get_device_properties(0).total_memory)
        r,arrays,features,tasks=load(args.prepared,args.prepared_sha256)
        ids=np.asarray([np.flatnonzero(arrays['optimizer_fit']&(arrays['task_index']==index))[0] for index in range(10)])
        selected={key:arrays[key][ids].copy() for key in ('physical_base','physical_actions','valid','canonical_current_qpos14','role_features','teacher_eef_now','task_index')}
        selected_features={'predicted_goal':features['predicted_goal'][ids].copy()}
        # Shuffle is deliberately never used by this FIT-current gate.
        selected_features['shuffled_predicted_goal']=np.zeros_like(selected_features['predicted_goal'])
        selected_features.update(shuffled_role_ids=np.arange(10,dtype=np.int64),monitor_ids=np.arange(10,dtype=np.int64))
        del arrays,features
        data=tensors(selected,selected_features,'cuda');del selected,selected_features
        stats=build_statistics(r['statistics'],'cuda')
        for mode,control in (('role_goal',False),('state_action_control',True)):
            model,contract=load_checkpoint(training/mode,admission['weights'][mode]['sha256'],device='cuda')
            first=batch(data,[0],stats,tasks)
            compiled,comparison=compile_and_compare(model,first,state_action_only=control,backend='inductor',warm_calls=5)
            rows=[]
            with torch.inference_mode():
                for index,task in enumerate(tasks):
                    inputs=batch(data,[index],stats,tasks)
                    expected=model(**inputs,state_action_only=control);actual=compiled(**inputs,state_action_only=control)
                    errors={}
                    for key in expected:
                        if expected[key].dtype==torch.bool:require(torch.equal(expected[key],actual[key]),'Actual same-input gate/limits differ: '+task+'/'+key)
                        else:
                            difference=(expected[key]-actual[key]).double()
                            errors[key]=dict(max_abs=float(difference.abs().max()),RMSE=float(difference.square().mean().sqrt()))
                            require(errors[key]['max_abs']<=1e-5 and errors[key]['RMSE']<=2e-6,'Original FP32 all10 compiler gate failed: '+task+'/'+key)
                    rejected=(~actual['accepted'])[:,None,:,None].expand(1,50,2,7).reshape_as(inputs['native_base'])
                    require(torch.equal(actual['native_actions'][rejected],inputs['native_base'][rejected]),'Rejected complete-hand fallback changed')
                    rows.append(dict(task=task,optimizer_FIT_source_row=int(ids[index]),parity=errors,
                        accepted=actual['accepted'].tolist(),rejected_hand_all7_bitexact=True))
            report['branches'][mode]=dict(contract=ref(training/mode/'checkpoint.json'),weights=ref(training/mode/'coupled-all14.safetensors'),
                compiler=comparison,all10_same_input_rows=rows,current_native_world_input_qualified=False)
            write(output/'receipt.json',report)
            del model,compiled
        report['status']='both_trained_modes_all10_FIT_current_partial_Inductor_numeric_pass_no_robot_promotion'
        report['GPU_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
    except BaseException as error:
        report.update(status='original_failed_preserved_no_fallback',failure=dict(type=type(error).__name__,message=str(error)))
        raise
    finally:
        signal.alarm(0);report['elapsed_seconds']=time.monotonic()-started;write(output/'receipt.json',report)
    print(json.dumps(dict(status=report['status'],receipt=ref(output/'receipt.json'))))


if __name__=='__main__':main()
