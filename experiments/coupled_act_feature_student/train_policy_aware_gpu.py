"""Separately admitted fixed GPU paired pilot; CPU recipe/model/objective retained."""
import argparse,json,os,time
from pathlib import Path
from .contracts import require,ref,verify


def admitted_GPU_UUID():
    return 'GPU-8e45cdd0-bbd6-9b4f-de12-e7125a11bed9'


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for key in ('cache','cache-receipt-sha256','protocol','protocol-sha256','admission','admission-sha256','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--check-only',action='store_true');a=p.parse_args()
    require((a.check_only and os.environ.get('CUDA_VISIBLE_DEVICES')=='') or os.environ.get('CUDA_VISIBLE_DEVICES')==admitted_GPU_UUID(),'GPU student requires root-bound local device, or hidden CUDA for CPU FILE check')
    protocol_ref=ref(a.protocol);require(protocol_ref['sha256']==a.protocol_sha256,'Fixed predeclared paired policy-aware recipe')
    protocol=json.loads(Path(a.protocol).read_text())
    require(protocol['schema']=='prospective_ACT_feature_student_paired_policy_aware_training_v1'
        and protocol['epochs']==30 and protocol['batch_size']==128 and protocol['seed']==1701
        and protocol['AdamW_lr']==.001 and protocol['weight_decay']==.0001,'Fixed matched optimizer/210updates')
    admission_ref=ref(a.admission);require(admission_ref['sha256']==a.admission_sha256,'Root CPU training admission')
    admission=json.loads(Path(a.admission).read_text())
    require(admission['schema']=='root_authorized_ACT_feature_student_paired_policy_aware_GPU_training_v1'
        and admission['execution_authorized'] is True and admission['protocol']==protocol_ref,'Separate paired root GPU admission')
    verify(admission['matched_native_teacher_benefit'])
    require(admission['GPU_UUID']==admitted_GPU_UUID(),'Sole local3090 exact UUID')
    parity=json.loads(verify(admission['actual_CPU_GPU_feature_parity']).read_text())
    require(parity['status']=='all10_exact_current_CPU_GPU_ACT_feature_numeric_parity_PASS' and parity['plan']==protocol['feature_plan'],'Actual same-source ten-task CPU/GPU feature parity required before training')
    require(admission['CPU_tree_cap_bytes']==3*(1<<30) and admission['threads']==4 and admission['wall_seconds']==180 and admission['GPU_Torch_cap_bytes']==2*(1<<30) and admission['whole_GPU_cap_bytes']==4*(1<<30),'GPU2GiB_Torch/whole4GiB/host3GiB/thread4/wall180sec')
    root=Path(__file__).resolve().parents[2]
    for relative,digest in protocol['source_sha256'].items():require(ref(root/relative)['sha256']==digest,'Fixed action-aware source differs')
    for relative,digest in admission['GPU_source_sha256'].items():require(ref(root/relative)['sha256']==digest,'Pinned additive GPU source differs')
    cache=ref(Path(a.cache)/'receipt.json');require(cache['sha256']==a.cache_receipt_sha256 and cache==admission['cache'],'Exact actual1000 cache required')
    from .data import load,moments,batch
    from .model import Student
    from .policy_aware import policy_aware_forward
    from .policy_aware_objective import objective,WEIGHTS
    from .policy_aware_data import fields
    from .extract import resource_gate
    import numpy as np,torch
    from safetensors.torch import load_file
    from experiments.coupled_goal_act_all14.model import CoupledGoalACT
    from experiments.coupled_goal_act_all14.units import ARM,GRIP
    from experiments.goal_latent_act_arm_sequence_full.fk import CommandFK,pose_errors
    require(protocol['loss_weights']==WEIGHTS,'Prospective policy loss weights changed')
    plan,arrays,masks=load(a.cache,cache);require(ref(verify(protocol['feature_plan']))==ref(verify(json.loads(verify(cache).read_text())['plan'])),'Feature plan and action-aware plan differ')
    if a.check_only:print(json.dumps(dict(status='paired_policy_aware_source_actual_cache_and_root_admission_valid',pretrained_models_loaded=0,optimizer_updates=0)));return
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    device='cuda:0';torch.cuda.set_per_process_memory_fraction((2*(1<<30))/torch.cuda.get_device_properties(0).total_memory,0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    # cuDNN eval-GRU cannot backward; eager GRU keeps actual motor eval/frozen.
    torch.backends.cudnn.enabled=False
    start=time.monotonic();out=Path(a.output);require(not out.exists(),'Fresh fixed paired output');out.mkdir(parents=True)
    actor=CoupledGoalACT().to(device).eval().requires_grad_(False);actor.load_state_dict(load_file(verify(plan['teacher_weights']),device='cpu'),strict=True)
    state_actor=CoupledGoalACT().to(device).eval().requires_grad_(False);state_actor.load_state_dict(load_file(verify(protocol['state_control_weights']),device='cpu'),strict=True)
    original={k:v.clone() for k,v in actor.state_dict().items()};contract=json.loads(verify(plan['teacher_contract']).read_text())
    fk=CommandFK(contract['geometry'],contract['urdf']).to(device).eval().requires_grad_(False)
    statistics=moments(arrays,masks['optimizer']);motor={k:v.to(device) for k,v in fields(plan['rows'],plan['teacher_statistics']).items()};indices=np.flatnonzero(masks['optimizer'])
    def motor_inputs(ids):return {k:v[ids] for k,v in motor.items() if k not in ('expert_target','expert_valid')}
    def errors(actions,ids):
        target=motor['expert_target'][ids,:10];valid=motor['expert_valid'][ids,:10];mask=valid[...,None]
        x=actions[:,:10]
        arm=((x[...,list(ARM)]-target[...,list(ARM)]).abs().mean(-1)*valid).sum()/valid.sum().clamp_min(1)
        grip=((x[...,list(GRIP)]-target[...,list(GRIP)]).abs()*mask).sum()/mask.expand(-1,-1,2).sum().clamp_min(1)
        pos,rot=pose_errors(fk(x[...,list(ARM)]),fk(target[...,list(ARM)]));denom=mask.expand_as(pos).sum().clamp_min(1)
        return dict(arm_MAE_rad=float(arm),opening_MAE_m=float(grip),FK_position_MAE_m=float((pos*mask).sum()/denom),FK_rotation_MAE_rad=float((rot*mask).sum()/denom))
    results={}
    for mode,state_only,aware in (('distill_visual',False,False),('distill_qpos_control',True,False),('policy_aware_visual',False,True)):
        torch.manual_seed(1701);student=Student(qpos_only=state_only).to(device);optimizer=torch.optim.AdamW(student.parameters(),lr=.001,weight_decay=.0001)
        rng=np.random.default_rng(1701);updates=0;loss_log=[]
        for epoch in range(30):
            order=rng.permutation(indices)
            for first in range(0,len(order),128):
                resource_gate(start,180,3*(1<<30));ids=order[first:first+128];values=batch(arrays,ids,statistics,device=device)
                prediction=student(values['visual'],values['qpos_features'],values['visual_global']);teacher={k:values[k] for k in prediction}
                if aware:
                    result=policy_aware_forward(actor,projected_roles=prediction['projected_roles'],goal_delta=prediction['predicted_goal_delta'],qpos_features=values['qpos_features'],**motor_inputs(ids))
                    loss,parts=objective(prediction,result,teacher,motor['expert_target'][ids],motor['expert_valid'][ids],fk)
                else:
                    parts={key:(prediction[key]-teacher[key]).square().mean() for key in prediction};loss=sum(parts.values())
                require(bool(torch.isfinite(loss)),'Finite fixed objective');optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();updates+=1
                loss_log.append(dict(step=updates,total=float(loss.detach()),parts={key:float(value.detach()) for key,value in parts.items()}))
        require(updates==210,'Exactly30×7 optimizer updates');student.eval();metrics={}
        with torch.inference_mode():
            for split in ('diagnostic','gate'):
                bytask={}
                for task in plan['tasks']:
                    ids=np.array([i for i,row in enumerate(plan['rows']) if row['task']==task and masks[split][i]])
                    values=batch(arrays,ids,statistics,device=device);prediction=student(values['visual'],values['qpos_features'],values['visual_global'])
                    student_actions=policy_aware_forward(actor,projected_roles=prediction['projected_roles'],goal_delta=prediction['predicted_goal_delta'],qpos_features=values['qpos_features'],**motor_inputs(ids))
                    teacher_actions=policy_aware_forward(actor,projected_roles=values['projected_roles'],goal_delta=values['predicted_goal_delta'],qpos_features=values['qpos_features'],**motor_inputs(ids))
                    state_actions=policy_aware_forward(state_actor,projected_roles=values['projected_roles'],goal_delta=values['predicted_goal_delta'],qpos_features=values['qpos_features'],state_action_only=True,**motor_inputs(ids))
                    delta=(prediction['predicted_goal_delta']-values['predicted_goal_delta'])*torch.tensor(plan['teacher_statistics']['eef_scale'],device=device);delta=delta.reshape(-1,2,9)
                    rejected=(~student_actions['accepted'])[:,None,:,None].expand(-1,50,2,7)
                    bytask[task]=dict(roles_RMSE=float((prediction['projected_roles']-values['projected_roles']).square().mean().sqrt()),
                        physical_goal_translation_RMSE_m=float(delta[...,:3].square().sum(-1).mean().sqrt()),rotation6D_RMSE=float(delta[...,3:].square().mean().sqrt()),
                        student=errors(student_actions['physical_actions'],ids),teacher=errors(teacher_actions['physical_actions'],ids),parent=errors(teacher_actions['parent_physical_actions'],ids),state_control=errors(state_actions['physical_actions'],ids),
                        hand_acceptance_agreement=float((student_actions['accepted']==teacher_actions['accepted']).float().mean()),
                        false_accepts_relative_to_teacher=int((student_actions['accepted']&~teacher_actions['accepted']).sum()),
                        rejected_native7_bitexact=torch.equal(student_actions['native_actions'].reshape(-1,50,2,7)[rejected],motor['native_base'][ids].reshape(-1,50,2,7)[rejected]))
                metrics[split]=bytask
        target=out/(mode+'.pt');torch.save(dict(model=student.state_dict(),statistics=statistics,pooling='spatial2x3',qpos_only=state_only,feature_plan=protocol['feature_plan'],protocol=protocol_ref,fixed_epoch=30),target)
        results[mode]=dict(weights=ref(target),optimizer_updates=updates,loss_log=loss_log,metrics=metrics)
        del student,optimizer
    for key,value in actor.state_dict().items():require(torch.equal(value,original[key]),'Frozen actual motor actor mutated')
    require(all(p.grad is None for p in actor.parameters()),'Frozen actor gradient boundary violated')
    torch.cuda.synchronize();require(torch.cuda.max_memory_reserved()<=2*(1<<30),'Declared Torch2GiB cap exceeded')
    (out/'receipt.json').write_text(json.dumps(dict(schema='actual_paired_ACT_feature_student_distill_and_policy_aware_GPU_v1',status='fixed30_three_candidates_complete_exploratory_no_robot_promotion',protocol=protocol_ref,admission=admission_ref,cache=cache,results=results,elapsed_seconds=time.monotonic()-start,
        teacher_weights=plan['teacher_weights'],motor_frozen_byte_equivalent=True,optimizer_episode_count=800,unused_gate_episode_count=100,unused_diagnostic_episode_count=100,MONITOR_selection=False,
        original_teacher5percent_criterion='FAILED_unchanged',GPU_queries=1,GPU_launches=1,GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(),native_results=0,robot_quality_promoted=False),indent=2)+'\n')


if __name__=='__main__':main()
