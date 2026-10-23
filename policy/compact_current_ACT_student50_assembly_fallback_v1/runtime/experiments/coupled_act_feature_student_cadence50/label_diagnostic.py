"""CPU held-out saved-label readback; never a closed-loop cadence50 claim."""
import argparse
import json
import os
from pathlib import Path
import time
from experiments.coupled_act_feature_student.contracts import ref,verify,require
from experiments.goal_latent_act_campaign.common import write

def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--campaign',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU label diagnostic must hide CUDA')
    campaign_ref=ref(a.campaign);campaign=json.loads(verify(campaign_ref).read_text());cfg=json.loads(verify(next(r['deployment'] for r in campaign['plans'] if r['mode']=='student')).read_text())
    out=Path(a.output).absolute();require(not out.exists(),'Fresh saved-label diagnostic');out.mkdir(parents=True)
    import numpy as np,torch
    from safetensors.torch import load_file
    from experiments.coupled_act_feature_student.data import load,batch
    from experiments.coupled_act_feature_student.policy_aware_data import fields
    from experiments.coupled_act_feature_student.model import Student
    from experiments.coupled_act_feature_student.projected import projected_forward
    from experiments.coupled_act_feature_student_runtime.contracts import validate_saved
    from experiments.coupled_goal_act_all14.model import CoupledGoalACT
    from experiments.coupled_goal_act_all14.units import ARM,GRIP
    from experiments.goal_latent_act_arm_sequence_full.fk import CommandFK,pose_errors
    torch.set_num_threads(4);torch.set_num_interop_threads(1);start=time.monotonic()
    cache=cfg['assets']['student_cache_receipt'];plan,arrays,masks=load(Path(cache['path']).parent,cache)
    saved=torch.load(verify(cfg['assets']['student_weights']),map_location='cpu',weights_only=False)
    training=json.loads(verify(cfg['assets']['student_training_receipt']).read_text());validate_saved(saved,cfg,training)
    student=Student('spatial2x3',qpos_only=False).eval().requires_grad_(False);student.load_state_dict(saved['model'],strict=True)
    actor=CoupledGoalACT().eval().requires_grad_(False);actor.load_state_dict(load_file(verify(cfg['assets']['teacher_weights']),device='cpu'),strict=True)
    contract=json.loads(verify(cfg['assets']['teacher_contract']).read_text());fk=CommandFK(contract['geometry'],contract['urdf']).eval().requires_grad_(False)
    rows=[];aggregates={};fallback=True
    def values(actions,target,valid,lo,hi):
        x=actions[:,lo:hi];y=target[:,lo:hi];v=valid[:,lo:hi];m=v[...,None];den=float(v.sum())
        arm=float(((x[...,list(ARM)]-y[...,list(ARM)]).abs().mean(-1)*v).sum())
        grip=float(((x[...,list(GRIP)]-y[...,list(GRIP)]).abs()*m).sum())/2
        pos,rot=pose_errors(fk(x[...,list(ARM)]),fk(y[...,list(ARM)]))
        pos=float((pos*m).sum())/2;rot=float((rot*m).sum())/2
        return dict(valid_commands=den,sums=dict(arm_MAE_rad=arm,opening_MAE_m=grip,FK_position_MAE_m=pos,FK_rotation_MAE_rad=rot))
    with torch.inference_mode():
        for split in ('gate','diagnostic'):
            ids=np.flatnonzero(masks[split]);require(len(ids)==100,'Original held-out100 FIT rows per split')
            selected=[plan['rows'][i] for i in ids];motor=fields(selected,plan['teacher_statistics']);v=batch(arrays,ids,saved['statistics'])
            predicted=student(v['visual'],v['qpos_features'],v['visual_global'])
            common={k:value for k,value in motor.items() if k not in ('expert_target','expert_valid')};common['qpos_features']=v['qpos_features']
            result=projected_forward(actor,projected_roles=predicted['projected_roles'],goal_delta=predicted['predicted_goal_delta'],**common)
            rejected=(~result['accepted'])[:,None,:,None].expand(-1,50,2,7)
            exact=torch.equal(result['native_actions'].reshape(-1,50,2,7)[rejected],motor['native_base'].reshape(-1,50,2,7)[rejected]);require(exact,'Rejected native7 changed');fallback&=exact
            aggregates[split]={}
            for task in plan['tasks']:
                local=np.array([i for i,row in enumerate(selected) if row['task']==task]);require(len(local)==10,'Exactly10 held-out episodes per task')
                row=dict(split=split,task=task,episodes=10,accepted_hands=int(result['accepted'][local].sum()),rejected_native7_bitexact=exact,spans={})
                for label,lo,hi in (('first10',0,10),('remaining40',10,50),('all50',0,50)):
                    metrics={}
                    for name,actions in (('student',result['physical_actions']),('parent',result['parent_physical_actions'])):
                        sums=values(actions[local],motor['expert_target'][local],motor['expert_valid'][local],lo,hi)
                        metrics[name]=dict(valid_commands=int(sums['valid_commands']),**{k:s/max(1,sums['valid_commands']) for k,s in sums['sums'].items()})
                        total=aggregates[split].setdefault(label,{}).setdefault(name,dict(valid_commands=0,sums={k:0. for k in sums['sums']}))
                        total['valid_commands']+=int(sums['valid_commands'])
                        for k,s in sums['sums'].items():total['sums'][k]+=s
                    metrics['student_relative_change_vs_parent']={k:metrics['student'][k]/max(1e-12,metrics['parent'][k])-1 for k in metrics['student'] if k!='valid_commands'}
                    row['spans'][label]=metrics
                rows.append(row)
            for label,arms in aggregates[split].items():
                for name,total in arms.items():total.update({k:s/max(1,total['valid_commands']) for k,s in total.pop('sums').items()})
                arms['student_relative_change_vs_parent']={k:arms['student'][k]/max(1e-12,arms['parent'][k])-1 for k in arms['student'] if k!='valid_commands'}
    require(not torch.cuda.is_initialized(),'CPU diagnostic initialized CUDA')
    body=dict(schema='same_trained_current_ACT_student_all50_heldout_CPU_label_diagnostic_v1',
        status='actual_gate100_diagnostic100_all50_label_readback_no_closed_loop_or_cadence_qualification',original_campaign=campaign_ref,
        student_weights=cfg['assets']['student_weights'],teacher_weights=cfg['assets']['teacher_weights'],cache=cache,
        rows=rows,aggregates=aggregates,all_rejected_native7_bitexact=fallback,epochs_added=0,optimizer_updates=0,
        input_scope='same_saved_current_ACT_maps_measured_qpos_and_current_FK_only; labels_used_for_reporting_only',
        action_outputs_generated50=True,executing50_closed_loop_not_tested=True,new_score_or_promotion=False,
        original_first10_training_selection_unchanged=True,CUDA_initialized=False,GPU_queries=0,GPU_launches=0,worlds=0,
        ACT_or_Qwen_models_constructed=0,CPU_small_models_constructed=3,seconds=time.monotonic()-start)
    write(out/'receipt.json',body);print(json.dumps(dict(receipt=ref(out/'receipt.json'),aggregates=aggregates,seconds=body['seconds'])))

if __name__=='__main__':main()
