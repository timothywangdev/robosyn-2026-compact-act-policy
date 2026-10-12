"""Additive native protocol rebinding; preserve the original template bytes."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from .prepare import prepare,ORDER,BRANCHES
from .contracts import HORIZONS,require,ref,verify
from experiments.goal_latent_act_campaign.common import write


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    for name in ('base','output','run-root'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--candidate');args=parser.parse_args()
    output=Path(args.output).absolute();require(not output.exists(),'Fresh additive Student protocol packet');output.mkdir(parents=True)
    candidate=None if args.candidate is None else json.loads(Path(args.candidate).read_text())
    original=prepare(args.base,output/'preserved-v1-template',args.run_root,candidate)
    base=json.loads(verify(original).read_text())
    protocol=dict(schema='coupled_ACT_current_map_student_development_protocol_v1',source_candidate_selection_pending=candidate is None,
        generator=2026102401,motor_seed=17,episodes_per_task_mode=1,tasks=list(ORDER),modes=list(BRANCHES),
        native_generated=50,prefix_executed=10,official_horizons=HORIZONS,
        native_ACT_implementation='unchanged_released_FP32_eager',student_features='read_only_same_forward_current_ACT_backbone_maps_high_right_left',
        actor_inputs=['current_RGB_via_same_ACT_forward','current_qpos','current_FK'],Qwen_model_inference=False,
        feature_targets='frozen_current_teacher_projected_role96_and_predicted_goal18_only',future_targets_at_inference=False,
        input_normalization='optimizer800_visual_moments_and_frozen_teacher_qpos_FK_physical_native_moments',
        hand_atomic_all7=True,rejected_native_hand_bitexact=True,accepted_opening_inverse_once=True,
        compilation='attempt_current_Student_numeric_plus_full_frozen_motor_gate_then_measured_fastest',
        teacher_native_equivalence_claim=False,robot_quality_promoted=False)
    write(output/'protocol.json',protocol);rows=[]
    for old in base['plans']:
        row=deepcopy(old);command=json.loads(verify(old['command']).read_text());command['bindings']['protocol']=ref(output/'protocol.json')
        command_path=output/(old['mode']+'-'+old['task']+'-command.json');write(command_path,command)
        plan=json.loads(verify(old['plan']).read_text());job=plan['jobs'][0];job['command']=ref(command_path)
        for flag,value in (('--command-record',str(command_path)),('--command-sha256',ref(command_path)['sha256'])):
            require(job['argv'].count(flag)==1,'Original wrapper flag differs');job['argv'][job['argv'].index(flag)+1]=value
        plan_path=output/(old['mode']+'-'+old['task']+'-plan.json');write(plan_path,plan)
        if candidate is not None:
            from .supervisor import validate_plan
            validate_plan(plan)
        row.update(command=ref(command_path),plan=ref(plan_path));rows.append(row)
    result=deepcopy(base);result.update(plans=rows,protocol=ref(output/'protocol.json'),preserved_original_template=original,
        new_semantic_student_protocol_explicit=True,teacher_native_equivalence_claim=False)
    write(output/'campaign.json',result);print(json.dumps(ref(output/'campaign.json')))


if __name__=='__main__':main()
