"""Actual selected Student packet with scoped native CNN flags and pairing hashes."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from .prepare import prepare,ORDER,BRANCHES
from .contracts import HORIZONS,require,ref,verify
from experiments.goal_latent_act_campaign.common import write


def build(base_path,output,run_root,candidate):
    out=Path(output).absolute();require(not out.exists(),'Fresh actual native precision packet');out.mkdir(parents=True)
    original=prepare(base_path,out/'preserved-v1-template',run_root,candidate)
    base=json.loads(verify(original).read_text());source=Path(base['source_root']);pins=deepcopy(base['source_sha256'])
    bridge=source/'policy/coupled_ACT_current_student_v2';bridge.mkdir(parents=True)
    (bridge/'__init__.py').write_text('from experiments.coupled_act_feature_student_runtime.policy_v2 import get_model, eval, reset_model, close_model\n')
    policy=bridge/'deploy_policy.yml';policy.write_text('policy_name: coupled_ACT_current_student_v2\nstate_obs_path: robot/qpos\nstrict_action_dim: true\nact_step: 10\n')
    pins[str((bridge/'__init__.py').relative_to(source))]=ref(bridge/'__init__.py')['sha256']
    protocol=dict(schema='coupled_ACT_current_map_student_development_protocol_v2',source_candidate_selection_pending=False,
        generator=2026102401,motor_seed=17,episodes_per_task_mode=1,tasks=list(ORDER),modes=list(BRANCHES),
        native_generated=50,prefix_executed=10,official_horizons=HORIZONS,
        native_ACT_implementation='released_FP32_eager_with_explicit_native_default_CNN_flags',
        native_ACT_precision=dict(matmul_allow_tf32=False,cudnn_allow_tf32=True),
        correction_precision=dict(matmul_allow_tf32=False,cudnn_allow_tf32=False),precision_restored_on_exception=True,
        actor_inputs=['current_RGB_via_same_ACT_forward','current_qpos','current_FK'],Qwen_model_inference=False,
        feature_targets='frozen_current_teacher_projected_role96_and_predicted_goal18_only',future_targets_at_inference=False,
        hand_atomic_all7=True,rejected_native_hand_bitexact=True,accepted_opening_inverse_once=True,
        compilation='attempt_current_Student_numeric_plus_full_frozen_motor_gate_then_measured_fastest',
        initial_source_pairing='actual_qpos14_and_pixel_dtype_shape_sha256_per_model_episode',
        early_actions='first7_exact_float32_output_buffer_sha256',hashes_are_features=False,
        teacher_native_equivalence_claim=False,robot_quality_promoted=False)
    write(out/'protocol.json',protocol);rows=[]
    for old in base['plans']:
        cfg=json.loads(verify(old['deployment']).read_text());cfg.update(source_sha256=pins,
            native_ACT_precision=protocol['native_ACT_precision'],pairing_initial_current_buffers_logged=True)
        deployment=out/(old['mode']+'-'+old['task']+'-deployment.json');write(deployment,cfg)
        command=json.loads(verify(old['command']).read_text());command['bindings'].update(protocol=ref(out/'protocol.json'),
            deployment=ref(deployment),policy=ref(bridge/'__init__.py'),policy_config=ref(policy))
        inner=command['inner_argv']
        for flag,value in (('--config',str(policy)),('--policy_name','coupled_ACT_current_student_v2'),
                           ('--coupled_student_manifest',str(deployment)),('--coupled_student_manifest_sha256',ref(deployment)['sha256'])):
            require(inner.count(flag)==1,'Original evaluator flag differs');inner[inner.index(flag)+1]=value
        command_path=out/(old['mode']+'-'+old['task']+'-command.json');write(command_path,command)
        plan=json.loads(verify(old['plan']).read_text());job=plan['jobs'][0];job['command']=ref(command_path)
        for flag,value in (('--command-record',str(command_path)),('--command-sha256',ref(command_path)['sha256'])):
            require(job['argv'].count(flag)==1,'Original wrapper flag differs');job['argv'][job['argv'].index(flag)+1]=value
        plan_path=out/(old['mode']+'-'+old['task']+'-plan.json');write(plan_path,plan)
        from .supervisor import validate_plan
        validate_plan(plan)
        row=deepcopy(old);row.update(deployment=ref(deployment),command=ref(command_path),plan=ref(plan_path));rows.append(row)
    result=deepcopy(base);result.update(plans=rows,protocol=ref(out/'protocol.json'),source_sha256=pins,
        preserved_original_template=original,new_semantic_student_protocol_explicit=True,
        native_CNN_precision_restoration_source_bound=True,initial_buffer_pairing_logged=True,
        candidate_selection=candidate['candidate_selection'],student_GPU_parity=candidate['student_GPU_parity'],
        student_GPU_parity_wait=candidate['student_GPU_parity_wait'],teacher_native_equivalence_claim=False)
    write(out/'campaign.json',result);return ref(out/'campaign.json')


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    for key in ('base','output','run-root','candidate'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args();print(json.dumps(build(args.base,args.output,args.run_root,json.loads(Path(args.candidate).read_text()))))
if __name__=='__main__':main()
