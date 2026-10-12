"""Concrete native templates; actual candidate bytes are mandatory to finalize."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
from .contracts import SCHEMA,HORIZONS,load_manifest,require,ref,verify,CAMERAS
from experiments.goal_latent_act_campaign.common import write

ROOT=Path(__file__).resolve().parents[2]
ORDER=('click_bell','drawer_open_place','handle_basket','item_assembly','items_handover',
       'manipulate_pipette','mixer_operating','sample_loading','table_rearrangement','water_pouring')
BRANCHES=('parent','student')


def prepare(base_path,output,run_root,candidate=None):
    base_ref=ref(base_path);base=json.loads(verify(base_ref).read_text());out=Path(output).absolute();run_root=Path(run_root).absolute()
    require(not out.exists() and not run_root.exists(), 'Fresh new student preparation/result roots')
    out.mkdir(parents=True);source=out/'source';source.mkdir()
    donor=Path(base['source_root']);pins={}
    for relative,digest in base['source_sha256'].items():
        require(ref(donor/relative)['sha256']==digest,'Frozen donor source differs')
        target=source/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(donor/relative,target);pins[relative]=digest
    for folder in ('coupled_act_feature_student','coupled_act_feature_student_runtime'):
        for original in sorted((ROOT/'experiments'/folder).glob('*.py')):
            relative=str(original.relative_to(ROOT));target=source/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(original,target);pins[relative]=ref(target)['sha256']
    bridge=source/'policy/coupled_ACT_current_student_v1';bridge.mkdir(parents=True)
    (bridge/'__init__.py').write_text('from experiments.coupled_act_feature_student_runtime.policy import get_model, eval, reset_model, close_model\n')
    policy=bridge/'deploy_policy.yml';policy.write_text('policy_name: coupled_ACT_current_student_v1\nstate_obs_path: robot/qpos\nstrict_action_dim: true\nact_step: 10\n')
    pins[str((bridge/'__init__.py').relative_to(source))]=ref(bridge/'__init__.py')['sha256']
    rows=[]
    for task in ORDER:
        for mode in BRANCHES:
            old=next(r for r in base['plans'] if r['task']==task and r['mode']==('parent' if mode=='parent' else 'goal'))
            original=json.loads(verify(old['deployment']).read_text());a=original['assets']
            assets={k:a[k] for k in ('official_ACT','inference_timer','ACT_model_source','geometry','urdf','state_evidence')}
            assets['runtime_admission']=None if candidate is None else candidate['runtime_admissions'][mode+'-'+task]
            if mode=='student':
                assets.update(teacher_weights=a['coupled_weights'],teacher_contract=a['coupled_contract'])
                for key in ('student_weights','student_training_receipt','student_cache_receipt','student_feature_plan',
                            'student_original_training_wait','student_independent_training_drain'):
                    assets[key]=None if candidate is None else candidate[key]
            cfg=dict(schema=('prospective_'+SCHEMA if candidate is None else SCHEMA),mode=mode,task=task,
                source_root=str(source),source_sha256=pins,assets=assets,act_root=original['act_root'],act_files=original['act_files'],
                instruction=original['instruction'],scratch_root=str(run_root/mode/task/'actor-scratch'),
                native_chunk_generated=50,prefix_executed=10,motor_seed=17,official_horizon=HORIZONS[task],
                camera_order=['observation.images.'+c for c in CAMERAS],act_core_mode='eager',act_preprocess_mode='native_every_action',
                compile_policy='attempt_current_numeric_then_fastest_qualified',gripper_codec=original['gripper_codec'],episode_wall_seconds=1800,
                current_inputs=['current_RGB_via_same_ACT_forward','current_qpos','current_FK'],future_inputs_allowed=False,
                Qwen_models_loaded=0,ACT_models_loaded=1,student_candidate=None if candidate is None else candidate['student_candidate'],
                student_qpos_only=None if candidate is None else candidate['student_qpos_only'],execution_authorized=False)
            deployment=out/f'{mode}-{task}-deployment.json';write(deployment,cfg)
            command=json.loads(verify(old['command']).read_text());native=run_root/mode/task/'native'
            command.update(schema='coupled_ACT_student_native50_prefix10_original_command_v1',mode=mode)
            inner=command['inner_argv']
            inner=[x.replace('--coupled_all14_manifest','--coupled_student_manifest') for x in inner]
            for flag,value in (('--config',str(policy)),('--policy_name','coupled_ACT_current_student_v1'),
                ('--coupled_student_manifest',str(deployment)),('--coupled_student_manifest_sha256',ref(deployment)['sha256']),
                ('--eval_result_dir',str(native/'evaluator-results')),('--model_name','coupled-current-student-'+mode)):
                require(inner.count(flag)==1,'Original evaluator flag missing/duplicated: '+flag);inner[inner.index(flag)+1]=value
            command['inner_argv']=inner
            env=json.loads(verify(command['bindings']['native_environment']).read_text())
            env['PYTHONPATH']=env['PYTHONPATH'].replace(str(donor),str(source))
            environment=out/f'{mode}-{task}-environment.json';write(environment,env)
            command['bindings'].update(deployment=ref(deployment),wrapper=ref(source/'experiments/coupled_act_feature_student_runtime/native.py'),
                policy=ref(bridge/'__init__.py'),policy_config=ref(policy),native_environment=ref(environment))
            command_path=out/f'{mode}-{task}-command.json';write(command_path,command)
            plan=json.loads(verify(old['plan']).read_text());plan.update(schema='coupled_ACT_student_local3090_native32GiB_supervisor_v1',output_root=str(native.parent))
            job=plan['jobs'][0];job.update(name=mode+'-'+task,output_root=str(native),max_owned_gpu_processes=1,command=ref(command_path))
            job['argv']=[str(Path(command['native_root'])/'.venvs/sim/bin/python'),'-B',command['bindings']['wrapper']['path'],
                '--command-record',str(command_path),'--command-sha256',ref(command_path)['sha256'],'--output',str(native)]
            plan_path=out/f'{mode}-{task}-plan.json';write(plan_path,plan)
            if candidate is not None:
                load_manifest(deployment,ref(deployment)['sha256'])
                from .supervisor import validate_plan
                validate_plan(plan)
            rows.append(dict(task=task,mode=mode,deployment=ref(deployment),command=ref(command_path),plan=ref(plan_path),output=str(native)))
    campaign=dict(schema='coupled_ACT_student_exact_all10_parent_student_CPU_packet_v1',execution_authorized=False,
        source_root=str(source),source_sha256=pins,plans=rows,borrowed_teacher_campaign=base_ref,
        candidate=None if candidate is None else candidate,status='prospective_pending_actual_student_bytes_and_ROOT_admission' if candidate is None else 'CPU_actual_candidate_bound_pending_independent_review_and_GPU_admission',
        generator=2026102401,motor_seed=17,native_generated=50,prefix_executed=10,official_horizons=HORIZONS,
        GPU_launches=0,Qwen_models_loaded=0,new_model_weights_fabricated=False,quality_promoted=False)
    write(out/'campaign.json',campaign);return ref(out/'campaign.json')


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for key in ('base','output','run-root'):p.add_argument('--'+key,required=True)
    p.add_argument('--candidate');a=p.parse_args()
    candidate=None if a.candidate is None else json.loads(Path(a.candidate).read_text())
    print(json.dumps(prepare(a.base,a.output,a.run_root,candidate)))


if __name__=='__main__':main()
