"""Exact same actual trained candidate; executor50 is a separate CPU packet."""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import shutil
from .contracts import SCHEMA,HORIZONS,load_manifest,require,ref,verify
from .native import COMMAND
from .supervisor import SCHEMA as GUARD_SCHEMA,validate_plan
from experiments.goal_latent_act_campaign.common import write

ROOT=Path(__file__).resolve().parents[2]
ORDER=('click_bell','drawer_open_place','handle_basket','item_assembly','items_handover',
       'manipulate_pipette','mixer_operating','sample_loading','table_rearrangement','water_pouring')
BRANCHES=('student',)

def build(base_path,output,run_root):
    oldref=ref(base_path);base=json.loads(verify(oldref).read_text());out=Path(output).absolute();run=Path(run_root).absolute()
    require(not out.exists() and not run.exists(),'Fresh executor50 source/results roots');out.mkdir(parents=True);source=out/'source';source.mkdir()
    pins={};oldsource=Path(base['source_root'])
    for name,digest in base['source_sha256'].items():
        src=oldsource/name;require(ref(src)['sha256']==digest,'Frozen executor10 source changed')
        dst=source/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);pins[name]=digest
    for src in Path(__file__).parent.glob('*.py'):
        name=str(src.relative_to(ROOT));dst=source/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);pins[name]=ref(dst)['sha256']
    bridge=source/'policy/coupled_ACT_current_student_cadence50_v1';bridge.mkdir(parents=True)
    (bridge/'__init__.py').write_text('from experiments.coupled_act_feature_student_cadence50.policy import get_model, eval, reset_model, close_model\n')
    policy=bridge/'deploy_policy.yml';policy.write_text('policy_name: coupled_ACT_current_student_cadence50_v1\nstate_obs_path: robot/qpos\nstrict_action_dim: true\nact_step: 50\n')
    pins[str((bridge/'__init__.py').relative_to(source))]=ref(bridge/'__init__.py')['sha256']
    protocol=dict(schema='current_ACT_student_execute50_prospective_native_protocol_v1',tasks=list(ORDER),modes=list(BRANCHES),
        generator=2026102401,motor_seed=17,episodes_per_task=1,native_generated=50,prefix_executed=50,official_horizons=HORIZONS,
        changed_policy='executor_cadence_only_current_observation_and_correction_every50_native_actions',
        original_student_weights=base['candidate']['student_weights'],old_generated50_correction_math_retained=True,
        current_inputs=['current_RGB_via_same_ACT_forward','current_qpos','current_FK'],future_inputs_allowed=False,
        no_second_CNN_forward=True,Qwen_models_loaded=0,hand_atomic_all7=True,raw_native_fallback_bitexact=True,
        numerical_first_current_compile_attempt_retained=True,native_execution50_qualified=False,
        original_prefix10_robot_or_quality_qualification_reused=False,robot_quality_promoted=False,execution_authorized=False)
    write(out/'protocol.json',protocol);rows=[]
    for task in ORDER:
        old=next(r for r in base['plans'] if r['task']==task and r['mode']=='student')
        cfg=json.loads(verify(old['deployment']).read_text());cfg.update(schema=SCHEMA,source_root=str(source),source_sha256=pins,
            prefix_executed=50,scratch_root=str(run/task/'actor-scratch'),executor_variant='same_trained_current_ACT_student_execute50_v1',
            original_prefix10_quality_qualification_reused=False,execution_authorized=False)
        dep=out/(task+'-deployment.json');write(dep,cfg);load_manifest(dep,ref(dep)['sha256'])
        cmd=json.loads(verify(old['command']).read_text());cmd['schema']=COMMAND
        env=json.loads(verify(cmd['bindings']['native_environment']).read_text());env['PYTHONPATH']=env['PYTHONPATH'].replace(str(oldsource),str(source))
        ep=out/(task+'-environment.json');write(ep,env)
        cmd['bindings'].update(deployment=ref(dep),protocol=ref(out/'protocol.json'),wrapper=ref(source/'experiments/coupled_act_feature_student_cadence50/native.py'),
            policy=ref(bridge/'__init__.py'),policy_config=ref(policy),native_environment=ref(ep))
        native=run/task/'native'
        for flag,value in (('--config',str(policy)),('--policy_name','coupled_ACT_current_student_cadence50_v1'),('--act_step','50'),
            ('--coupled_student_manifest',str(dep)),('--coupled_student_manifest_sha256',ref(dep)['sha256']),
            ('--model_name','same-trained-student-execute50-v1'),('--eval_result_dir',str(native/'evaluator-results'))):
            require(cmd['inner_argv'].count(flag)==1,'Original flag absent/ambiguous');cmd['inner_argv'][cmd['inner_argv'].index(flag)+1]=value
        cp=out/(task+'-command.json');write(cp,cmd)
        plan=json.loads(verify(old['plan']).read_text());plan.update(schema=GUARD_SCHEMA,output_root=str(native.parent))
        job=plan['jobs'][0];job.update(output_root=str(native),command=ref(cp))
        job['argv']=[str(Path(cmd['native_root'])/'.venvs/sim/bin/python'),'-B',cmd['bindings']['wrapper']['path'],
            '--command-record',str(cp),'--command-sha256',ref(cp)['sha256'],'--output',str(native)]
        pp=out/(task+'-plan.json');write(pp,plan);validate_plan(plan)
        rows.append(dict(task=task,mode='student',deployment=ref(dep),command=ref(cp),plan=ref(pp),output=str(native),original_prefix10_slot=old))
    campaign=deepcopy(base);campaign.update(schema='coupled_ACT_student_execute50_exact_all10_CPU_packet_v1',source_root=str(source),source_sha256=pins,
        plans=rows,protocol=ref(out/'protocol.json'),prefix_executed=50,execution_authorized=False,original_prefix10_campaign=oldref,
        status='CPU_actual_same_trained_weights_execute50_pending_independent_review_and_separate_native_admission',
        cadence50_robot_qualification=False,quality_promoted=False)
    write(out/'campaign.json',campaign);print(json.dumps(dict(campaign=ref(out/'campaign.json'),plans=len(rows),sources=len(pins),GPU_queries=0,worlds=0)))

if __name__=='__main__':
    p=argparse.ArgumentParser(allow_abbrev=False)
    for k in ('base','output','run-root'):p.add_argument('--'+k,required=True)
    a=p.parse_args();build(a.base,a.output,a.run_root)
