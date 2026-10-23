"""Prospective all10 schedule, then exact manifests only after real training proof."""
import argparse
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
from .contracts import SCHEMA,STATE,ACTION,INPUTS,require,sha,load_manifest
from .native import COMMAND,GENERATOR
from experiments.goal_latent_act_campaign.common import ORDER,ref,write

ROOT = Path(__file__).resolve().parents[2]
NATIVE = Path('/tmp/ra5')
BRANCHES = ('parent','goal','goal_free')


def dependencies(root):
    names = ['experiments.coupled_goal_act_all14_runtime.'+p.stem for p in Path(__file__).parent.glob('*.py')]
    names += ['experiments.coupled_goal_act_all14.'+p.stem for p in (root/'experiments/coupled_goal_act_all14').glob('*.py')]
    found = {}
    while names:
        name=names.pop(); path=root.joinpath(*name.split('.')).with_suffix('.py')
        if not path.is_file(): path=root.joinpath(*name.split('.'))/'__init__.py'
        if not path.is_file() or path in found: continue
        found[path]=sha(path)
        package=name.split('.') if path.name=='__init__.py' else name.split('.')[:-1]
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node,ast.Import): names += [n.name for n in node.names if n.name.startswith(('experiments.','policy.'))]
            elif isinstance(node,ast.ImportFrom):
                target=node.module or ''
                if node.level: target='.'.join(package[:len(package)-node.level+1]+([target] if target else []))
                if target.startswith(('experiments.','policy.')):
                    names.append(target); names += [target+'.'+n.name for n in node.names if n.name!='*']
        for parent in path.parents:
            if parent==root: break
            if (parent/'__init__.py').is_file(): names.append('.'.join(parent.relative_to(root).parts))
    return {str(p.relative_to(root)):digest for p,digest in sorted(found.items())}


def protocol():
    horizons={task:json.loads((NATIVE/'upstream/RoboSynChallenge/configs'/task/'random/gym_config.json').read_text())['max_episode_steps'] for task in ORDER}
    return dict(schema='coupled_all14_all10_matched_parent_role_state_native_pilot_v1',execution_authorized=False,
        generator=GENERATOR,motor_seed=17,task_order=list(ORDER),branches=list(BRANCHES),episodes_per_task_per_branch=1,
        native_generated=50,prefix_executed=10,full_official_horizons=horizons,expert_max_attempts=100,
        fresh_current_RGB_state_roles_goal_every10=True,no_future_labels=True,MONITOR_selection=False,
        accepted_hand_all7_corrected=True,rejected_hand_all7_raw_native_bitexact=True,
        parent_control='raw native ACT generated50/executed10; identical task/scene protocol',
        scene_pairing='actual accepted scene+instruction+expert steps+horizon+generator+motor must match; unavailable is never score zero',
        historical_reuse='only identical accepted-scene and generated50/executed10 protocol; no aggregate leaderboard reuse as matched score',
        numeric_compiler_limits=dict(max_abs=1e-5,RMSE=2e-6,Boolean_decisions_bitexact=True,warm_calls=5),
        compilation='new affine/reliability segments; native GRU kernel; existing qualified FP32 eager ACT core',
        deployment_budget=dict(single5090_total_GiB=32,single5090_target_GiB=28,local3090_total_GiB=24,
            actual_new_combined_fit_qualified=False,whole_device_renderer_observer_motor_required=True),
        original_wait_reap_and_independent_whole_device_drain_required=True,
        scalar_full_horizon_for_all_tasks=False,quality_promoted=False)


def prepare(output,run_root,training_root,role_admission,state_admission,wait,drain,pair,gpu_uuid):
    output=Path(output).absolute(); run_root=Path(run_root).absolute()
    require(not output.exists() and not run_root.exists(), 'Fresh immutable preparation/future results required')
    output.mkdir(parents=True)
    p=protocol(); write(output/'protocol.json',p)
    closure=dependencies(ROOT)
    training_root=Path(training_root).resolve(strict=True)
    contracts={mode:json.loads((training_root/name/'checkpoint.json').read_text()) for mode,name in (('goal','role_goal'),('goal_free','state_action_control'))}
    head_path=Path(contracts['goal']['highlevel_contract']['path'])
    head=json.loads(head_path.read_text())
    for relative,digest in {**head['source_pins'],**contracts['goal']['source_pins'],**contracts['goal_free']['source_pins']}.items():
        require(sha(ROOT/relative)==digest, 'Final original scientific source changed: '+relative); closure[relative]=digest
    source=output/'source'; source.mkdir()
    for relative,digest in closure.items():
        target=source/relative; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/relative,target)
        require(sha(target)==digest, 'Copied exact source differs')
    policy=source/'policy/coupled_goal_act_all14_v1'; policy.mkdir(parents=True)
    shutil.copy2(ROOT/'experiments/coupled_goal_act_all14_runtime/policy_shim.py',policy/'__init__.py')
    config=policy/'deploy_policy.yml'
    config.write_text('policy_name: coupled_goal_act_all14_v1\nstate_obs_path: robot/qpos\nstrict_action_dim: true\nact_step: 10\n')
    env_source=ROOT/'outputs/architecture-reassessment/goal-latent-act-prefix10-pilot-preparation-v1/CPU-packet-v2/native-environment-private.json'
    env=json.loads(env_source.read_text())
    env.update(PYTHONPATH=os.pathsep.join(map(str,[source,NATIVE/'upstream/lerobot-sim/src',NATIVE/'upstream/EmbodiChain',NATIVE/'upstream/RoboSynChallenge',NATIVE/'upstream/RoboSynChallenge/policy'])),
        CUDA_VISIBLE_DEVICES=gpu_uuid,EMBODICHAIN_SIM_EXIT_PROCESS='0')
    env.pop('TORCHINDUCTOR_EMULATE_PRECISION_CASTS',None)
    env_path=output/'native-environment-private.json';write(env_path,env);env_path.chmod(0o600)
    templates=ROOT/'outputs/architecture-reassessment/goal-latent-act-prefix10-pilot-preparation-v1/CPU-packet-v2'
    plans=[]
    for task in ORDER:
        for mode in BRANCHES:
            cfg=json.loads((templates/f'parent10-{task}-deployment.json').read_text())
            cfg.update(schema=SCHEMA,mode=mode,source_root=str(source),source_sha256=closure,
                state_contract=STATE,action_contract=ACTION,actor_inputs=INPUTS,
                correction_compiler_backend='inductor',require_compile_speedup=True,
                native_grippers_bitexact=False,rejected_hand_all7_bitexact=True,hand_atomic_all7=True,
                qwen_python_path=[str(source)],runtime_family='coupled_all14')
            for key in ('arm_correction_bound_rad',): cfg.pop(key,None)
            if mode!='parent':
                name='role_goal' if mode=='goal' else 'state_action_control'; c=contracts[mode]
                cfg['assets'].update(coupled_contract=ref(training_root/name/'checkpoint.json'),
                    coupled_weights=ref(training_root/name/'coupled-all14.safetensors'),training_receipt=ref(training_root/name/'receipt.json'),
                    original_training_wait=ref(wait),independent_training_drain=ref(drain),training_pair=ref(pair),
                    runtime_admission=ref(role_admission if mode=='goal' else state_admission),
                    highlevel_contract=c['highlevel_contract'],highlevel_weights=c['highlevel_weights'])
            path=output/f'{mode}-{task}-deployment.json';write(path,cfg);load_manifest(path,sha(path))
            gym=NATIVE/'upstream/RoboSynChallenge/configs'/task/'random/gym_config.json'
            target=run_root/mode/task/'native'
            inner=[str(NATIVE/'upstream/RoboSynChallenge/scripts/eval_policy.py'),'--config',str(config),'--overrides',
                '--policy_name','coupled_goal_act_all14_v1','--task_name',task,'--setting','random','--model_name','coupled_all14_'+mode,
                '--max_episodes','1','--seed',str(GENERATOR),'--max_steps',str(p['full_official_horizons'][task]),'--act_step','10',
                '--state_obs_path','robot/qpos','--strict_action_dim','True','--eval_expert_check','True',
                '--eval_max_seed_attempts_per_episodes','100','--eval_reset_sync_steps','1','--renderer','hybrid','--headless','True',
                '--eval_video_log','True','--coupled_all14_manifest',str(path),'--coupled_all14_manifest_sha256',sha(path),
                '--eval_result_dir',str(target/'evaluator-results')]
            bindings=dict(deployment=ref(path),wrapper=ref(source/'experiments/coupled_goal_act_all14_runtime/native.py'),
                policy=ref(policy/'__init__.py'),policy_config=ref(config),protocol=ref(output/'protocol.json'),gym=ref(gym),
                bootstrap=ref(NATIVE/'experiments/simulator_cleanup_repair_v1/native_bootstrap_v2.py'),
                hook=ref(NATIVE/'experiments/simulator_cleanup_repair_v1/constraint_teardown.py'),
                SDK=ref(NATIVE/'upstream/EmbodiChain/embodichain/lab/sim/sim_manager.py'),
                cache_installer=ref(NATIVE/'experiments/goal_latent_act_submission/trained_hybrid/cleanup_v2/run_task.py'),
                cache_implementation=ref(NATIVE/'experiments/goal_latent_act_submission/trained_hybrid/runtime_cache.py'),
                evaluator=ref(NATIVE/'upstream/RoboSynChallenge/scripts/eval_policy.py'),native_environment=ref(env_path))
            command=dict(schema=COMMAND,task=task,mode=mode,bindings=bindings,GPU_uuid=gpu_uuid,native_root=str(NATIVE),
                cwd=str(NATIVE/'upstream/RoboSynChallenge'),inner_argv=inner,official_horizon=p['full_official_horizons'][task])
            command_path=output/f'{mode}-{task}-command.json';write(command_path,command)
            argv=[str(NATIVE/'.venvs/sim/bin/python'),'-B',str(source/'experiments/coupled_goal_act_all14_runtime/native.py'),
                '--command-record',str(command_path),'--command-sha256',sha(command_path),'--output',str(target)]
            from .supervisor import SCHEMA as SUPERVISOR,validate_plan
            plan=dict(schema=SUPERVISOR,gpu_uuid=gpu_uuid,admission_memory_floor_gib=32,
                whole_GPU_cap_bytes=24*(1<<30)-256*(1<<20),owned_tree_RSS_cap_bytes=32*(1<<30),
                borrowed_guard=ref(source/'experiments/coupled_goal_act_all14/supervisor.py'),output_root=str(target.parent),
                jobs=[dict(name=mode+'-'+task,argv=argv,cwd=str(ROOT),output_root=str(target),wall_seconds=1800,
                    max_owned_gpu_processes=2 if mode=='goal' else 1,command=ref(command_path))])
            validate_plan(plan);plan_path=output/f'{mode}-{task}-plan.json';write(plan_path,plan)
            plans.append(dict(mode=mode,task=task,command=ref(command_path),deployment=ref(path),plan=ref(plan_path),output=str(target),
                original_argv=argv,prospective_process_count=2 if mode=='goal' else 1,official_horizon=p['full_official_horizons'][task]))
    packet=dict(schema='coupled_all14_exact_all10_parent_role_state_CPU_packet_v1',execution_authorized=False,
        protocol=ref(output/'protocol.json'),source_root=str(source),source_sha256=closure,plans=plans,
        independent_CPU_review_pending=True,fresh_GPU_owner_admission_pending=True,GPU_launches=0,worlds_constructed=0)
    write(output/'campaign.json',packet)
    return packet


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False);parser.add_argument('--output',required=True)
    parser.add_argument('--plan-only',action='store_true')
    for key in ('run-root','training-root','role-admission','state-admission','wait','drain','pair','gpu-uuid'):parser.add_argument('--'+key)
    args=parser.parse_args()
    if args.plan_only:
        out=Path(args.output);require(not out.exists(),'Fresh conditional protocol output');out.mkdir(parents=True)
        write(out/'prospective-protocol.json',protocol());print(json.dumps(dict(status='CPU_conditional_all10_protocol_no_execution',GPU_launches=0)));return
    require(all(getattr(args,key.replace('-','_')) for key in ('run-root','training-root','role-admission','state-admission','wait','drain','pair','gpu-uuid')), 'Actual final training/root evidence required')
    result=prepare(args.output,args.run_root,args.training_root,args.role_admission,args.state_admission,args.wait,args.drain,args.pair,args.gpu_uuid)
    print(json.dumps(dict(status='CPU_exact_all10_manifest_packet_pending_independent_review_and_fresh_GPU_owner',commands=len(result['plans']),GPU_launches=0)))


if __name__=='__main__': main()
