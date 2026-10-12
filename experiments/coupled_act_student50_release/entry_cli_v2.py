"""Verified selected package -> submitted YAML/four hooks -> official evaluator."""
import argparse,json,os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.coupled_act_student50_release.binder import verify_package,ref,write
from experiments.coupled_act_feature_student_cadence50_submission.contracts import materialize

POLICY='compact_current_ACT_student50_release_v1'

def command(root,public,task,seed,episodes,output,attempts=100):
    root=Path(root).resolve(strict=True);public=Path(public).resolve(strict=True);out=Path(output).absolute()
    if out.exists() or out.is_symlink() or not 0<=seed<2**32 or episodes<=0 or attempts<=0:
        raise ValueError('Fresh output and explicit valid organizer controls required')
    spec,verification=verify_package(root)
    row=next((r for r in spec['tasks'] if r['task']==task),None)
    if row is None:raise ValueError('Supported exact all10 task required')
    out.mkdir(parents=True);scratch=out/'binding-check';scratch.mkdir()
    template=root/row['template'];path,cfg=materialize(root,template,scratch)
    gym=json.loads((root/f'upstream/RoboSynChallenge/configs/{task}/random/gym_config.json').read_text())
    if gym['max_episode_steps']!=row['official_horizon'] or gym['env']['dataset']['lerobot']['params']['instruction']['lang']!=cfg['instruction']:
        raise ValueError('Exact official instruction/horizon required')
    yaml=public/'policy'/POLICY/'deploy_policy.yml'
    if not yaml.is_file() or yaml.is_symlink() or ('policy_name: '+POLICY) not in yaml.read_text():
        raise ValueError('Submitted public YAML required')
    argv=[str(root/'.venvs/sim/bin/python'),'-B',str(root/'experiments/simulator_cleanup_repair_v1/native_bootstrap_v2.py'),'--',
      str(root/'upstream/RoboSynChallenge/scripts/eval_policy.py'),'--config',str(yaml),'--overrides',
      '--policy_name',POLICY,'--task_name',task,'--setting','random','--model_name','compact-student-selected50',
      '--seed',str(seed),'--max_episodes',str(episodes),'--max_steps',str(row['official_horizon']),'--act_step','50',
      '--state_obs_path','robot/qpos','--strict_action_dim','True','--eval_expert_check','True',
      '--eval_max_seed_attempts_per_episodes',str(attempts),'--eval_reset_sync_steps','1','--renderer','hybrid',
      '--headless','True','--eval_video_log','True','--compact_student_submission_root',str(root),
      '--eval_result_dir',str(out/'official-results')]
    receipt=dict(schema='Student50_submitted_public_fourhooks_official_entry_CPU_v2',argv=argv,package=verification['package'],
      public_YAML=ref(yaml),public_fourhooks=ref(public/'policy'/POLICY/'deploy_policy.py'),task=task,
      official_horizon=row['official_horizon'],scene_seed=seed,motor_seed=17,episodes=episodes,
      native_generated=50,prefix_executed=50,automatic_binding=True,manual_admission_required=False,
      scientific_actor_changed=False,models_constructed=0,worlds_constructed=0,GPU_queries=0)
    write(out/'command.json',receipt)
    return receipt,spec

def native_environment(root,public,inherited):
    env=dict(inherited)
    for key in ('PYTHONPATH','PYTHONHOME','PYTHONSTARTUP','PYTHONUSERBASE','TORCHINDUCTOR_EMULATE_PRECISION_CASTS'):env.pop(key,None)
    env.update(PYTHONPATH=os.pathsep.join(map(str,[public,root,root/'upstream/lerobot-sim/src',root/'upstream/EmbodiChain',root/'upstream/RoboSynChallenge'])),
      ROBOSYN_STUDENT50_PACKAGE_ROOT=str(root),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONNOUSERSITE='1',
      PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',EMBODICHAIN_ROOT=str(root/'upstream/EmbodiChain'),
      EMBODICHAIN_DATA_ROOT=str(root/'assets/embodichain/data'),EMBODICHAIN_SIM_EXIT_PROCESS='0',
      TORCH_HOME=str(root/'cache/torch'),TORCHINDUCTOR_COMPILE_THREADS='1')
    return env

def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--root',default=str(Path(__file__).resolve().parents[2]/'artifact'))
    for name in ('task','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--episodes',type=int,required=True)
    p.add_argument('--expert-seed-attempts',type=int,default=100);p.add_argument('--check-only',action='store_true');a=p.parse_args()
    root=Path(a.root).resolve(strict=True);public=Path(__file__).resolve().parents[2]
    receipt,spec=command(root,public,a.task,a.seed,a.episodes,a.output,a.expert_seed_attempts)
    if a.check_only:print(json.dumps(receipt));return
    env=native_environment(root,public,os.environ)
    from experiments.goal_latent_act_submission.trained_hybrid.runtime_cache import prepare
    env,cache=prepare(root,spec,env,scratch_parent='/tmp');write(Path(a.output)/'private-runtime-cache-proof.json',cache)
    os.chdir(root/'upstream/RoboSynChallenge');os.execvpe(receipt['argv'][0],receipt['argv'],env)

if __name__=='__main__':main()
