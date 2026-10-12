"""Portable command construction; native execution needs separate root admission."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.coupled_act_feature_student_cadence50_submission.contracts import materialize,require,ref,sha,write

def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for key in ('root','task','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--mode',choices=('student',),required=True);p.add_argument('--setting',default='random')
    p.add_argument('--seed',type=int,required=True);p.add_argument('--episodes',type=int,required=True)
    p.add_argument('--expert-seed-attempts',type=int,default=100);p.add_argument('--check-only',action='store_true')
    p.add_argument('--admission');p.add_argument('--admission-sha256');a=p.parse_args()
    root=Path(a.root).resolve(strict=True);spec=json.loads((root/'submission.json').read_text())
    row=next((r for r in spec['tasks'] if (r['task'],r['mode'])==(a.task,a.mode)),None)
    require(row is not None and a.setting=='random' and 0<=a.seed<2**32 and a.episodes>0 and a.expert_seed_attempts>0,'Explicit organizer controls required')
    out=Path(a.output).absolute();require(not out.exists() and not out.is_symlink(),'Fresh command output');out.mkdir(parents=True)
    scratch=out/'binding-check';scratch.mkdir();template=root/row['template'];path,cfg=materialize(root,template,scratch)
    gym=json.loads((root/f'upstream/RoboSynChallenge/configs/{a.task}/{a.setting}/gym_config.json').read_text())
    require(gym['max_episode_steps']==row['official_horizon'] and gym['env']['dataset']['lerobot']['params']['instruction']['lang']==cfg['instruction'],'Exact native horizon/instruction')
    argv=[str(root/'.venvs/sim/bin/python'),'-B',str(root/'experiments/simulator_cleanup_repair_v1/native_bootstrap_v2.py'),'--',
        str(root/'upstream/RoboSynChallenge/scripts/eval_policy.py'),'--config',str(root/'policy/compact_current_ACT_student50_v1/deploy_policy.yml'),
        '--overrides','--policy_name','compact_current_ACT_student50_v1','--task_name',a.task,'--setting',a.setting,
        '--model_name','compact-student-unpromoted-'+a.mode,'--seed',str(a.seed),'--max_episodes',str(a.episodes),
        '--max_steps',str(row['official_horizon']),'--act_step','50','--state_obs_path','robot/qpos','--strict_action_dim','True',
        '--eval_expert_check','True','--eval_max_seed_attempts_per_episodes',str(a.expert_seed_attempts),
        '--eval_reset_sync_steps','1','--renderer','hybrid','--headless','True','--eval_video_log','True',
        '--compact_student_submission_root',str(root),'--compact_student_submission_template',str(template),
        '--compact_student_portable_scratch',str(out/'actor-scratch'),'--compact_student_portable_admission',str(a.admission),
        '--compact_student_portable_admission_sha256',str(a.admission_sha256),'--eval_result_dir',str(out/'official-results')]
    receipt=dict(schema='compact_ACT_student50_portable_entry_CPU_command_v1',package=ref(root/'submission.json'),template=ref(template),
        operational=ref(path),argv=argv,task=a.task,mode=a.mode,official_horizon=row['official_horizon'],scene_seed=a.seed,motor_seed=17,
        native_generated=50,prefix_executed=50,models_constructed=0,GPU_queries=0,worlds_constructed=0,submission_ready=False)
    write(out/'command.json',receipt)
    if a.check_only:print(json.dumps(receipt));return
    require(a.admission and a.admission_sha256 and sha(a.admission)==a.admission_sha256,'Separate portable native root admission required')
    proof=json.loads(Path(a.admission).read_text())
    require(proof['schema']=='root_authorized_compact_ACT_student50_portable_native_v1' and proof['execution_authorized'] is True
        and proof['robot_quality_promoted'] is False and proof['package']==receipt['package'] and proof['template']==receipt['template']
        and proof['entry_source']==ref(root/'experiments/coupled_act_feature_student_cadence50_submission/policy.py'),'Exact portable package/entry admission')
    env=os.environ.copy()
    for key in ('PYTHONPATH','PYTHONHOME','PYTHONSTARTUP','PYTHONUSERBASE','TORCHINDUCTOR_EMULATE_PRECISION_CASTS'):env.pop(key,None)
    env.update(PYTHONPATH=os.pathsep.join(map(str,[root,root/'upstream/lerobot-sim/src',root/'upstream/EmbodiChain',root/'upstream/RoboSynChallenge'])),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        EMBODICHAIN_ROOT=str(root/'upstream/EmbodiChain'),EMBODICHAIN_DATA_ROOT=str(root/'assets/embodichain/data'),
        EMBODICHAIN_SIM_EXIT_PROCESS='0',TORCH_HOME=str(root/'cache/torch'),TORCHINDUCTOR_COMPILE_THREADS='1')
    from experiments.goal_latent_act_submission.trained_hybrid.runtime_cache import prepare
    env,cache=prepare(root,spec,env,scratch_parent='/tmp');write(out/'private-runtime-cache-proof.json',cache)
    os.chdir(root/'upstream/RoboSynChallenge');os.execvpe(argv[0],argv,env)

if __name__=='__main__':main()
