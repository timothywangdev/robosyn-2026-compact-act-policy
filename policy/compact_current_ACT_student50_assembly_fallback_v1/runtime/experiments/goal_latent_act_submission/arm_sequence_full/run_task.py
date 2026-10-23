"""Native command for the typed experimental ARM12 actor, no alternate inference."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
# Keep FILE invocation outside actor folders containing a sibling queue module.
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from experiments.goal_latent_act_submission.arm_sequence_full.contracts import *
from experiments.goal_latent_act_submission.run_task import metadata_check,interpreter

BOOTSTRAP='00a17e1ac358e13407a12a6b597fed60bf8a06b900c725ad238728aed164b607'
HOOK='06db4bd673f6e1f9c7beb8bacc03e0ca85a67d1d50b3d6838e129f5ea699490f'


def command(root,cfg,manifest,task,setting,seed,episodes,budget,output,*,python=None):
    require(task in TASKS and cfg['task']==task and cfg['mode'] in MODES,'Exact typed mode/task')
    original=json.loads(child(root,f'upstream/RoboSynChallenge/configs/{task}/{setting}/gym_config.json').read_text())
    horizon=original['max_episode_steps'];instruction=original['env']['dataset']['lerobot']['params']['instruction']['lang']
    require(setting=='random' and instruction==cfg['instruction'],'Native instruction/random setting differs')
    require(type(seed)is int and 0<=seed<2**32 and type(episodes)is int and episodes>0 and type(budget)is int and budget>0,'Explicit organizer arguments')
    bootstrap=child(root,'experiments/simulator_cleanup_repair_v1/native_bootstrap_v2.py')
    require(sha(bootstrap)==BOOTSTRAP and sha(child(root,'experiments/simulator_cleanup_repair_v1/constraint_teardown.py'))==HOOK,'Qualified native cleanup source pair differs')
    spec=json.loads((root/'submission.json').read_text())
    record=json.loads(child(root,spec['banks'][cfg['mode']][task]['native_command_template']).read_text())
    require(record['schema']=='arm12_portable_native_command_template_v1','Original native command template required')
    replacements={'{evaluator}':str(child(root,'upstream/RoboSynChallenge/scripts/eval_policy.py')),
        '{config}':str(root/'policy/goal_latent_act/deploy_policy.yml'),'{manifest}':str(manifest),
        '{manifest_sha}':sha(manifest),'{results}':str(output/'official-results')}
    inner=[replacements.get(value,value)for value in record['inner_argv']]
    for key,value in [('--task_name',task),('--setting',setting),('--seed',seed),('--max_episodes',episodes),
                      ('--max_steps',horizon),('--eval_max_seed_attempts_per_episodes',budget)]:
        require(inner.count(key)==1,'Unique original native flag required: '+key);inner[inner.index(key)+1]=str(value)
    # A supplied interpreter is solely for explicit CPU reuse review. Real
    # execution defaults to the contained, newly installed isolated environment.
    return [str(python if python is not None else interpreter(root,'sim')),'-B',str(bootstrap),'--',*inner]


def environment(root,spec):
    env=os.environ.copy()
    for key in ('PYTHONPATH','PYTHONHOME','PYTHONSTARTUP','PYTHONUSERBASE'):env.pop(key,None)
    env.update(PYTHONPATH=os.pathsep.join(map(str,[root,root/'upstream/lerobot-sim/src',root/'upstream/EmbodiChain',root/'upstream/RoboSynChallenge'])),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PIP_NO_INDEX='1',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        TORCH_HOME=str(root/'cache/torch'),EMBODICHAIN_ROOT=str(root/'upstream/EmbodiChain'),
        EMBODICHAIN_DATA_ROOT=str(child(root,spec['embodichain_data_root'])),EMBODICHAIN_SIM_EXIT_PROCESS='0')
    cache_file=child(root,'experiments/goal_latent_act_submission/trained_hybrid/runtime_cache.py')
    spec_module=importlib.util.spec_from_file_location('_arm_portable_private_native_cache',cache_file)
    cache=importlib.util.module_from_spec(spec_module);spec_module.loader.exec_module(cache)
    return cache.prepare(root,spec,env,scratch_parent='/tmp')


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--root',required=True);p.add_argument('--task',choices=sorted(TASKS),required=True)
    p.add_argument('--mode',choices=sorted(MODES),required=True);p.add_argument('--setting',default='random')
    p.add_argument('--seed',type=int,required=True);p.add_argument('--episodes',type=int,required=True)
    p.add_argument('--expert-seed-attempts',type=int,default=100);p.add_argument('--output',required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    root=Path(a.root).resolve(strict=True);spec=json.loads((root/'submission.json').read_text());verify_stage(root,spec);metadata_check(root,spec)
    output=Path(a.output).absolute();require(not output.exists(),'Fresh output only');output.mkdir(parents=True)
    scratch=output/'scratch';scratch.mkdir()
    cfg=materialize(root,json.loads(child(root,spec['banks'][a.mode][a.task]['template']).read_text()),scratch)
    manifest=output/'deployment.json';json_write(manifest,cfg);load_native(root,manifest,sha(manifest))
    argv=command(root,cfg,manifest,a.task,a.setting,a.seed,a.episodes,a.expert_seed_attempts,output)
    receipt=dict(schema='arm12_portable_native_invocation_v1',status='CPU_science_source_path_binding_no_model_or_robot',mode=a.mode,task=a.task,
        manifest=reference(manifest),argv=argv,models_constructed=0,GPU_launches=0,primary_variant=None,offline_promotion_criterion_passed=False)
    json_write(output/'command.json',receipt)
    if not a.execute:print(json.dumps(receipt));return
    env,cache_proof=environment(root,spec);json_write(output/'private-runtime-cache-proof.json',cache_proof)
    os.chdir(root/'upstream/RoboSynChallenge');os.execvpe(argv[0],argv,env)


if __name__=='__main__':main()
