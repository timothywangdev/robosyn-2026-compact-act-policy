"""Official custom-policy entrypoint; materialize pinned paths then native exec."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.goal_latent_act_submission.contracts import *


def verify_stage(root,spec):
    require(spec['schema']in (SCHEMA,SCHEMA50) and set(spec['tasks'])==TASKS and spec['RTX5090_qualified']is False,'Explicit all10 unqualified portable stage required')
    if spec['schema']==SCHEMA50:
        require(spec['parent_schema']==PARENT50 and all(spec['protocol'][k]==50 for k in ('generated','executed','refresh'))
            and spec['protocol']['discarded']==0,'Explicit native50 portable protocol required')
    for name,row in spec['files'].items():
        path=child(root,name);require(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Installed source/model changed: '+name)
    parent=PARENT50 if spec['schema']==SCHEMA50 else PARENT
    for task,row in spec['tasks'].items():
        template=json.loads(child(root,row['template']).read_text())
        require(template['task']==task and template['schema']==PROTOCOLS[parent][1] and template['parent_schema']==parent,
                'Portable all10 task template and stage protocol differ')


def interpreter(root,name):
    path=root/'.venvs'/name/'bin/python'
    require(path.parent.resolve().is_relative_to(root) and path.is_file() and os.access(path,os.X_OK),'Missing isolated '+name+' interpreter')
    return path


def metadata_check(root,spec):
    program="""import importlib.metadata,json,sys
sys.path[:0]=json.loads(sys.argv[1]);expected=json.loads(sys.argv[2])
for name,version in expected.items():
 actual=importlib.metadata.version(name)
 assert actual==version,(name,actual,version)
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
print(json.dumps({'packages':len(expected),'models_constructed':0}))
"""
    for name,row in spec['environments'].items():
        python=interpreter(root,'sim'if name=='sim'else'qwen35')
        extra=[str(root/'dependencies/qwen-overlay')]if name=='qwen-overlay'else[]
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
        result=subprocess.run([str(python),'-I','-B','-c',program,json.dumps(extra),json.dumps(row['packages'])],env=env,
            stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=120)
        require(result.returncode==0,'Installed '+name+' dependency versions differ: '+result.stderr[-4000:])


def command(root,cfg,manifest,task,setting,seed,episodes,budget,output):
    gym=child(root,f'upstream/RoboSynChallenge/configs/{task}/{setting}/gym_config.json')
    original=json.loads(gym.read_text());horizon=original['max_episode_steps']
    instruction=original['env']['dataset']['lerobot']['params']['instruction']['lang']
    require(setting=='random' and instruction==cfg['instruction'],'Only source-bound original random setting/instruction admitted')
    require(type(horizon)is int and horizon>0 and type(seed)is int and 0<=seed<2**32
        and type(episodes)is int and episodes>0 and type(budget)is int and budget>0,'Explicit valid organizer controls required')
    return [str(interpreter(root,'sim')),'-B','scripts/eval_policy.py','--config',str(root/'policy/goal_latent_act_submission/deploy_policy.yml'),
        '--overrides','--task_name',task,'--setting',setting,'--model_name','trained-goal-latent-ACT-prefix'+str(cfg['prefix_executed']),
        '--seed',str(seed),'--max_episodes',str(episodes),'--max_steps',str(horizon),
        '--eval_expert_check','True','--eval_max_seed_attempts_per_episodes',str(budget),
        '--goal_latent_manifest',str(manifest),'--goal_latent_manifest_sha256',sha(manifest),
        '--eval_result_dir',str(output/'official-results')]


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--root',default=str(Path(__file__).resolve().parents[2]));p.add_argument('--task',choices=sorted(TASKS),required=True)
    p.add_argument('--setting',default='random');p.add_argument('--seed',type=int,required=True);p.add_argument('--episodes',type=int,required=True)
    p.add_argument('--expert-seed-attempts',type=int,default=100);p.add_argument('--output',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    root=Path(a.root).resolve(strict=True);spec=json.loads((root/'submission.json').read_text());verify_stage(root,spec);metadata_check(root,spec)
    output=Path(a.output).absolute();require(not output.exists() and not output.is_symlink(),'Fresh invocation output required')
    output.mkdir(parents=True);scratch=output/'scratch';scratch.mkdir()
    template=json.loads(child(root,spec['tasks'][a.task]['template']).read_text());cfg=materialize(root,template,scratch)
    manifest=output/'deployment.json';json_write(manifest,cfg);load_native(root,manifest,sha(manifest))
    argv=command(root,cfg,manifest,a.task,a.setting,a.seed,a.episodes,a.expert_seed_attempts,output)
    receipt=dict(schema='goal_latent_act_submission_invocation_v1',status='CPU_binding_and_dependencies_pass_no_model',task=a.task,
        argv=argv,manifest_sha256=sha(manifest),models_constructed=0,RTX5090_qualified=False)
    json_write(output/'command.json',receipt)
    if not a.execute:
        print(json.dumps(receipt));return
    env=os.environ.copy()
    for key in ('PYTHONPATH','PYTHONHOME','PYTHONSTARTUP','PYTHONUSERBASE'):env.pop(key,None)
    env.update(PYTHONPATH=os.pathsep.join(map(str,[root,root/'upstream/lerobot-sim/src',root/'upstream/EmbodiChain',root/'upstream/RoboSynChallenge'])),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        TORCH_HOME=str(root/'cache/torch'),EMBODICHAIN_ROOT=str(root/'upstream/EmbodiChain'),
        EMBODICHAIN_DATA_ROOT=str(child(root,spec['embodichain_data_root'])),EMBODICHAIN_SIM_EXIT_PROCESS='0',
        EMBODICHAIN_MATERIAL_CACHE_DIR=str(output/'materials'))
    # Caller/organizer selects its admitted GPU. No device remapping, auto rental,
    # hidden future inputs, checkpoint switching, or online download is performed.
    os.chdir(root/'upstream/RoboSynChallenge');os.execvpe(argv[0],argv,env)


if __name__=='__main__':main()
