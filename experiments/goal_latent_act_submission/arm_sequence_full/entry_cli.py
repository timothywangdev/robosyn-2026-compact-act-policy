"""Organizer five-argument entry for an explicitly selected local ARM payload.

The checkpoint directory is a materialized offline payload. No network fetch or
implicit primary selection occurs. Cold installation uses only its bound wheels.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from experiments.goal_latent_act_submission.arm_sequence_full.contracts import MODES,TASKS,require,verify_stage


def native_argv(root,mode,task,setting,seed,episodes,budget,output,execute):
    require(mode in MODES and task in TASKS and setting=='random','Explicit ARM mode/official task/random setting required')
    require(type(seed)is int and 0<=seed<2**32 and type(episodes)is int and episodes>0
            and type(budget)is int and budget>0,'Explicit valid organizer controls required')
    argv=[str(root/'.venvs/sim/bin/python'),'-B',str(root/'experiments/goal_latent_act_submission/arm_sequence_full/run_task.py'),
          '--root',str(root),'--mode',mode,'--task',task,'--setting',setting,'--seed',str(seed),
          '--episodes',str(episodes),'--expert-seed-attempts',str(budget),'--output',str(output)]
    return argv+(['--execute']if execute else [])


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for name in ('task','setting','checkpoint_root','model_name','gpu_id'):p.add_argument(name)
    p.add_argument('--root');p.add_argument('--mode',choices=sorted(MODES),required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--episodes',type=int,required=True)
    p.add_argument('--expert-seed-attempts',type=int,default=100);p.add_argument('--output',required=True)
    p.add_argument('--base-python',default=sys.executable);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--routing-only',action='store_true');p.add_argument('--reuse-installed',action='store_true')
    a=p.parse_args();root=Path(a.root or a.checkpoint_root).resolve(strict=True);output=Path(a.output).absolute()
    spec=json.loads((root/'submission.json').read_text());verify_stage(root,spec,full=not a.routing_only)
    require(not output.exists() and not output.is_symlink(),'Fresh organizer output required')
    require(a.gpu_id.isdecimal(),'Explicit nonnegative organizer CUDA index required')
    argv=native_argv(root,a.mode,a.task,a.setting,a.seed,a.episodes,a.expert_seed_attempts,output,not a.prepare_only)
    if a.routing_only:
        print(json.dumps(dict(schema='arm12_visible_organizer_entry_v1',status='actual_CLI_routing_only_no_install_or_model',
            mode=a.mode,task=a.task,native_argv=argv,organizer_model_label=a.model_name,
            primary_variant=None,offline_promotion_criterion_passed=False,GPU_launches=0)))
        return
    cpu_env=os.environ.copy();cpu_env.update(CUDA_VISIBLE_DEVICES='',PIP_NO_INDEX='1',HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
    for key in ('PYTHONPATH','PYTHONHOME','PYTHONSTARTUP','PYTHONUSERBASE'):cpu_env.pop(key,None)
    lock=os.open(root/'.offline-entry-install.lock',os.O_CREAT|os.O_RDWR,0o600)
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.reuse_installed:
            from experiments.goal_latent_act_submission.run_task import metadata_check
            metadata_check(root,spec)
        else:
            require(not(root/'.venvs').exists(),'Prior install retained; use a fresh payload or explicit --reuse-installed')
            log=root/'entry-cold-install.log';fd=os.open(log,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
            with os.fdopen(fd,'wb')as stream:
                result=subprocess.run([a.base_python,'-B',str(root/'experiments/goal_latent_act_submission/arm_sequence_full/install.py'),
                    '--root',str(root),'--base-python',a.base_python,'--execute'],cwd=root,env=cpu_env,
                    stdin=subprocess.DEVNULL,stdout=stream,stderr=stream,timeout=1800)
            require(result.returncode==0,'Original offline installer failed; logs/partial environment retained, no retry')
    finally:
        os.close(lock)
    env=dict(cpu_env if a.prepare_only else os.environ)
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PIP_NO_INDEX='1',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
    if not a.prepare_only:env['CUDA_VISIBLE_DEVICES']=a.gpu_id
    for key in ('PYTHONPATH','PYTHONHOME','PYTHONSTARTUP','PYTHONUSERBASE'):env.pop(key,None)
    os.execvpe(argv[0],argv,env)


if __name__=='__main__':main()
