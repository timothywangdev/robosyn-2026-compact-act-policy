"""Offline cold environment installation into a fresh packaged stage."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from experiments.goal_latent_act_submission.arm_sequence_full.contracts import *
from experiments.goal_latent_act_submission.arm_sequence_full.run_task import verify_stage,metadata_check


def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--root',required=True);p.add_argument('--base-python',required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args();root=Path(a.root).resolve(strict=True)
    spec=json.loads((root/'submission.json').read_text());verify_stage(root,spec)
    require(Path(a.base_python).is_absolute() and Path(a.base_python).is_file(),'Explicit existing Python3.11 bootstrap required')
    require(not(root/'.venvs').exists(),'Fresh cold virtual environments only')
    wheels=child(root,'dependencies/wheels');require(wheels.is_dir(),'Pinned offline wheel directory required')
    for row in spec['environments'].values():
        lines=child(root,row['lock']).read_text().splitlines()
        require((lines or not row['packages']) and all(re.fullmatch(r'[A-Za-z0-9_.-]+==[A-Za-z0-9_.+!-]+ --hash=sha256:[0-9a-f]{64}',line)for line in lines),
                'Flat offline exact requirement hashes required; no remote URLs or resolver choices')
    result=subprocess.run([a.base_python,'-I','-B','-c','import sys; assert sys.version_info[:2]==(3,11)'],capture_output=True,timeout=20)
    require(result.returncode==0,'Exact Python3.11 bootstrap required')
    phases=[]
    bootstrap={}
    for name in ('sim','qwen35'):
        python=root/'.venvs'/name/'bin/python'
        all_lines=child(root,spec['environments'][name]['lock']).read_text().splitlines()
        bootstrap[name]=[line for line in all_lines if line.split('==')[0].lower()in {'pip','setuptools','wheel','numpy','cython'}]
        require({'pip','setuptools','wheel','numpy'}<={line.split('==')[0].lower()for line in bootstrap[name]},'Pinned cold bootstrap packages required')
        phases.append([a.base_python,'-I','-m','venv',str(python.parents[1])])
        phases.append([str(python),'-m','pip','install','--no-index','--find-links',str(wheels),'--no-deps','--require-hashes',
            '--no-cache-dir','--no-build-isolation','-r',str(root/'installation'/('bootstrap-'+name+'.lock'))])
        phases.append([str(python),'-m','pip','install','--no-index','--find-links',str(wheels),'--no-deps','--require-hashes',
            '--no-cache-dir','--no-build-isolation','-r',str(child(root,spec['environments'][name]['lock']))])
    python=root/'.venvs/qwen35/bin/python'
    if spec['environments']['qwen-overlay']['packages']:
        phases.append([str(python),'-m','pip','install','--no-index','--find-links',str(wheels),'--no-deps','--require-hashes',
            '--no-cache-dir','--no-build-isolation','--target',str(root/'dependencies/qwen-overlay'),'-r',str(child(root,spec['environments']['qwen-overlay']['lock']))])
    for repo in ('lerobot-sim','EmbodiChain','RoboSynChallenge'):
        phases.append([str(root/'.venvs/sim/bin/python'),'-m','pip','install','--no-index','--no-deps','--no-build-isolation',
            '-e',str(child(root,'upstream/'+repo))])
    if not a.execute:
        print(json.dumps(dict(status='offline_install_plan_only',phases=phases,GPU_launches=0)));return
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PIP_NO_INDEX='1',PYTHONDONTWRITEBYTECODE='1')
    for key in ('PYTHONPATH','PYTHONHOME'):env.pop(key,None)
    logs=root/'installation';logs.mkdir(exist_ok=False)
    for name,lines in bootstrap.items():(logs/('bootstrap-'+name+'.lock')).write_text('\n'.join(lines)+'\n')
    receipt=dict(schema='goal_latent_act_offline_install_v1',status='running',phases=[],models_constructed=0,RTX5090_qualified=False)
    try:
        for i,argv in enumerate(phases):
            with(logs/f'{i:02d}.log').open('xb')as log:
                r=subprocess.run(argv,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,timeout=900)
            receipt['phases'].append(dict(argv=argv,exit_code=r.returncode));json_write(logs/'receipt.json',receipt)
            require(r.returncode==0,'Cold install phase failed; no retry')
        metadata_check(root,spec);receipt['status']='offline_install_metadata_pass_no_model'
    except BaseException as error:
        receipt.update(status='failed_preserved_no_retry',error=dict(type=type(error).__name__,message=str(error)));raise
    finally:json_write(logs/'receipt.json',receipt)


if __name__=='__main__':main()
