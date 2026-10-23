"""Bounded actual offline installation plus all10 imports; CUDA always hidden."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.goal_latent_act_submission.contracts import require, reference, json_write


def identity(pid):
    try:
        value=(Path('/proc')/str(pid)/'stat').read_text();tail=value[value.rindex(')')+2:].split()
        return dict(pid=pid,birth_ticks=int(tail[19]),group=int(tail[2]),session=int(tail[3]))
    except FileNotFoundError:return None


def members(group):
    result=[]
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            try:row=identity(int(path.name))
            except ProcessLookupError:continue
            if row and row['group']==group:result.append(row)
    return result


def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--root',required=True);p.add_argument('--base-python',required=True)
    p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.root).resolve(strict=True);output=Path(a.output).absolute()
    require(not output.exists(),'Fresh bounded CPU proof only');output.mkdir(parents=True)
    package=root/'experiments/goal_latent_act_submission'
    phases=[('cold-install',[sys.executable,'-B',str(package/'install.py'),'--root',str(root),'--base-python',a.base_python,'--execute'],1200),
            ('all10-imports',[sys.executable,'-B',str(Path(__file__).with_name('check_cpu.py')),'--root',str(root),'--output',str(output/'qualification')],300)]
    receipt=dict(schema='goal-latent-act-bounded-actual-cold-CPU-v1',status='running',controller=identity(os.getpid()),
        submission=reference(root/'submission.json'),phases=[],models_constructed=0,GPU_launches=0,RTX5090_qualified=False)
    started=time.monotonic();env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PIP_NO_INDEX='1',PYTHONDONTWRITEBYTECODE='1')
    try:
        for name,argv,cap in phases:
            phase=dict(name=name,argv=argv,wall_cap_seconds=cap);receipt['phases'].append(phase)
            with(output/(name+'.stdout.txt')).open('xb')as stdout,(output/(name+'.stderr.txt')).open('xb')as stderr:
                child=subprocess.Popen(argv,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
                owner=identity(child.pid);require(owner and owner['group']==owner['session']==child.pid,'Exact fresh CPU child group required')
                phase['owner']=owner;json_write(output/'receipt.json',receipt);begin=time.monotonic()
                try:code=child.wait(timeout=cap)
                except subprocess.TimeoutExpired:
                    phase['timed_out']=True
                    require(identity(child.pid)==owner,'CPU child identity changed before bounded termination')
                    os.killpg(child.pid,signal.SIGTERM)
                    try:code=child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        require(identity(child.pid)==owner,'CPU child identity changed before bounded kill')
                        os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=10)
                phase.update(original_child_exit_code=code,seconds=time.monotonic()-begin,child_reaped=True,
                    independently_observed_group_after_reap=members(child.pid),original_process_absent=identity(child.pid)is None)
                json_write(output/'receipt.json',receipt)
                require(code==0 and phase['original_process_absent']and not phase['independently_observed_group_after_reap'],
                        'Original CPU phase failed or did not drain; preserve attempt')
        receipt['status']='actual_offline_install_all10_strict_bindings_and_imports_pass_no_model'
    except BaseException as error:
        receipt.update(status='failed_preserved_no_retry',error=dict(type=type(error).__name__,message=str(error)));raise
    finally:receipt['seconds']=time.monotonic()-started;json_write(output/'receipt.json',receipt)
    print(json.dumps(dict(status=receipt['status'],seconds=receipt['seconds'],GPU_launches=0,models_constructed=0)))


if __name__=='__main__':main()
