"""Actual CPU-only original tests/preparation/plan with literal waits and drains."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from .common import ROOT,ref,write,require,sha
from experiments.goal_latent_act_full_data.run_training_pair import identity,group_members


def original(argv,out,name):
    with (out/(name+'.stdout.log')).open('x') as stdout,(out/(name+'.stderr.log')).open('x') as stderr:
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONUNBUFFERED='1')
        start=time.monotonic();p=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
        owned=identity(p.pid);require(owned is not None,'Actual CPU child birth required');code=p.wait()
    remaining=group_members(p.pid);absent=identity(p.pid) is None
    row=dict(argv=argv,identity=owned,original_exit_code=code,original_child_reaped=True,original_process_absent=absent,
        independently_observed_group_after_reap=remaining,signals_sent=[],natural_completion_no_signals_sent=True,
        elapsed_seconds=time.monotonic()-start,stdout=ref(out/(name+'.stdout.log')),stderr=ref(out/(name+'.stderr.log')))
    write(out/(name+'-original-wait.json'),row)
    require(code==0 and absent and remaining==[],'Actual original CPU qualification failed: '+name)
    return row


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for k in ('python','parent-prepared','parent-prepared-sha256','protocol','output','prepared-output','future-GPU-output'):
        p.add_argument('--'+k,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);runs=[]
    runs.append(original([a.python,'-B','-m','experiments.coupled_goal_act_all14.test_cpu'],out,'fixtures'))
    runs.append(original([a.python,'-B','-m','experiments.coupled_goal_act_all14.compile_cpu','--output',str(out/'compiled-paths.json')],out,'compiled-paths'))
    runs.append(original([a.python,'-B','-m','experiments.coupled_goal_act_all14.prepare','--parent-prepared',a.parent_prepared,
        '--parent-prepared-sha256',a.parent_prepared_sha256,'--protocol',a.protocol,'--output',a.prepared_output],out,'prepare'))
    prepared=Path(a.prepared_output)/'receipt.json'
    runs.append(original([a.python,'-B','-m','experiments.coupled_goal_act_all14.launch','--prepared',a.prepared_output,
        '--prepared-sha256',sha(prepared),'--python',a.python,'--output',a.future_GPU_output],out,'actual-plan'))
    write(out/'receipt.json',dict(schema='coupled-goal-ACT-all14-original-CPU-qualification-v1',
        status='all_four_original_CPU_children_exit0_natural_complete',runs=runs,prepared=ref(prepared),
        compiled_paths=ref(out/'compiled-paths.json'),GPU_queries=0,GPU_launches=0,robot_promotion=False))
    print('all_four_original_CPU_children_exit0_natural_complete',flush=True)


if __name__=='__main__':main()
