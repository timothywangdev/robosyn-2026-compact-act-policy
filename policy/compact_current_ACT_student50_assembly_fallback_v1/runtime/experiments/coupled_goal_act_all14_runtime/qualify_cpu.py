"""Genuine CPU children and source closure; no native/model GPU admission."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .contracts import require
from .prepare import ROOT,dependencies
from experiments.goal_latent_act_campaign.common import ref,write
from experiments.goal_latent_act_submission.cold_cpu import identity,members


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False);parser.add_argument('--output',required=True);args=parser.parse_args()
    output=Path(args.output).absolute();require(not output.exists(),'Fresh CPU evidence required');output.mkdir(parents=True)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
        PYTHONPATH=str(ROOT),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    children=[];start=time.monotonic()
    commands=[('tests',[sys.executable,'-B','-m','experiments.coupled_goal_act_all14_runtime.test_cpu']),
        ('standalone-FILE',[sys.executable,'-B',str(ROOT/'experiments/coupled_goal_act_all14_runtime/native.py'),'--help']),
        ('typed-campaign-CLI',[sys.executable,'-B','-m','experiments.coupled_goal_act_all14_runtime.controller','--help']),
        ('trained-compiler-CLI',[sys.executable,'-B','-m','experiments.coupled_goal_act_all14_runtime.compiler_assay','--help']),
        ('actual-trained-statistics',[sys.executable,'-B','-m','experiments.coupled_goal_act_all14_runtime.compiler_assay',
            '--check-statistics-only','--prepared',str(ROOT/'outputs/architecture-reassessment/coupled-goal-act-all14-preparation-v1/CPU-prepared-v3'),
            '--prepared-sha256','f177eb813eea5e170e3216c41ad560fc29ec7b9d39ec3868ed3c616736d705d4']),
        ('conditional-all10',[sys.executable,'-B','-m','experiments.coupled_goal_act_all14_runtime.prepare',
            '--plan-only','--output',str(output/'conditional-all10')])]
    for name,argv in commands:
        with (output/(name+'.stdout.log')).open('xb') as stdout,(output/(name+'.stderr.log')).open('xb') as stderr:
            child=subprocess.Popen(argv,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
            original=identity(child.pid);require(original and original['pid']==original['group']==original['session'],'Exact original CPU child identity required')
            code=child.wait(timeout=120)
        row=dict(name=name,argv=argv,original=original,original_child_exit_code=code,child_reaped=True,
            original_absent=identity(child.pid) is None,group_after_reap=members(original['group']),
            stdout=ref(output/(name+'.stdout.log')),stderr=ref(output/(name+'.stderr.log')))
        write(output/(name+'-original-wait.json'),row);children.append(row)
        require(code==0 and row['original_absent'] and row['group_after_reap']==[], 'CPU original failed or did not drain')
    result=dict(schema='coupled_goal_ACT_all14_runtime_actual_CPU_qualification_v1',
        status='CPU_current_snapshot_fullhand_queue_units_partial_compile_ABI_qualified_no_robot_admission',
        owner=identity(os.getpid()),children=children,source_sha256=dependencies(ROOT),
        actual_tests=13,small_synthetic_CoupledGoalACT_CPU_only=True,ACT_models_constructed=0,Qwen_models_constructed=0,
        all30_flag_validator_fixture_contract_only=True,final_trained_manifest_admission_qualified=False,
        GPU_queries=0,GPU_launches=0,native_worlds_constructed=0,CUDA_initialized=False,
        trained_weight_GPU_parity_qualified=False,combined_memory_fit_qualified=False,
        genuine_new_training_wait_pending=True,actual_native_FILE_bootstrap_with_final_manifest_pending=True,
        robot_execution_authorized=False,elapsed_seconds=time.monotonic()-start)
    write(output/'receipt.json',result)
    print(json.dumps(dict(status=result['status'],receipt=ref(output/'receipt.json'))))


if __name__=='__main__':main()
