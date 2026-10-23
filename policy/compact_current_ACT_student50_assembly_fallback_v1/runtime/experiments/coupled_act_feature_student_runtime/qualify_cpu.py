"""Genuine CPU child waits for prospective source, causal tests and FILE ABI."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .contracts import require,ref,verify,HORIZONS
from experiments.goal_latent_act_submission.cold_cpu import identity,members
from experiments.goal_latent_act_campaign.common import write


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False);parser.add_argument('--campaign',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU qualification hides CUDA')
    campaign_ref=ref(args.campaign);campaign=json.loads(verify(campaign_ref).read_text());source=Path(campaign['source_root'])
    require(campaign['execution_authorized'] is False and campaign['candidate'] is None and len(campaign['plans'])==20, 'Only prospective candidate-free20 CPU packet')
    for name,digest in campaign['source_sha256'].items():require(ref(source/name)['sha256']==digest,'Prospective staged source changed: '+name)
    output=Path(args.output).absolute();require(not output.exists(),'Fresh actual CPU proof root');output.mkdir(parents=True)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(source),PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',STUDENT_RUNTIME_CPU_CAMPAIGN=campaign_ref['path'])
    results=[]
    def child(label,argv):
        started=time.monotonic()
        with (output/(label+'.stdout.log')).open('xb') as stdout,(output/(label+'.stderr.log')).open('xb') as stderr:
            process=subprocess.Popen(argv,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
            owner=identity(process.pid);code=process.wait(timeout=120)
        body=dict(label=label,original=owner,original_child_exit_code=code,child_reaped=True,
            original_absent=identity(process.pid) is None,group_after_reap=members(owner['group']),
            elapsed_seconds=time.monotonic()-started,signals=[],stdout=ref(output/(label+'.stdout.log')),stderr=ref(output/(label+'.stderr.log')))
        write(output/(label+'-original-wait.json'),body);results.append(body)
        require(code==0 and body['original_absent'] and body['group_after_reap']==[], 'Actual CPU child did not naturally succeed/drain: '+label)
    child('twelve-causal-tests',[sys.executable,'-B','-m','experiments.coupled_act_feature_student_runtime.test_cpu'])
    child('terminal-boundary-tests',[sys.executable,'-B','-m','experiments.coupled_act_feature_student_runtime.test_terminal_cpu','--output',str(output/'actual-terminal-boundaries')])
    code="""import json,sys,importlib.util
from pathlib import Path
from experiments.coupled_act_feature_student_runtime.contracts import load_manifest,HORIZONS
root=Path(sys.argv[1]);p=json.loads(Path(sys.argv[2]).read_text());rows=[]
for row in p['plans']:
    cfg=json.loads(Path(row['deployment']['path']).read_text());command=json.loads(Path(row['command']['path']).read_text());plan=json.loads(Path(row['plan']['path']).read_text())
    assert cfg['official_horizon']==HORIZONS[row['task']] and cfg['motor_seed']==17
    assert cfg['Qwen_models_loaded']==0 and cfg['ACT_models_loaded']==1 and cfg['assets']['runtime_admission'] is None
    assert cfg['schema'].startswith('prospective_') and cfg['execution_authorized'] is False
    assert plan['jobs'][0]['max_owned_gpu_processes']==1
    inner=command['inner_argv']
    for flag,value in [('--seed','2026102401'),('--act_step','10'),('--max_steps',str(HORIZONS[row['task']])),('--policy_name','coupled_ACT_current_student_v1')]:assert inner.count(flag)==1 and inner[inner.index(flag)+1]==value
    try:load_manifest(row['deployment']['path'],row['deployment']['sha256'])
    except ValueError:pass
    else:raise AssertionError('Prospective missing candidate was executable')
    rows.append({'task':row['task'],'mode':row['mode'],'candidate_missing_execution_rejected':True})
spec=importlib.util.spec_from_file_location('actual_student_policy_FILE',root/'policy/coupled_ACT_current_student_v1/__init__.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
assert all(callable(getattr(module,k)) for k in ('get_model','eval','reset_model','close_model'))
assert 'torch' not in sys.modules
Path(sys.argv[3]).write_text(json.dumps({'schema':'ACT_student_all20_prospective_FILE_CPU_v1','cases':rows,'Torch_imported':False,'models_constructed':0,'worlds_constructed':0,'GPU_queries':0},indent=2)+'\\n')
"""
    child('all20-prospective-and-FILE',[sys.executable,'-B','-c',code,str(source),campaign_ref['path'],str(output/'all20-FILE.json')])
    child('actual-native-FILE-help',[sys.executable,'-B',str(source/'experiments/coupled_act_feature_student_runtime/native.py'),'--help'])
    receipt=dict(schema='coupled_ACT_current_student_runtime_actual_CPU_qualification_v1',status='twelve_causal_tests_terminal_guards_all20_prospective_and_actual_FILE_PASS',
        campaign=campaign_ref,source_sha256=campaign['source_sha256'],actual_original_children=results,
        twelve_synthetic_CPU_tests=True,actual_source_FK_current_vs_qualified_state_runtime_bitexact=True,
        pretrained_model_loads=0,synthetic_small_models_constructed=True,optimizer_updates=0,
        candidate_weights_exist=False,actual_trained_student_GPU_parity_pending=True,CPU_vs_GPU_feature_parity_pending=True,
        Qwen_models_loaded=0,second_ACT_or_CNN_forward=False,native_worlds=0,GPU_queries=0,GPU_launches=0,
        actual_all10_robot_results_pending=True,teacher_offline5percent_criterion='FAILED_unchanged',quality_promoted=False)
    write(output/'receipt.json',receipt);print(json.dumps(ref(output/'receipt.json')))


if __name__=='__main__':main()
