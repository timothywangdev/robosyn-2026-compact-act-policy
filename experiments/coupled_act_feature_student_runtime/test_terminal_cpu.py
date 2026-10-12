"""Literal CPU SIGSEGV wait and strict terminal-boundary fixtures; no CUDA."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from .controller import classify_original_terminal
from .contracts import require
from experiments.goal_latent_act_campaign.common import ref,write
from experiments.goal_latent_act_submission.cold_cpu import identity,members


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False);parser.add_argument('--output',required=True);args=parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU terminal fixtures must hide CUDA')
    output=Path(args.output).absolute();require(not output.exists(),'Fresh CPU wait proof required');output.mkdir(parents=True)
    with (output/'native.stderr.log').open('xb') as stderr:
        child=subprocess.Popen([sys.executable,'-B','-c',
            'import os,resource,signal; resource.setrlimit(resource.RLIMIT_CORE,(0,0)); os.kill(os.getpid(),signal.SIGSEGV)'],
            stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=stderr,start_new_session=True)
        captured=identity(child.pid);code=child.wait(timeout=10)
    wait=dict(original=captured,original_child_exit_code=code,child_reaped=True,
        original_absent=identity(child.pid) is None,group_after_reap=members(captured['group']))
    require(code==-11 and wait['original_absent'] and wait['group_after_reap']==[],'Literal natural CPU SIGSEGV wait did not drain')
    write(output/'actual-native-original-wait.json',wait)
    plan={'fixture':'explicit_no_native_model_or_world'}
    native=dict(original_child_exit_code=code,child_reaped=True,signals=[],cleanup_errors=[],
        drain=dict(natural=True,after_reap=[],signals=[],signal_refusals=[]))
    body=dict(all_owned_groups_drained=True,exit_code=1,cpu_fixture=False,plan=plan,jobs=[native],
        failure=dict(type='RuntimeError',message='Original child exited nonzero; no sequential retry'))
    # GPU rows below are declared CPU fixtures; no query is performed.
    snapshots=[{'compute_pids':[]},{'compute_pids':[]}]
    def call(b=body,views=snapshots,absent=True,group=(),raw_stdout='',raw_stderr=''):
        return classify_original_terminal(b['exit_code'],b,plan,original_absent=absent,group_after_reap=list(group),snapshots=views,
            raw_stdout=raw_stdout,raw_stderr=raw_stderr)
    require(call()=='unavailable_natural_native_SIGSEGV','Clean literal SIGSEGV must be unavailable, not score zero or retry')
    normal=deepcopy(body);normal['exit_code']=0;normal['jobs'][0]['original_child_exit_code']=0
    require(call(normal)=='natural_original0','Natural0 classifier changed')
    variants=[]
    for name,mutate in (
        ('signaled',lambda b:b['jobs'][0]['signals'].append('SIGTERM')),
        ('signal_refusal',lambda b:b['jobs'][0]['drain']['signal_refusals'].append('reused identity')),
        ('live_owned_group',lambda b:b['jobs'][0]['drain']['after_reap'].append({'pid':123})),
        ('not_reaped',lambda b:b['jobs'][0].update(child_reaped=False)),
        ('cleanup_error',lambda b:b['jobs'][0]['cleanup_errors'].append('failed')),
        ('resource_failure',lambda b:b['failure'].update(message='Whole-device memory cap exceeded')),
        ('compiler_failure',lambda b:b['jobs'][0].update(original_child_exit_code=1)),
        ('asymmetric_native0_supervisor1',lambda b:b['jobs'][0].update(original_child_exit_code=0)),
        ('wrong_plan',lambda b:b.update(plan={'different':True})),
        ('fixture_laundered',lambda b:b.update(cpu_fixture=True)),
    ):
        changed=deepcopy(body);mutate(changed)
        try:call(changed)
        except (ValueError,AssertionError):variants.append(name)
        else:raise AssertionError('Unsafe continuation: '+name)
    for name,kwargs in (
        ('fresh_GPU_not_empty',dict(views=[{'compute_pids':[]},{'compute_pids':[123]}])),
        ('only_one_GPU_snapshot',dict(views=[{'compute_pids':[]}])),
        ('live_supervisor',dict(absent=False)),
        ('live_supervisor_group',dict(group=[123])),
        ('preceding_python_policy_failure',dict(raw_stderr='Traceback (most recent call last):\nValueError: failed numerical gate\nFatal Python error: Segmentation fault')),
        ('preceding_logged_contract_failure',dict(raw_stdout='DecodeRejected: action bounds failed\n',raw_stderr='Fatal Python error: Segmentation fault')),
        ('active_compiler_failure_stack',dict(raw_stderr='Fatal Python error: Segmentation fault\n  in compile_and_compare')),
        ('flushed_policy_error_before_SDK_cleanup',dict(raw_stderr='COUPLED_ALL14_POLICY_EXCEPTION_CAPTURED eval ValueError\nFatal Python error: Segmentation fault')),
    ):
        try:call(**kwargs)
        except (ValueError,AssertionError):variants.append(name)
        else:raise AssertionError('Unsafe continuation: '+name)
    import contextlib
    import io
    from experiments.coupled_goal_act_all14_runtime.policy_shim_eager import guarded,EXCEPTION_MARKER
    hooks=[]
    for name in ('get_model','eval','reset_model','close_model'):
        error=ValueError('explicit CPU lifecycle exception fixture');stream=io.StringIO()
        def throws():raise error
        try:
            with contextlib.redirect_stderr(stream):guarded(name,throws)
        except ValueError as caught:require(caught is error,'Lifecycle marker changed the original exception')
        else:raise AssertionError('Lifecycle exception was swallowed')
        require(EXCEPTION_MARKER+' '+name+' ValueError' in stream.getvalue(),'Missing pre-cleanup error marker')
        hooks.append(dict(hook=name,marker_emitted_before_original_propagation=True))
    result=dict(schema='coupled_ACT_student_isolated_literal_SIGSEGV_CPU_boundary_v1',status='CPU_original_wait_and_20_terminal_cases_plus_four_lifecycle_markers_pass',
        original_native_wait=ref(output/'actual-native-original-wait.json'),clean_SIGSEGV_unavailable=True,
        natural0_pass=True,unsafe_cases_rejected=variants,explicit_fixture_GPU_rows=True,GPU_queries=0,GPU_launches=0,
        Torch_imported='torch' in sys.modules,models_constructed=0,worlds_constructed=0,automatic_retries=0,lifecycle_exception_hooks=hooks)
    require(result['Torch_imported'] is False,'CPU terminal helper unexpectedly imported Torch')
    write(output/'receipt.json',result);print(json.dumps(ref(output/'receipt.json')))


if __name__=='__main__':main()
