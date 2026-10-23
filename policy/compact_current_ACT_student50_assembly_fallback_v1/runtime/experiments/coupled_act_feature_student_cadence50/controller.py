"""One root-authorized all20 campaign, genuine per-slot waits, no original retry."""
import argparse
import json
import os
from pathlib import Path
import signal
import re
import subprocess
import sys
import time
from .contracts import require,verify_ref,module
from .prepare import ORDER,BRANCHES
from .supervisor import SCHEMA,UUID,validate_plan
from experiments.goal_latent_act_campaign.common import ref,write
from experiments.goal_latent_act_submission.cold_cpu import identity,members


def classify_original_terminal(code,body,plan,*,original_absent,group_after_reap,snapshots,raw_stdout='',raw_stderr=''):
    """Isolate only a naturally reaped literal SIGSEGV, never a scored retry."""
    require(original_absent is True and group_after_reap==[] and body['all_owned_groups_drained'] is True
        and body['exit_code']==code and body['cpu_fixture'] is False and body['plan']==plan,
        'Original wait/plan/drain differs')
    require(len(body['jobs'])==1,'Exactly one original native attempt required')
    native=body['jobs'][0]
    require(native['child_reaped'] is True and native['signals']==[] and native['cleanup_errors']==[]
        and native['drain']['natural'] is True and native['drain']['after_reap']==[]
        and native['drain']['signals']==[] and native['drain'].get('signal_refusals',[])==[],
        'No signaled/live/failed-cleanup continuation')
    require(len(snapshots)==2 and all(view['compute_pids']==[] for view in snapshots),
        'Two fresh whole-GPU-empty snapshots required')
    if code==0 and native['original_child_exit_code']==0:return 'natural_original0'
    if code==1 and native['original_child_exit_code']==-11:
        require(body.get('failure',{}).get('type')=='RuntimeError'
            and body['failure']['message']=='Original child exited nonzero; no sequential retry',
            'A resource/contract/runtime monitor failure must stop even if native also crashed')
        raw=raw_stdout+'\n'+raw_stderr
        require('COUPLED_ALL14_POLICY_EXCEPTION_CAPTURED' not in raw and 'Traceback (most recent call last):' not in raw
            and re.search(r'(?m)^(?:[\w.]+(?:Error|Exception)|DecodeRejected):',raw) is None
            and not any(marker in raw for marker in ('CUDA error:','CUDA out of memory',
                'Declared compiled acceleration gate failed','Compiled FP32 output parity failed',
                'in qualify_compiler','in correct_native50','in compile_and_compare')),
            'Preceding actor/compiler/contract/resource failure evidence must stop, even after native SIGSEGV')
        return 'unavailable_natural_native_SIGSEGV'
    require(False,'Original nonzero outside the isolated natural SIGSEGV allowance; stop without replay')


def validate(admission):
    require(admission['schema']=='root_authorized_coupled_ACT_student_execute50_native_all10_campaign_v1'
        and admission['execution_authorized'] is True and admission['controller']==ref(__file__), 'Exact root campaign/controller admission required')
    campaign=json.loads(verify_ref(admission['campaign']).read_text())
    require(campaign['schema']=='coupled_ACT_student_execute50_exact_all10_CPU_packet_v1'
        and [(row['task'],row['mode']) for row in campaign['plans']]==[(task,mode) for task in ORDER for mode in BRANCHES], 'Exact full20 schedule required')
    require(admission['total_wall_seconds']==7200 and admission['GPU_uuid']==UUID,'Two-hour local single-GPU campaign cap required')
    require(admission['predecessor_genuine_wait_consumed'] is True and admission['predecessor_processes_drained'] is True
        and admission['independent_CPU_review_qualified'] is True and admission['actual_student_candidate_qualified'] is True,
        'Genuine pair terminal, independent review and actual compilation attempt required')
    for binding in admission['predecessor_proofs']:verify_ref(binding)
    verify_ref(admission['independent_CPU_review'])
    verify_ref(admission['supervisor'])
    require(admission['supervisor']['path'].endswith('/experiments/coupled_act_feature_student_cadence50/supervisor.py'), 'Exact additive native supervisor required')
    root=Path(campaign['source_root'])
    for relative,digest in campaign['source_sha256'].items():
        require(ref(root/relative)['sha256']==digest,'Immutable runtime source changed: '+relative)
    for row in campaign['plans']:
        plan=json.loads(verify_ref(row['plan']).read_text());validate_plan(plan)
    return campaign


def run(admission_ref,output):
    admission=json.loads(verify_ref(admission_ref).read_text());campaign=validate(admission)
    output=Path(output).absolute();require(not output.exists(),'Fresh original campaign output required');output.mkdir(parents=True)
    started=time.monotonic();deadline=started+7200;child=None;owner=None
    report=dict(schema='actual_coupled_ACT_student_execute50_native_all10_original_controller_v1',admission=admission_ref,
        owner=identity(os.getpid()),status='running_original',jobs=[],quality_promoted=False,accepted_scene_pairing_qualified=False)
    def save():report['elapsed_seconds']=time.monotonic()-started;write(output/'receipt.json',report)
    def interrupted(sig,_frame):raise InterruptedError('Original controller interrupted '+str(sig))
    prior=signal.signal(signal.SIGTERM,interrupted)
    guard=module('_coupled_original_campaign_drain',verify_ref(json.loads(verify_ref(campaign['plans'][0]['plan']).read_text())['borrowed_guard']))
    predecessor=admission['predecessor_proofs'];save()
    try:
        for index,row in enumerate(campaign['plans']):
            require(time.monotonic()<deadline-60,'Campaign wall cap reached; preserve unattempted slots')
            plan=json.loads(verify_ref(row['plan']).read_text());validate_plan(plan)
            slot=dict(schema='root_authorized_coupled_ACT_student_execute50_local_native_slot_v1',execution_authorized=True,
                authorized_by_original_campaign=admission_ref,plan=row['plan'],runtime_supervisor=admission['supervisor'],
                predecessor_genuine_wait_consumed=True,predecessor_processes_drained=True,predecessor_proofs=predecessor,
                independent_CPU_review_qualified=True,independent_CPU_review=admission['independent_CPU_review'])
            slot_path=output/f'{index:02d}-slot-admission.json';write(slot_path,slot)
            argv=[sys.executable,'-B','-m','experiments.coupled_act_feature_student_cadence50.supervisor','--plan',row['plan']['path'],
                '--execute','--root-admission',str(slot_path),'--root-admission-sha256',ref(slot_path)['sha256']]
            env=os.environ.copy();env.pop('PYTHONHOME',None);env.update(PYTHONPATH=campaign['source_root'],PYTHONDONTWRITEBYTECODE='1')
            job=dict(index=index,task=row['task'],mode=row['mode'],plan=row['plan'],status='starting_original');report['jobs'].append(job);save()
            with (output/f'{index:02d}.stdout.log').open('xb') as stdout,(output/f'{index:02d}.stderr.log').open('xb') as stderr:
                child=subprocess.Popen(argv,cwd=campaign['source_root'],env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
                owner=identity(child.pid);require(owner and owner['pid']==owner['group']==owner['session'],'Exact original supervisor identity required')
                job['owner']=owner;save();code=child.wait(timeout=min(1850,max(1,deadline-time.monotonic()-45)))
            job.update(original_supervisor_exit_code=code,child_reaped=True,original_absent=identity(child.pid) is None,group_after_reap=members(owner['group']))
            original=Path(plan['output_root'])/'receipt.json';body=json.loads(original.read_text());job['original_supervisor_receipt']=ref(original)
            require(job['original_absent'] and job['group_after_reap']==[] and body['all_owned_groups_drained'] is True
                and body['exit_code']==code and body['cpu_fixture'] is False and body['plan']==plan,'Original wait/plan/drain differs')
            require(len(body['jobs'])==1,'Exactly one original native attempt required');native=body['jobs'][0]
            require(native['child_reaped'] is True and native['signals']==[] and native['cleanup_errors']==[]
                and native['drain']['natural'] is True and native['drain']['after_reap']==[]
                and native['drain']['signals']==[] and native['drain'].get('signal_refusals',[])==[],'No signaled/live/failed-cleanup continuation')
            snapshots=[guard.gpu_snapshot(UUID),guard.gpu_snapshot(UUID)]
            require(all(view['compute_pids']==[] for view in snapshots),'Two fresh whole-GPU-empty snapshots required')
            wait=dict(schema='coupled_ACT_student_genuine_original_native_slot_wait_v1',owner=owner,original_child_exit_code=code,
                child_reaped=True,original_absent=job['original_absent'],group_after_reap=job['group_after_reap'],
                original_native_exit_code=native['original_child_exit_code'],original_supervisor_receipt=ref(original),fresh_GPU_snapshots=snapshots)
            wait_path=output/f'{index:02d}-original-wait.json';write(wait_path,wait);job['genuine_original_wait']=ref(wait_path)
            raw_logs={suffix:Path(plan['output_root'])/'logs'/(plan['jobs'][0]['name']+'.'+suffix+'.log') for suffix in ('stdout','stderr')}
            for path in raw_logs.values():require(path.is_file() and not path.is_symlink() and path.stat().st_size<=32*(1<<20),'Raw native log missing or too large for strict terminal classification')
            job['original_native_logs']={key:ref(path) for key,path in raw_logs.items()}
            classification=classify_original_terminal(code,body,plan,original_absent=job['original_absent'],
                group_after_reap=job['group_after_reap'],snapshots=snapshots,
                raw_stdout=raw_logs['stdout'].read_text(errors='replace'),raw_stderr=raw_logs['stderr'].read_text(errors='replace'))
            if classification=='unavailable_natural_native_SIGSEGV':
                job.update(status=classification,episodes=0,successes=None,native_SIGSEGV_cause_not_inferred=True,
                    original_failed_attempt_preserved=True,retried=False)
                predecessor=[ref(wait_path),ref(original)];save();child=None;owner=None
                continue
            metrics=list((Path(plan['jobs'][0]['output_root'])/'evaluator-results').rglob('evaluation_metrics.json'))
            if not metrics:job.update(status='unavailable_missing_official_metric_after_natural_original0',episodes=0,successes=None)
            else:
                require(len(metrics)==1,'Ambiguous official metrics');metric=json.loads(metrics[0].read_text());summary=metric['summary']
                require(metric['schema_version']==1 and metric['config']['task']==row['task'] and summary['episode_count']==1
                    and type(summary['success_count']) is int and summary['success_count'] in (0,1)
                    and summary['success_rate']==summary['success_count'],'Only actual official one-episode metrics qualify')
                job.update(status='official_original_one_episode_qualified',episodes=1,successes=summary['success_count'],metric=ref(metrics[0]))
            predecessor=[ref(wait_path),ref(original)];save();child=None;owner=None
        report['status']='all10_execute50_original_slots_terminal_pairing_pending';report['exit_code']=0
    except BaseException as error:
        report.update(status='original_campaign_failed_preserved_unattempted_retained',exit_code=1,failure=dict(type=type(error).__name__,message=str(error)))
        if child is not None:
            # Only this captured original supervisor is interrupted; its proven
            # original ledger owns cleanup of its native/Qwen group.
            current=identity(child.pid)
            if current and owner and guard.same_identity(current,owner):child.send_signal(signal.SIGTERM)
            try:report['cleanup_original_supervisor_exit_code']=child.wait(timeout=45)
            except subprocess.TimeoutExpired:report['cleanup_original_supervisor_wait_pending']=True
        raise
    finally:signal.signal(signal.SIGTERM,prior);save()


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--admission',required=True);parser.add_argument('--admission-sha256',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--check-only',action='store_true');args=parser.parse_args();binding=ref(args.admission)
    require(binding['sha256']==args.admission_sha256,'Exact original campaign admission required')
    if args.check_only:validate(json.loads(verify_ref(binding).read_text()));print(json.dumps(dict(status='CPU_actual_all20_campaign_valid',GPU_queries=0)));return
    run(binding,args.output)


if __name__=='__main__':main()
