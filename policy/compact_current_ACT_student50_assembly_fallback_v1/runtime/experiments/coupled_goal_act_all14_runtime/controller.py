"""One root-authorized all30 campaign, genuine per-slot waits, no original retry."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .contracts import require,verify_ref,module
from .prepare import ORDER,BRANCHES
from .supervisor import SCHEMA,UUID,validate_plan
from experiments.goal_latent_act_campaign.common import ref,write
from experiments.goal_latent_act_submission.cold_cpu import identity,members


def validate(admission):
    require(admission['schema']=='root_authorized_coupled_all14_native_all30_campaign_v1'
        and admission['execution_authorized'] is True and admission['controller']==ref(__file__), 'Exact root campaign/controller admission required')
    campaign=json.loads(verify_ref(admission['campaign']).read_text())
    require(campaign['schema']=='coupled_all14_exact_all10_parent_role_state_CPU_packet_v1'
        and [(row['task'],row['mode']) for row in campaign['plans']]==[(task,mode) for task in ORDER for mode in BRANCHES], 'Exact full30 schedule required')
    require(admission['total_wall_seconds']==7200 and admission['GPU_uuid']==UUID,'Two-hour local single-GPU campaign cap required')
    require(admission['predecessor_genuine_wait_consumed'] is True and admission['predecessor_processes_drained'] is True
        and admission['independent_CPU_review_qualified'] is True and admission['compiler_assay_qualified'] is True,
        'Genuine pair terminal, independent review and actual trained compiler assay required')
    for binding in admission['predecessor_proofs']:verify_ref(binding)
    verify_ref(admission['independent_CPU_review'])
    compiled=json.loads(verify_ref(admission['compiler_assay']).read_text())
    require(compiled['status']=='both_trained_modes_all10_FIT_current_partial_Inductor_numeric_pass_no_robot_promotion', 'No silent compiler fallback or failed assay promotion')
    for branch in compiled['branches'].values():
        require(len(branch['all10_same_input_rows'])==10 and branch['compiler']['backend']=='inductor'
            and all(row['graphs']>0 and row['calls']>=6 for row in branch['compiler']['graph_and_kernel_calls'].values()), 'Actual compiled whole-model paths/all10 inputs required')
    verify_ref(admission['supervisor'])
    require(admission['supervisor']['path'].endswith('/experiments/coupled_goal_act_all14_runtime/supervisor.py'), 'Exact additive native supervisor required')
    root=Path(campaign['source_root'])
    for relative,digest in campaign['source_sha256'].items():
        require(ref(root/relative)['sha256']==digest,'Immutable runtime source changed: '+relative)
    for row in campaign['plans']:
        plan=json.loads(verify_ref(row['plan']).read_text());validate_plan(plan)
        if row['mode']!='parent':
            from .contracts import load_manifest
            cfg=load_manifest(row['deployment']['path'],row['deployment']['sha256'])
            branch='role_goal' if row['mode']=='goal' else 'state_action_control'
            require(compiled['branches'][branch]['weights']==cfg['assets']['coupled_weights']
                and compiled['branches'][branch]['contract']==cfg['assets']['coupled_contract'],'Compiler assay did not qualify these exact trained actor bytes')
    return campaign


def run(admission_ref,output):
    admission=json.loads(verify_ref(admission_ref).read_text());campaign=validate(admission)
    output=Path(output).absolute();require(not output.exists(),'Fresh original campaign output required');output.mkdir(parents=True)
    started=time.monotonic();deadline=started+7200;child=None;owner=None
    report=dict(schema='actual_coupled_all14_native_all30_original_controller_v1',admission=admission_ref,
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
            slot=dict(schema='root_authorized_coupled_all14_local_native_slot_v1',execution_authorized=True,
                authorized_by_original_campaign=admission_ref,plan=row['plan'],runtime_supervisor=admission['supervisor'],
                predecessor_genuine_wait_consumed=True,predecessor_processes_drained=True,predecessor_proofs=predecessor,
                independent_CPU_review_qualified=True,independent_CPU_review=admission['independent_CPU_review'])
            slot_path=output/f'{index:02d}-slot-admission.json';write(slot_path,slot)
            argv=[sys.executable,'-B','-m','experiments.coupled_goal_act_all14_runtime.supervisor','--plan',row['plan']['path'],
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
            wait=dict(schema='coupled_all14_genuine_original_native_slot_wait_v1',owner=owner,original_child_exit_code=code,
                child_reaped=True,original_absent=job['original_absent'],group_after_reap=job['group_after_reap'],
                original_native_exit_code=native['original_child_exit_code'],original_supervisor_receipt=ref(original),fresh_GPU_snapshots=snapshots)
            wait_path=output/f'{index:02d}-original-wait.json';write(wait_path,wait);job['genuine_original_wait']=ref(wait_path)
            require(code==0 and native['original_child_exit_code']==0,'Original nonzero preserved; stop without automatic replay')
            metrics=list((Path(plan['jobs'][0]['output_root'])/'evaluator-results').rglob('evaluation_metrics.json'))
            if not metrics:job.update(status='unavailable_missing_official_metric_after_natural_original0',episodes=0,successes=None)
            else:
                require(len(metrics)==1,'Ambiguous official metrics');metric=json.loads(metrics[0].read_text());summary=metric['summary']
                require(metric['schema_version']==1 and metric['config']['task']==row['task'] and summary['episode_count']==1
                    and type(summary['success_count']) is int and summary['success_count'] in (0,1)
                    and summary['success_rate']==summary['success_count'],'Only actual official one-episode metrics qualify')
                job.update(status='official_original_one_episode_qualified',episodes=1,successes=summary['success_count'],metric=ref(metrics[0]))
            predecessor=[ref(wait_path),ref(original)];save();child=None;owner=None
        report['status']='all30_original_slots_terminal_pairing_pending';report['exit_code']=0
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
    if args.check_only:validate(json.loads(verify_ref(binding).read_text()));print(json.dumps(dict(status='CPU_actual_all30_campaign_valid',GPU_queries=0)));return
    run(binding,args.output)


if __name__=='__main__':main()
