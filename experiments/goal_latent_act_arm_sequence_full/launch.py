"""Hash-bound local-only matched pair; execution requires root admission."""
import argparse
import fcntl
import json
import os
from pathlib import Path
from .common import PAIR, PREP, ROOT, require, sha, ref, verify, write, verify_sources
from . import supervisor

UUID='GPU-8e45cdd0-bbd6-9b4f-de12-e7125a11bed9'


def make_plan(prepared,expected,interpreter,output):
    target=Path(output).absolute()
    jobs=[]
    for name,control in (('role_goal',False),('state_action_control',True)):
        argv=[str(Path(interpreter).absolute()),'-B','-m','experiments.goal_latent_act_arm_sequence_full.train',
              '--prepared',str(Path(prepared).absolute()),'--prepared-sha256',expected,'--output',str(target/name),'--execute']
        if control:argv.append('--state-action-only')
        jobs.append(dict(name=name,argv=argv,cwd=str(ROOT),output_root=str(target/name),wall_seconds=1700,max_owned_gpu_processes=1))
    return dict(schema=supervisor.SCHEMA,gpu_uuid=UUID,admission_memory_floor_gib=32,output_root=str(target),jobs=jobs)


def predecessor(value,expected_owner):
    proof=json.loads(verify(value).read_text())
    require(type(proof['original_outer_exit_code']) is int and proof['signals_sent']==[]
            and proof['all_recorded_groups_sessions_live_members']==[] and proof['fresh_GPU_snapshot']['compute_pids']==[]
            and proof['independent_shared_flock_acquired_and_released'] is True, 'Genuine whole predecessor terminal/drain/GPU-empty proof required')
    waited=json.loads(verify(proof['original_outer_wait']).read_text())
    owner=waited['owner'];require((owner['pid'],owner.get('birth_ticks',owner.get('birth')))==tuple(expected_owner), 'Exact required preceding owner differs')
    require(waited.get('original_controller_exit_code',waited.get('original_outer_exit_code'))==proof['original_outer_exit_code']
            and waited['original_controller_reaped'] is True and waited.get('original_controller_absent',waited.get('original_process_absent')) is True
            and waited['independently_observed_group_after_reap']==[] and waited.get('signals',waited.get('signals_sent'))==[], 'Original actual parent wait/reap/drain required')
    found=False
    for row in proof['identities']:
        recorded=row['recorded'];require(row['exact_recorded_identity_absent'] is True and row['current'] is None, 'Any predecessor ownership remains unqualified')
        birth=recorded.get('birth_ticks',recorded.get('birth'))
        found|=(recorded['pid'],birth)==tuple(expected_owner)
        try:current=supervisor.identity(recorded['pid'])
        except (FileNotFoundError,ProcessLookupError):current=None
        require(current is None or current['birth_ticks']!=birth, 'Original predecessor birth is still alive')
        require(supervisor.group_members(recorded['group'])==[], 'Predecessor group still has live members')
    require(found,'Predecessor terminal proof omits actual required controller')
    return proof


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for k in ('prepared','prepared-sha256','python','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--root-admission');p.add_argument('--root-admission-sha256');p.add_argument('--execute',action='store_true');a=p.parse_args()
    require(sha(Path(a.prepared)/'receipt.json')==a.prepared_sha256,'Exact actual CPU preparation required')
    prepared=json.loads((Path(a.prepared)/'receipt.json').read_text());require(prepared['schema']==PREP and prepared['status']=='CPU_full_source_current_only_FK_parity_verified','Qualified full-data temporal-arm preparation required')
    verify_sources(prepared['source_pins']);value=make_plan(a.prepared,a.prepared_sha256,a.python,a.output);supervisor.validate_plan(value)
    if not a.execute:
        print(json.dumps(dict(status='CPU_plan_valid_pending_root_GPU_admission',plan=value,GPU_queries=0,GPU_launches=0)));return
    require(a.root_admission and a.root_admission_sha256 and sha(a.root_admission)==a.root_admission_sha256,'Exact prospective root admission mandatory')
    admission=json.loads(Path(a.root_admission).read_text())
    require(admission['schema']=='goal-latent-act-temporal-arm-local3090-root-admission-v1' and admission['GPU_training_authorized'] is True
            and admission['prepared']==ref(Path(a.prepared)/'receipt.json') and admission['plan']==value
            and admission['supervisor_sha256']==sha(supervisor.__file__)
            and admission['total_train_report_cap_seconds']==3600,'Root admission must bind exact pair/source/budget')
    expected={'matched_parent1703':(2507170,57572830),'fast_leaf_table':(2545539,57660109)}
    require(set(admission['predecessors'])==set(expected),'Both actual preceding campaigns must drain')
    for key,owner in expected.items():predecessor(admission['predecessors'][key],owner)
    with Path('/tmp/robosyn-local3090-experiment.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for key,owner in expected.items():predecessor(admission['predecessors'][key],owner)
        result=supervisor.run_plan(value)
        report=dict(schema=PAIR,status='failed_original_preserved_no_retry',supervisor_receipt=ref(Path(a.output)/'receipt.json'),
                    original_supervisor_exit_code=result['exit_code'],GPU_training_authorized_root=ref(a.root_admission),robot_launches=0)
        if result['exit_code']==0:
            reports={name:json.loads((Path(a.output)/name/'receipt.json').read_text()) for name in expected_branch_names()}
            factual,control=reports.values()
            for key in ('history','gate_history'):
                require([v['FIT_order_sha256'] for v in factual[key]]==[v['FIT_order_sha256'] for v in control[key]],'Actual matched FIT optimizer order differs')
            require(all(r['status']=='completed_fixed30_temporal_arm20_physical_FIT_gate' and r['reload_predictions_bitexact'] is True
                        and (r['optimizer_updates'],r['gate_updates'])==(11820,1000) for r in reports.values()),'Genuine paired completed scientific receipts required')
            metric=factual['diagnostic_FIT']['metrics'];free=control['diagnostic_FIT']['metrics']
            c=metric['candidate'];pmetric=metric['parent'];state=free['candidate']
            report.update(status='both_original_GPU_children_exit0_natural_complete',branches={k:ref(Path(a.output)/k/'receipt.json') for k in reports},
                diagnostic_FIT_criterion_met=c['arm_MAE_rad']<=.95*pmetric['arm_MAE_rad'] and c['command_FK_position_L2_m']<=.95*pmetric['command_FK_position_L2_m']
                    and c['arm_MAE_rad']<=.95*state['arm_MAE_rad'] and c['command_FK_position_L2_m']<=.95*state['command_FK_position_L2_m']
                    and c['command_FK_rotation_geodesic_rad']<=pmetric['command_FK_rotation_geodesic_rad'],
                MONITOR_report_only=True,robot_deployment_authorized=False,robot_success_claim=False)
        write(Path(a.output)/'pair-receipt.json',report)
        raise SystemExit(result['exit_code'])


def expected_branch_names():return ('role_goal','state_action_control')


if __name__=='__main__':main()
