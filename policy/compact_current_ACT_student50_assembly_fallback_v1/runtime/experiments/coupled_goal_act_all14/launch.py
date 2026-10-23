"""Source-qualified original matched pair; root owns fresh GPU admission and wait."""
import argparse
import fcntl
import json
from pathlib import Path
from .common import ROOT,PREP,require,ref,sha,verify,write,verify_sources
from . import supervisor

UUID='GPU-8e45cdd0-bbd6-9b4f-de12-e7125a11bed9'
PAIR='coupled-goal-ACT-all14-original-GPU-pair-v1'


def make_plan(prepared,expected,interpreter,output):
    target=Path(output).absolute();jobs=[]
    for name,control in (('role_goal',False),('state_action_control',True)):
        argv=[str(Path(interpreter).absolute()),'-B','-m','experiments.coupled_goal_act_all14.train',
              '--prepared',str(Path(prepared).absolute()),'--prepared-sha256',expected,'--output',str(target/name),'--execute']
        if control:argv.append('--state-action-only')
        jobs.append(dict(name=name,argv=argv,cwd=str(ROOT),output_root=str(target/name),wall_seconds=1250,max_owned_gpu_processes=1))
    value=dict(schema=supervisor.SCHEMA,gpu_uuid=UUID,admission_memory_floor_gib=32,output_root=str(target),jobs=jobs)
    return supervisor.validate_plan(value)


def criterion(role,state,horizon):
    # Original5% ARM/FK benefit versus BOTH parent and state control retained;
    # no original failure is relabeled. Opening no-regression is an ADDITION.
    c=role['diagnostic_FIT']['metrics'][str(horizon)]['candidate']
    p=role['diagnostic_FIT']['metrics'][str(horizon)]['parent']
    s=state['diagnostic_FIT']['metrics'][str(horizon)]['candidate']
    return (c['arm_MAE_rad']<=.95*p['arm_MAE_rad'] and c['FK_position_L2_m']<=.95*p['FK_position_L2_m']
        and c['arm_MAE_rad']<=.95*s['arm_MAE_rad'] and c['FK_position_L2_m']<=.95*s['FK_position_L2_m']
        and c['FK_orientation_geodesic_rad']<=p['FK_orientation_geodesic_rad']
        and c['opening_MAE_m']<=p['opening_MAE_m']+.0001 and c['opening_MAE_m']<=s['opening_MAE_m']+.0001)


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for k in ('prepared','prepared-sha256','python','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--root-admission');p.add_argument('--root-admission-sha256');p.add_argument('--execute',action='store_true')
    a=p.parse_args();prepared=Path(a.prepared)/'receipt.json'
    require(sha(prepared)==a.prepared_sha256,'Exact new CPU preparation required')
    r=json.loads(prepared.read_text());require(r['schema']==PREP and r['status']=='CPU_all10_source_units_current_features_and_fixture_qualified','Actual CPU source/units qualification required')
    verify_sources(r['source_pins']);plan=make_plan(a.prepared,a.prepared_sha256,a.python,a.output)
    if not a.execute:
        print(json.dumps(dict(status='CPU_plan_valid_pending_root_GPU_admission',plan=plan,GPU_queries=0,GPU_launches=0)));return
    require(a.root_admission and a.root_admission_sha256 and sha(a.root_admission)==a.root_admission_sha256,'Exact fresh root admission required')
    admission=json.loads(Path(a.root_admission).read_text())
    require(admission['schema']=='coupled-goal-ACT-all14-local3090-root-admission-v1' and admission['GPU_training_authorized'] is True
            and admission['prepared']==ref(prepared) and admission['plan']==plan
            and admission['supervisor_sha256']==sha(supervisor.__file__)
            and admission['source_pins']==r['source_pins'],'Root admission must bind this exact pair/source')
    # Root separately consumes genuine predecessor foreground and records all
    # process/GPU/flock drain. Preserve literal nonzero outcomes where present.
    require(admission['predecessor_original_wait_consumed'] is True and admission['whole_predecessor_process_families_absent'] is True
            and admission['predecessor_signals_sent']==[] and len(admission['predecessor_proofs'])>0,'Actual natural predecessor wait/drain mandatory')
    for value in admission['predecessor_proofs']:verify(value)
    require(admission['independent_review']['qualified'] is True,'Independent CPU review mandatory')
    verify(admission['independent_review']['receipt'])
    with Path('/tmp/robosyn-local3090-experiment.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);result=supervisor.run_plan(plan)
        report=dict(schema=PAIR,status='failed_original_preserved_no_retry',supervisor_receipt=ref(Path(a.output)/'receipt.json'),
            original_supervisor_exit_code=result['exit_code'],root_admission=ref(a.root_admission),robot_deployment_authorized=False,robot_success_claim=False)
        if result['exit_code']==0:
            reports={name:json.loads((Path(a.output)/name/'receipt.json').read_text()) for name in ('role_goal','state_action_control')}
            role,state=reports.values()
            for key in ('history','gate_history'):
                require([v['FIT_order_sha256'] for v in role[key]]==[v['FIT_order_sha256'] for v in state[key]],'Matched actual optimizer/gate order differs')
            require(all(v['status']=='completed_fixed30_coupled_all14_20_independent_FIT_gate' and v['reload_predictions_bitexact'] is True
                        and (v['optimizer_updates'],v['gate_updates'])==(11820,1000) for v in reports.values()),'Actual fixed30/20 branch completion required')
            report.update(status='both_original_GPU_children_exit0_natural_complete',branches={n:ref(Path(a.output)/n/'receipt.json') for n in reports},
                diagnostic_FIT_original50_plus_opening_criterion_met=criterion(role,state,50),
                diagnostic_FIT_executed10_plus_opening_criterion_met=criterion(role,state,10),MONITOR_report_only=True)
            report['diagnostic_FIT_criterion_met']=report['diagnostic_FIT_original50_plus_opening_criterion_met'] and report['diagnostic_FIT_executed10_plus_opening_criterion_met']
        write(Path(a.output)/'pair-receipt.json',report);raise SystemExit(result['exit_code'])


if __name__=='__main__':main()
