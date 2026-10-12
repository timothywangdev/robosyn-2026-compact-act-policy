"""Five sequential CPU children; exact corpus admission and original wait proofs."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from experiments.goal_latent_act.prepare import write_json
from experiments.goal_latent_act_campaign.common import ref,require
from experiments.goal_latent_act_full_data.run_training_pair import identity,group_members
from . import data

SCHEMA='goal-latent-act-commanded50-full63000-original-CPU-pilot-v1'

def run_cpu_child(argv,output,name,report,timeout=1850):
    output=Path(output);tick=time.monotonic();child=None;row=None
    try:
        with (output/(name+'.stdout.log')).open('x') as stdout,(output/(name+'.stderr.log')).open('x') as stderr:
            child=subprocess.Popen(argv,stdout=stdout,stderr=stderr,start_new_session=True)
            owned=identity(child.pid);require(owned and owned['pid']==owned['group']==owned['session'],'Fresh CPU child identity required')
            row=dict(name=name,argv=argv,identity=owned,original_exit_code=None,signals_sent=[],status='actual_CPU_child_running')
            report['runs'].append(row);write_json(output/'receipt.json',report)
            print(json.dumps(dict(name=name,pid=owned['pid'],birth=owned['birth'])),flush=True)
            code=child.wait(timeout=timeout)
        row['original_exit_code']=code
    finally:
        if child is not None and row is not None:
            if child.poll() is None:
                owned=row['identity'];current=identity(child.pid)
                require(current==owned,'Exact fresh CPU identity required before cleanup')
                os.killpg(owned['group'],signal.SIGTERM);row['signals_sent'].append(int(signal.SIGTERM))
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    require(identity(child.pid)==owned,'Exact owned CPU identity before KILL')
                    os.killpg(owned['group'],signal.SIGKILL);row['signals_sent'].append(int(signal.SIGKILL));child.wait(timeout=5)
            row.update(original_exit_code=child.returncode,original_child_reaped=True,original_process_absent=identity(child.pid) is None,
                independently_observed_group_after_reap=group_members(row['identity']['group']),
                natural_completion_no_signals_sent=not row['signals_sent'],elapsed_seconds=time.monotonic()-tick,status='captured_original_exit')
            receipt=output/name/'receipt.json'
            if receipt.exists():row['child_receipt']=ref(receipt)
            write_json(output/'receipt.json',report)
    require(row['original_exit_code']==0 and row['original_process_absent'] and not row['independently_observed_group_after_reap'] and not row['signals_sent'],'Actual original0/reap/natural empty CPU group required')
    require('child_receipt' in row,'Original CPU child must persist scientific receipt')
    return row

def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for k in ('snapshot','snapshot-sha256','geometry','urdf','campaign','campaign-sha256','protocol','protocol-sha256','source-seal','source-seal-sha256','python','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--campaign-original-wait','--corpus-admission',dest='campaign_original_wait',required=True)
    p.add_argument('--campaign-original-wait-sha256','--corpus-admission-sha256',dest='campaign_original_wait_sha256',required=True)
    a=p.parse_args();started=time.monotonic()
    require(data.sha(a.source_seal)==a.source_seal_sha256,'Exact reviewed full CPU source seal required')
    seal=json.loads(Path(a.source_seal).read_text())
    require(seal['schema']=='goal-latent-act-full63000-CPU-source-seal-v1' and seal['source_pins']==data.pins(),'Reviewed full scientific source closure changed')
    data.verify_terminal(a.campaign,a.campaign_sha256,a.campaign_original_wait,a.campaign_original_wait_sha256)
    require(data.sha(a.protocol)==a.protocol_sha256,'Exact preregistered full-data protocol required')
    protocol=json.loads(Path(a.protocol).read_text())
    require(protocol['schema']=='goal-latent-act-command50-sequence-full63000-preregistered-v1' and protocol['source_counts']=={'FIT':63000,'MONITOR':7000} and protocol['partitions']=={'optimizer_FIT':50400,'diagnostic_FIT':6300,'gate_FIT':6300,'MONITOR':7000},'Exact independent full-data protocol required')
    available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
    require(available>=12*(1<<30),'Full70k CPU preparation requires12GiB available RAM')
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    report=dict(schema=SCHEMA,status='actual_CPU_pilot_running',controller=identity(os.getpid()),started_UTC=datetime.now(timezone.utc).isoformat(),
        protocol=ref(a.protocol),source_seal=ref(a.source_seal),source_pins=data.pins(),host_MemAvailable_admission_bytes=available,runs=[],GPU_queries=0,GPU_launches=0,robot_launches=0,
        source_FIT=63000,source_MONITOR=7000,optimizer_FIT=50400,diagnostic_FIT=6300,gate_FIT=6300,total_wall_budget_seconds=4500,MONITOR_selection=False)
    def expired(signum,*_):raise TimeoutError('Full CPU pilot bounded/interrupted '+str(signum))
    old={s:signal.signal(s,expired) for s in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)};signal.alarm(4500)
    def run(name,module,args,timeout=1850):
        for source,digest in report['source_pins'].items():require(data.sha(data.ROOT/source)==digest,'Frozen full source changed before child')
        return run_cpu_child([a.python,'-B','-m',module,*args],output,name,report,timeout)
    try:
        run('prepared','experiments.goal_latent_act_command_goal_training_full.data',
            ['--snapshot',a.snapshot,'--snapshot-sha256',a.snapshot_sha256,'--geometry',a.geometry,'--urdf',a.urdf,
             '--campaign',a.campaign,'--campaign-sha256',a.campaign_sha256,'--campaign-original-wait',a.campaign_original_wait,
             '--campaign-original-wait-sha256',a.campaign_original_wait_sha256,'--output',str(output/'prepared')],950)
        prepared=ref(output/'prepared/receipt.json')
        run('head50','experiments.goal_latent_act_command_goal_training_full.train_head',
            ['--data',str(output/'prepared'),'--data-sha256',prepared['sha256'],'--horizon','50','--output',str(output/'head50')])
        report['status']='head_completed_original0_natural_CPU';write_json(output/'head-original-wait.json',report)
        head=json.loads((output/'head50/checkpoint.json').read_text())
        run('sequence_prepared','experiments.goal_latent_act_sequence_contact_training_full.prepare',
            ['--data',str(output/'prepared'),'--data-sha256',prepared['sha256'],'--highlevel',str(output/'head50'),
             '--highlevel-sha256',head['weights_sha256'],'--original-pilot',str(output/'head-original-wait.json'),
             '--original-pilot-sha256',data.sha(output/'head-original-wait.json'),'--protocol',a.protocol,
             '--protocol-sha256',a.protocol_sha256,'--output',str(output/'sequence_prepared')],950)
        seq=ref(output/'sequence_prepared/receipt.json')
        for name,control in [('role_goal',False),('state_action_control',True)]:
            args=['--prepared',str(output/'sequence_prepared'),'--prepared-sha256',seq['sha256'],'--output',str(output/name)]
            if control:args.append('--state-action-only')
            run(name,'experiments.goal_latent_act_sequence_contact_training_full.train',args)
        factual=json.loads((output/'role_goal/receipt.json').read_text());control=json.loads((output/'state_action_control/receipt.json').read_text())
        for key in ('history','gate_history'):
            require([r['FIT_order_sha256'] for r in factual[key]]==[r['FIT_order_sha256'] for r in control[key]],'Actual role/control fixed budgets and FIT order differ')
        metrics=factual['final_sequence_diagnostic_FIT']['metrics'];free=control['final_sequence_diagnostic_FIT']['metrics']
        candidate=metrics['candidate']['MAE_m'];parent=metrics['parent']['MAE_m'];state=free['candidate']['MAE_m']
        timing=factual['final_gate_FIT']['per_task']['item_assembly']['closed_boundary_timing']['candidate']['same_direction_recall_within_5']
        report.update(status='all_original_exit0_natural_CPU_pilot_complete',completed_UTC=datetime.now(timezone.utc).isoformat(),
            result=dict(diagnostic_FIT_parent_MAE_m=parent,diagnostic_FIT_role_goal_MAE_m=candidate,diagnostic_FIT_state_control_MAE_m=state,
                Assembly_gate_FIT_within5_recall=timing,predeclared_offline_diagnostic_criteria_met=candidate<=.9*parent and candidate<=.95*state and timing>=.5,
                robot_deployment_authorized=False,MONITOR_report_only=True,robot_success_claim=False),actual_trained_heads=1,actual_trained_sequences=2)
    except BaseException as error:
        report.update(status='failed_original_preserved_no_retry',failure=dict(type=type(error).__name__,message=str(error)));raise
    finally:
        signal.alarm(0)
        for s,h in old.items():signal.signal(s,h)
        report['elapsed_seconds']=time.monotonic()-started;write_json(output/'receipt.json',report)

if __name__=='__main__':main()
