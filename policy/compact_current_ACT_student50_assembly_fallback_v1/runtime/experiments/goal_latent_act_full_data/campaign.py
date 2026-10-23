"""Sequential resumable all-data campaign, immutable bounded child shards.

No failed/partial shard is reused or retried. A resume accepts only published,
hash-bound natural-exit shards; independent supervisor performs GPU admission.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
from experiments.goal_latent_act_full_data import produce as p, publish

SCHEMA='goal-latent-act-full-data-campaign-v1'
ROOT=p.ROOT
BASE=ROOT/'outputs/architecture-reassessment/goal-latent-act-full-data-cache-v1'
PACKED=ROOT/'outputs/architecture-reassessment/goal-latent-act-first-training-v1/data-v1'
BALANCED=ROOT/'outputs/architecture-reassessment/goal-latent-act-balanced-cache-v1'
RUNTIME_UNITS=BALANCED/'runtime-unit-qualification-v1/receipt.json'
PRODUCER_SHA='8cdb861d4deddb5fbf5ca6b6618ddedc8c504e37c6c553cef1a25954db114987'
SUPERVISOR_SHA='76857b6d3fe77ed401c6e65886dc841b04b214c3888cb95976e87f9ff4cc61f7'
SEED_SHA='b34e4b61afbe4d37a39371822ec57179cbc3950fe42ac277ef29ad824b5aa1d1'
SEED_SUPERVISOR_SHA='a5d9f42317dff62c9e7243ae626e1ff6fd09e7b770c0654f9d6fdd0a652a86c7'

def schedule():
    rows=[dict(number=i+1,fit_start=16+9*i,fit_count=9,monitor_start=902+i,monitor_count=1) for i in range(98)]
    rows.append(dict(number=99,fit_start=898,fit_count=2,monitor_start=1000,monitor_count=0))
    return rows

def seed():
    receipt=BALANCED/'actual-full1260-v2/receipt.json';supervisor=BALANCED/'actual-full1260-supervisor-v2/receipt.json'
    p.require(p.sha(receipt)==SEED_SHA and p.sha(supervisor)==SEED_SUPERVISOR_SHA,'Exact completed1260 seed required')
    s=json.loads(supervisor.read_text());r=json.loads(receipt.read_text())
    p.require(s['original_child_exit_code']==s['exit_code']==0 and s['child_reaped'] is True
              and s['drain']['natural'] is True and s['drain']['drained'] is True
              and s['independent_group_after_reap']==[] and s['independent_GPU_after']['compute_pids']==[], 'Seed captured exit/drain required')
    p.require(r['status']=='balanced_experimental_cache_complete' and r['fit']==1120 and r['monitor']==140,'Actual seed exact split required')
    p.require(len(r['records'])==1260 and len({x['clip_id'] for x in r['records']})==1260,'Unique exact seed required')
    for row in r['records']:p.check_ref(row['file'])
    return dict(receipt=p.ref(receipt),supervisor=p.ref(supervisor),fit=1120,monitor=140),r['records']

def checked_publication(path):
    value=json.loads(path.read_text());p.require(value['schema']==publish.SCHEMA and value['status']=='published_after_original_exit0_natural_drain','Published original-exit shard required')
    for name in ('source_plan','source_child_receipt','supervisor','statistics','unit_clarification','runtime_unit_qualification'):p.check_ref(value[name])
    supervision=json.loads(Path(value['supervisor']['path']).read_text());publish.terminal_supervision(supervision,value['source_plan']['sha256'])
    records=[]
    for t in value['tasks']:
        p.check_ref(t);task=json.loads(Path(t['path']).read_text())
        for r in task['records']:p.check_ref(r['file']);records.append(r)
    p.require(len(records)==value['records'] and sum(x['split']=='fit' for x in records)==value['fit'] and sum(x['split']=='monitor' for x in records)==value['monitor'],'Published counts differ')
    return value,records

def main():
    parser=argparse.ArgumentParser(allow_abbrev=False);parser.add_argument('--gpu-uuid',required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--deadline-utc',required=True);parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    deadline=dt.datetime.fromisoformat(args.deadline_utc.replace('Z','+00:00'));p.require(deadline.tzinfo is not None,'Explicit UTC deadline required')
    p.require(p.sha(ROOT/'experiments/goal_latent_act_full_data/produce.py')==PRODUCER_SHA
              and p.sha(ROOT/'experiments/goal_latent_act_full_data/run_local.py')==SUPERVISOR_SHA,'Immutable actual first-shard sources required')
    work=schedule();p.require(sum((x['fit_count']+x['monitor_count'])*70 for x in work)+1260==70000,'Exact full target required')
    if not args.execute:print(json.dumps(dict(schema=SCHEMA,target_FIT=63000,target_MONITOR=7000,new_shards=len(work),GPU_queries=0)));return 0
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(schema=SCHEMA,status='running',controller_pid=__import__('os').getpid(),deadline_utc=args.deadline_utc,
        sources={str(ROOT/'experiments/goal_latent_act_full_data/produce.py'):PRODUCER_SHA,str(ROOT/'experiments/goal_latent_act_full_data/run_local.py'):SUPERVISOR_SHA,
                 str(Path(__file__).resolve()):p.sha(__file__),str(Path(publish.__file__).resolve()):p.sha(publish.__file__)},
        completed_shards=[],target_FIT=63000,target_MONITOR=7000,optimizer_updates=0,robot_quality_claim=False)
    started=time.monotonic();child=None
    def save():report['elapsed_seconds']=time.monotonic()-started;p.write(args.output/'receipt.json',report)
    def interrupted(*_):raise InterruptedError('Owned campaign interruption; bounded supervisor must drain')
    previous={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
    for s in previous:signal.signal(s,interrupted)
    try:
        original,all_records=seed();report['seed']=original;report.update(FIT=1120,MONITOR=140);save()
        for spec in work:
            name='shard%04d'%spec['number'];publication=BASE/'published'/name/'receipt.json'
            if publication.exists():
                result,records=checked_publication(publication)
                p.require(result['episodes']=={'fit':list(range(spec['fit_start'],spec['fit_start']+spec['fit_count'])),'monitor':list(range(spec['monitor_start'],spec['monitor_start']+spec['monitor_count']))},'Wrong episode coverage in published shard')
            else:
                p.require((deadline-dt.datetime.now(dt.timezone.utc)).total_seconds()>1900,'Insufficient deadline for bounded next job; stop without launch')
                plan_path=BASE/(name+'-plan.json');data=BASE/(name+'-data');supervised=BASE/(name+'-supervisor')
                p.require(not plan_path.exists() and not data.exists() and not supervised.exists(),'Preserve existing unqualified/failed output; no retries')
                report['current_shard']=dict(spec,status='CPU_preparing');save()
                plan=p.prepare(PACKED,BALANCED/'released-assets-v1/receipt.json',spec['fit_start'],spec['fit_count'],spec['monitor_start'],spec['monitor_count'],data.resolve())
                p.require(p.sha(RUNTIME_UNITS)==publish.RUNTIME_UNITS_SHA,'Exact runtime-unit qualification required')
                plan['runtime_unit_qualification']=p.ref(RUNTIME_UNITS);plan['input_files'].append(plan['runtime_unit_qualification']);p.write(plan_path,plan)
                report['current_shard'].update(status='GPU_supervisor_launch_pending',plan=p.ref(plan_path));save()
                command=[sys.executable,'-B',str(ROOT/'experiments/goal_latent_act_full_data/run_local.py'),'--plan',str(plan_path),
                    '--plan-sha256',p.sha(plan_path),'--gpu-uuid',args.gpu_uuid,'--output',str(supervised),'--execute']
                with (args.output/(name+'.log')).open('xb') as log:
                    child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                    report['current_shard'].update(status='bounded_supervisor_running',pid=child.pid,argv=command);save()
                    try:code=child.wait(timeout=1835)
                    except BaseException as wait_error:
                        # The exact Popen supervisor owns its own descendant cleanup.
                        child.terminate()
                        try:code=child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            report['supervisor_timeout_cleanup_unproven']=True;save();raise
                        report['supervisor_interrupted_original_exit']=code;save();raise wait_error
                report['current_shard']['captured_original_supervisor_exit_code']=code;save()
                p.require(code==0,'Original supervisor failed; preserve data and stop, no retry')
                result=publish.publish(plan_path,p.sha(plan_path),supervised/'receipt.json',publication.parent)
                result,records=checked_publication(publication)
            previous={r['clip_id'] for r in all_records};p.require(not previous.intersection(r['clip_id'] for r in records),'Duplicate seed/shard records forbidden')
            all_records.extend(records);report['completed_shards'].append(p.ref(publication));report['FIT']+=result['fit'];report['MONITOR']+=result['monitor'];save()
            request=BASE/'switch-successor-after-current.json'
            if request.exists():
                report.update(status='completed_shards_preserved_successor_requested',successor_request=p.ref(request));save();return 0
        p.require(len(all_records)==70000 and report['FIT']==63000 and report['MONITOR']==7000,'Exact all-data split target required')
        packed_rows=json.loads((PACKED/'rows.json').read_text());expected={(r['clip_id'],r['task'],r['revision'],r['episode'],r['frame'],r['split']) for r in packed_rows}
        actual={(r['clip_id'],r['task'],r['revision'],r['episode'],r['frame'],r['split']) for r in all_records}
        p.require(expected==actual,'All70000 exact role/teacher identities required')
        report.update(status='all63000_FIT_7000_MONITOR_frozen_ACT_records_published',coverage_exact=True)
        p.write(args.output/'index.json',all_records);report['index']=p.ref(args.output/'index.json');save()
        return 0
    except BaseException as error:
        report.update(status='stopped_preserved_no_retry',failure=dict(type=type(error).__name__,message=str(error)));save();raise
    finally:
        for s,h in previous.items():signal.signal(s,h)
if __name__=='__main__':raise SystemExit(main())
