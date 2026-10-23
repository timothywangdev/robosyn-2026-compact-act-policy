"""Publish a frozen shard only after independently captured natural exit/drain.

This module never initializes CUDA. Child/task completion alone is insufficient.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from experiments.goal_latent_act_full_data import produce as p

SCHEMA = 'goal-latent-act-full-data-published-shard-v1'
RUNTIME_UNITS_SHA = '78fab9ff1e61ba7b315f6c85c1894ea3a075847a16c6ccbcf54d355554fab3c7'
ARM = np.array([0,1,2,3,4,5,7,8,9,10,11,12])

def terminal_supervision(supervision, plan_sha):
    p.require(supervision['schema']=='goal-latent-act-full-data-local-supervision-v1', 'Full-data supervision schema required')
    p.require(supervision['plan_sha256']==plan_sha, 'Supervisor bound a different plan')
    p.require(supervision['exit_code']==0 and supervision['original_child_exit_code']==0
              and supervision.get('child_reaped') is True and not supervision.get('cleanup_errors'), 'Captured original exit0/reap required')
    p.require(supervision['drain']['drained'] is True and supervision['drain']['natural'] is True,
              'Successful natural owned drain required')
    p.require(supervision['independent_group_after_reap']==[]
              and supervision['independent_GPU_after']['compute_pids']==[], 'Independent empty group/GPU required')

def publish(plan_file, plan_sha, supervision_file, output):
    plan_file=Path(plan_file);supervision_file=Path(supervision_file);output=Path(output)
    p.require(p.sha(plan_file)==plan_sha, 'Exact plan bytes required')
    plan=json.loads(plan_file.read_text());supervision=json.loads(supervision_file.read_text())
    terminal_supervision(supervision,plan_sha)
    p.require(plan['schema']==p.SCHEMA, 'Full-data source schema required')
    for name,digest in plan['source_sha256'].items():p.require(p.sha(name)==digest,'Bound source changed '+name)
    runtime=plan['runtime_unit_qualification'];p.check_ref(runtime)
    p.require(runtime['sha256']==RUNTIME_UNITS_SHA, 'Runtime grip qualification required')
    p.check_ref(plan['unit_clarification']);p.require(plan['unit_clarification']['sha256']==p.UNITS_SHA,'Native-grip clarification required')
    child_file=Path(plan['output'])/'receipt.json';child=json.loads(child_file.read_text())
    p.require(child['status']=='full_data_shard_complete_pending_original_exit0_publication', 'Complete original frozen child required')
    p.require(child['plan_sha256']==hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest(),'Producer semantic plan binding differs')
    p.require(child['optimizer_updates']==0 and child['native_release_sources_unchanged'] is True,'Frozen zero-training source required')
    expected={r['clip_id']:r for r in plan['rows']};records=child['records']
    p.require(len(expected)==len(records)==len({r['clip_id'] for r in records}), 'Exact unique complete shard required')
    sums=np.zeros(12,np.float64);squares=np.zeros(12,np.float64);count=0;fit=monitor=0
    tasks={name:[] for name in p.TASK_REVISIONS}
    for r in records:
        source=expected[r['clip_id']]
        p.require(all(r[k]==source[k] for k in source), 'Complete source/current-RGB/role/teacher identity differs')
        p.require(r['split']==('fit' if r['episode']<900 else 'monitor'), 'Independent split differs')
        p.check_ref(r['file'])
        with np.load(r['file']['path'],allow_pickle=False) as z:
            p.validate_payload(z)
            p.require(z['physical_base'].shape==z['physical_actions'].shape==(50,14)
                      and z['valid'].shape==(50,) and z['valid'].dtype==np.bool_
                      and z['role_features'].shape==(3,4096), 'Exact cached ABI required')
            p.require(int(z['valid'].sum())==r['valid_rows']>=32,'Valid real next commands required')
            if r['split']=='fit':
                values=z['physical_actions'][z['valid']][:,ARM].astype(np.float64)
                sums+=values.sum(0);squares+=(values*values).sum(0);count+=len(values)
        fit+=r['split']=='fit';monitor+=r['split']=='monitor';tasks[r['task']].append(r)
    p.require(fit==child['fit'] and monitor==child['monitor'],'Reported split counts differ')
    p.require(not output.exists(), 'Fresh immutable publication required');output.mkdir(parents=True)
    published=[]
    for t in child['tasks']:
        p.check_ref(t['task_receipt']);p.check_ref(t['normalizer'])
        p.require(t['frozen_unchanged'] is True and t['latent_calls']==t['records']==len(tasks[t['task']]),'Exact frozen zero-latent forward counts required')
        task=json.loads(Path(t['task_receipt']['path']).read_text())
        p.require(task['records']==tasks[t['task']] and task['frozen_unchanged'] is True,'Immutable task receipt differs')
        file=output/(t['task']+'.json')
        p.write(file,dict(schema=SCHEMA,status='published_after_original_exit0_natural_drain',task=t['task'],
            records=tasks[t['task']],normalizer=t['normalizer'],source_task_receipt=t['task_receipt'],
            source_plan=p.ref(plan_file),supervisor=p.ref(supervision_file),unit_clarification=plan['unit_clarification'],
            runtime_unit_qualification=runtime,arm12_loss_only=True,native_grip_passthrough=True))
        published.append(p.ref(file))
    mean=sums/count;scale=np.sqrt(np.maximum(squares/count-mean*mean,0))
    statistics=dict(schema='goal-latent-act-completed-shard-fit-arm12-moments-v1',rows=count,
        sum=sums.tolist(),sum_squares=squares.tolist(),mean=mean.tolist(),scale=scale.tolist(),
        arm_indices=ARM.tolist(),units='radians',source='valid physical target rows from FIT only',monitor_rows_used=0)
    p.write(output/'fit-arm-moments.json',statistics)
    report=dict(schema=SCHEMA,status='published_after_original_exit0_natural_drain',fit=fit,monitor=monitor,
        records=len(records),episodes=plan['episodes'],tasks=published,source_plan=p.ref(plan_file),
        source_child_receipt=p.ref(child_file),supervisor=p.ref(supervision_file),source_sha256=plan['source_sha256'],
        statistics=p.ref(output/'fit-arm-moments.json'),unit_clarification=plan['unit_clarification'],runtime_unit_qualification=runtime,
        arm12_loss_only=True,native_grip_passthrough=True,optimizer_updates=0,robot_quality_claim=False,
        publisher_source_sha256=p.sha(__file__))
    p.write(output/'receipt.json',report)
    return report

def main():
    a=argparse.ArgumentParser(allow_abbrev=False);a.add_argument('--plan',type=Path,required=True);a.add_argument('--plan-sha256',required=True)
    a.add_argument('--supervisor',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    report=publish(args.plan,args.plan_sha256,args.supervisor,args.output)
    print(json.dumps(dict(status=report['status'],FIT=report['fit'],MONITOR=report['monitor'],receipt=str(args.output/'receipt.json'))))
if __name__=='__main__':main()
