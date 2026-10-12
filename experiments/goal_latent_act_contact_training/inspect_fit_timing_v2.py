"""FIT-only binary closed/open boundary timing; v1 class-path audit preserved."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import signal
import time
import numpy as np
import torch
from experiments.goal_latent_act_command_goal_training import data
from experiments.goal_latent_act_campaign.common import ref,require
from experiments.goal_latent_act.prepare import write_json,sha
from .contracts import load_checkpoint
from .train import batch
from .model import opening_classes


def events(target,pred,valid):
    opening_classes(target);opening_classes(pred)
    actual=(target<=.005).numpy();prediction=(pred<=.005).numpy();valid=valid.numpy()
    c=Counter();lags=[]
    for row in range(len(actual)):
        for hand in (0,1):
            n=int(valid[row].sum());a=actual[row,:n,hand];p=prediction[row,:n,hand]
            positions=np.flatnonzero(a[1:]!=a[:-1])+1;predicted=np.flatnonzero(p[1:]!=p[:-1])+1
            c['target_events']+=len(positions);c['predicted_events']+=len(predicted)
            for t in positions:
                direction=(int(a[t-1]),int(a[t]));c['event_'+str(direction)]+=1
                matches=[int(k) for k in predicted if (int(p[k-1]),int(p[k]))==direction]
                if not matches:c['no_same_direction_prediction']+=1;continue
                nearest=min(matches,key=lambda k:abs(k-t));lag=nearest-int(t);lags.append(lag)
                c['same_direction_prediction_exists']+=1
                for tolerance in (0,1,2,5):c['same_direction_within_'+str(tolerance)]+=abs(lag)<=tolerance
    return dict(c),lags


def aggregate(c,lags):
    n=c['target_events'];r=dict(c)
    r.update({f'same_direction_recall_within_{k}':c[f'same_direction_within_{k}']/max(1,n) for k in (0,1,2,5)})
    r['nearest_direction_lag_frames_median']=float(np.median(lags)) if lags else None
    r['nearest_direction_abs_lag_frames_median']=float(np.median(np.abs(lags))) if lags else None
    r['lag_frame_histogram']={str(k):int(v) for k,v in sorted(Counter(lags).items())}
    return r


def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--pilot',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);require(not out.exists(),'Fresh FIT timing output required');started=time.monotonic()
    torch.set_num_threads(8);torch.set_num_interop_threads(1);signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('FIT-only timing180s bound')));signal.alarm(180)
    pilot=Path(a.pilot);r=json.loads((pilot/'prepared/receipt.json').read_text())
    source,records,arrays,tasks=data.load(Path(r['source_data']['path']).parent,r['source_data']['sha256'])
    with np.load(pilot/'prepared/features.npz',allow_pickle=False) as z:features={k:z[k].copy() for k in z.files}
    stats={k:torch.tensor(v,dtype=torch.float32) for k,v in r['statistics'].items() if k in ('qpos_mean','qpos_scale','base_mean','base_scale','eef_scale')};stats['tasks']=tasks
    models={name:load_checkpoint(pilot/name)[0] for name in ('role_goal','state_action_control')}
    result={}
    with torch.no_grad():
        for split in ('residual_fit','gate_fit'):
            ids=np.flatnonzero(features[split]);require(all(records[i]['split']=='fit' and records[i]['episode']<900 for i in ids),'FIT-only timing rows required')
            counters=defaultdict(Counter);lags=defaultdict(list)
            for start in range(0,len(ids),128):
                current=ids[start:start+128];x=batch(arrays,features,current,stats);target=torch.from_numpy(arrays['physical_actions'][current][:,:,[6,13]]);valid=torch.from_numpy(arrays['valid'][current])
                candidates={'parent':x['parent_grips_m']}
                for name,m in models.items():candidates[name]=m(**x,state_action_only=name=='state_action_control')['grips_m']
                for j,task in enumerate(tasks):
                    mask=torch.from_numpy(arrays['task_index'][current]==j)
                    for name,pred in candidates.items():
                        c,l=events(target[mask],pred[mask],valid[mask]);counters[task,name].update(c);lags[task,name].extend(l)
            result[split]=dict(rows=len(ids),demonstrations=len({(records[i]['task'],records[i]['episode']) for i in ids}),
                per_task={t:{name:aggregate(counters[t,name],lags[t,name]) for name in ('parent','role_goal','state_action_control')} for t in tasks})
    report=dict(schema='physical-opening-FIT-only-closed-boundary-timing-diagnostic-v2',status='completed_CPU_readback_no_training',
                data=ref(pilot/'prepared/receipt.json'),model_contracts={n:ref(pilot/n/'checkpoint.json') for n in models},
                source=ref(__file__),splits=result,MONITOR_rows_used=0,optimizer_updates=0,GPU_queries=0,GPU_launches=0,
                CUDA_initialized=torch.cuda.is_initialized(),elapsed_seconds=time.monotonic()-started,
                closed_boundary_m=.005,original_three_class_audit='FIT-timing-diagnostic.json preserved; direct3class path misses smoothed intermediate transitions',interpretation='Binary closed/open threshold timing from FIT commanded labels; conditional nearest-event lag NOT actual contact/tracking latency')
    require(not report['CUDA_initialized'],'CPU only');write_json(out,report);signal.alarm(0)
    print(json.dumps(dict(status=report['status'],rows={k:v['rows'] for k,v in result.items()},elapsed_seconds=report['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
