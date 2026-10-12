"""Immutable CPU training snapshot from accepted original-exit cache shards."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from experiments.goal_latent_act_full_data import campaign, produce as p
from experiments.goal_latent_act.train_correction import live_original

SCHEMA='goal-latent-act-published-cache-training-snapshot-v1'
KEYS=('physical_base','normalized_base','physical_actions','normalized_actions','valid',
      'canonical_current_qpos14','role_features','teacher_eef_now','teacher_eef_future')

def prepare(output,min_new_shards=4):
    output=Path(output);p.require(not output.exists(),'Fresh accepted-cache snapshot required')
    seed,records=campaign.seed();sources=[seed]
    publications=sorted((campaign.BASE/'published').glob('shard*/receipt.json'))
    p.require(len(publications)>=min_new_shards,'At least four independently published new shards required')
    normalizers={}
    seed_raw=json.loads(Path(seed['receipt']['path']).read_text())
    for t in seed_raw['tasks']:p.check_ref(t['normalizer']);normalizers[t['task']]=t['normalizer']
    for publication in publications:
        r,values=campaign.checked_publication(publication)
        outer=json.loads(Path(r['supervisor']['path']).read_text())
        p.require(not live_original(outer['child']) and not live_original(outer['controller']),'Original producer/controller births must be absent')
        plan=json.loads(Path(r['source_plan']['path']).read_text())
        for path,digest in plan['source_sha256'].items():p.require(p.sha(path)==digest,'Published scientific source changed')
        p.check_ref(r['unit_clarification']);p.check_ref(r['runtime_unit_qualification'])
        for ref in r['tasks']:
            t=json.loads(Path(ref['path']).read_text());p.check_ref(t['normalizer'])
            with np.load(normalizers[t['task']]['path'],allow_pickle=False) as a,np.load(t['normalizer']['path'],allow_pickle=False) as b:
                p.require(set(a.files)==set(b.files) and all(p.same_array_bytes(a[k],b[k]) for k in a.files),'Native task normalization buffers differ across shards')
        records.extend(values);sources.append(dict(publication=p.ref(publication)))
    p.require(len(records)==len({r['clip_id'] for r in records}),'Duplicate accepted records forbidden')
    records.sort(key=lambda r:(list(p.TASK_REVISIONS).index(r['task']),r['episode'],r['frame']))
    count=Counter((r['task'],r['split']) for r in records)
    p.require(len({count[t,'fit'] for t in p.TASK_REVISIONS})==1 and len({count[t,'monitor'] for t in p.TASK_REVISIONS})==1,'Allten equally balanced snapshot required')
    for r in records:
        p.require((r['split']=='fit' and 0<=r['episode']<900) or (r['split']=='monitor' and 900<=r['episode']<1000),'Actual independent FIT/MONITOR namespace required')
    output.mkdir(parents=True)
    result=dict(schema=SCHEMA,status='all_accepted_sources_hash_verified_no_live_partial',sources=sources,records=records,
        normalizers=normalizers,fit=sum(r['split']=='fit' for r in records),monitor=sum(r['split']=='monitor' for r in records),
        tasks=list(p.TASK_REVISIONS),counts={t:{s:count[t,s] for s in ('fit','monitor')} for t in p.TASK_REVISIONS},
        unit_clarification=p.ref(p.UNITS),runtime_unit_qualification=p.ref(campaign.RUNTIME_UNITS),
        source_sha256=p.sha(__file__),GPU_queries=0,live_partial_records_used=0)
    p.write(output/'snapshot.json',result)
    return result

def load(snapshot,expected_sha):
    snapshot=Path(snapshot);p.require(p.sha(snapshot)==expected_sha,'Exact fixed training snapshot required')
    r=json.loads(snapshot.read_text());p.require(r['schema']==SCHEMA and r['status']=='all_accepted_sources_hash_verified_no_live_partial','Accepted immutable snapshot required')
    p.check_ref(r['unit_clarification']);p.check_ref(r['runtime_unit_qualification'])
    p.require(r['unit_clarification']['sha256']==p.UNITS_SHA,'Native-grip clarification required')
    # Recheck every publication/exit receipt rather than trusting a snapshot claim.
    for source in r['sources']:
        if 'publication' in source:
            p.check_ref(source['publication']);campaign.checked_publication(Path(source['publication']['path']))
        else:p.check_ref(source['receipt']);p.check_ref(source['supervisor']);campaign.seed()
    records=r['records'];arrays={key:[] for key in KEYS};tasks=sorted(r['normalizers']);seen=set()
    norms={}
    for task,ref in r['normalizers'].items():
        p.check_ref(ref)
        with np.load(ref['path'],allow_pickle=False) as z:norms[task]={k:z[k].copy() for k in z.files}
    for record in records:
        p.require(record['clip_id'] not in seen,'Duplicate training record');seen.add(record['clip_id']);p.check_ref(record['file'])
        with np.load(record['file']['path'],allow_pickle=False) as z:
            p.validate_payload(z);row={k:z[k].copy() for k in KEYS}
        for key in ('physical_base','normalized_base','physical_actions','normalized_actions'):
            p.require(row[key].shape==(50,14) and row[key].dtype==np.float32 and np.isfinite(row[key]).all(),'Finite exact native50 FP32 arrays required')
        p.require(row['valid'].dtype==np.bool_ and row['valid'].shape==(50,) and row['valid'][:32].all(),'Literal future command/EOS mask required')
        p.require(row['role_features'].shape==(3,4096) and row['role_features'].dtype==np.float16
                  and row['canonical_current_qpos14'].shape==(14,) and row['teacher_eef_now'].shape==row['teacher_eef_future'].shape==(2,4,4),'Current role/state/FK ABI required')
        n=norms[record['task']];mean=n['action_mean'];scale=n['action_std']
        p.require(np.allclose(row['physical_base'],row['normalized_base']*scale+mean,atol=2e-6,rtol=1e-6),'Native ACT release denormalization differs')
        arm=np.array([0,1,2,3,4,5,7,8,9,10,11,12]);mask=row['valid']
        p.require(np.allclose(row['physical_actions'][mask][:,arm],(row['normalized_actions']*scale+mean)[mask][:,arm],atol=2e-6,rtol=1e-6),'Physical arm-only target units differ')
        for key in KEYS:arrays[key].append(row[key])
    arrays={key:np.stack(values) for key,values in arrays.items()}
    arrays['fit']=np.array([x['split']=='fit' for x in records]);arrays['task_index']=np.array([tasks.index(x['task']) for x in records])
    p.require(int(arrays['fit'].sum())==r['fit'] and int((~arrays['fit']).sum())==r['monitor'],'Snapshot split counts differ')
    return r,records,arrays,tasks
