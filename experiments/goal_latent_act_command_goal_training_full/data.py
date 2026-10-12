"""Exact completed70k source; independent episode-level optimizer/diagnostic/gate."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import signal
import numpy as np
import torch
from experiments.goal_latent_act_full_data import aggregate,campaign_resume_v3 as corpus,publish
from experiments.goal_latent_act_full_data.run_training_pair import identity,group_members
from experiments.goal_latent_act_campaign.common import ref,verify,require,write
from experiments.goal_latent_act.prepare import sha
from .contracts import DATA

ROOT=Path(__file__).resolve().parents[2]
COUNTS=(50400,6300,6300,7000)
GEOMETRY_SHA='68b1c42869efab392ed3dfbbac0336c86e51eda265cb6f9f9822f44f6025e947'
URDF_SHA='788b87e05dd5aa3e18aad177a87aa9692e43c5eb9dfa0f688f7bc7c8ad821dd2'

def pins():
    from experiments.goal_latent_act_command_goal_training.data import pins as shared
    result=shared()
    for name in ('experiments/goal_latent_act_sequence_contact_training/model.py',
                 'experiments/goal_latent_act_contact_training/model.py',
                 'experiments/goal_latent_act_contact_training/prepare.py',
                 'experiments/goal_latent_act_contact_training/inspect_fit_timing_v2.py',
                 'experiments/goal_latent_act_full_data/run_training_pair.py',
                 'experiments/goal_latent_act_full_data/campaign_resume_v3.py',
                 'experiments/goal_latent_act_full_data/publish.py',
                 'experiments/goal_latent_act_data/prepare_assets.py',
                 'experiments/goal_latent_act_campaign/common.py',
                 'policy/wam_adaptation/multitask_data.py'):
        result[name]=sha(ROOT/name)
    for namespace in ('goal_latent_act_command_goal_training_full','goal_latent_act_sequence_contact_training_full'):
        for p in sorted((ROOT/'experiments'/namespace).glob('*.py')):
            result[str(p.relative_to(ROOT))]=sha(p)
    return result

def split(records,tasks):
    require(len(records)==70000 and len(tasks)==10 and len(set(tasks))==10,'Literal full source70k/all10 required')
    require(set(tasks)==set(aggregate.p.TASK_REVISIONS),'Official all10 task identities required')
    seen=set();episodes=Counter()
    for r in records:
        require(r['task'] in tasks and r['revision']==aggregate.p.TASK_REVISIONS[r['task']],'Exact task/revision required')
        require((r['split']=='fit' and 0<=r['episode']<900) or (r['split']=='monitor' and 900<=r['episode']<1000),'FIT/MONITOR namespace differs')
        key=(r['task'],r['episode'],r['frame']);require(key not in seen,'Duplicate exact source anchor forbidden');seen.add(key)
        episodes[r['task'],r['episode']]+=1
    require(set(episodes)=={(t,e) for t in tasks for e in range(1000)} and set(episodes.values())=={7},'Exactly7 anchors from every official1000 demonstrations/task required')
    fit=np.array([r['split']=='fit' for r in records])
    optimizer=fit&np.array([r['episode']%10<8 for r in records]);diagnostic=fit&np.array([r['episode']%10==8 for r in records])
    gate=fit&np.array([r['episode']%10==9 for r in records]);monitor=np.flatnonzero(~fit)
    require((int(optimizer.sum()),int(diagnostic.sum()),int(gate.sum()),len(monitor))==COUNTS,'Predeclared80/10/10 FIT split required')
    return optimizer,diagnostic,gate,monitor

def verify_terminal(campaign_path,campaign_sha,wait_path,wait_sha):
    require(sha(campaign_path)==campaign_sha and sha(wait_path)==wait_sha,'Exact complete campaign and captured original wait required')
    c=json.loads(Path(campaign_path).read_text());w=json.loads(Path(wait_path).read_text())
    require(c['status']=='all63000_FIT_7000_MONITOR_frozen_ACT_records_published' and c['coverage_exact'] is True and (c['FIT'],c['MONITOR'])==(63000,7000),'Actual full campaign completion required; no partial substitution')
    require(w['campaign']==ref(campaign_path),'Corpus/wait proof must bind exact final campaign')
    owned=c['controller_identity'];recorded=w['identity']
    require((recorded['pid'],recorded['birth'],recorded['group'],recorded['session'])==(owned['pid'],owned['birth_ticks'],owned['group'],owned['session']),'Captured original producer-controller identity differs')
    if w['schema']=='goal-latent-act-full-cache-campaign-original-wait-v1':
        require(w['original_exit_code']==0 and w['original_child_reaped'] is True and w['natural_completion_no_signals_sent'] is True and w['signals_sent']==[] and w['independently_observed_group_after_reap']==[],'Original campaign exit0/reap/natural drain required')
    else:
        require(w['schema']=='goal-latent-act-full-corpus-admission-v1' and w['mode']=='complete_corpus_from_original_qualified_shards_outer_wait_unknown','Only explicitly typed qualified-corpus alternative allowed')
        require(w['coordinator_original_outer_exit_code'] is None and w['coordinator_original_outer_wait_captured'] is False,'Unknown coordinator wait must remain explicitly unknown')
        require(w['corpus_index']==c['index'] and w['source_counts']=={'FIT':63000,'MONITOR':7000},'Exact final corpus index/scope binding required')
        require((recorded['pid'],recorded['birth'])==(155029,54941326),'Alternative applies only to reviewed original coordinator birth')
        qualify_corpus(c)
    current=identity(recorded['pid'])
    require(current is None or current['birth']!=recorded['birth'],'Original campaign birth still live')
    require(not group_members(recorded['group']),'Original campaign group not independently drained')
    for path,digest in c['sources'].items():require(sha(path)==digest,'Actual producer source changed')
    verify(c['index']);return c

def qualify_corpus(c):
    """Verify all original shard exits and identities; never invent outer exit0."""
    seed,records=corpus.seed();require(c['seed']==seed,'Exact independently qualified1260 seed required')
    require(len(c['completed_shards'])==99 and len({r['path'] for r in c['completed_shards']})==99,'Exactly99 immutable qualified shard receipts required')
    schedule=corpus.schedule()
    for spec,reference in zip(schedule,c['completed_shards']):
        value=json.loads(verify(reference).read_text())
        require(value['schema']==publish.SCHEMA and value['status']=='published_after_original_exit0_natural_drain','Only original-exit-qualified publication allowed')
        for key in ('source_plan','source_child_receipt','supervisor','statistics','unit_clarification','runtime_unit_qualification'):verify(value[key])
        plan=json.loads(verify(value['source_plan']).read_text())
        require(plan['schema']==aggregate.p.SCHEMA and plan['source_sha256']==value['source_sha256'],'Original publication and exact producer source bindings differ')
        for path,digest in plan['source_sha256'].items():require(sha(path)==digest,'Original qualified shard source changed')
        supervision=json.loads(verify(value['supervisor']).read_text());publish.terminal_supervision(supervision,value['source_plan']['sha256'])
        require(value['episodes']=={'fit':list(range(spec['fit_start'],spec['fit_start']+spec['fit_count'])),'monitor':list(range(spec['monitor_start'],spec['monitor_start']+spec['monitor_count']))},'Exact full-corpus shard episode schedule differs')
        for item in value['tasks']:records.extend(json.loads(verify(item).read_text())['records'])
    index=json.loads(verify(c['index']).read_text())
    keys=lambda rows:{(r['clip_id'],r['task'],r['revision'],r['episode'],r['frame'],r['split'],r['file']['sha256']) for r in rows}
    require(len(records)==70000 and len(keys(records))==70000 and keys(records)==keys(index),'Every exact record/payload SHA must match final corpus; count alone insufficient')
    split(index,list(aggregate.p.TASK_REVISIONS))
    return dict(seed_original_exit0=True,qualified_shards=99,records=70000,FIT=63000,MONITOR=7000)

def pose_features(poses):
    return np.concatenate((poses[...,:3,3],poses[...,:3,:2].swapaxes(-1,-2).reshape(len(poses),2,6)),-1).reshape(len(poses),18)

def optimization_statistics(arrays,optimizer,valid):
    selected=optimizer&valid;require(bool(selected.any()),'Actual terminal-valid optimizer rows required')
    q=arrays['canonical_current_qpos14'][selected];eef=pose_features(arrays['teacher_eef_now'][selected])
    return dict(schema='command-goal-full-optimization-FIT-current-zscore-v1',qpos_mean=q.mean(0).tolist(),qpos_scale=np.maximum(q.std(0),.01).tolist(),eef_mean=eef.mean(0).tolist(),eef_scale=np.maximum(eef.std(0),.01).tolist(),rows=int(selected.sum()),state_units='arm radians; measured grippers normalized opening; arm-major14',eef_units='robot_root_meters and rigid rotation',coordinate_frame='robot_root_meters',normalizer_labels_scope='terminal-valid optimizer FIT episodes%10<8 only',historical_collector_lineage_certified=False)

def prepare(a):
    output=Path(a.output);require(not output.exists(),'Fresh full-data preparation required')
    campaign=verify_terminal(a.campaign,a.campaign_sha256,a.campaign_original_wait,a.campaign_original_wait_sha256)
    require(sha(a.geometry)==GEOMETRY_SHA and sha(a.urdf)==URDF_SHA,'Exact qualified physical FK sources required')
    declared=json.loads(Path(a.snapshot).read_text());require((declared['fit'],declared['monitor'])==(63000,7000),'Full snapshot source counts required')
    source,records,arrays,tasks=aggregate.load(a.snapshot,a.snapshot_sha256)
    optimizer,diagnostic,gate,monitor=split(records,tasks)
    final_index=json.loads(verify(campaign['index']).read_text())
    def keys(rows):return {(r['clip_id'],r['task'],r['revision'],r['episode'],r['frame'],r['split']) for r in rows}
    require(keys(records)==keys(final_index),'Snapshot must exactly cover completed campaign identities')
    spec=importlib.util.spec_from_file_location('full_command_goal_geometry',a.geometry);g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
    chain=g.read_chain(a.urdf);tcp=g.tcp();n=len(records);valid=arrays['valid'][:,49];goal=np.empty((n,2,4,4),np.float32);maximum=0.
    for i in range(n):
        q=arrays['canonical_current_qpos14'][i];current=np.stack([g.fk_link(chain,q[:6],'left')@tcp,g.fk_link(chain,q[7:13],'right')@tcp])
        maximum=max(maximum,float(np.abs(current-arrays['teacher_eef_now'][i]).max()))
        if not valid[i]:goal[i]=arrays['teacher_eef_now'][i];continue
        command=arrays['physical_actions'][i,49];goal[i]=np.stack([g.fk_link(chain,command[:6],'left')@tcp,g.fk_link(chain,command[7:13],'right')@tcp])
    require(maximum<2e-6,'Root-frame current FK teacher parity differs')
    selected=valid&optimizer;translation=np.linalg.norm(goal[selected,...,:3,3]-arrays['teacher_eef_now'][selected,...,:3,3],axis=-1)
    rotation=goal[selected,...,:3,:3]@arrays['teacher_eef_now'][selected,...,:3,:3].swapaxes(-1,-2)
    angle=np.arccos(np.clip((np.trace(rotation,axis1=-2,axis2=-1)-1)/2,-1,1))
    bounds=dict(translation_max_m=float(translation.max()),rotation_max_rad=float(angle.max()),scope='terminal-valid optimization FIT only; fixed bounds .60m/2rad')
    require(bounds['translation_max_m']<=.60+2e-6 and bounds['rotation_max_rad']<=2.+2e-6,'Full optimization FIT exceeds fixed physical bounds; stop for versioned review, no clipping')
    statistics=optimization_statistics(arrays,optimizer,valid);payload={k:arrays[k] for k in ('physical_base','physical_actions','valid','canonical_current_qpos14','role_features','teacher_eef_now','fit','task_index')}
    payload.update(head_valid=valid,command_goal50=goal,optimizer_fit=optimizer,diagnostic_fit=diagnostic,gate_fit=gate)
    output.mkdir(parents=True);np.savez(output/'arrays.npz',**payload);write(output/'rows.json',records);write(output/'statistics.json',statistics)
    masks={'optimizer':optimizer,'diagnostic':diagnostic,'gate':gate,'monitor':~arrays['fit']}
    counts={t:{s:int(sum(r['task']==t and mask[i] and valid[i] for i,r in enumerate(records))) for s,mask in masks.items()} for t in tasks}
    result=dict(schema=DATA,status='prepared_accepted_exact_command_FK_labels',snapshot=ref(a.snapshot),geometry=ref(a.geometry),urdf=ref(a.urdf),campaign=ref(a.campaign),campaign_original_wait=ref(a.campaign_original_wait),source_pins=pins(),tasks=tasks,source_FIT=63000,source_MONITOR=7000,optimizer_source_FIT=50400,diagnostic_source_FIT=6300,gate_source_FIT=6300,head_optimizer_FIT=int(selected.sum()),head_MONITOR=int((valid&~arrays['fit']).sum()),terminal_invalid_FIT=int((~valid&arrays['fit']).sum()),terminal_invalid_MONITOR=int((~valid&~arrays['fit']).sum()),head_per_task=counts,statistics=statistics,normalizers=source['normalizers'],unit_clarification=source['unit_clarification'],runtime_unit_qualification=source['runtime_unit_qualification'],coordinate_frame='robot_root_meters',target_type='commanded FK; NOT measured future EEF',frame_offsets={'command_goal50':50},maximum_current_FK_teacher_matrix_difference=maximum,FIT_target_bounds=bounds,CUDA_initialized=torch.cuda.is_initialized(),GPU_queries=0,GPU_launches=0,artifacts={name:ref(output/name) for name in ('arrays.npz','rows.json','statistics.json')})
    require(not result['CUDA_initialized'],'CPU only preparation required');write(output/'receipt.json',result);return result

def load(root,expected):
    root=Path(root);require(sha(root/'receipt.json')==expected,'Exact full-data receipt required');r=json.loads((root/'receipt.json').read_text())
    require(r['schema']==DATA and r['status']=='prepared_accepted_exact_command_FK_labels','New full source schema required')
    for name,row in r['artifacts'].items():require(verify(row)==root.resolve()/name,'Artifact binding differs')
    for name,digest in r['source_pins'].items():require(sha(ROOT/name)==digest,'Scientific source changed: '+name)
    for key in ('snapshot','geometry','urdf','unit_clarification','runtime_unit_qualification','campaign','campaign_original_wait'):verify(r[key])
    records=json.loads((root/'rows.json').read_text())
    with np.load(root/'arrays.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    optimizer,diagnostic,gate,monitor=split(records,r['tasks'])
    require(arrays['role_features'].shape==(70000,3,4096) and arrays['role_features'].dtype==np.float16,'Exact actual current role features required')
    require(np.array_equal(arrays['head_valid'],arrays['valid'][:,49]),'Terminal-valid mask differs')
    for name,mask in [('optimizer_fit',optimizer),('diagnostic_fit',diagnostic),('gate_fit',gate)]:require(np.array_equal(arrays[name],mask),'Saved independent partition differs')
    require((r['source_FIT'],r['source_MONITOR'])==(63000,7000),'Full source counts differ')
    return r,records,arrays,r['tasks']

def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for k in ('snapshot','snapshot-sha256','geometry','urdf','campaign','campaign-sha256','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--campaign-original-wait','--corpus-admission',dest='campaign_original_wait',required=True)
    p.add_argument('--campaign-original-wait-sha256','--corpus-admission-sha256',dest='campaign_original_wait_sha256',required=True)
    a=p.parse_args();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Full CPU preparation900s cap')));signal.alarm(900)
    try:r=prepare(a);print(json.dumps(dict(status=r['status'],head_optimizer_FIT=r['head_optimizer_FIT'],head_MONITOR=r['head_MONITOR'])))
    finally:signal.alarm(0)

if __name__=='__main__':main()
