"""Small new preparation references immutable arrays and qualifies actual all10 units."""
import argparse
import copy
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import numpy as np
import torch
import pyarrow.parquet as pq
from experiments.goal_latent_act_arm_sequence_full.prepare import load as old_load
from experiments.goal_latent_act_arm_sequence_full.fk import CommandFK
from .common import ROOT,PREP,require,ref,verify,write,sha,sources,validate_protocol
from .units import ARM,GRIP,LOWER6,UPPER6,scale_for_tasks,native_to_physical
from .model import CoupledGoalACT
from .data import tensors,batch
from .objective import loss


def prepare(a):
    start=time.monotonic();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    protocol=validate_protocol(json.loads(Path(a.protocol).read_text()))
    old,arrays,features,tasks=old_load(a.parent_prepared,a.parent_prepared_sha256)
    source=json.loads(verify(old['source_data']).read_text())
    verify(source['unit_clarification']);verify(source['runtime_unit_qualification'])
    root=ET.parse(verify(old['urdf'])).getroot();bounds=[]
    for joint in root.findall('joint'):
        if joint.attrib['type']=='revolute':
            limit=joint.find('limit');bounds.append((float(limit.attrib['lower']),float(limit.attrib['upper'])))
    require(tuple(v[0] for v in bounds)==LOWER6 and tuple(v[1] for v in bounds)==UPPER6,'Exact pinned six-joint URDF limits required')
    target=arrays['physical_actions'][...,list(GRIP)]
    require(np.isfinite(target).all() and float(target.min())>=0 and float(target.max())<=.050001,'Teacher grippers are already physical metres')
    # Old normalized native-grip moments are NOT reused. Decode first and then
    # compute new physical fourteen-channel moments on only50400 optimizer rows.
    opt=arrays['optimizer_fit'];base=arrays['physical_base'][opt].copy()
    scale=np.where(arrays['task_index'][opt]==tasks.index('drawer_open_place'),np.float32(.05),np.float32(1.))
    base[...,list(GRIP)]=np.minimum(np.maximum(base[...,list(GRIP)],0),(.05/scale)[:,None,None])*scale[:,None,None]
    mean=base.mean((0,1));floor=np.full(14,.01,np.float32);floor[list(GRIP)]=.001
    std=np.maximum(base.std((0,1)),floor);stats=copy.deepcopy(old['statistics'])
    stats['base_mean']=mean.tolist();stats['base_scale']=std.tolist()
    stats['scope']='50400 optimizer FIT current qpos/native arms and source-decoded physical opening moments only'
    stats['action_features_units']='arm radians; parent grip physical metres; current measured grip remains normalized opening'
    stats['base_scale_floor_units']=dict(arm_rad=.01,opening_m=.001)
    del base
    # Independently decode original expert parquet rows from one FIT episode/task.
    records=json.loads(verify(source['artifacts']['rows.json']).read_text());audits=[];selected=[]
    for j,task in enumerate(tasks):
        ids=np.flatnonzero(opt&(arrays['task_index']==j))[:3];selected.extend(ids.tolist())
        record=records[int(ids[0])];path=ROOT/'datasets/official-cache'/task/record['revision']/f"data/chunk-000/episode_{record['episode']:06d}.parquet"
        raw=np.asarray(pq.read_table(path,columns=['action'])['action'].to_pylist(),dtype=np.float64)
        physical=raw.copy();physical[:,list(GRIP)]=np.clip(raw[:,list(GRIP)],0,1)*.05 if task=='drawer_open_place' else np.clip(raw[:,list(GRIP)],0,.05)
        reference=ref(path)
        for i in ids:
            row=records[int(i)];require(row['episode']==record['episode'] and row['parquet_sha256']==reference['sha256'],'Exact all10 source parquet identity differs')
            n=min(50,len(raw)-row['frame']-1);expected=np.zeros((50,14),np.float32);expected[:n]=physical[row['frame']+1:row['frame']+1+n]
            require(np.array_equal(expected,arrays['physical_actions'][i]) and np.array_equal(np.arange(50)<n,arrays['valid'][i]),'Actual expert target conversion/horizon/validity mismatch')
        audits.append(dict(task=task,rows=ids.tolist(),source=reference,target_source_decode_once_verified=True))
    del records
    # Actual source current-only features use the same equations at train/runtime.
    selected=np.asarray(selected);small={k:arrays[k][selected].copy() for k in ('physical_base','physical_actions','valid','canonical_current_qpos14','role_features','teacher_eef_now','task_index')}
    feature={k:features[k][selected].copy() for k in ('predicted_goal',)}
    d={k:torch.from_numpy(v) for k,v in small.items()};d['predicted_goal']=torch.from_numpy(feature['predicted_goal'])
    st={k:torch.tensor(stats[k],dtype=torch.float32) for k in ('qpos_mean','qpos_scale','base_mean','base_scale','current_eef_mean','current_eef_scale','eef_scale')}
    x=batch(d,np.arange(len(selected)),st,tasks);m=CoupledGoalACT();result=m(**x,gated=False)
    fk=CommandFK(old['geometry'],old['urdf']);value=loss(result,d['physical_actions'],d['valid'],fk);value.backward()
    require(bool(torch.isfinite(value)) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in m.parameters()),'Actual all10 objective/gradients finite required')
    require(float(m.output.bias.grad[list(GRIP)].abs().sum())>0,'Actual teacher opening gradients required')
    with torch.no_grad():
        gated=m(**x);require(torch.equal(gated['native_actions'],x['native_base']),'Actual all10 zero-init native parent mismatch')
    pins=sources();pins.update(old['source_pins']);pins.update(source['source_pins'])
    report=dict(schema=PREP,status='CPU_all10_source_units_current_features_and_fixture_qualified',parent_ARM12_preparation=ref(Path(a.parent_prepared)/'receipt.json'),
        source_data=old['source_data'],source_features=old['source_features'],highlevel_contract=old['highlevel_contract'],highlevel_weights=old['highlevel_weights'],
        geometry=old['geometry'],urdf=old['urdf'],protocol=ref(a.protocol),source_pins=pins,statistics=stats,tasks=tasks,
        source_counts=protocol['source_counts'],partitions=protocol['partitions'],
        expert_grip_mapping='Drawer clamp01*.05; all other source commands clamp0..05; physical target cache already decoded',
        parent_ACT_grip_mapping='Drawer native normalized opening decoded*.05 once; other native grips clamp0..05 metres',
        measured_current_grip_units='normalized opening; never interpreted as metres',
        actual_source_parquet_checks=audits,actual_current_source_rows=len(selected),actual_objective=float(value.detach()),
        actual_grip_output_gradient_abs_sum=float(m.output.bias.grad[list(GRIP)].abs().sum()),
        new_physical_base_moments_optimizer_only=True,official_URDF_joint_limits_rad=bounds,
        source_cache=source['artifacts'],CUDA_initialized=torch.cuda.is_initialized(),GPU_queries=0,GPU_launches=0,
        elapsed_seconds=time.monotonic()-start)
    require(not report['CUDA_initialized'],'CPU-only actual preparation required');write(out/'receipt.json',report)
    print(json.dumps(dict(status=report['status'],rows=len(selected),elapsed_seconds=report['elapsed_seconds'])),flush=True)


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for k in ('parent-prepared','parent-prepared-sha256','protocol','output'):p.add_argument('--'+k,required=True)
    prepare(p.parse_args())


if __name__=='__main__':main()
