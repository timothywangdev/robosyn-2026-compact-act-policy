"""Reuse immutable current-only caches; normalize decoded openings on optimizer FIT."""
import json
import numpy as np
import torch
from experiments.goal_latent_act_arm_sequence_full.prepare import load as old_load
from experiments.goal_latent_act_full_data.correction_prefix import eef_features
from .common import PREP,require,verify,verify_sources,validate_protocol
from .units import GRIP,scale_for_tasks,native_to_physical


def load(prepared,expected):
    import pathlib
    path=pathlib.Path(prepared)/'receipt.json';from .common import sha
    require(sha(path)==expected,'Exact all14 CPU preparation required')
    r=json.loads(path.read_text());require(r['schema']==PREP and r['status']=='CPU_all10_source_units_current_features_and_fixture_qualified','Distinct all14 preparation required')
    verify_sources(r['source_pins']);validate_protocol(json.loads(verify(r['protocol']).read_text()))
    old=r['parent_ARM12_preparation'];old_r,arrays,features,tasks=old_load(str(verify(old).parent),old['sha256'])
    require(tasks==r['tasks'] and old_r['source_data']==r['source_data'] and old_r['source_features']==r['source_features'],'Same exact cache/head pairing required')
    for key,count in (('optimizer_fit',50400),('diagnostic_fit',6300),('gate_fit',6300),('fit',63000)):
        require(int(arrays[key].sum())==count,'Exact independent partition changed: '+key)
    require(not bool((arrays['optimizer_fit']&arrays['gate_fit']).any()) and not bool((arrays['optimizer_fit']&arrays['diagnostic_fit']).any()),'Optimizer/gate/diagnostic overlap')
    return r,arrays,features,tasks


def tensors(arrays,features,device):
    fields=('physical_base','physical_actions','valid','canonical_current_qpos14','role_features','teacher_eef_now','task_index')
    result={k:torch.from_numpy(arrays[k]).to(device) for k in fields}
    for key in ('predicted_goal','shuffled_predicted_goal','shuffled_role_ids','monitor_ids'):
        result[key]=torch.from_numpy(features[key]).to(device)
    return result


def batch(data,ids,stats,tasks,shuffled=False):
    ids=torch.as_tensor(ids,dtype=torch.long,device=data['physical_base'].device)
    q=data['canonical_current_qpos14'][ids];native=data['physical_base'][ids]
    now=eef_features(data['teacher_eef_now'][ids]);role_ids=ids;goal=data['predicted_goal'][ids]
    if shuffled:
        lookup=torch.searchsorted(data['monitor_ids'],ids)
        require(torch.equal(data['monitor_ids'][lookup],ids),'Role shuffle is MONITOR report-only')
        role_ids=data['shuffled_role_ids'][lookup];goal=data['shuffled_predicted_goal'][lookup]
    scale=scale_for_tasks(data['task_index'][ids],tasks)
    physical=native_to_physical(native,scale)
    return dict(roles=data['role_features'][role_ids].float(),qpos_features=(q-stats['qpos_mean'])/stats['qpos_scale'],
                current_eef_features=(now-stats['current_eef_mean'])/stats['current_eef_scale'],
                goal_delta=(eef_features(goal)-now)/stats['eef_scale'],
                native_base_features=(physical-stats['base_mean'])/stats['base_scale'],native_base=native,grip_scale=scale)
