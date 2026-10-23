"""Reuse exact completed full-data arrays and current-only head predictions."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from experiments.goal_latent_act_command_goal_training_full import data
from experiments.goal_latent_act_command_goal_training_full.contracts import validate_head
from experiments.goal_latent_act_full_data.correction_prefix import eef_features
from .common import PREP, ref, verify, write, sha, require, sources, validate_protocol
from .fk import CommandFK, geometry
from .model import ARM


def qualify_original(training, outer):
    r = json.loads(verify(training).read_text()); w = json.loads(verify(outer).read_text())
    require(r['schema']=='goal-latent-act-commanded50-full63000-original-CPU-pilot-v1' and r['status']=='all_original_exit0_natural_CPU_pilot_complete', 'Actual original full-source CPU completion required')
    require(w['schema']=='goal-latent-act-full63000-controller-original-wait-v1' and w['owner']==r['controller']
            and w['scientific_receipt']==training and w['original_controller_exit_code']==0 and w['original_controller_reaped'] is True
            and w['original_process_absent'] is True and w['independently_observed_group_after_reap']==[]
            and w['natural_completion_no_signals_sent'] is True and w['signals_sent']==[], 'Genuine original outer0/reap/natural drain required')
    for row in r['runs']:
        require(row['original_exit_code']==0 and row['original_child_reaped'] is True and row['original_process_absent'] is True
                and row['signals_sent']==[] and row['natural_completion_no_signals_sent'] is True
                and row['independently_observed_group_after_reap']==[], 'Original full CPU child failed or lacks drain')
        verify(row['child_receipt'])
    return r


def load(prepared, expected):
    root = Path(prepared); require(sha(root/'receipt.json')==expected, 'Exact new temporal-arm preparation required')
    r = json.loads((root/'receipt.json').read_text()); require(r['schema']==PREP and r['status']=='CPU_full_source_current_only_FK_parity_verified', 'Qualified new arm preparation required')
    from .common import verify_sources
    verify_sources(r['source_pins']); validate_protocol(json.loads(verify(r['protocol']).read_text()))
    qualify_original(r['original_training'],r['original_outer_wait'])
    original,records,arrays,tasks=data.load(Path(r['source_data']['path']).parent,r['source_data']['sha256'])
    verify(r['source_features'])
    with np.load(r['source_features']['path'],allow_pickle=False) as z:
        features={k:z[k] for k in ('predicted_goal','shuffled_role_ids','shuffled_predicted_goal','monitor_ids')}
    require(features['predicted_goal'].shape==(70000,2,4,4) and np.isfinite(features['predicted_goal']).all(), 'Exact frozen current-only head outputs required')
    return r,arrays,features,tasks


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for name in ('original-training','original-outer-wait','data','head','sequence-prepared','protocol','output'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args(); started=time.monotonic();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    protocol=validate_protocol(json.loads(Path(a.protocol).read_text()))
    original=qualify_original(ref(a.original_training),ref(a.original_outer_wait))
    r,records,arrays,tasks=data.load(a.data,sha(Path(a.data)/'receipt.json'))
    head=json.loads((Path(a.head)/'checkpoint.json').read_text());validate_head(head)
    require(ref(Path(a.head)/'receipt.json')==original['runs'][1]['child_receipt'], 'Exact previously completed head required')
    require(sha(Path(a.head)/'highlevel.safetensors')==head['weights_sha256'] and head['data_receipt']==ref(Path(a.data)/'receipt.json'), 'Same frozen full-data head identity required')
    oldseq=json.loads((Path(a.sequence_prepared)/'receipt.json').read_text())
    require(ref(Path(a.sequence_prepared)/'receipt.json')==original['runs'][2]['child_receipt']
            and oldseq['status']=='CPU_source_labels_and_current_only_head_verified'
            and oldseq['source_data']==head['data_receipt']
            and oldseq['highlevel_contract']==ref(Path(a.head)/'checkpoint.json')
            and oldseq['highlevel_weights']==ref(Path(a.head)/'highlevel.safetensors'), 'Exact actual current-only head cache pairing required')
    verify(oldseq['artifacts']['features.npz'])
    with np.load(oldseq['artifacts']['features.npz']['path'],allow_pickle=False) as z:
        require(np.array_equal(z['residual_fit'],arrays['optimizer_fit']) and np.array_equal(z['gate_fit'],arrays['gate_fit'])
                and np.array_equal(z['diagnostic_fit'],arrays['diagnostic_fit']) and np.array_equal(z['monitor_ids'],np.flatnonzero(~arrays['fit'])), 'Old current-only predictions and new source partition differ')
    # Existing moments were computed on these literal50400 optimization rows.
    opt=arrays['optimizer_fit'];stats=oldseq['statistics']
    checks={'qpos_mean':arrays['canonical_current_qpos14'][opt].mean(0),
            'qpos_scale':np.maximum(arrays['canonical_current_qpos14'][opt].std(0),.01),
            'base_mean':arrays['physical_base'][opt].mean((0,1)),
            'base_scale':np.maximum(arrays['physical_base'][opt].std((0,1)),.01),
            'current_eef_mean':eef_features(torch.from_numpy(arrays['teacher_eef_now'][opt])).numpy().mean(0),
            'current_eef_scale':np.maximum(eef_features(torch.from_numpy(arrays['teacher_eef_now'][opt])).numpy().std(0),.01)}
    for key,value in checks.items():require(np.array_equal(value,np.asarray(stats[key],dtype=np.float32)), 'Optimization-only moments mismatch: '+key)
    g=geometry(r['geometry']);chain=g.read_chain(verify(r['urdf']));fk=CommandFK(r['geometry'],r['urdf'])
    # Actual source current and valid command states from allten tasks.
    selected=np.concatenate([np.flatnonzero(opt&(arrays['task_index']==j))[:2] for j in range(10)])
    q=np.concatenate((arrays['canonical_current_qpos14'][selected][:,None,list(ARM)],arrays['physical_actions'][selected][:,protocol['FK_indices']][:,:,list(ARM)]),1)
    with torch.no_grad(): actual=fk(torch.from_numpy(q)).numpy()
    expected=np.empty_like(actual)
    for i in range(len(q)):
        for k in range(q.shape[1]):
            for j,side in enumerate(('left','right')):expected[i,k,j]=g.fk_link(chain,q[i,k,j*6:(j+1)*6],side)@g.tcp()
    maximum=float(np.abs(actual-expected).max());require(maximum<2e-6,'Torch FK does not match exact qualified NumPy chain')
    gradient=torch.from_numpy(q[:2].copy()).requires_grad_(True);fk(gradient)[...,:3,3].square().sum().backward()
    require(torch.isfinite(gradient.grad).all() and bool((gradient.grad!=0).any()), 'Physical FK gradients unavailable')
    receipt=dict(schema=PREP,status='CPU_full_source_current_only_FK_parity_verified',original_training=ref(a.original_training),
        original_outer_wait=ref(a.original_outer_wait),source_data=ref(Path(a.data)/'receipt.json'),source_features=oldseq['artifacts']['features.npz'],
        highlevel_contract=ref(Path(a.head)/'checkpoint.json'),highlevel_weights=ref(Path(a.head)/'highlevel.safetensors'),
        geometry=r['geometry'],urdf=r['urdf'],protocol=ref(a.protocol),source_pins=sources(),statistics=stats,tasks=tasks,
        counts=protocol['partitions'],source_counts=protocol['source_counts'],
        FK_qualification=dict(all_tasks=10,rows=20,poses=int(np.prod(actual.shape[:-2])),maximum_matrix_absolute_difference=maximum,
                              threshold=2e-6,finite_nonzero_gradients=True,coordinate_frame='robot_root_meters',target_semantics='commanded FK; not measured future EEF'),
        normalization_recomputed_bitexact_from_optimizer_only=True,CUDA_initialized=torch.cuda.is_initialized(),GPU_queries=0,GPU_launches=0,
        elapsed_seconds=time.monotonic()-started)
    require(not receipt['CUDA_initialized'],'CPU-only preparation required');write(out/'receipt.json',receipt)
    print(json.dumps(dict(status=receipt['status'],FK_max=maximum,elapsed_seconds=receipt['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
