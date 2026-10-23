"""Additive training labels from the SAME declared FIT cache; no new frames."""
import numpy as np
import torch
from .contracts import require,verify,partition
from experiments.coupled_goal_act_all14.units import native_to_physical
from experiments.goal_latent_act_full_data.correction_prefix import eef_features


def fields(rows,statistics):
    require(rows and all(r['split']=='fit' and r['partition']==partition(r['episode']) for r in rows),'Declared episode-disjoint FIT rows only')
    samples=[]
    for row in rows:
        with np.load(verify(row['file']),allow_pickle=False) as archive:
            samples.append({key:archive[key].copy() for key in ('physical_base','physical_actions','valid','teacher_eef_now')})
    native=torch.from_numpy(np.stack([s['physical_base'] for s in samples])).float()
    # The original field name physical_base stores the ACT's native codec.
    # Expert physical_actions was converted to metres upstream and stays so.
    scale=torch.tensor([.05 if r['task']=='drawer_open_place' else 1. for r in rows],dtype=torch.float32)
    physical=native_to_physical(native,scale)
    target=torch.from_numpy(np.stack([s['physical_actions'] for s in samples])).float()
    valid=torch.from_numpy(np.stack([s['valid'] for s in samples])).bool()
    now=eef_features(torch.from_numpy(np.stack([s['teacher_eef_now'] for s in samples])).float())
    return dict(native_base=native,grip_scale=scale,
        native_base_features=(physical-torch.tensor(statistics['base_mean']))/torch.tensor(statistics['base_scale']),
        current_eef_features=(now-torch.tensor(statistics['current_eef_mean']))/torch.tensor(statistics['current_eef_scale']),
        expert_target=target,expert_valid=valid)
