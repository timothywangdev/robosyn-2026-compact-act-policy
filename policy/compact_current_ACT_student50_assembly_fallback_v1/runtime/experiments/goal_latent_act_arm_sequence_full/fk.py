"""Differentiable exact translation of the independently pinned URDF chain."""
import importlib.util
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .common import require, verify


def geometry(reference):
    path = verify(reference)
    spec = importlib.util.spec_from_file_location('arm_sequence_qualified_numpy_geometry', path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


class CommandFK(nn.Module):
    def __init__(self, geometry_ref, urdf_ref, dtype=torch.float32):
        super().__init__()
        g = geometry(geometry_ref); chain = g.read_chain(verify(urdf_ref))
        require(sum(axis is not None for _,axis in chain) == 6, 'Exact six revolute joint chain required')
        self.register_buffer('origins', torch.tensor(np.stack([origin for origin,_ in chain]), dtype=dtype))
        self.register_buffer('mounts', torch.tensor(np.stack([g.mount(side) for side in ('left','right')]), dtype=dtype))
        self.register_buffer('tcp', torch.tensor(g.tcp(), dtype=dtype))
        self.moving = tuple(axis is not None for _,axis in chain)
        skew = []
        for _,axis in chain:
            a = np.zeros(3) if axis is None else np.asarray(axis)/np.linalg.norm(axis)
            x,y,z = a
            skew.append(np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]]))
        self.register_buffer('skew', torch.tensor(np.stack(skew), dtype=dtype))

    def forward(self, arms):
        require(arms.shape[-1] == 12 and arms.dtype == self.origins.dtype, 'Physical arm12 matching FK dtype required')
        q = arms.reshape(*arms.shape[:-1], 2, 6)
        T = self.mounts.expand(*q.shape[:-2], 2,4,4)
        j = 0
        for i,moving in enumerate(self.moving):
            T = T @ self.origins[i]
            if moving:
                angle = q[...,j,None,None]; K = self.skew[i]
                R = torch.eye(3,dtype=q.dtype,device=q.device) + angle.sin()*K + (1-angle.cos())*(K@K)
                top = torch.cat((R, torch.zeros(*R.shape[:-1],1,dtype=q.dtype,device=q.device)), -1)
                bottom = torch.zeros(*R.shape[:-2],1,4,dtype=q.dtype,device=q.device)
                bottom[...,0,3] = 1
                T = T @ torch.cat((top,bottom),-2)
                j += 1
        return T @ self.tcp


def pose_errors(prediction, target):
    position = torch.linalg.vector_norm(prediction[...,:3,3]-target[...,:3,3],dim=-1)
    relative = prediction[...,:3,:3].transpose(-1,-2) @ target[...,:3,:3]
    v = torch.stack((relative[...,2,1]-relative[...,1,2],relative[...,0,2]-relative[...,2,0],relative[...,1,0]-relative[...,0,1]),-1)
    sine = torch.linalg.vector_norm(v,dim=-1)/2
    cosine = (relative.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2
    return position, torch.atan2(sine,cosine)


def hand_errors(prediction, target, valid, fk, indices=(0,9,31,49)):
    arms = (0,1,2,3,4,5,7,8,9,10,11,12)
    p = prediction[...,list(arms)].reshape(len(prediction),50,2,6)
    t = target[...,list(arms)].reshape_as(p)
    mask = valid[...,None]
    joint = ((p-t).abs().mean(-1)*mask).sum(1)/mask.sum(1).clamp_min(1)
    poses = fk(prediction[:,indices][...,list(arms)])
    with torch.no_grad(): labels = fk(target[:,indices][...,list(arms)])
    position,rotation = pose_errors(poses,labels)
    m = valid[:,indices,None]
    denom = m.sum(1).clamp_min(1)
    return dict(joint=joint,position=(position*m).sum(1)/denom,rotation=(rotation*m).sum(1)/denom,
                poses=poses,labels=labels,FK_valid=m)


def physical_gate_labels(candidate, parent, target, valid, fk):
    c = hand_errors(candidate,target,valid,fk); p = hand_errors(parent,target,valid,fk)
    return ((c['joint']+.001<p['joint']) & (c['position']+.0001<p['position']) &
            (c['rotation']<=p['rotation']+.001)).float().detach()
