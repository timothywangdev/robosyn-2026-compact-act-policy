"""Physical teacher labels and terminal masks remain exclusively training-side."""
import torch
from experiments.goal_latent_act_arm_sequence_full.fk import hand_errors
from .common import require
from .units import ARM,GRIP


def opening_error(actions,target,valid):
    mask=valid[...,None]
    return ((actions[...,list(GRIP)]-target[...,list(GRIP)]).abs()*mask).sum(1)/mask.sum(1).clamp_min(1)


def loss(result,target,valid,fk):
    require(target.shape==result['ungated_physical_actions'].shape and valid.shape==target.shape[:2], 'Literal command50 targets/mask required')
    candidate=result['ungated_physical_actions'];mask=valid[...,None].expand(-1,-1,12)
    value=torch.nn.functional.smooth_l1_loss((candidate[...,list(ARM)]-target[...,list(ARM)])[mask]/.1,torch.zeros_like(candidate[...,list(ARM)][mask]))
    e=hand_errors(candidate,target,valid,fk);m=e['FK_valid'].expand(e['poses'].shape[:-2])
    delta=(e['poses'][...,:3,3]-e['labels'][...,:3,3])/.02
    pos=torch.nn.functional.smooth_l1_loss(delta[m],torch.zeros_like(delta[m]))
    rot=(e['poses'][...,:3,:3]-e['labels'][...,:3,:3]).square().mean((-1,-2))
    gm=valid[...,None].expand(-1,-1,2)
    opening=torch.nn.functional.smooth_l1_loss((candidate[...,list(GRIP)]-target[...,list(GRIP)])[gm]/.05,torch.zeros_like(target[...,list(GRIP)][gm]))
    pair=(valid[:,1:]&valid[:,:-1])[...,None].expand(-1,-1,2)
    diff=(candidate[:,1:,list(GRIP)]-candidate[:,:-1,list(GRIP)])-(target[:,1:,list(GRIP)]-target[:,:-1,list(GRIP)])
    increment=torch.nn.functional.smooth_l1_loss(diff[pair]/.01,torch.zeros_like(diff[pair])) if bool(pair.any()) else diff.sum()*0
    return value+.25*pos+.10*rot[m].mean()+opening+.10*increment


def gate_labels(result,target,valid,fk):
    candidate=result['ungated_physical_actions'];parent=result['parent_physical_actions']
    c=hand_errors(candidate,target,valid,fk);p=hand_errors(parent,target,valid,fk)
    # Original ARM12 criteria unchanged. Each gate now controls ALL7 channels,
    # and adds physical-opening no-regression; no impossible strict improvement
    # is demanded of an already perfect/inactive gripper.
    original=((c['joint']+.001<p['joint'])&(c['position']+.0001<p['position'])&(c['rotation']<=p['rotation']+.001))
    opening=opening_error(candidate,target,valid)<=opening_error(parent,target,valid)+.0001
    return (original&opening&result['within_URDF_limits']).float().detach()
