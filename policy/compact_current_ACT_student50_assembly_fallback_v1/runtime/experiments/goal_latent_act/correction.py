"""A learned physical-goal correction of frozen ACT; native grips unchanged."""
import torch
from torch import nn
from torch.nn import functional as F

VERSION = 'goal-latent-act-bounded-arm12-reliability-v1'
ARM = (0,1,2,3,4,5,7,8,9,10,11,12)
GRIP = (6,13)


def eef_features(pose):
    if pose.ndim != 4 or pose.shape[1:] != (2,4,4): raise ValueError('Dual rigid EEF poses required')
    return torch.cat((pose[...,:3,3],pose[...,:3,:2].transpose(-1,-2).flatten(-2)),-1).flatten(1)


class GoalArmCorrection(nn.Module):
    def __init__(self, hidden=128, max_correction_rad=.15):
        super().__init__()
        if hidden < 16 or not 0 < max_correction_rad <= .3: raise ValueError('Explicit bounded physical arm corrections')
        self.config=dict(hidden=hidden,max_correction_rad=max_correction_rad)
        self.residual=nn.Sequential(nn.Linear(14+14+18+1,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,12))
        nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)
        self.reliability=nn.Sequential(nn.Linear(14+18+14+12+4,hidden),nn.GELU(),nn.Linear(hidden,1))
        nn.init.zeros_(self.reliability[-1].weight);nn.init.constant_(self.reliability[-1].bias,-4.)

    def forward(self, *, physical_base, qpos, goal_pose, current_eef, qpos_mean, qpos_scale,
                eef_scale, base_mean, base_scale, goal_free=False, gated=True, threshold=.5):
        b=physical_base.shape[0]
        if physical_base.shape!=(b,50,14) or qpos.shape!=(b,14) or goal_pose.shape!=(b,2,4,4):
            raise ValueError('Frozen native50/current14/physical dualgoal dimensions')
        if not 0 < threshold < 1: raise ValueError('Predeclared reliability threshold')
        q=(qpos.detach()-qpos_mean)/qpos_scale
        goal=(eef_features(goal_pose.detach())-eef_features(current_eef.detach()))/eef_scale
        if goal_free:goal=goal*0
        base=physical_base.detach()
        features=(base-base_mean)/base_scale
        position=torch.linspace(-1,1,50,device=base.device,dtype=base.dtype).reshape(1,50,1).expand(b,-1,-1)
        x=torch.cat((features,q[:,None].expand(-1,50,-1),goal[:,None].expand(-1,50,-1),position),-1)
        delta=self.config['max_correction_rad']*torch.tanh(self.residual(x))
        raw=base.clone();raw[:,:,list(ARM)]=raw[:,:,list(ARM)]+delta
        summary=torch.cat((q,goal,features[:,:10].mean(1),delta[:,:10].mean(1),
                           delta[:,:10].square().mean((1,2))[:,None],delta[:,:10].abs().amax((1,2))[:,None],
                           delta[:,:10].mean((1,2))[:,None],delta[:,:10].std((1,2),unbiased=False)[:,None]),-1)
        logits=self.reliability(summary.detach()).squeeze(-1)
        probability=logits.sigmoid();accept=probability>=threshold
        output=torch.where(accept[:,None,None],raw,base) if gated else raw
        if not torch.equal(output[:,:,list(GRIP)],base[:,:,list(GRIP)]):raise ValueError('Original ACT grippers changed')
        return dict(physical_actions=output,ungated_physical_actions=raw,delta_arm_rad=delta,
                    reliability_logits=logits,reliability=probability,accepted=accept,goal_features=goal)


def arm_loss(prediction, target, valid, *, first=50):
    if prediction.shape!=target.shape or prediction.shape[1:]!=(50,14) or valid.shape!=prediction.shape[:2] or valid.dtype!=torch.bool:
        raise ValueError('Native50 targets plus detached training-only validity mask')
    selected=valid[:,:first,None].expand(-1,-1,12)
    error=(prediction[:,:first,list(ARM)]-target.detach()[:,:first,list(ARM)])/.1
    return F.smooth_l1_loss(error[selected],torch.zeros_like(error[selected]))


def per_record_error(prediction,target,valid,*,first=10):
    value=(prediction[:,:first,list(ARM)]-target.detach()[:,:first,list(ARM)]).abs()
    mask=valid[:,:first,None].expand_as(value)
    return (value*mask).sum((1,2))/mask.sum((1,2)).clamp_min(1)
