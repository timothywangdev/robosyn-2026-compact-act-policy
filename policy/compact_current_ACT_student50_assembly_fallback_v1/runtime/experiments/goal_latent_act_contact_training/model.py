"""Physical opening residual and FIT reliability; contact means opening-state proxy."""
import torch
from torch import nn

SCHEMA = 'goal-latent-act-current-role-physical-opening50-v1'
ARM = (0,1,2,3,4,5,7,8,9,10,11,12)
GRIP = (6,13)


def map_native_grips(base, task_index, tasks):
    if base.ndim != 3 or base.shape[1:] != (50,14) or task_index.shape != (len(base),):
        raise ValueError('Native50 plus exact task identity required')
    if not torch.isfinite(base).all() or (task_index < 0).any() or (task_index >= len(tasks)).any():
        raise ValueError('Finite exact task-native actions required')
    grip = base[...,list(GRIP)]
    drawer = task_index == tasks.index('drawer_open_place')
    return torch.where(drawer[:,None,None], grip.clamp(0,1)*.05, grip.clamp(0,.05))


def opening_classes(target):
    if target.shape[-1] != 2 or not torch.isfinite(target).all() or (target < 0).any() or (target > .050001).any():
        raise ValueError('Physical source-clamped grip labels required; native ratios rejected')
    return torch.where(target <= .005, 0, torch.where(target >= .045, 2, 1)).long()


class RolePhysicalOpening(nn.Module):
    def __init__(self, hidden=128, role_projection=32, max_correction_m=.05):
        super().__init__()
        if hidden != 128 or role_projection != 32 or max_correction_m != .05:
            raise ValueError('Fixed predeclared contact architecture required')
        self.config = dict(hidden=hidden, role_projection=role_projection, max_correction_m=max_correction_m)
        self.roles = nn.Linear(4096,role_projection)
        context = 3*role_projection+14+18
        self.trunk = nn.Sequential(nn.Linear(context+14+2+1,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU())
        self.residual = nn.Linear(hidden,2)
        self.contact = nn.Linear(hidden,6)
        nn.init.zeros_(self.residual.weight);nn.init.zeros_(self.residual.bias)
        self.reliability = nn.Sequential(nn.Linear(context+14+2*3,hidden),nn.GELU(),nn.Linear(hidden,2))
        nn.init.zeros_(self.reliability[-1].weight);nn.init.constant_(self.reliability[-1].bias,-4.)

    def forward(self, *, roles, qpos_features, goal_delta, native_base_features, native_base,
                parent_grips_m, state_action_only=False, gated=True, threshold=.5):
        b=len(native_base)
        for value,shape in [(roles,(b,3,4096)),(qpos_features,(b,14)),(goal_delta,(b,18)),
                            (native_base_features,(b,50,14)),(native_base,(b,50,14)),(parent_grips_m,(b,50,2))]:
            if value.shape != shape or not torch.isfinite(value).all():
                raise ValueError('Finite current-only typed contact features required')
        opening_classes(parent_grips_m)
        if threshold != .5:raise ValueError('Fixed FIT gate threshold required')
        role_context=torch.tanh(self.roles(roles.detach())).flatten(1)
        if state_action_only:role_context=role_context*0;goal_delta=goal_delta*0
        context=torch.cat((role_context,qpos_features.detach(),goal_delta.detach()),-1)
        time=torch.linspace(-1,1,50,dtype=native_base.dtype,device=native_base.device).reshape(1,50,1).expand(b,-1,-1)
        inputs=torch.cat((context[:,None].expand(-1,50,-1),native_base_features.detach(),parent_grips_m.detach()/.05,time),-1)
        hidden=self.trunk(inputs)
        delta=self.config['max_correction_m']*torch.tanh(self.residual(hidden))
        raw=(parent_grips_m.detach()+delta).clamp(0,.05)
        summary=torch.cat((context,native_base_features.detach().mean(1),delta.detach().mean(1)/.05,
                           delta.detach().amin(1)/.05,delta.detach().amax(1)/.05),-1)
        logits=self.reliability(summary.detach());accepted=logits.sigmoid()>=threshold
        output=torch.where(accepted[:,None,:],raw,parent_grips_m) if gated else raw
        physical=native_base.detach().clone();physical[...,list(GRIP)]=output
        if not torch.equal(physical[...,list(ARM)],native_base[...,list(ARM)]):
            raise ValueError('Frozen native ACT arms changed')
        return dict(physical_actions=physical,grips_m=output,ungated_grips_m=raw,
                    contact_logits=self.contact(hidden).reshape(b,50,2,3),reliability_logits=logits,
                    accepted=accepted,delta_m=delta)


def per_hand_mae(pred,target,valid,first=50):
    opening_classes(target)
    if pred.shape != target.shape or valid.shape != pred.shape[:2] or valid.dtype != torch.bool:
        raise ValueError('Physical opening50 and training-only bool validity required')
    mask=valid[:,:first,None]
    return ((pred[:,:first]-target[:,:first]).abs()*mask).sum(1)/mask.sum(1).clamp_min(1)


def contact_loss(pred,target,valid):
    classes=opening_classes(target);mask=valid[...,None].expand_as(target)
    mse=((pred['ungated_grips_m'][mask]-target.detach()[mask])/.05).square().mean()
    auxiliary=torch.nn.functional.cross_entropy(pred['contact_logits'][mask],classes[mask])
    return mse+.05*auxiliary
