"""Current roles/command50 goal condition a bounded native50 arm trajectory."""
import torch
from torch import nn
from .common import require, MODEL

ARM = (0,1,2,3,4,5,7,8,9,10,11,12)
GRIP = (6,13)


class GoalArmSequence(nn.Module):
    def __init__(self, hidden=64, role_projection=32, max_correction_rad=.15):
        super().__init__()
        require((hidden,role_projection,max_correction_rad)==(64,32,.15), 'Fixed sequence64/role32/bound.15 required')
        self.config = dict(hidden=hidden,role_projection=role_projection,max_correction_rad=max_correction_rad)
        self.roles = nn.Linear(4096,32)
        self.initial = nn.Sequential(nn.Linear(96+14+18+18,64),nn.Tanh())
        self.sequence = nn.GRU(14+1,64,batch_first=True)
        self.output = nn.Linear(64,12)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
        self.reliability = nn.Sequential(nn.Linear(96+14+18+18+14+12+4,128),nn.GELU(),nn.Linear(128,2))
        nn.init.zeros_(self.reliability[-1].weight); nn.init.constant_(self.reliability[-1].bias,-4.)

    def forward(self,*,roles,qpos_features,current_eef_features,goal_delta,native_base_features,native_base,
                state_action_only=False,gated=True,threshold=.5):
        b = len(native_base)
        for x,shape in ((roles,(b,3,4096)),(qpos_features,(b,14)),(current_eef_features,(b,18)),
                        (goal_delta,(b,18)),(native_base_features,(b,50,14)),(native_base,(b,50,14))):
            require(x.shape == shape and x.dtype == torch.float32 and bool(torch.isfinite(x).all()), 'Finite current-only FP32 sequence features required')
        require(type(state_action_only) is bool and threshold == .5, 'Explicit paired control/fixed gate required')
        role = torch.tanh(self.roles(roles.detach())).flatten(1)
        if state_action_only:
            role = role*0; goal_delta = goal_delta*0
        context = torch.cat((role,qpos_features.detach(),current_eef_features.detach(),goal_delta.detach()),-1)
        time = torch.linspace(-1,1,50,device=native_base.device,dtype=native_base.dtype).view(1,50,1).expand(b,-1,-1)
        hidden,_ = self.sequence(torch.cat((native_base_features.detach(),time),-1),self.initial(context)[None])
        delta = self.config['max_correction_rad']*torch.tanh(self.output(hidden))
        raw = native_base.detach().clone(); raw[...,list(ARM)] = raw[...,list(ARM)] + delta
        summary = torch.cat((context,native_base_features.mean(1).detach(),delta.mean(1).detach(),
                             delta.square().mean((1,2))[:,None].detach(),delta.abs().amax((1,2))[:,None].detach(),
                             delta.mean((1,2))[:,None].detach(),delta.std((1,2),unbiased=False)[:,None].detach()),-1)
        logits = self.reliability(summary.detach()); accepted = logits.sigmoid() >= threshold
        arms = raw[...,list(ARM)].reshape(b,50,2,6)
        original = native_base[...,list(ARM)].reshape_as(arms)
        corrected = torch.where(accepted[:,None,:,None],arms,original) if gated else arms
        result = native_base.detach().clone(); result[...,list(ARM)] = corrected.reshape(b,50,12)
        require(torch.equal(result[...,list(GRIP)],native_base[...,list(GRIP)]), 'Original native grippers changed')
        return dict(native_actions=result,ungated_native_actions=raw,delta_arm_rad=delta,
                    reliability_logits=logits,accepted=accepted)
