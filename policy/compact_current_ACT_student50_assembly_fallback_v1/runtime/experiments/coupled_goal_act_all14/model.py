"""One shared GRU produces synchronized arm-radian and opening-metre residuals."""
import torch
from torch import nn
from .common import require, CONFIG
from .units import ARM, GRIP, LOWER6, UPPER6, native_to_physical, physical_to_native, hand_atomic


class CoupledGoalACT(nn.Module):
    def __init__(self, hidden=64, role_projection=32, max_correction_rad=.15, max_opening_correction_m=.05):
        super().__init__()
        self.config = dict(hidden=hidden,role_projection=role_projection,max_correction_rad=max_correction_rad,
                           max_opening_correction_m=max_opening_correction_m)
        require(self.config==CONFIG, 'Fixed new shared GRU64 all14 architecture required')
        self.roles = nn.Linear(4096,32)
        self.initial = nn.Sequential(nn.Linear(96+14+18+18,64),nn.Tanh())
        self.sequence = nn.GRU(14+2+1,64,batch_first=True)
        self.output = nn.Linear(64,14)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        self.reliability = nn.Sequential(nn.Linear(96+14+18+18+14+14+4,128),nn.GELU(),nn.Linear(128,2))
        nn.init.zeros_(self.reliability[-1].weight);nn.init.constant_(self.reliability[-1].bias,-4.)
        lo=torch.tensor(LOWER6,dtype=torch.float32);hi=torch.tensor(UPPER6,dtype=torch.float32)
        # Inward representable limits make accepted FP32 commands obey the exact
        # decimal URDF limits. Invalid proposals fall back, never silently clip.
        import numpy as np
        for i,(lower,upper) in enumerate(zip(LOWER6,UPPER6)):
            if float(lo[i])<lower:lo[i]=torch.nextafter(lo[i],torch.tensor(float('inf')))
            if float(hi[i])>upper:hi[i]=torch.nextafter(hi[i],torch.tensor(float('-inf')))
        self.register_buffer('lower6',lo);self.register_buffer('upper6',hi)

    def forward(self,*,roles,qpos_features,current_eef_features,goal_delta,
                native_base_features,native_base,grip_scale,state_action_only=False,gated=True,threshold=.5):
        b=len(native_base)
        for x,shape in ((roles,(b,3,4096)),(qpos_features,(b,14)),(current_eef_features,(b,18)),
                        (goal_delta,(b,18)),(native_base_features,(b,50,14)),(native_base,(b,50,14))):
            require(x.shape==shape and x.dtype==torch.float32 and bool(torch.isfinite(x).all()), 'Finite current-only FP32 features required')
        require(type(state_action_only) is bool and type(gated) is bool and threshold==.5,'Fixed paired control and gate required')
        parent=native_to_physical(native_base,grip_scale)
        role=torch.tanh(self.roles(roles.detach())).flatten(1)
        if state_action_only:role=role*0;goal_delta=goal_delta*0
        context=torch.cat((role,qpos_features.detach(),current_eef_features.detach(),goal_delta.detach()),-1)
        time=torch.linspace(-1,1,50,device=native_base.device,dtype=native_base.dtype).view(1,50,1).expand(b,-1,-1)
        hidden,_=self.sequence(torch.cat((native_base_features.detach(),parent[...,list(GRIP)].detach()/.05,time),-1),self.initial(context)[None])
        dimensionless=torch.tanh(self.output(hidden))
        delta_arm=.15*dimensionless[...,list(ARM)]
        delta_grip=.05*dimensionless[...,list(GRIP)]
        proposal=parent.detach().clone()
        proposal[...,list(ARM)]=proposal[...,list(ARM)]+delta_arm
        proposal[...,list(GRIP)]=(proposal[...,list(GRIP)]+delta_grip).clamp(0,.05)
        native_proposal=physical_to_native(proposal,grip_scale)
        summary=torch.cat((context,native_base_features.mean(1).detach(),dimensionless.mean(1).detach(),
                           dimensionless.square().mean((1,2))[:,None].detach(),dimensionless.abs().amax((1,2))[:,None].detach(),
                           dimensionless.mean((1,2))[:,None].detach(),dimensionless.std((1,2),unbiased=False)[:,None].detach()),-1)
        logits=self.reliability(summary.detach())
        arms=proposal[...,list(ARM)].reshape(b,50,2,6)
        within=((arms>=self.lower6)&(arms<=self.upper6)).all(-1).all(1)
        accepted=(logits.sigmoid()>=threshold)&within
        native=hand_atomic(native_proposal,native_base,accepted) if gated else native_proposal
        physical=hand_atomic(proposal,parent,accepted) if gated else proposal
        return dict(native_actions=native,physical_actions=physical,ungated_physical_actions=proposal,
                    ungated_native_actions=native_proposal,parent_physical_actions=parent,delta_arm_rad=delta_arm,
                    delta_opening_m=delta_grip,reliability_logits=logits,accepted=accepted,within_URDF_limits=within)
