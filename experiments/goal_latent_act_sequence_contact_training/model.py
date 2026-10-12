"""Current-conditioned action-sequence hazards; predicted recurrence only."""
import torch
from torch import nn
from experiments.goal_latent_act_contact_training.model import ARM,GRIP,opening_classes

SCHEMA='goal-latent-act-current-role-sequence-opening50-v1'


class SequencePhysicalOpening(nn.Module):
    def __init__(self,hidden=64,role_projection=32):
        super().__init__()
        if hidden!=64 or role_projection!=32:raise ValueError('Predeclared GRU64/role32 required')
        self.config=dict(hidden=hidden,role_projection=role_projection)
        self.roles=nn.Linear(4096,32);context=96+14+18+18
        self.initial=nn.Sequential(nn.Linear(context,64),nn.Tanh())
        self.sequence=nn.GRU(14+2+1,64,batch_first=True)
        self.hazards=nn.Linear(64,4);nn.init.zeros_(self.hazards.weight);nn.init.constant_(self.hazards.bias,-4.)
        self.reliability=nn.Sequential(nn.Linear(context+14+6+4,128),nn.GELU(),nn.Linear(128,2))
        nn.init.zeros_(self.reliability[-1].weight);nn.init.constant_(self.reliability[-1].bias,-4.)

    def forward(self,*,roles,qpos_features,current_eef_features,goal_delta,native_base_features,native_base,
                parent_grips_m,measured_opening,state_action_only=False,gated=True,threshold=.5):
        b=len(native_base)
        for x,shape in [(roles,(b,3,4096)),(qpos_features,(b,14)),(current_eef_features,(b,18)),(goal_delta,(b,18)),
                        (native_base_features,(b,50,14)),(native_base,(b,50,14)),(parent_grips_m,(b,50,2)),(measured_opening,(b,2))]:
            if x.shape!=shape or not torch.isfinite(x).all():raise ValueError('Typed finite current-only sequence features required')
        opening_classes(parent_grips_m)
        if (measured_opening<0).any() or (measured_opening>1).any() or threshold!=.5:raise ValueError('Measured normalized opening and fixed gate required')
        role=torch.tanh(self.roles(roles.detach())).flatten(1)
        if state_action_only:role=role*0;goal_delta=goal_delta*0
        context=torch.cat((role,qpos_features.detach(),current_eef_features.detach(),goal_delta.detach()),-1)
        time=torch.linspace(-1,1,50,dtype=native_base.dtype,device=native_base.device).view(1,50,1).expand(b,-1,-1)
        x=torch.cat((native_base_features.detach(),parent_grips_m.detach()/.05,time),-1)
        hidden,_=self.sequence(x,self.initial(context)[None])
        logits=self.hazards(hidden).reshape(b,50,2,2);hazard=logits.sigmoid();u=measured_opening.detach();values=[]
        for t in range(50):
            u=u*(1-hazard[:,t,:,0])+(1-u)*hazard[:,t,:,1];values.append(u)
        raw=(.05*torch.stack(values,1)).clamp(0,.05)
        summary=torch.cat((context,native_base_features.mean(1).detach(),raw.mean(1).detach()/.05,
                           raw.amin(1).detach()/.05,raw.amax(1).detach()/.05,hazard.mean(1).flatten(1).detach()),-1)
        gate=self.reliability(summary.detach());accepted=gate.sigmoid()>=threshold
        output=torch.where(accepted[:,None],raw,parent_grips_m) if gated else raw
        physical=native_base.detach().clone();physical[...,list(GRIP)]=output
        if not torch.equal(physical[...,list(ARM)],native_base[...,list(ARM)]):raise ValueError('Native ACT arms changed')
        return dict(physical_actions=physical,grips_m=output,ungated_grips_m=raw,hazard_logits=logits,hazards=hazard,
                    reliability_logits=gate,accepted=accepted)


def hazard_targets(target,measured_opening,valid):
    opening_classes(target)
    if valid.shape!=target.shape[:2] or valid.dtype!=torch.bool:raise ValueError('Literal valid50 mask required')
    closed=target<=.005;previous=torch.cat(((measured_opening<=.1)[:,None],closed[:,:-1]),1)
    return torch.stack((~previous&closed,previous&~closed),-1).float()


def loss(pred,target,measured_opening,valid,positive_weights):
    event=hazard_targets(target,measured_opening,valid);mask=valid[...,None].expand_as(target)
    mse=((pred['ungated_grips_m'][mask]-target.detach()[mask])/.05).square().mean()
    selected=valid[...,None,None].expand_as(event)
    raw=torch.nn.functional.binary_cross_entropy_with_logits(pred['hazard_logits'],event,reduction='none',pos_weight=positive_weights)
    return mse+.2*raw[selected].mean()
