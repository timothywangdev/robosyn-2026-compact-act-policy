"""Fixed executed10 FIT-label loss, with no expert labels at inference."""
import torch
from .contracts import require
from experiments.coupled_goal_act_all14.units import ARM,GRIP

WEIGHTS=dict(role_distillation=1.,goal_distillation=1.,arm=1.,opening=1.,FK_position=.25,FK_rotation=.10)


def objective(student,result,teacher,target,valid,fk):
    require(set(student)==set(teacher)=={'projected_roles','predicted_goal_delta'},'Only causal student/teacher fields')
    require(target.shape==result['ungated_physical_actions'].shape and target.shape[1:]==(50,14)
        and valid.shape==target.shape[:2] and valid.dtype==torch.bool,'Exact commanded50 FIT targets and bool mask')
    require(target.dtype==torch.float32 and bool(torch.isfinite(target).all()) and bool(valid[:,:10].any()),'Finite valid first10 expert commands')
    require(bool(((target[...,list(GRIP)][valid]>=0)&(target[...,list(GRIP)][valid]<=.050001)).all()),'Expert openings already metres, no second task conversion')
    candidate=result['ungated_physical_actions'][:,:10];label=target.detach()[:,:10];mask=valid[:,:10]
    arm_mask=mask[...,None].expand(-1,-1,12);grip_mask=mask[...,None].expand(-1,-1,2)
    def smooth(value):return torch.nn.functional.smooth_l1_loss(value,torch.zeros_like(value))
    values=dict(role_distillation=(student['projected_roles']-teacher['projected_roles'].detach()).square().mean(),
        goal_distillation=(student['predicted_goal_delta']-teacher['predicted_goal_delta'].detach()).square().mean(),
        arm=smooth(((candidate[...,list(ARM)]-label[...,list(ARM)])/.1)[arm_mask]),
        opening=smooth(((candidate[...,list(GRIP)]-label[...,list(GRIP)])/.05)[grip_mask]))
    poses=fk(candidate[...,list(ARM)])
    with torch.no_grad():label_poses=fk(label[...,list(ARM)])
    pose_mask=mask[...,None].expand(-1,-1,2)
    values['FK_position']=smooth(((poses[...,:3,3]-label_poses[...,:3,3])/.02)[pose_mask])
    values['FK_rotation']=((poses[...,:3,:3]-label_poses[...,:3,:3]).square().mean((-1,-2)))[pose_mask].mean()
    require(all(bool(torch.isfinite(value)) for value in values.values()),'Every fixed objective component finite')
    return sum(WEIGHTS[key]*value for key,value in values.items()),values
