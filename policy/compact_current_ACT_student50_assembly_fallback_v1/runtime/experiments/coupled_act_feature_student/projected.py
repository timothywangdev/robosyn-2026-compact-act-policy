"""New projected-input ABI; fixed coupled GRU/reliability math retained."""
import torch
from .contracts import require
from experiments.coupled_goal_act_all14.units import ARM,GRIP,native_to_physical,physical_to_native,hand_atomic

def projected_forward(actor, *, projected_roles, qpos_features, current_eef_features, goal_delta,
                      native_base_features, native_base, grip_scale, state_action_only=False, gated=True, threshold=.5):
    b=len(native_base)
    for x,shape in ((projected_roles,(b,96)),(qpos_features,(b,14)),(current_eef_features,(b,18)),
                    (goal_delta,(b,18)),(native_base_features,(b,50,14)),(native_base,(b,50,14))):
        require(x.shape==shape and x.dtype==torch.float32 and bool(torch.isfinite(x).all()), 'Finite current-only projected FP32 features required')
    require(type(state_action_only) is bool and type(gated) is bool and threshold==.5,'Fixed paired control and gate required')
    parent = native_to_physical(native_base, grip_scale)
    role = projected_roles.detach()
    if state_action_only:
        role = role * 0
        goal_delta = goal_delta * 0
    context = torch.cat((role, qpos_features.detach(), current_eef_features.detach(), goal_delta.detach()), -1)
    time = torch.linspace(-1, 1, 50, device=native_base.device, dtype=native_base.dtype).view(1, 50, 1).expand(b, -1, -1)
    hidden, _ = actor.sequence(torch.cat((native_base_features.detach(), parent[..., list(GRIP)].detach() / 0.05, time), -1), actor.initial(context)[None])
    dimensionless = torch.tanh(actor.output(hidden))
    delta_arm = 0.15 * dimensionless[..., list(ARM)]
    delta_grip = 0.05 * dimensionless[..., list(GRIP)]
    proposal = parent.detach().clone()
    proposal[..., list(ARM)] = proposal[..., list(ARM)] + delta_arm
    proposal[..., list(GRIP)] = (proposal[..., list(GRIP)] + delta_grip).clamp(0, 0.05)
    native_proposal = physical_to_native(proposal, grip_scale)
    summary = torch.cat((context, native_base_features.mean(1).detach(), dimensionless.mean(1).detach(), dimensionless.square().mean((1, 2))[:, None].detach(), dimensionless.abs().amax((1, 2))[:, None].detach(), dimensionless.mean((1, 2))[:, None].detach(), dimensionless.std((1, 2), unbiased=False)[:, None].detach()), -1)
    logits = actor.reliability(summary.detach())
    arms = proposal[..., list(ARM)].reshape(b, 50, 2, 6)
    within = ((arms >= actor.lower6) & (arms <= actor.upper6)).all(-1).all(1)
    accepted = (logits.sigmoid() >= threshold) & within
    native = hand_atomic(native_proposal, native_base, accepted) if gated else native_proposal
    physical = hand_atomic(proposal, parent, accepted) if gated else proposal
    return dict(native_actions=native, physical_actions=physical, ungated_physical_actions=proposal, ungated_native_actions=native_proposal, parent_physical_actions=parent, delta_arm_rad=delta_arm, delta_opening_m=delta_grip, reliability_logits=logits, accepted=accepted, within_URDF_limits=within)
