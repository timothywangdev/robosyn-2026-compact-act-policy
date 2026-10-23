"""Current-only shared object observer, learned +10 goal and WAM interface.

This is an isolated trainable component, not an enabled deployment policy.
Geometry/transport labels are passed only to loss functions. All ten tasks use
one model; the original-instruction foundation representation conditions it.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .observer import SpatialGeometryObserver, partial_carrier_loss, measured_transport_loss

GOAL_SLOTS = tuple(range(10, 19)) + tuple(range(44, 53))
POSE_SLOTS = tuple(range(9)) + tuple(range(34, 43))
CURRENT_SLOTS = tuple(range(10)) + tuple(range(34, 44))
CARRIER_SLOTS = (19, 20, 21)


class CurrentObjectGoal(nn.Module):
    def __init__(self, token_dim, intent_dim, hidden_dim=128, max_goal_translation_m=.15,
                 max_goal_rotation_rad=.5):
        super().__init__()
        if not 0 < max_goal_translation_m <= 1 or not 0 < max_goal_rotation_rad <= 3.14159:
            raise ValueError('Explicit positive prospective goal bounds required')
        self.observer = SpatialGeometryObserver(token_dim, intent_dim, hidden_dim)
        self.intent_projection = nn.Linear(intent_dim, hidden_dim)
        self.goal_head = nn.Sequential(nn.Linear(12 + 14 + hidden_dim, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 6))
        # Start from a current-object goal, not an arbitrary absolute location.
        nn.init.zeros_(self.goal_head[-1].weight)
        nn.init.zeros_(self.goal_head[-1].bias)
        self.max_goal_translation_m = float(max_goal_translation_m)
        self.max_goal_rotation_rad = float(max_goal_rotation_rad)

    def forward(self, *, current_tokens, token_coordinates, token_valid, current_qpos, intent):
        prediction = self.observer(current_tokens=current_tokens,
            token_coordinates=token_coordinates, token_valid=token_valid,
            current_qpos=current_qpos, intent=intent)
        geometry = torch.cat((prediction['positions'].flatten(1), prediction['axes'].flatten(1)), -1)
        context = self.intent_projection(intent.detach())
        output = self.goal_head(torch.cat((geometry, current_qpos.detach(), context), -1))
        displacement = self.max_goal_translation_m * torch.tanh(output[:, :3])
        displacement = displacement / torch.clamp(displacement.norm(dim=-1, keepdim=True) / self.max_goal_translation_m, min=1.)
        # Bound the vector norm, not each of three components independently.
        rotation_vector = self.max_goal_rotation_rad * torch.tanh(output[:, 3:])
        norm = rotation_vector.norm(dim=-1, keepdim=True)
        rotation_vector = rotation_vector / torch.clamp(norm / self.max_goal_rotation_rad, min=1.)
        rotation = rotation_from_vector(rotation_vector)
        prediction.update(goal_position=prediction['positions'][:, 0] + displacement,
            goal_axis=(rotation @ prediction['axes'][:, 0, :, None])[..., 0],
            goal_displacement=displacement, goal_rotation_vector=rotation_vector,
            carrier_probabilities=prediction['carrier_logits'].softmax(-1))
        return prediction


def skew(vector):
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*vector.shape[:-1], 3, 3)


def rotation_from_vector(vector):
    # torch.sinc avoids the zero-angle division/gradient singularity.
    theta = vector.norm(dim=-1)
    cross = skew(vector)
    eye = torch.eye(3, dtype=vector.dtype, device=vector.device).expand_as(cross)
    return eye + torch.sinc(theta / torch.pi)[..., None, None] * cross + .5 * torch.sinc(theta / (2 * torch.pi)).square()[..., None, None] * (cross @ cross)


def minimal_axis_rotation(current, desired):
    """Minimal directed-axis rotation; axial twist deliberately not supervised."""
    if current.shape != desired.shape or current.shape[-1] != 3:
        raise ValueError('Matched directed-axis vectors required')
    if any(not bool(torch.isfinite(x).all()) or bool((x.norm(dim=-1) < 1e-5).any()) for x in (current, desired)):
        raise ValueError('Finite nonzero predicted axes required')
    a, b = F.normalize(current, dim=-1), F.normalize(desired, dim=-1)
    cosine = (a * b).sum(-1).clamp(-1, 1)
    cross = skew(torch.linalg.cross(a, b))
    eye = torch.eye(3, dtype=a.dtype, device=a.device).expand_as(cross)
    ordinary = eye + cross + cross @ cross / (1 + cosine).clamp_min(1e-6)[..., None, None]
    # Deterministic perpendicular axis only for the antipodal singularity.
    basis = F.one_hot(a.abs().argmin(-1), 3).to(a.dtype)
    axis = F.normalize(torch.linalg.cross(a, basis), dim=-1)
    antipodal = 2 * axis[..., :, None] * axis[..., None, :] - eye
    return torch.where((cosine < -1 + 1e-5)[..., None, None], antipodal, ordinary)


def wam_object_condition(prediction, *, current_eef_root, arm_from_root,
                         current_proprio, mean, scale):
    """Append two rigid transport candidates and a carrier distribution.

    All poses are computed from predicted CURRENT object state/goal and CURRENT
    qpos FK. Neither hand is chosen in Python; WAM learns how to use both
    candidates and all three carrier probabilities. No rotations are averaged.
    The41-field contract needs separately trained WAM proprio columns.
    """
    batch = prediction['positions'].shape[0]
    if current_eef_root.shape != (batch, 2, 4, 4) or arm_from_root.shape not in ((2, 4, 4), (batch, 2, 4, 4)) or current_proprio.shape != (batch, 80):
        raise ValueError('Current FK/calibration/proprio shape contract')
    if mean.shape != (80,) or scale.shape != (80,) or not bool((scale > 0).all()):
        raise ValueError('Fit-only physical normalizer80 required')
    if any(not bool(torch.isfinite(v).all()) for v in (current_eef_root, arm_from_root, current_proprio, mean, scale)):
        raise ValueError('Finite current/calibration inputs required')
    for pose in (current_eef_root, arm_from_root):
        r=pose[..., :3, :3]
        eye=torch.eye(3,dtype=r.dtype,device=r.device).expand_as(r)
        bottom=torch.tensor([0,0,0,1],dtype=pose.dtype,device=pose.device).expand_as(pose[...,3,:])
        if not torch.allclose(r.transpose(-1,-2)@r,eye,atol=1e-4,rtol=0) or not torch.allclose(torch.linalg.det(r),torch.ones_like(r[...,0,0]),atol=1e-4,rtol=0) or not torch.allclose(pose[...,3,:],bottom,atol=1e-4,rtol=0):
            raise ValueError('Rigid CURRENT FK and arm-frame calibration required')
    rotation = minimal_axis_rotation(prediction['axes'][:, 0], prediction['goal_axis'])
    eef = current_eef_root.detach()
    goal = torch.zeros_like(eef)
    goal[..., :3, :3] = rotation[:, None] @ eef[..., :3, :3]
    relative = eef[..., :3, 3] - prediction['positions'][:, None, 0]
    goal[..., :3, 3] = (rotation[:, None] @ relative[..., None])[..., 0] + prediction['goal_position'][:, None]
    goal[..., 3, 3] = 1
    arm_goal = arm_from_root.detach() @ goal
    xyz = arm_goal[..., :3, 3]
    rot6 = arm_goal[..., :3, :2].transpose(-1, -2).flatten(-2)
    physical = torch.cat((xyz, rot6), -1).flatten(1)
    normalized = (physical - mean.detach()[list(POSE_SLOTS)]) / scale.detach()[list(POSE_SLOTS)]
    out = current_proprio.detach().clone()
    # Prepared WAM proprio is BF16, while the learned geometry stays FP32.
    # Indexed assignment requires equal dtypes; Tensor.to retains autograd.
    out[:, list(GOAL_SLOTS)] = normalized.to(out)
    out[:, list(CARRIER_SLOTS)] = prediction['carrier_probabilities'].to(out)
    mask = torch.zeros(batch, 80, dtype=torch.bool, device=out.device)
    mask[:, list(CURRENT_SLOTS + GOAL_SLOTS + CARRIER_SLOTS)] = True
    return out, mask


def masked_object_losses(prediction, *, labels, position_scale_m=.02,
                         carrier_weight=1., transport_weight=1.):
    """Use per-object/per-window validity only in detached training losses.

    Partial valid single-object rows retain current geometry supervision.
    Missing labels never become zeros or actor features. Same-budget control
    sets carrier/transport weights0 with otherwise identical model/losses.
    """
    batch = prediction['positions'].shape[0]
    current_valid = labels['object_valid']
    future_valid = labels['future_object_valid']
    if any(mask.shape != (batch, 2) or mask.dtype != torch.bool for mask in (current_valid, future_valid)):
        raise ValueError('Per-object bool availability masks required')
    if min(carrier_weight, transport_weight) < 0 or position_scale_m <= 0:
        raise ValueError('Positive position scale and nonnegative weights required')
    zero = sum(x.sum() for x in prediction.values()) * 0
    losses = {name: zero for name in ('position', 'axis', 'carrier', 'goal_position', 'goal_axis', 'transport')}
    if bool(current_valid.any()):
        positions = labels['positions'][current_valid].detach().to(prediction['positions'])
        axes = labels['axes'][current_valid].detach().to(prediction['axes'])
        if not bool(torch.isfinite(positions).all()) or not bool(torch.isfinite(axes).all()) or not torch.allclose(axes.norm(dim=-1), torch.ones_like(axes[:, 0]), atol=1e-4, rtol=0):
            raise ValueError('Valid current pose labels must be finite with unit axes')
        losses['position'] = F.smooth_l1_loss(prediction['positions'][current_valid] / position_scale_m, positions / position_scale_m)
        losses['axis'] = (1 - (prediction['axes'][current_valid] * axes).sum(-1)).mean()
    window = current_valid[:, 0] & future_valid[:, 0]
    if bool(window.any()):
        target_position = labels['future_positions'][window, 0].detach().to(prediction['positions'])
        target_axis = labels['future_axes'][window, 0].detach().to(prediction['axes'])
        if not bool(torch.isfinite(target_position).all()) or not bool(torch.isfinite(target_axis).all()) or not torch.allclose(target_axis.norm(dim=-1), torch.ones_like(target_axis[:, 0]), atol=1e-4, rtol=0):
            raise ValueError('Valid future source labels must be finite/unit')
        pred = {k: v[window] for k, v in prediction.items()}
        losses['goal_position'] = F.smooth_l1_loss(pred['goal_position'] / position_scale_m, target_position / position_scale_m)
        losses['goal_axis'] = (1 - (pred['goal_axis'] * target_axis).sum(-1)).mean()
        losses['carrier'] = carrier_weight * partial_carrier_loss(pred['carrier_logits'], labels['admissible'][window].detach())
        losses['transport'] = transport_weight * measured_transport_loss(pred,
            eef_now=labels['eef_now'][window].detach().to(prediction['positions']), eef_future=labels['eef_future'][window].detach().to(prediction['positions']),
            pen_future_position=target_position, pen_future_axis=target_axis,
            carrier=labels['carrier'][window].detach(), position_scale_m=position_scale_m)
    return losses
