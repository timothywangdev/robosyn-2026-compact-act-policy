"""Current-only Qwen role latents -> physical goals, with sparse object labels.

EEF inputs are measured-current FK, never future or simulator object poses.
The dual-arm goal head has supervision on all ten tasks; object/carrier heads
retain the original validity masks where named object poses are unavailable.
"""
import torch
from torch import nn
from torch.nn import functional as F
from policy.object_relative_hierarchy.multitask_model import rotation_from_vector, masked_object_losses

VERSION = 'cached-current-qwen-role-dual-eef-object-goal-v1'


def bounded(vector, limit):
    value = limit * torch.tanh(vector)
    return value / (value.norm(dim=-1, keepdim=True) / limit).clamp_min(1.)


class RolePhysicalGoal(nn.Module):
    def __init__(self, hidden=128, eef_translation_m=.20, eef_rotation_rad=.75,
                 object_translation_m=.15, object_rotation_rad=.5):
        super().__init__()
        if hidden < 16 or min(eef_translation_m, eef_rotation_rad, object_translation_m, object_rotation_rad) <= 0:
            raise ValueError('Positive explicit physical bounds and hidden width required')
        self.config = dict(hidden=hidden, eef_translation_m=eef_translation_m,
                           eef_rotation_rad=eef_rotation_rad, object_translation_m=object_translation_m,
                           object_rotation_rad=object_rotation_rad)
        self.role_projection = nn.Linear(4096, hidden)
        self.trunk = nn.Sequential(nn.Linear(3 * hidden + 14 + 18, 2 * hidden), nn.GELU(),
                                   nn.Linear(2 * hidden, 2 * hidden), nn.GELU())
        self.geometry = nn.Linear(2 * hidden, 12)
        self.carrier = nn.Linear(2 * hidden, 3)
        self.object_goal = nn.Linear(2 * hidden, 6)
        self.eef_goal = nn.Linear(2 * hidden, 12)
        for layer in (self.object_goal, self.eef_goal):
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)

    def forward(self, *, roles, qpos, eef_now, qpos_mean, qpos_scale, eef_mean, eef_scale,
                role_control='factual'):
        batch = roles.shape[0]
        if roles.shape != (batch, 3, 4096) or qpos.shape != (batch, 14) or eef_now.shape != (batch, 2, 4, 4):
            raise ValueError('Exact current role/qpos/measured FK dimensions required')
        if role_control not in ('factual', 'zero'):
            raise ValueError('Explicit factual or role-zero control')
        r = roles.detach().float()
        r = F.layer_norm(r, (4096,))
        if role_control == 'zero': r = r * 0
        r = self.role_projection(r).flatten(1)
        q = (qpos.detach() - qpos_mean) / qpos_scale
        pose = torch.cat((eef_now[..., :3, 3], eef_now[..., :3, :2].transpose(-1, -2).flatten(-2)), -1).flatten(1)
        pose = (pose.detach() - eef_mean) / eef_scale
        h = self.trunk(torch.cat((r, q, pose), -1))
        geom = self.geometry(h).reshape(batch, 2, 6)
        axes = F.normalize(geom[..., 3:], dim=-1, eps=1e-6)
        obj = self.object_goal(h)
        od = bounded(obj[:, :3], self.config['object_translation_m'])
        ov = bounded(obj[:, 3:], self.config['object_rotation_rad'])
        eo = self.eef_goal(h).reshape(batch, 2, 6)
        ed = bounded(eo[..., :3], self.config['eef_translation_m'])
        ev = bounded(eo[..., 3:], self.config['eef_rotation_rad'])
        goal = torch.zeros_like(eef_now)
        goal[..., :3, :3] = rotation_from_vector(ev) @ eef_now[..., :3, :3].detach()
        goal[..., :3, 3] = eef_now[..., :3, 3].detach() + ed
        goal[..., 3, 3] = 1
        logits = self.carrier(h)
        return dict(positions=geom[..., :3], axes=axes, axis_norm=geom[..., 3:].norm(dim=-1),
                    carrier_logits=logits, carrier_probabilities=logits.softmax(-1),
                    goal_position=geom[:, 0, :3] + od,
                    goal_axis=(rotation_from_vector(ov) @ axes[:, 0, :, None])[..., 0],
                    goal_displacement=od, goal_rotation_vector=ov,
                    eef_goal_pose=goal, eef_goal_displacement=ed, eef_goal_rotation_vector=ev)


def physical_loss(prediction, labels):
    auxiliary = masked_object_losses(prediction, labels=labels, carrier_weight=1., transport_weight=1.)
    target = labels['eef_future'].detach()
    auxiliary['eef_goal_position'] = F.smooth_l1_loss(prediction['eef_goal_pose'][..., :3, 3] / .02, target[..., :3, 3] / .02)
    auxiliary['eef_goal_rotation'] = (prediction['eef_goal_pose'][..., :3, :3] - target[..., :3, :3]).square().mean()
    return sum(auxiliary.values()), auxiliary


def goal_metrics(prediction, labels):
    target = labels['eef_future']
    value = prediction['eef_goal_pose']
    rot = value[..., :3, :3].transpose(-1, -2) @ target[..., :3, :3]
    angle = ((rot.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1).acos()
    return {'eef_goal_position_l2_m': (value[..., :3, 3] - target[..., :3, 3]).norm(dim=-1).mean(-1),
            'eef_goal_rotation_rad': angle.mean(-1)}
