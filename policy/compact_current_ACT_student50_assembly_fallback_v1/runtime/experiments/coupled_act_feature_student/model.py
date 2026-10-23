"""Frozen-ACT coarse-spatial feature student; qpos-only is a matched control."""
import torch
from torch import nn
from .contracts import require


class Student(nn.Module):
    def __init__(self, pooling='spatial2x3', qpos_only=False):
        super().__init__()
        require(pooling in ('spatial2x3', 'global') and type(qpos_only) is bool, 'Declared matched student mode')
        self.pooling, self.qpos_only = pooling, qpos_only
        width = 3 * 512 * (6 if pooling == 'spatial2x3' else 1)
        self.hidden = nn.Sequential(nn.Linear(width + 14, 256), nn.GELU())
        self.roles = nn.Linear(256, 96)
        self.goal = nn.Linear(256, 18)

    def forward(self, visual, qpos_features, visual_global=None):
        require(visual.ndim == 5 and visual.shape[1:] == (3, 512, 2, 3)
                and qpos_features.shape == (len(visual), 14), 'Current spatial512x2x3/normalizedqpos14 required')
        require(visual.dtype == qpos_features.dtype == torch.float32
                and bool(torch.isfinite(visual).all()) and bool(torch.isfinite(qpos_features).all()), 'Finite FP32 student input')
        if self.pooling == 'global':
            require(visual_global is not None and visual_global.shape == (len(visual), 3, 512)
                    and visual_global.dtype == torch.float32 and bool(torch.isfinite(visual_global).all()),
                    'Exact raw-map global average required; do not average adaptive cells')
        features = visual_global if self.pooling == 'global' else visual
        if self.qpos_only:
            features = features * 0
        hidden = self.hidden(torch.cat((features.flatten(1), qpos_features), -1))
        return dict(projected_roles=torch.tanh(self.roles(hidden)), predicted_goal_delta=self.goal(hidden))


def teacher_targets(teacher, current_roles, current_eef, predicted_goal, eef_scale):
    """Teacher predictions only; this never reads command/future ground truth."""
    from experiments.goal_latent_act_full_data.correction_prefix import eef_features
    require(current_roles.shape[1:] == (3, 4096) and current_eef.shape[1:] == predicted_goal.shape[1:] == (2, 4, 4),
            'Exact current teacher queries/FK/predicted goal inputs')
    require(eef_scale.shape == (18,) and bool((eef_scale > 0).all()), 'Frozen teacher EEF scale required')
    with torch.no_grad():
        return dict(projected_roles=torch.tanh(teacher.roles(current_roles.float())).flatten(1).detach(),
            predicted_goal_delta=((eef_features(predicted_goal) - eef_features(current_eef)) / eef_scale).detach())
