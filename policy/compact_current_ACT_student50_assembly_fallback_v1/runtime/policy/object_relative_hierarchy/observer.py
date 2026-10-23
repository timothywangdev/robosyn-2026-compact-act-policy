"""Small spatial-token observer shared across Qwen/WAM/Smol feature controls.

Backbones and runtime are deliberately absent. All forward inputs are current
allowed observation representations. Future/privileged data enter losses only.
"""
import torch
from torch import nn
from torch.nn import functional as F

FEATURE_NORMALIZATION = 'token-and-object-slot-layernorm-v1'


class SpatialGeometryObserver(nn.Module):
    def __init__(self, token_dim, intent_dim, hidden_dim=128, heads=4, layers=2):
        super().__init__()
        if min(token_dim, intent_dim, hidden_dim, heads, layers) <= 0 or hidden_dim % heads:
            raise ValueError('Positive dimensions; hidden width divisible by heads')
        self.token_dim, self.intent_dim = token_dim, intent_dim
        self.token_projection = nn.Linear(token_dim, hidden_dim)
        self.coordinate_projection = nn.Linear(5, hidden_dim)  # camera onehot3 + xy2
        self.context_projection = nn.Linear(intent_dim + 14, hidden_dim)
        self.object_queries = nn.Parameter(torch.randn(2, hidden_dim) * .02)
        self.blocks = nn.ModuleList([nn.TransformerDecoderLayer(hidden_dim, heads, 2 * hidden_dim, dropout=0., activation='gelu', batch_first=True, norm_first=True) for _ in range(layers)])
        self.position_head = nn.Linear(hidden_dim, 3)
        self.axis_head = nn.Linear(hidden_dim, 3)
        self.carrier_head = nn.Sequential(nn.Linear(2 * hidden_dim + 14, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3))

    def forward(self, *, current_tokens, token_coordinates, token_valid, current_qpos, intent):
        """tokens[B,N,D]; coordinates[B,N,5]; bool valid[B,N]; q[B,14].

        Coordinates explicitly identify three current cameras and spatial xy.
        Backbones must provide a verified current-only token ordering; this head
        cannot establish provenance from a tensor's values alone.
        """
        if current_tokens.ndim != 3:
            raise ValueError('Current spatial tokens require[B,N,D]')
        batch, count, dim = current_tokens.shape
        if dim != self.token_dim or token_coordinates.shape != (batch, count, 5) or token_valid.shape != (batch, count) or token_valid.dtype != torch.bool or current_qpos.shape != (batch, 14) or intent.shape != (batch, self.intent_dim):
            raise ValueError('Current token/context dimension contract')
        if not bool(token_valid.any(-1).all()):
            raise ValueError('Each sample needs a visible token')
        if any(not bool(torch.isfinite(value).all()) for value in (current_tokens, token_coordinates, current_qpos, intent)):
            raise ValueError('Finite current inputs required')
        cameras = token_coordinates[..., :3][token_valid]
        xy = token_coordinates[..., 3:][token_valid]
        if not bool(((cameras == 0) | (cameras == 1)).all()) or not bool((cameras.sum(-1) == 1).all()) or bool((xy.abs() > 1).any()):
            raise ValueError('Onehot camera and normalized spatial xy required')
        current_tokens, token_coordinates = current_tokens.detach(), token_coordinates.detach()
        current_qpos, intent = current_qpos.detach(), intent.detach()
        # The frozen vision tower's unmerged tokens have an uncalibrated scale.
        # Normalize each current token, retaining camera/coordinate identities.
        memory = self.token_projection(F.layer_norm(current_tokens, (self.token_dim,))) + self.coordinate_projection(token_coordinates)
        context = self.context_projection(torch.cat((current_qpos, intent), -1))
        slots = self.object_queries[None].expand(batch, -1, -1) + context[:, None]
        for block in self.blocks:
            slots = block(slots, memory, memory_key_padding_mask=~token_valid)
        # Pre-norm decoder residuals otherwise propagate arbitrary magnitudes
        # into metric geometry and saturate the three-way carrier softmax.
        slots = F.layer_norm(slots, (slots.shape[-1],))
        raw_axes = self.axis_head(slots)
        return dict(positions=self.position_head(slots), axes=F.normalize(raw_axes, dim=-1, eps=1e-6),
                    axis_norm=raw_axes.norm(dim=-1), carrier_logits=self.carrier_head(torch.cat((slots.flatten(1), current_qpos), -1)))


def partial_carrier_loss(logits, admissible):
    """Set-label likelihood; no preference among admissible explanations."""
    if logits.ndim != 2 or logits.shape[1] != 3 or admissible.shape != logits.shape or admissible.dtype != torch.bool or not bool(admissible.any(-1).all()) or not bool(torch.isfinite(logits).all()):
        raise ValueError('Finite three-class logits and nonempty bool admissible sets required')
    admissible = admissible.detach()
    selected = logits.masked_fill(~admissible, float('-inf'))
    per_row = torch.logsumexp(logits, -1) - torch.logsumexp(selected, -1)
    return per_row.mean()


def geometry_loss(prediction, *, positions, axes, carrier, admissible=None, position_scale_m=.02):
    """Training-only current pose labels and weak carrier labels; -1 masked."""
    if positions.shape != prediction['positions'].shape or axes.shape != prediction['axes'].shape or carrier.shape != (positions.shape[0],):
        raise ValueError('Training label shapes')
    if position_scale_m <= 0 or not torch.isfinite(positions).all() or not torch.isfinite(axes).all() or not torch.allclose(axes.norm(dim=-1), torch.ones_like(axes[..., 0]), atol=1e-4, rtol=0):
        raise ValueError('Finite position/unit-axis labels required')
    if carrier.dtype != torch.long or bool(((carrier < -1) | (carrier > 2)).any()):
        raise ValueError('Carrier target must be -1/0/1/2')
    positions, axes, carrier = positions.detach(), axes.detach(), carrier.detach()
    pos = F.smooth_l1_loss(prediction['positions'] / position_scale_m, positions / position_scale_m)
    axis = (1 - (prediction['axes'] * axes).sum(-1)).mean()
    valid = carrier >= 0
    carry = partial_carrier_loss(prediction['carrier_logits'], admissible) if admissible is not None else (F.cross_entropy(prediction['carrier_logits'][valid], carrier[valid]) if bool(valid.any()) else prediction['carrier_logits'].sum() * 0)
    return dict(position=pos, axis=axis, carrier=carry)


def measured_transport_loss(prediction, *, eef_now, eef_future, pen_future_position, pen_future_axis, carrier, position_scale_m=.02):
    """Auxiliary observer loss using measured future FK ONLY during training.

    Explicit weak labels select an unambiguous hand, avoiding a learnable gate
    that could suppress its own loss. This is not action-effect certification.
    """
    batch = prediction['positions'].shape[0]
    if eef_now.shape != (batch, 2, 4, 4) or eef_future.shape != eef_now.shape or pen_future_position.shape != (batch, 3) or pen_future_axis.shape != (batch, 3) or carrier.shape != (batch,):
        raise ValueError('Transport label dimension contract')
    if position_scale_m <= 0 or any(not bool(torch.isfinite(x).all()) for x in (eef_now, eef_future, pen_future_position, pen_future_axis)):
        raise ValueError('Finite transport labels required')
    if carrier.dtype != torch.long or bool(((carrier < -1) | (carrier > 2)).any()) or not torch.allclose(pen_future_axis.norm(dim=-1), torch.ones_like(pen_future_axis[:, 0]), atol=1e-4, rtol=0):
        raise ValueError('Valid carrier labels and unit future axis required')
    for transform in (eef_now, eef_future):
        rotation = transform[..., :3, :3]
        if not torch.allclose(rotation.transpose(-1, -2) @ rotation, torch.eye(3, device=rotation.device, dtype=rotation.dtype).expand_as(rotation), atol=1e-4, rtol=0) or not torch.allclose(torch.linalg.det(rotation), torch.ones_like(rotation[..., 0, 0]), atol=1e-4, rtol=0):
            raise ValueError('Rigid measured EEF rotations required')
        bottom = torch.tensor([0, 0, 0, 1], device=transform.device, dtype=transform.dtype).expand_as(transform[..., 3, :])
        if not torch.allclose(transform[..., 3, :], bottom, atol=1e-4, rtol=0):
            raise ValueError('Homogeneous measured EEF poses required')
    eef_now, eef_future = eef_now.detach(), eef_future.detach()
    pen_future_position, pen_future_axis, carrier = pen_future_position.detach(), pen_future_axis.detach(), carrier.detach()
    valid = (carrier == 0) | (carrier == 1)
    if not bool(valid.any()):
        return prediction['positions'].sum() * 0 + prediction['axes'].sum() * 0
    rows, hand = torch.arange(batch, device=carrier.device)[valid], carrier[valid]
    now, future = eef_now[rows, hand], eef_future[rows, hand]
    rotation = future[:, :3, :3] @ now[:, :3, :3].transpose(-1, -2)
    p = (rotation @ (prediction['positions'][valid, 0] - now[:, :3, 3])[..., None])[..., 0] + future[:, :3, 3]
    axis = (rotation @ prediction['axes'][valid, 0, :, None])[..., 0]
    return F.smooth_l1_loss(p / position_scale_m, pen_future_position[valid] / position_scale_m) + (1 - (axis * pen_future_axis[valid]).sum(-1)).mean()
