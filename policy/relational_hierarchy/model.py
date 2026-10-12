"""Three goal slots and an identity-initialized SmolVLA prefix adapter.

This module imports only PyTorch and the standard library. It neither loads a
checkpoint nor patches a policy. Goal features must be derived externally from
the current permitted observation/task, never from demonstration futures.
"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class BridgeOutput:
    prefix_embeddings: Tensor
    prefix_padding_mask: Tensor
    prefix_attention_mask: Tensor
    goal_slots: Tensor
    valid_slots: Tensor

    def prefix(self) -> tuple[Tensor, Tensor, Tensor]:
        """The same three-tensor contract as the pinned embed_prefix method."""
        return self.prefix_embeddings, self.prefix_padding_mask, self.prefix_attention_mask


def _boolean_mask(value: Tensor, shape: tuple[int, ...], name: str, device: torch.device):
    if value.shape != shape or value.dtype != torch.bool or value.device != device:
        raise ValueError(f"{name} must be bool with shape {shape} on {device}")


def _finite(value: Tensor, name: str):
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain finite floating-point values")


class RelationalGoalBridge(nn.Module):
    """Read role-specific goals and alter only existing motor-state tokens.

    Role order is left arm, right arm, object-to-destination relation. Both
    controls use exactly the same parameters: structured mode restricts each
    query to its matching goal plus a null key; unstructured mode permits all
    goal keys. The original ordered inputs and role embeddings remain present
    in both controls. Goal drop is an explicit caller-provided mask, not RNG.
    """

    roles = ("left_arm", "right_arm", "object_destination")

    def __init__(self, prefix_dim: int, goal_dim: int, hidden_dim: int = 128,
                 num_heads: int = 4, effect_dim: int = 16):
        super().__init__()
        for name, value in (("prefix_dim", prefix_dim), ("goal_dim", goal_dim),
                            ("hidden_dim", hidden_dim), ("num_heads", num_heads),
                            ("effect_dim", effect_dim)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.prefix_dim, self.goal_dim = prefix_dim, goal_dim
        self.hidden_dim, self.num_heads, self.effect_dim = hidden_dim, num_heads, effect_dim
        self.goal_projection = nn.Linear(goal_dim, hidden_dim)
        self.context_projection = nn.Linear(prefix_dim, hidden_dim)
        self.role_queries = nn.Parameter(torch.empty(3, hidden_dim))
        self.role_keys = nn.Parameter(torch.empty(3, hidden_dim))
        nn.init.normal_(self.role_queries, std=0.02)
        nn.init.normal_(self.role_keys, std=0.02)
        self.goal_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.0, batch_first=True)
        self.context_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.0, batch_first=True)
        self.slot_norm = nn.LayerNorm(hidden_dim)
        self.slot_ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.motor_projection = nn.Linear(3 * hidden_dim, prefix_dim, bias=False)
        # Adding new attention tokens would not be an identity transformation.
        # A zero residual on existing tokens preserves all native mask/position logic.
        nn.init.zeros_(self.motor_projection.weight)
        self.effect_heads = nn.ModuleList(nn.Linear(hidden_dim, effect_dim) for _ in self.roles)

    def forward(self, prefix_embeddings: Tensor, prefix_padding_mask: Tensor,
                prefix_attention_mask: Tensor, goal_features: Tensor, goal_valid: Tensor,
                *, mode: str = "structured", drop_goals: Tensor | None = None) -> BridgeOutput:
        """Condition a prefix without future inputs or any internal random draws.

        prefix_embeddings: (B, P, prefix_dim), the unmodified embed_prefix output.
        goal_features: (B, 3, goal_dim), externally encoded current goal evidence.
        goal_valid: (B, 3), unknown roles are False and carry finite placeholder data.
        drop_goals: optional (B,) ablation mask; True returns the native prefix row.
        """
        if mode not in ("structured", "unstructured"):
            raise ValueError("mode must be structured or unstructured")
        if prefix_embeddings.ndim != 3 or prefix_embeddings.shape[-1] != self.prefix_dim:
            raise ValueError("prefix_embeddings must have shape (B, P, prefix_dim)")
        batch, length, _ = prefix_embeddings.shape
        if batch == 0 or length == 0:
            raise ValueError("prefix batch and sequence must be nonempty")
        device = prefix_embeddings.device
        _finite(prefix_embeddings, "prefix_embeddings")
        _boolean_mask(prefix_padding_mask, (batch, length), "prefix_padding_mask", device)
        _boolean_mask(prefix_attention_mask, (batch, length), "prefix_attention_mask", device)
        if goal_features.shape != (batch, 3, self.goal_dim) or goal_features.device != device:
            raise ValueError("goal_features must have shape (B, 3, goal_dim) on the prefix device")
        _finite(goal_features, "goal_features")
        _boolean_mask(goal_valid, (batch, 3), "goal_valid", device)
        if self.motor_projection.weight.device != device or self.motor_projection.weight.dtype != prefix_embeddings.dtype:
            raise ValueError("Move the bridge to the prefix device/dtype before calling it")
        state_positions = prefix_padding_mask & prefix_attention_mask
        if not bool(state_positions.any(dim=1).all()):
            raise ValueError("Pinned Smol prefix must contain at least one valid state token per row")
        valid = goal_valid
        if drop_goals is not None:
            _boolean_mask(drop_goals, (batch,), "drop_goals", device)
            valid = valid & ~drop_goals[:, None]
        # Mask before projection, too: unknown placeholder magnitudes cannot affect any slot.
        features = torch.where(valid[..., None], goal_features, torch.zeros_like(goal_features))
        goal_keys = self.goal_projection(features.to(prefix_embeddings.dtype)) + self.role_keys[None]
        null = goal_keys.new_zeros(batch, 1, self.hidden_dim)
        keys = torch.cat((goal_keys, null), dim=1)
        allowed = valid[:, None, :].expand(batch, 3, 3)
        if mode == "structured":
            allowed = allowed & torch.eye(3, dtype=torch.bool, device=device)[None]
        # Every query can attend a null key, avoiding all-masked softmax rows.
        allowed = torch.cat((allowed, torch.ones(batch, 3, 1, dtype=torch.bool, device=device)), dim=-1)
        blocked = (~allowed).repeat_interleave(self.num_heads, dim=0)
        queries = self.role_queries[None].expand(batch, -1, -1)
        slots, _ = self.goal_attention(queries, keys, keys, attn_mask=blocked, need_weights=False)
        context = self.context_projection(prefix_embeddings)
        observed, _ = self.context_attention(slots, context, context,
                                             key_padding_mask=~prefix_padding_mask, need_weights=False)
        slots = self.slot_norm(slots + observed)
        slots = slots + self.slot_ffn(slots)
        slots = torch.where(valid[..., None], slots, torch.zeros_like(slots))
        delta = self.motor_projection(slots.flatten(1))[:, None, :]
        change = state_positions & valid.any(dim=1)[:, None]
        conditioned = torch.where(change[..., None], prefix_embeddings + delta, prefix_embeddings)
        return BridgeOutput(conditioned, prefix_padding_mask, prefix_attention_mask, slots, valid)

    def future_effect_loss(self, output: BridgeOutput, targets: Tensor, observed: Tensor) -> Tensor:
        """Training-only auxiliary loss; targets never enter forward/inference.

        Callers define and document effect coordinates, normalize them using the
        training split, and mark only verified observations. These are achieved
        effects, not inferred contact labels or counterfactual expert actions.
        """
        if not self.training:
            raise RuntimeError("Future-effect supervision is training-only")
        shape = (*output.valid_slots.shape, self.effect_dim)
        if targets.shape != shape or targets.device != output.goal_slots.device:
            raise ValueError("targets must have shape (B, 3, effect_dim) on the slot device")
        _boolean_mask(observed, shape, "observed", output.goal_slots.device)
        _finite(targets, "targets")
        prediction = torch.stack([head(output.goal_slots[:, i]) for i, head in enumerate(self.effect_heads)], dim=1)
        mask = observed & output.valid_slots[..., None]
        errors = F.smooth_l1_loss(prediction.float(), targets.detach().float(), reduction="none")
        return (errors * mask).sum() / mask.sum().clamp_min(1)


def verified_contrastive_loss(goal_embeddings: Tensor, effect_embeddings: Tensor,
                              verified_positive: Tensor, verified_negative: Tensor,
                              *, temperature: float = 0.1) -> Tensor:
    """Multi-positive contrastive loss using only externally verified pairs.

    No diagonal positives, off-diagonal negatives, temporal negatives, or labels
    are inferred. Rows without a verified negative contribute no contrast loss.
    A negative row without any verified positive is invalid supervision.
    """
    if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if goal_embeddings.ndim != 2 or effect_embeddings.ndim != 2 or goal_embeddings.shape[1] != effect_embeddings.shape[1]:
        raise ValueError("Embedding matrices must have the same feature dimension")
    if goal_embeddings.device != effect_embeddings.device:
        raise ValueError("Embedding devices differ")
    _finite(goal_embeddings, "goal_embeddings"); _finite(effect_embeddings, "effect_embeddings")
    shape = (goal_embeddings.shape[0], effect_embeddings.shape[0])
    if not all(shape) or goal_embeddings.shape[1] == 0:
        raise ValueError("Embedding matrices must be nonempty")
    _boolean_mask(verified_positive, shape, "verified_positive", goal_embeddings.device)
    _boolean_mask(verified_negative, shape, "verified_negative", goal_embeddings.device)
    if bool((verified_positive & verified_negative).any()):
        raise ValueError("A pair cannot be both positive and negative")
    positive_rows, negative_rows = verified_positive.any(1), verified_negative.any(1)
    if bool((negative_rows & ~positive_rows).any()):
        raise ValueError("Verified negatives require a verified positive in the same row")
    rows = positive_rows & negative_rows
    if not bool(rows.any()):
        return goal_embeddings.sum() * 0 + effect_embeddings.sum() * 0
    logits = F.normalize(goal_embeddings.float(), dim=-1) @ F.normalize(effect_embeddings.float(), dim=-1).T
    logits = logits[rows] / temperature
    positive, negative = verified_positive[rows], verified_negative[rows]
    denominator = torch.logsumexp(logits.masked_fill(~(positive | negative), -torch.inf), dim=1)
    numerator = torch.logsumexp(logits.masked_fill(~positive, -torch.inf), dim=1)
    return (denominator - numerator).mean()
