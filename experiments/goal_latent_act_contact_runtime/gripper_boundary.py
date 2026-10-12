"""Explicit physical grip commands around the released native50 action ABI.

Drawer's existing evaluation boundary performs its normalized-opening inverse.
New contact outputs therefore return to Drawer native coordinates here, so the
existing inverse runs exactly once. Other tasks preserve the native arm command
and submit the learned opening in metres to the robot's physical joint clamp.
This changes gripper behavior and must be evaluated as a new candidate.
"""

ARMS = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
GRIPS = (6, 13)
DRAWER = 'drawer_normalized_opening_inverse_v1'
IDENTITY = 'physical_meters_identity_v1'
TASKS = ('click_bell', 'drawer_open_place', 'item_assembly', 'items_handover',
         'table_rearrangement', 'mixer_operating', 'sample_loading',
         'water_pouring', 'manipulate_pipette', 'handle_basket')


def _require(value, message):
    if not value:
        raise ValueError(message)


def _validate_actions(actions, task, codec):
    import torch
    _require(task in TASKS, 'An explicit competition task is required')
    _require(codec in (DRAWER, IDENTITY), 'Explicit source-qualified codec required')
    _require((task == 'drawer_open_place') == (codec == DRAWER),
             'Drawer inverse must match the actual task')
    _require(isinstance(actions, torch.Tensor) and actions.dtype == torch.float32
             and actions.ndim == 3 and actions.shape[1:] == (50, 14)
             and actions.shape[0] > 0 and bool(torch.isfinite(actions).all()),
             'Finite FP32 full native50 action batch required')


def native_grips_to_physical(actions, *, task, codec):
    """Apply the source-declared native route before contact loss/features."""
    _validate_actions(actions, task, codec)
    native = actions[:, :, GRIPS]
    if codec == DRAWER:
        return native.clamp(0.0, 1.0) * 0.05
    return native.clamp(0.0, 0.05)


def apply_physical_grips(actions, physical_grips, *, task, codec):
    """Return new contact commands in the unchanged downstream action ABI."""
    import torch
    _validate_actions(actions, task, codec)
    _require(isinstance(physical_grips, torch.Tensor)
             and physical_grips.dtype == actions.dtype
             and physical_grips.device == actions.device
             and physical_grips.shape == (actions.shape[0], 50, 2)
             and bool(torch.isfinite(physical_grips).all())
             and bool(((physical_grips >= 0.0) & (physical_grips <= 0.05)).all()),
             'Contact output must be bounded physical metres on the same device')
    output = actions.clone()
    output[:, :, GRIPS] = physical_grips / 0.05 if codec == DRAWER else physical_grips
    _require(torch.equal(output[:, :, ARMS], actions[:, :, ARMS]),
             'A contact-only stage must preserve every native arm command')
    return output
