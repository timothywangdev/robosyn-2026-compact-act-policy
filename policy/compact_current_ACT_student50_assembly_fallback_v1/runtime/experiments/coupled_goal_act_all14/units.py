"""One explicit native-to-metres conversion; no task/phase labels enter the model."""
import torch
from .common import require

ARM = (0,1,2,3,4,5,7,8,9,10,11,12)
GRIP = (6,13)
LOWER6 = (-2.618,0.,-2.967,-1.745,-1.22,-2.0944)
UPPER6 = (2.618,3.14,0.,1.745,1.22,2.0944)


def scale_for_tasks(task_indices, tasks):
    require(task_indices.dtype==torch.long and len(tasks)==10 and len(set(tasks))==10,
            'Exact declared task codec indices required')
    require(bool(((task_indices>=0)&(task_indices<len(tasks))).all()), 'Unknown task codec')
    # Codec metadata, not a learned task index feature. The expert targets are
    # already physically converted upstream; this applies ONLY to native ACT.
    drawer = task_indices == tasks.index('drawer_open_place')
    return torch.where(drawer, .05, 1.).to(torch.float32)


def native_to_physical(native, grip_scale):
    require(native.ndim==3 and native.shape[1:]==(50,14) and native.dtype==torch.float32,
            'Native generated50 FP32 arm-major14 required')
    require(grip_scale.shape==(len(native),) and grip_scale.dtype==native.dtype and grip_scale.device==native.device,
            'Explicit source codec scale on same device required')
    require(bool(((grip_scale==.05)|(grip_scale==1.)).all()), 'Only exact Drawer/identity codecs allowed')
    physical = native.clone()
    # This exactly implements the existing source route: Drawer clamp01*.05;
    # other native grip values clamp0..05, arm coordinates stay unchanged.
    limit = .05/grip_scale
    physical[...,list(GRIP)] = torch.minimum(native[...,list(GRIP)].clamp_min(0),limit[:,None,None])*grip_scale[:,None,None]
    return physical


def physical_to_native(physical, grip_scale):
    result = physical.clone()
    result[...,list(GRIP)] = physical[...,list(GRIP)]/grip_scale[:,None,None]
    return result


def hand_atomic(candidate_native, original_native, accepted):
    require(accepted.shape==(len(original_native),2) and accepted.dtype==torch.bool,
            'Exactly one decision for each complete seven-channel hand required')
    candidate = candidate_native.reshape(-1,50,2,7)
    original = original_native.reshape_as(candidate)
    return torch.where(accepted[:,None,:,None],candidate,original).reshape_as(original_native)
