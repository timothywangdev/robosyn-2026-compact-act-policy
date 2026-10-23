"""Read-only actual ACT layer4 hook with a fresh per-replan identity."""
from .contracts import CAMERAS, require


class CurrentACTFeatures:
    def __init__(self, backbone, camera_order):
        require(tuple(camera_order) == tuple('observation.images.' + c for c in CAMERAS),
                'Original ACT camera order high/right/left required')
        self.rows = []; self.active = None
        self.handle = backbone.register_forward_hook(self.capture)

    def begin(self, *, task, episode, action, input_sha256, batch_size):
        require(self.active is None, 'Previous current-replan capture not consumed')
        require(type(action) is int and action >= 0 and type(batch_size) is int and batch_size > 0,
                'Actual current action/batch required')
        self.rows = []; self.active = dict(task=task, episode=episode, action=action,
            input_sha256=input_sha256, batch_size=batch_size)

    def capture(self, _backbone, _inputs, output):
        import torch
        import torch.nn.functional as F
        require(self.active is not None and len(self.rows) < 3, 'Only three current camera callbacks')
        require(isinstance(output, dict) and 'feature_map' in output, 'Actual layer4 feature map required')
        value = output['feature_map']
        require(value.ndim == 4 and value.shape[:2] == (self.active['batch_size'], 512)
                and value.shape[2] >= 2 and value.shape[3] >= 3
                and value.dtype == torch.float32 and bool(torch.isfinite(value).all()), 'Finite native FP32 layer4[B,512,H,W]')
        # Neither the output nor the actor's tensor is modified or retained.
        detached = value.detach()
        self.rows.append((F.adaptive_avg_pool2d(detached, (2, 3)).clone(),
                          detached.mean((-2, -1)).clone()))

    def finish(self, expected_identity):
        import torch
        require(self.active == expected_identity and len(self.rows) == 3, 'Exact fresh current replan identity/three cameras')
        spatial = torch.stack([r[0] for r in self.rows], dim=1)
        global_mean = torch.stack([r[1] for r in self.rows], dim=1)
        self.rows = []; self.active = None
        return spatial, global_mean

    def reset(self):
        self.rows = []; self.active = None

    def close(self):
        self.handle.remove(); self.reset()
