"""Recover current sensor bytes through the original ACT device operation."""
import hashlib
import numpy as np
from .contracts import CAMERAS, require


def current_inputs(batch, state_evidence):
    import torch
    keys = {'observation.state', *('observation.images.'+c for c in CAMERAS)}
    require(set(batch) == keys, 'Only current ACT RGB/state batch permitted')
    require(state_evidence['conversion'] == 'identity; source-declared state grippers already normalized',
            'Source-qualified current state conversion required')
    state = batch['observation.state']
    require(isinstance(state, torch.Tensor) and state.dtype == torch.float32
            and state.shape == (1,14) and bool(torch.isfinite(state).all()), 'Original finite current ACT state required')
    observer = {'qpos': state[0].detach().cpu().numpy().copy()}
    proof = {'current_RGB_only': True, 'ACT_inputs_changed': False,
             'roundtrip_operation': 'uint8.to(original_ACT_device,float32) / 255.0', 'images': {}}
    for camera in CAMERAS:
        value = batch['observation.images.'+camera]
        require(isinstance(value, torch.Tensor) and value.dtype == torch.float32
                and value.shape == (1,3,480,640) and bool(torch.isfinite(value).all())
                and bool(((value >= 0) & (value <= 1)).all()), 'Original current normalized ACT RGB640 required')
        recovered = torch.round(value.detach() * 255.0).to(torch.uint8)
        # CUDA scalar division and NumPy CPU division differ in their final bit
        # for some byte values. Validate using the exact released ACT operation
        # on the original device; do not alter motor tensors or use a tolerance.
        reconstructed = recovered.to(device=value.device, dtype=torch.float32) / 255.0
        require(torch.equal(value, reconstructed), 'Current RGB is not an exact sensor-byte ACT normalization')
        image = recovered[0].permute(1,2,0).contiguous().cpu().numpy().copy()
        require(image.dtype == np.uint8 and image.shape == (480,640,3), 'Exact current RGB sensor bytes required')
        observer[camera] = image
        proof['images'][camera] = {'sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                                  'source_device': str(value.device), 'exact_roundtrip': True}
    return observer, proof
