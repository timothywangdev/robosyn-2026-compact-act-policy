"""Small CPU-only validation shared by the runner and both runtime entrypoints."""
import hashlib
import math
from pathlib import Path


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_training_proof(step, config, proof):
    """Accept trained expert artifacts or strictly evidenced frozen-base bridges."""
    completed = proof.get('completed_step')
    if type(completed) is not int or completed <= 0 or proof.get('frozen_unchanged') is not True:
        raise ValueError('A completed, genuinely trained checkpoint is required')
    change = proof.get('base_max_change')
    if proof.get('freeze_base', False) is not True:
        if proof.get('training_scope') == 'bridge_only' or not _finite(change) or change <= 0:
            raise ValueError('Updated expert parameters required for an expert-training artifact')
        return 'expert_and_bridge' if config.get('mode') != 'native' else 'expert_only'
    if config.get('mode') not in ('structured', 'unstructured') or config.get('bridge_used') is not True:
        raise ValueError('Frozen native training is a no-op and is not deployable')
    if proof.get('mode') != config['mode']:
        raise ValueError('Trained bridge mode differs from deployment mode')
    if proof.get('training_scope') != 'bridge_only' or proof.get('base_trainable_count') != 0 or proof.get('base_trainable_names') != []:
        raise ValueError('Frozen-base optimizer scope differs')
    if not _finite(change) or change != 0 or proof.get('all_base_state_unchanged') is not True:
        raise ValueError('All frozen base parameters and buffers must be unchanged')
    before, after = proof.get('base_state_sha256_before'), proof.get('base_state_sha256_after')
    if not _sha(before) or before != after:
        raise ValueError('Complete frozen-base state hash mismatch')
    initial, final = proof.get('bridge_state_sha256_before'), proof.get('bridge_state_sha256_after')
    if not _sha(initial) or not _sha(final) or initial == final:
        raise ValueError('Bridge state must actually change')
    for field in ('bridge_max_change', 'motor_projection_max_abs'):
        if not _finite(proof.get(field)) or proof[field] <= 0:
            raise ValueError('Trained nonzero bridge conditioning required')
    if not _finite(proof.get('motor_projection_initial_max_abs')) or proof['motor_projection_initial_max_abs'] != 0:
        raise ValueError('Bridge motor projection must begin at the native identity')
    step = Path(step)
    if not _sha(proof.get('bridge_safetensors_sha256')) or file_sha256(step / 'bridge.safetensors') != proof['bridge_safetensors_sha256']:
        raise ValueError('Saved bridge bytes differ from training proof')
    files = proof.get('immutable_base_files_sha256')
    if not isinstance(files, dict) or not {'model.safetensors', 'config.json'} <= set(files):
        raise ValueError('Pinned immutable checkpoint files required')
    if files['model.safetensors'] != proof.get('base_sha256'):
        raise ValueError('Exported base does not match the starting checkpoint')
    for name, expected in files.items():
        if Path(name).name != name or not _sha(expected) or file_sha256(step / 'policy' / name) != expected:
            raise ValueError('Exported frozen checkpoint bytes changed')
    return 'bridge_only'
