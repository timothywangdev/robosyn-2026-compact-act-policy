"""New arm-only native50 consumer; no older checkpoint may be relabeled."""
import json
from pathlib import Path
from safetensors.torch import load_file
from .common import MODEL, require, sha, verify_sources, validate_protocol
from .model import GoalArmSequence


def validate(c):
    require(c['schema']==MODEL and c['model_config']==dict(hidden=64,role_projection=32,max_correction_rad=.15), 'New typed temporal-arm checkpoint required')
    require(c['native_chunk_generated']==c['prefix_executed']==50 and c['native_grippers_bitexact'] is True, 'Declared native50/exact native grips required')
    require(c['source_counts']==dict(FIT=63000,MONITOR=7000) and c['partitions']==dict(optimizer_FIT=50400,diagnostic_FIT=6300,gate_FIT=6300,MONITOR=7000), 'Independent full source partitions required')
    require((c['selected_epoch'],c['gate_epochs'],c['optimizer_updates'],c['gate_updates'],c['batch_size'],c['seed'])==(30,20,11820,1000,128,1701), 'Actual fixed matched budgets required')
    require(c['MONITOR_selection'] is False and c['use_future_labels_at_inference'] is False and c['coordinate_frame']=='robot_root_meters', 'Current-only root-frame/report-only monitor required')
    require(c['state_action_only'] in (True,False) and c['control_zero_channels']==(['roles','predicted_goal_delta'] if c['state_action_only'] else []), 'Control must remove both foundation-derived channels')
    require(c['command_goal_offset']==50 and c['gate_threshold']==.5, 'Bounded fixed command50 interface required')
    validate_protocol(c['protocol']);verify_sources(c['source_pins'])


def load_checkpoint(root, expected=None, device='cpu'):
    root=Path(root);c=json.loads((root/'checkpoint.json').read_text());validate(c)
    require(sha(root/'arm-sequence.safetensors')==c['weights_sha256'] and (expected is None or expected==c['weights_sha256']), 'Exact temporal arm weights required')
    model=GoalArmSequence(**c['model_config']);model.load_state_dict(load_file(str(root/'arm-sequence.safetensors')),strict=True)
    return model.eval().requires_grad_(False).to(device),c
