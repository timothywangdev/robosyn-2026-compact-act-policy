"""New coupled weights cannot be confused with arm-only or contact-only models."""
import json
from pathlib import Path
from safetensors.torch import load_file
from .common import MODEL,CONFIG,PARTITIONS,require,sha,verify_sources,validate_protocol
from .model import CoupledGoalACT


def validate(c):
    require(c['schema']==MODEL and c['model_config']==CONFIG,'New coupled all14 checkpoint required')
    require(c['partitions']==PARTITIONS and c['source_counts']==dict(FIT=63000,MONITOR=7000),'Exact full source split required')
    require((c['selected_epoch'],c['gate_epochs'],c['optimizer_updates'],c['gate_updates'],c['batch_size'],c['seed'])==(30,20,11820,1000,128,1701),'Actual matched fixed budgets required')
    require(c['state_action_only'] in (True,False) and c['control_zero_channels']==(['roles','predicted_goal_delta'] if c['state_action_only'] else []),'Foundation intervention must match training')
    require(c['coordinate_frame']=='robot_root_meters' and c['MONITOR_selection'] is False and c['use_future_labels_at_inference'] is False,'Current-only root-frame/report-only monitor required')
    require((c['native_chunk_generated'],c['prefix_executed'])==(50,10) and c['gate_threshold']==.5 and c['hand_atomic_all7'] is True,'Matched50/10 complete-hand gate required')
    require(c['native_grippers_bitexact'] is False and c['rejected_hand_all7_bitexact'] is True,'This changes accepted grippers; no relabeling ARM12 allowed')
    validate_protocol(c['protocol']);verify_sources(c['source_pins'])


def load_checkpoint(root,expected=None,device='cpu'):
    root=Path(root);c=json.loads((root/'checkpoint.json').read_text());validate(c)
    require(sha(root/'coupled-all14.safetensors')==c['weights_sha256'] and (expected is None or expected==c['weights_sha256']),'Exact coupled weights required')
    model=CoupledGoalACT(**c['model_config']);model.load_state_dict(load_file(str(root/'coupled-all14.safetensors')),strict=True)
    return model.eval().requires_grad_(False).to(device),c
