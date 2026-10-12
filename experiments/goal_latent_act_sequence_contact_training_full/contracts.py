"""Full63000 source checkpoint type; paired with the NEW optimization-only head."""
import json
from pathlib import Path
from safetensors.torch import load_file
from experiments.goal_latent_act.prepare import sha
from experiments.goal_latent_act_campaign.common import require
from experiments.goal_latent_act_command_goal_training_full.contracts import HEAD
from .model import SCHEMA,SequencePhysicalOpening

SELECTION='fixed_final30_full63000_sequence_final20_FIT_gate; MONITOR_reporting_only'

def validate(c):
    require(c['schema']==SCHEMA and c['selection']==SELECTION,'Exact separately typed full sequence required')
    require(c['arms_unchanged'] is True and c['output_units']=='physical_joint_opening_meters' and c['opening_bounds_m']==[0,.05],'Native arms and physical opening bounds required')
    require(c['use_future_labels_at_inference'] is False and c['coordinate_frame']=='robot_root_meters','Current-only root-frame features required')
    require(c['selected_epoch']==30 and c['gate_epochs']==20 and c['gate_threshold']==.5,'Fixed final30/20 gate required')
    require(c['highlevel_command_target_offset']==50 and c['highlevel_schema']==HEAD,'NEW full command50 head binding required')
    require(c['optimizer_updates']==11820 and c['gate_updates']==1000 and c['seed']==1701 and c['batch_size']==128,'Exact full matched budgets required')
    require(c['native_chunk_generated']==c['prefix_executed']==50 and c['MONITOR_selection'] is False,'Native50 and report-only MONITOR required')
    require(c['control_zero_channels']==(['roles','predicted_goal_delta'] if c['state_action_only'] else []),'State-only control must remove both role-derived channels')
    require((c['source_FIT'],c['source_MONITOR'],c['optimizer_FIT'],c['diagnostic_FIT'],c['gate_FIT'])==(63000,7000,50400,6300,6300),'Full source and explicit partition counts required')
    require(c['normalization_scope']=='optimizer FIT episodes%10<8 only; shared with high-level head','Hierarchy normalization may not consume diagnostic/gate/MONITOR')

def load_checkpoint(root,expected=None):
    root=Path(root);c=json.loads((root/'checkpoint.json').read_text());validate(c)
    require(sha(root/'opening.safetensors')==c['weights_sha256'],'Exact full sequence weights required')
    require(expected is None or expected==c['weights_sha256'],'Requested full sequence SHA differs')
    m=SequencePhysicalOpening(**c['model_config']);m.load_state_dict(load_file(str(root/'opening.safetensors')),strict=True)
    return m.requires_grad_(False).eval(),c
