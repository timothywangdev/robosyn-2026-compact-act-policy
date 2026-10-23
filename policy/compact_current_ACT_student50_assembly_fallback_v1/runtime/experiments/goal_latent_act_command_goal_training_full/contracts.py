"""Strict full-source head ABI; current features and literal command+50 FK."""
import json
from pathlib import Path
from safetensors.torch import load_file
from experiments.goal_latent_act.model import RolePhysicalGoal
from experiments.goal_latent_act.prepare import sha
from experiments.goal_latent_act_campaign.common import require

HEAD='goal-latent-act-commanded50-full63000-head-v1'
DATA='goal-latent-act-commanded50-full63000-FK-data-v1'
SELECTION='fixed_predeclared_final_epoch; MONITOR_reporting_only'

def validate_head(c):
    require(c['schema']==HEAD and c['selection']==SELECTION,'Exact new full head schema/final selection required')
    require(c['coordinate_frame']=='robot_root_meters' and c['use_future_labels_at_inference'] is False,'Current-only root-frame ABI required')
    require(c['command_target_index']==49 and c['command_target_frame_offset']==50,'Literal recorded command+50 required')
    require(c['target_semantics']=='qualified_FK_of_recorded_arm_joint_command_NOT_measured_future_EEF','Command FK is not measured future EEF')
    require(c['selected_epoch']==50 and c['seed']==1701 and c['batch_size']==256,'Fixed head budget required')
    require(c['source_FIT']==63000 and c['source_MONITOR']==7000,'Full source counts required')
    require(c['normalization_scope']=='terminal-valid optimizer FIT episodes%10<8 only','Independent head normalization scope required')
    require(c['statistics']['schema']=='command-goal-full-optimization-FIT-current-zscore-v1','Optimization-only statistics required')
    require(c['model_config']['eef_translation_m']==.60 and c['model_config']['eef_rotation_rad']==2.,'Unchanged physical bounds required')
    require(c['optimizer_updates']==((c['FIT_rows']+255)//256)*50 and c['examples_seen']==c['FIT_rows']*50,'Actual fixed valid-row budget required')

def load_head(root,expected_weights_sha=None):
    root=Path(root);c=json.loads((root/'checkpoint.json').read_text());validate_head(c)
    require(sha(root/'highlevel.safetensors')==c['weights_sha256'],'Exact full head weights required')
    require(expected_weights_sha is None or expected_weights_sha==c['weights_sha256'],'Requested full head SHA differs')
    model=RolePhysicalGoal(**c['model_config']);model.load_state_dict(load_file(str(root/'highlevel.safetensors')),strict=True)
    return model.requires_grad_(False).eval(),c
