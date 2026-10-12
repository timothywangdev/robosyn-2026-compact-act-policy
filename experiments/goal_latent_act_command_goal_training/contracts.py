"""Typed loaders reuse unchanged physical models without legacy relabeling."""
import json
from pathlib import Path
from safetensors.torch import load_file
from experiments.goal_latent_act.model import RolePhysicalGoal
from experiments.goal_latent_act_full_data.correction_prefix import GoalArmCorrectionPrefix
from experiments.goal_latent_act.prepare import sha
from experiments.goal_latent_act_campaign.common import require

HEAD='goal-latent-act-commanded-FK-EEF-head-v1'
MOTOR='goal-latent-act-commanded-FK-native50-motor-v1'
DATA='goal-latent-act-commanded-FK-matched-data-v1'
SELECTION='fixed_predeclared_final_epoch; MONITOR_reporting_only'


def _load(root,schema,filename,expected):
 root=Path(root);contract=json.loads((root/'checkpoint.json').read_text())
 require(contract['schema']==schema and sha(root/filename)==contract['weights_sha256'],'Exact separately typed weights required')
 require(expected is None or expected==contract['weights_sha256'],'Requested checkpoint SHA differs')
 require(contract['selection']==SELECTION and contract['coordinate_frame']=='robot_root_meters','Fixed final and robot-root contract required')
 require(contract['use_future_labels_at_inference'] is False,'Current-only inference required')
 return root,contract


def load_head(root,expected_weights_sha=None):
 root,c=_load(root,HEAD,'highlevel.safetensors',expected_weights_sha)
 require(c['command_target_index'] in (9,49) and c['command_target_frame_offset']==c['command_target_index']+1,
         'Literal next-command horizon required')
 require(c['target_semantics']=='qualified_FK_of_recorded_arm_joint_command_NOT_measured_future_EEF','Commanded target semantics differ')
 require(c['model_config']['eef_translation_m']==.60 and c['model_config']['eef_rotation_rad']==2.,'Predeclared FIT-only bounds required')
 model=RolePhysicalGoal(**c['model_config']);model.load_state_dict(load_file(str(root/'highlevel.safetensors')),strict=True)
 return model.requires_grad_(False).eval(),c


def load_motor(root,expected_weights_sha=None):
 root,c=_load(root,MOTOR,'correction.safetensors',expected_weights_sha)
 require(c['native_chunk_generated']==c['prefix_executed']==c['model_config']['execution_prefix']==50 and c['discarded']==0,
         'Native generated50/executed50 required')
 require(c['residual_loss_prefix']==c['gate_labels_prefix']==c['gate_summary_prefix']==50 and c['grippers_unchanged'] is True,
         'All arm/gate training boundaries50 plus bitexact native grips required')
 require(c['command_target_frame_offset'] in (10,50) and c['model_config']['max_correction_rad']==.15,'Typed horizon and bounded correction required')
 model=GoalArmCorrectionPrefix(**c['model_config']);model.load_state_dict(load_file(str(root/'correction.safetensors')),strict=True)
 return model.requires_grad_(False).eval(),c
