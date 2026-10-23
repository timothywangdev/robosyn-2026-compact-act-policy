"""Explicit commanded-FK semantics; never reinterpret measured-pose contracts."""
from .contracts import require

HEAD_SCHEMA = 'goal-latent-act-commanded-FK-EEF-head-v1'
MOTOR_SCHEMA = 'goal-latent-act-commanded-FK-native50-motor-v1'
TARGET = 'qualified_FK_of_recorded_arm_joint_command_NOT_measured_future_EEF'
FRAME = 'robot_root_meters'
SELECTION = 'fixed_predeclared_final_epoch; MONITOR_reporting_only'


def validate_head(contract, cfg):
    offset = cfg['command_target_frame_offset']
    require(type(offset) is int and offset in (10, 50), 'An explicit commanded horizon is required')
    require(contract['schema'] == HEAD_SCHEMA and contract['command_target_frame_offset'] == offset
            and contract['command_target_index'] == offset - 1
            and contract['target_semantics'] == cfg['target_semantics'] == TARGET
            and contract['coordinate_frame'] == cfg['coordinate_frame'] == FRAME,
            'The head predicts qualified commanded FK, not observed future poses')
    require(contract['use_future_labels_at_inference'] is False
            and contract['selection'] == SELECTION and contract['selected_epoch'] == 50,
            'Fixed FIT-trained final head with current-only inference required')
    require(contract['context_version'] == 'qwen35-canonical-current-role-hidden-v2'
            and contract.get('role_zero_control', False) is False
            and contract['trained_outputs'] == ['dual_eef_goal_pose'],
            'The actual current factual Qwen feature ABI is required')
    require(contract['model_config']['eef_translation_m'] == .60
            and contract['model_config']['eef_rotation_rad'] == 2.,
            'The predeclared FIT-derived command bounds must remain unchanged')


def validate_motor(contract, cfg):
    require(contract['schema'] == MOTOR_SCHEMA
            and contract['command_target_frame_offset'] == cfg['command_target_frame_offset']
            and contract['command_target_index'] == cfg['command_target_frame_offset'] - 1
            and contract['target_semantics'] == TARGET and contract['coordinate_frame'] == FRAME,
            'The motor must bind its actual commanded-goal head semantics')
    require(contract['use_future_labels_at_inference'] is False
            and contract['selection'] == SELECTION
            and contract['selected_epoch'] == 30
            and contract['optimizer_updates'] > 0 and contract['gate_updates'] > 0
            and contract['grippers_unchanged'] is True
            and contract['model_config']['max_correction_rad'] == .15,
            'Only bounded FIT-trained arm correction is admitted')
    require(all(contract[k] == 50 for k in ('prefix_executed', 'native_chunk_generated',
                'residual_loss_prefix', 'gate_labels_prefix', 'gate_summary_prefix'))
            and contract['model_config']['execution_prefix'] == 50 and contract['discarded'] == 0,
            'Native50 training, gate and execution must agree')


def validate_pair(head, motor, cfg):
    validate_head(head, cfg)
    validate_motor(motor, cfg)
    require(motor['highlevel_schema'] == HEAD_SCHEMA
            and motor['highlevel_weights_sha256'] == cfg['assets']['highlevel_weights']['sha256']
            and motor['highlevel_checkpoint_sha256'] == cfg['assets']['highlevel_contract']['sha256']
            and motor['highlevel_weights_file'] == 'highlevel.safetensors',
            'The motor must bind the exact actually trained head and contract')
    require(motor['highlevel_model_config'] == head['model_config']
            and motor['highlevel_statistics'] == head['statistics']
            and motor['highlevel_context_version'] == head['context_version']
            and head['geometry_sha256'] == motor['geometry_sha256'] == cfg['assets']['geometry']['sha256']
            and head['urdf_sha256'] == motor['urdf_sha256'] == cfg['assets']['urdf']['sha256'],
            'Current normalizer, head ABI and qualified FK identity must agree')
    require(all(motor['statistics'][k] == head['statistics'][k]
                for k in ('qpos_mean', 'qpos_scale', 'eef_scale')),
            'The head and motor must use the same FIT-only current feature normalizer')
    require(motor['goal_free'] == (cfg['mode'] == 'goal_free')
            and motor['weights_sha256'] == cfg['assets']['correction_weights']['sha256'],
            'The actual trained branch and bound motor weights must agree')
