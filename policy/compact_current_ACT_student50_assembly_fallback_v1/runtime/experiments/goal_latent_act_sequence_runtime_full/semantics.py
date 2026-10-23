"""Full63000 scientific consumer; old contracts cannot become new weights."""
import math
from .contracts import require

HEAD_SCHEMA = 'goal-latent-act-commanded50-full63000-head-v1'
CONTACT_SCHEMA = 'goal-latent-act-current-role-sequence-opening50-full63000-v1'
PAIR_SCHEMA = 'goal-latent-act-full63000-qualified-training-pair-v1'
TARGET = 'qualified_FK_of_recorded_arm_joint_command_NOT_measured_future_EEF'
STATS = {'qpos_mean':14,'qpos_scale':14,'base_mean':14,'base_scale':14,
         'current_eef_mean':18,'current_eef_scale':18,'eef_scale':18}


def validate_head(head, cfg):
    from experiments.goal_latent_act_command_goal_training_full.contracts import validate_head as scientific
    scientific(head)
    require(head['schema'] == HEAD_SCHEMA
            and head['weights_sha256'] == cfg['assets']['highlevel_weights']['sha256'],
            'NEW actual full63000 head weights required')
    require(head['command_target_frame_offset'] == cfg['command_target_frame_offset'] == 50
            and head['target_semantics'] == cfg['target_semantics'] == TARGET
            and head['coordinate_frame'] == cfg['coordinate_frame'] == 'robot_root_meters',
            'Literal commanded+50 FK differs; future measured labels prohibited')
    require(head['context_version'] == 'qwen35-canonical-current-role-hidden-v2'
            and head['feature_shape'] == [3,4096] and head['trained_outputs'] == ['dual_eef_goal_pose']
            and head['geometry_sha256'] == cfg['assets']['geometry']['sha256']
            and head['urdf_sha256'] == cfg['assets']['urdf']['sha256'],
            'Unchanged current canonical Qwen/physical FK ABI required')
    require(head['actor_inputs'] == ['current canonical Qwen roles','current canonical qpos','measured-current FK'],
            'Head may consume only current observation features')


def validate_pair(head, contact, cfg):
    from experiments.goal_latent_act_sequence_contact_training_full.contracts import validate as scientific
    validate_head(head, cfg); scientific(contact)
    require(cfg['mode'] in ('goal','goal_free'), 'Only actual paired role/state branches')
    require(contact['schema'] == CONTACT_SCHEMA
            and contact['weights_sha256'] == cfg['assets']['contact_weights']['sha256']
            and contact['highlevel_weights_sha256'] == cfg['assets']['highlevel_weights']['sha256']
            and contact['highlevel_contract_sha256'] == cfg['assets']['highlevel_contract']['sha256'],
            'Exact NEW full63000 head/sequence pairing required')
    require(contact['highlevel_context_version'] == head['context_version']
            and contact['source_data'] == head['data_receipt']
            and contact['model_config'] == {'hidden':64,'role_projection':32}
            and contact['arms_unchanged'] is cfg['arms_unchanged'] is True
            and contact['output_units'] == cfg['contact_output_units'] == 'physical_joint_opening_meters'
            and contact['native_chunk_generated'] == cfg['native_chunk_generated'] == cfg['prefix_executed'] == 50,
            'Same full source, physical opening, untouched arms and native50 required')
    require(cfg['task'] in contact['tasks'] and len(set(contact['tasks'])) == 10
            and contact['state_action_only'] == (cfg['mode']=='goal_free')
            and contact['control_zero_channels'] == (['roles','predicted_goal_delta'] if cfg['mode']=='goal_free' else []),
            'State/action control removes roles AND role-derived goal')
    require(contact['actor_inputs'] == ['current_roles','measured_current_qpos_and_opening',
            'measured_current_FK','predicted_command50_goal','frozen_native_ACT50']
            and contact['recurrence'] == 'predicted opening anchored measured-current opening; no future teacherforcing',
            'Current-only recurrent prediction ABI required')
    for key, width in STATS.items():
        values = contact['statistics'][key]
        require(len(values)==width and all(math.isfinite(x) for x in values), 'Finite actual optimization FIT statistics')
        if key.endswith('scale'): require(all(x>0 for x in values), 'Positive optimization FIT scales')
    for source, digest in {**head['source_pins'], **contact['source_pins']}.items():
        require(cfg['source_sha256'].get(source)==digest, 'Actual full scientific source changed: '+source)
