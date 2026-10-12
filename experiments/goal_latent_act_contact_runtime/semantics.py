"""A new sequence-opening actor; never admit legacy arm or pointwise weights."""
import math
from .contracts import require
from experiments.goal_latent_act_command_goal_runtime.semantics import HEAD_SCHEMA, validate_head

CONTACT_SCHEMA = 'goal-latent-act-current-role-sequence-opening50-v1'
SELECTION = 'fixed_final30_sequence_final20_FIT_gate; MONITOR_reporting_only'
STATS = {'qpos_mean':14,'qpos_scale':14,'base_mean':14,'base_scale':14,
         'current_eef_mean':18,'current_eef_scale':18,'eef_scale':18}


def validate_pair(head, contact, cfg):
    validate_head(head, cfg)
    require(cfg['command_target_frame_offset'] == 50,
            'This sequence was trained with the exact command50 goal')
    require(contact['schema'] == CONTACT_SCHEMA and contact['selection'] == SELECTION,
            'Separately typed fixed-final sequence weights required')
    require(contact['selected_epoch'] == 30 and contact['gate_epochs'] == 20
            and contact['gate_threshold'] == .5 and contact['optimizer_updates'] == 2010
            and contact['gate_updates'] == 180 and contact['MONITOR_selection'] is False,
            'The actual fixed FIT-only budget and reliability gate must agree')
    require(contact['model_config'] == {'hidden':64,'role_projection':32}
            and contact['arms_unchanged'] is True
            and contact['output_units'] == 'physical_joint_opening_meters'
            and contact['opening_bounds_m'] == [0,.05]
            and contact['coordinate_frame'] == cfg['coordinate_frame'] == 'robot_root_meters',
            'Contact-only physical opening and unchanged native arms required')
    require(contact['use_future_labels_at_inference'] is False
            and contact['native_chunk_generated'] == contact['prefix_executed'] == 50
            and contact['highlevel_command_target_offset'] == 50
            and contact['highlevel_weights_sha256'] == cfg['assets']['highlevel_weights']['sha256']
            and contact['highlevel_contract_sha256'] == cfg['assets']['highlevel_contract']['sha256']
            and contact['highlevel_context_version'] == head['context_version']
            and contact['weights_sha256'] == cfg['assets']['contact_weights']['sha256'],
            'Exact current-only trained sequence/head binding required')
    require(cfg['task'] in contact['tasks']
            and contact['state_action_only'] == (cfg['mode'] == 'goal_free')
            and contact['control_zero_channels'] == (['roles','predicted_goal_delta'] if cfg['mode']=='goal_free' else []),
            'The actual role/control branch must match its checkpoint')
    for key, width in STATS.items():
        values = contact['statistics'][key]
        require(len(values) == width and all(math.isfinite(v) for v in values),
                'Finite typed FIT statistics required: '+key)
        if key.endswith('scale'):
            require(all(v > 0 for v in values), 'Strictly positive FIT scale required')
    require(head['geometry_sha256'] == cfg['assets']['geometry']['sha256']
            and head['urdf_sha256'] == cfg['assets']['urdf']['sha256'],
            'Qualified command-head FK identity required')
    require(cfg['arms_unchanged'] is True and cfg['contact_output_units']=='physical_joint_opening_meters',
            'Explicit contact-only deployment behavior required')
    for relative, expected in contact['source_pins'].items():
        require(cfg['source_sha256'].get(relative) == expected,
                'The actually trained sequence source must remain exact: '+relative)
