"""New temporal-arm science; no gripper checkpoint or schema promotion."""
import math
from .contracts import require
from experiments.goal_latent_act_sequence_runtime_full.semantics import HEAD_SCHEMA, STATS, validate_head

ARM_SCHEMA='goal-latent-act-full63000-current-role-goal-temporal-arm12-v1'
PAIR_SCHEMA='goal-latent-act-full63000-temporal-arm12-qualified-training-pair-v1'
HEAD_WEIGHTS='dabf15d49de48c60fae91905c85ee076d98daed9cf5571ed3cce6a03d5927bc4'
HEAD_CONTRACT='94b0da439d115851e6cc269113ac761236652f131161a4e11fa56b838f236210'


def same_bytes(left,right):
    return all(left[k]==right[k] for k in ('bytes','sha256'))


def validate_pair(head,arm,cfg):
    from experiments.goal_latent_act_arm_sequence_full.contracts import validate
    validate_head(head,cfg);validate(arm)
    require(cfg['mode'] in ('goal','goal_free') and arm['schema']==ARM_SCHEMA,
            'Only actual newly trained temporal-arm branches required')
    require(cfg['assets']['highlevel_weights']['sha256']==HEAD_WEIGHTS
            and cfg['assets']['highlevel_contract']['sha256']==HEAD_CONTRACT
            and same_bytes(arm['highlevel_weights'],cfg['assets']['highlevel_weights'])
            and same_bytes(arm['highlevel_contract'],cfg['assets']['highlevel_contract'])
            and arm['weights_sha256']==cfg['assets']['arm_weights']['sha256'],
            'Exact actual full command50 head and temporal-arm weights required')
    require(same_bytes(arm['geometry'],cfg['assets']['geometry'])
            and same_bytes(arm['urdf'],cfg['assets']['urdf'])
            and cfg['native_grippers_bitexact'] is arm['native_grippers_bitexact'] is True
            and cfg['arm_correction_bound_rad']==arm['model_config']['max_correction_rad']==.15,
            'Same physical FK and bounded arm-only correction required')
    require(cfg['task'] in arm['tasks'] and len(set(arm['tasks']))==10
            and arm['state_action_only']==(cfg['mode']=='goal_free')
            and arm['actor_inputs']==['current_roles','current_qpos','current_root_FK','predicted_command50_goal','native_ACT50'],
            'Current-only same-task causal arm ABI required')
    for key,width in STATS.items():
        values=arm['statistics'][key]
        require(len(values)==width and all(math.isfinite(x) for x in values),'Finite optimization-FIT moments required')
        if key.endswith('scale'):require(all(x>0 for x in values),'Positive optimization-FIT scales required')
    for source,digest in {**head['source_pins'],**arm['source_pins']}.items():
        require(cfg['source_sha256'].get(source)==digest,'Exact actual scientific source pin required: '+source)
