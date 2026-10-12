"""Distinct typed experiment, immutable lineage, and fixed prospective budgets."""
from pathlib import Path
from experiments.goal_latent_act_arm_sequence_full.common import require, ref, verify, write, sha

ROOT = Path(__file__).resolve().parents[2]
PREP = 'coupled-goal-ACT-all14-source-qualified-preparation-v1'
MODEL = 'coupled-goal-ACT-all14-current-role-goal-GRU64-v1'
PARTITIONS = dict(optimizer_FIT=50400, diagnostic_FIT=6300, gate_FIT=6300, MONITOR=7000)
CONFIG = dict(hidden=64, role_projection=32, max_correction_rad=.15, max_opening_correction_m=.05)
ARM_GATE = dict(joint_improvement_rad=.001, position_improvement_m=.0001,
                rotation_regression_tolerance_rad=.001, opening_regression_tolerance_m=.0001,
                threshold=.5, hand_atomic_all7=True)
LOSS = dict(arm_smooth_l1_scale_rad=.1, position_smooth_l1_scale_m=.02,
            position_weight=.25, rotation_matrix_MSE_weight=.1,
            opening_smooth_l1_scale_m=.05, opening_weight=1.,
            opening_increment_smooth_l1_scale_m=.01, opening_increment_weight=.1)


def sources():
    names = [str(p.relative_to(ROOT)) for p in sorted(Path(__file__).parent.glob('*.py'))]
    names += ['experiments/goal_latent_act_arm_sequence_full/fk.py',
              'experiments/goal_latent_act_arm_sequence_full/common.py',
              'experiments/goal_latent_act_arm_sequence_full/prepare.py',
              'experiments/goal_latent_act_arm_sequence_full/train.py',
              'experiments/goal_latent_act_full_data/correction_prefix.py',
              'experiments/goal_latent_act_contact_runtime/gripper_boundary.py',
              'policy/wam_adaptation/multitask_data.py']
    return {name: sha(ROOT/name) for name in names}


def verify_sources(pins):
    for name, digest in pins.items():
        path = (ROOT/name).resolve(strict=True)
        require(path.is_relative_to(ROOT) and sha(path) == digest, 'Frozen source changed: '+name)


def validate_protocol(p):
    require(p['schema']=='coupled-goal-ACT-all14-preregistered-v1', 'Distinct all14 protocol required')
    require(p['source_counts']==dict(FIT=63000, MONITOR=7000) and p['partitions']==PARTITIONS, 'Exact independent 70k partitions required')
    require(p['model']==CONFIG and p['loss']==LOSS and p['gate_rule']==ARM_GATE, 'Prospective model/loss/gate changed')
    require((p['sequence_epochs'],p['gate_epochs'],p['seed'],p['batch_size'])==(30,20,1701,128), 'Fixed matched budgets required')
    require(p['FK_indices']==[0,9,31,49] and (p['native_generated'],p['native_executed'])==(50,10), 'Matched generated50/executed10 required')
    require(p['control_zero_channels']==['roles','predicted_goal_delta'] and p['MONITOR_selection'] is False, 'Exact causal matched control required')
    require(p['selection']=='fixed_final_epoch_no_MONITOR_selection', 'Fixed final selection required')
    require(p['units']['target_grips']=='already source-converted physical metres; never decoded twice', 'Explicit teacher units required')
    return p
