"""Explicit source closure and atomic, durable public experiment records."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREP = 'goal-latent-act-full63000-temporal-arm12-preparation-v1'
MODEL = 'goal-latent-act-full63000-current-role-goal-temporal-arm12-v1'
PAIR = 'goal-latent-act-full63000-temporal-arm12-original-GPU-pair-v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def ref(path):
    p = Path(path).absolute()
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))


def verify(value):
    p = Path(value['path'])
    require(p.is_absolute() and p.is_file() and not p.is_symlink(), 'Regular bound source required')
    require(p.stat().st_size == value['bytes'] and sha(p) == value['sha256'], 'Bound source changed: '+str(p))
    return p


def write(path, value):
    p = Path(path)
    temp = p.with_suffix(p.suffix+'.tmp')
    with temp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    temp.replace(p)
    fd = os.open(p.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def sources():
    names = ['experiments/goal_latent_act_full_data/correction_prefix.py',
             'experiments/goal_latent_act_full_data/run_training_pair.py',
             'experiments/goal_latent_act_command_goal_training_full/contracts.py',
             'experiments/goal_latent_act_command_goal_training_full/train_head.py',
             'experiments/goal_latent_act_command_goal_training_full/data.py',
             'experiments/goal_latent_act/model.py', 'experiments/goal_latent_act/prepare.py',
             'experiments/goal_latent_act/train.py']
    names += [str(p.relative_to(ROOT)) for p in sorted(Path(__file__).parent.glob('*.py'))]
    return {name: sha(ROOT/name) for name in names}


def verify_sources(pins):
    for name, digest in pins.items():
        p = (ROOT/name).resolve(strict=True)
        require(p.is_relative_to(ROOT) and sha(p) == digest, 'Frozen experiment source changed: '+name)


def validate_protocol(p):
    require(p['schema'] == 'goal-latent-act-full63000-temporal-arm12-preregistered-v1', 'New arm protocol required')
    require(p['source_counts'] == dict(FIT=63000, MONITOR=7000), 'Full source required')
    require(p['partitions'] == dict(optimizer_FIT=50400, diagnostic_FIT=6300, gate_FIT=6300, MONITOR=7000), 'Independent episode-level partitions required')
    require(p['model'] == dict(hidden=64, role_projection=32, max_correction_rad=.15), 'Fixed temporal architecture/bound required')
    require((p['sequence_epochs'], p['gate_epochs'], p['seed'], p['batch_size']) == (30,20,1701,128), 'Fixed matched budgets required')
    require(p['FK_indices'] == [0,9,31,49] and p['native_generated'] == p['native_executed'] == 50, 'Explicit native50/FK anchor scope required')
    require(p['control_zero_channels'] == ['roles','predicted_goal_delta'] and p['MONITOR_selection'] is False, 'Matched current-only intervention/report-only monitor required')
    require(p['selection'] == 'fixed_final_epoch_no_MONITOR_selection' and p['native_grippers_bitexact'] is True, 'Fixed final epoch and exact native grips required')
    require(p['loss'] == dict(arm_smooth_l1_scale_rad=.1, position_smooth_l1_scale_m=.02, position_weight=.25, rotation_matrix_MSE_weight=.1), 'Predeclared joint/FK objective changed')
    require(p['gate_rule'] == dict(joint_improvement_rad=.001,position_improvement_m=.0001,rotation_regression_tolerance_rad=.001,threshold=.5,handwise=True), 'Predeclared handwise physical gate changed')
    return p
