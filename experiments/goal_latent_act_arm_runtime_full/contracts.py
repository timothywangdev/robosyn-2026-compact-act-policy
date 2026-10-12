"""Explicit deployment bindings; no conversion of older scientific manifests."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

SCHEMA = 'goal-latent-act-commanded50-full63000-temporal-arm12-deployment-v1'
CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify_ref(ref):
    path = Path(ref['path'])
    require(path.is_absolute() and path.is_file() and not path.is_symlink(), 'Regular absolute bound asset required')
    require(path.stat().st_size == ref['bytes'] and sha(path) == ref['sha256'], 'Bound asset changed: ' + str(path))
    return path


def load_manifest(path, expected):
    path = Path(path).resolve(strict=True)
    require(sha(path) == expected, 'Exact deployment manifest required')
    cfg = json.loads(path.read_text())
    require(cfg['schema'] == SCHEMA and cfg['mode'] in ('goal', 'goal_free', 'parent'), 'Only separately typed full63000 paired branches/native parent required')
    require(cfg['native_chunk_generated'] == 50 and cfg['prefix_executed'] == 50, 'Unchanged explicit generated50/executed50 contract')
    require(cfg['motor_seed']==17, 'Actual predeclared Torch motor seed17 required')
    require(cfg['act_core_mode'] == 'eager', 'Explicit ACT core execution mode required')
    require(cfg['act_preprocess_mode'] == 'native_every_action', 'Explicit native/fast ACT preprocessing mode required')
    require(cfg['state_contract'] == 'arm-major14; arms radians; measured grippers normalized opening', 'Canonical measured state required')
    require(cfg['action_contract'] == 'arm-major14; arms radians; frozen ACT grippers native checkpoint units', 'Declared native boundary required')
    require(cfg['gripper_codec'] in ('drawer_normalized_opening_inverse_v1','physical_meters_identity_v1'), 'Explicit native-to-environment gripper codec required')
    require((cfg['task']=='drawer_open_place') == (cfg['gripper_codec']=='drawer_normalized_opening_inverse_v1'), 'Task-specific released ACT gripper contract differs')
    require(cfg['actor_inputs'] == ['current_RGB', 'original_instruction', 'current_qpos', 'current_FK'], 'Only current observations may enter inference')
    require(0 < cfg['worker_timeout_seconds'] <= 600 and 0 < cfg['episode_wall_seconds'] <= 1800, 'Bounded actual deployment required')
    root = Path(cfg['source_root']).resolve(strict=True)
    for relative, digest in cfg['source_sha256'].items():
        p = (root / relative).resolve(strict=True)
        require(p.is_relative_to(root) and sha(p) == digest, 'Runtime source changed: ' + relative)
    required = {'experiments/goal_latent_act_arm_runtime_full/'+name+'.py'
                for name in ('contracts','runtime','observer_worker','semantics','inference','qualification')}
    required.update({'experiments/goal_latent_act_sequence_runtime_full/'+name+'.py'
                    for name in ('current_snapshot','terminal','contracts','semantics')})
    required.update({'experiments/goal_latent_act_contact_runtime/'+name+'.py'
                    for name in ('current_snapshot','terminal','contracts')})
    required.update({'experiments/goal_latent_act_full_data/correction_prefix.py',
        'policy/object_relative_hierarchy/canonical_context.py','policy/relational_hierarchy/qwen_features.py',
        'policy/relational_hierarchy/qwen_contract_worker.py','policy/robosyn_sprint/qwen_planner.py',
        'policy/relational_hierarchy/runtime.py','policy/robosyn_sprint/act_drawer_adapter.py',
        'experiments/goal_latent_act/model.py'})
    require(required <= set(cfg['source_sha256']), 'Exact new arm runtime and unchanged dependencies required')
    for ref in cfg['assets'].values():
        verify_ref(ref)
    if cfg['mode'] != 'parent':
        trained = {'experiments/goal_latent_act_command_goal_training_full/contracts.py',
                   'experiments/goal_latent_act_arm_sequence_full/contracts.py',
                   'experiments/goal_latent_act_arm_sequence_full/model.py',
                   'experiments/goal_latent_act_arm_sequence_full/common.py'}
        require(trained <= set(cfg['source_sha256']), 'NEW full63000 training loader/source pins required')
        from .semantics import validate_pair
        head = json.loads(Path(cfg['assets']['highlevel_contract']['path']).read_text())
        arm = json.loads(Path(cfg['assets']['arm_contract']['path']).read_text())
        validate_pair(head, arm, cfg)
        from .qualification import validate_bound_training
        validate_bound_training(cfg)
    else:
        require(not any(k in cfg['assets'] for k in ('highlevel_weights','highlevel_contract','arm_weights','arm_contract','training_pair','robot_admission')),
                'Parent must use only original frozen native50 ACT; no trained hierarchy assets')
    act = Path(cfg['act_root']).resolve(strict=True)
    require({'model.safetensors', 'config.json'} <= set(cfg['act_files']), 'ACT weights/config inventory required')
    for name, ref in cfg['act_files'].items():
        require((act / name).resolve() == Path(ref['path']).resolve(), 'ACT asset root differs')
        verify_ref(ref)
    cfg['_manifest'] = path
    return cfg


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def load_official(cfg):
    timer = verify_ref(cfg['assets']['inference_timer'])
    present = sys.modules.get('policy.inference_timing')
    if present is None:
        module('policy.inference_timing', timer)
    else:
        require(sha(present.__file__) == sha(timer), 'Existing official timer differs')
    return module('goal_latent_released_ACT', verify_ref(cfg['assets']['official_ACT']))
