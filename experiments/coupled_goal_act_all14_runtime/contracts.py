"""Separate all14 deployment evidence; old ARM12 admissions cannot qualify it."""
import json
import math
from pathlib import Path
from experiments.goal_latent_act_arm_prefix10.contracts import CAMERAS, require, sha, verify_ref, module, load_official

SCHEMA = 'coupled-goal-ACT-all14-generated50-executed10-deployment-v1'
ADMISSION = 'root_authorized_coupled_all14_native_pilot_v1'
STATS = {'qpos_mean':14,'qpos_scale':14,'base_mean':14,'base_scale':14,
         'current_eef_mean':18,'current_eef_scale':18,'eef_scale':18}
STATE = 'arm-major14; arms radians; measured grippers normalized opening'
ACTION = 'arm-major14; arms radians; accepted hand opening metres inverted once to native ACT units; rejected whole7 raw ACT bitexact'
INPUTS = ['current_RGB','original_instruction','current_qpos','current_FK']


def same_ref(left, right):
    return all(left[k] == right[k] for k in ('bytes','sha256'))


def validate_training(cfg, contract):
    from experiments.coupled_goal_act_all14.contracts import validate
    validate(contract)
    a = cfg['assets']
    require(contract['weights_sha256'] == a['coupled_weights']['sha256'], 'Exact new all14 weight identity required')
    require(contract['state_action_only'] == (cfg['mode']=='goal_free'), 'Role/state intervention differs from training')
    require(cfg['task'] in contract['tasks'] and len(set(contract['tasks'])) == 10, 'Complete matched all10 training required')
    for key in ('geometry','urdf','highlevel_contract','highlevel_weights'):
        require(same_ref(contract[key],a[key]), 'Original high-level/FK asset differs: '+key)
    for name, width in STATS.items():
        values = contract['statistics'][name]
        require(len(values)==width and all(math.isfinite(x) for x in values), 'Finite optimization-FIT moments required')
        if name.endswith('scale'): require(all(x>0 for x in values), 'Positive optimization-FIT scale required')
    for source, digest in contract['source_pins'].items():
        require(cfg['source_sha256'].get(source)==digest, 'Actual model source changed: '+source)
    receipt = json.loads(verify_ref(a['training_receipt']).read_text())
    require(receipt['status']=='completed_fixed30_coupled_all14_20_independent_FIT_gate'
            and receipt['checkpoint']==contract and receipt['reload_predictions_bitexact'] is True
            and (receipt['optimizer_updates'],receipt['gate_updates'])==(11820,1000), 'Genuine completed new training required')
    pair = json.loads(verify_ref(a['training_pair']).read_text())
    branch = 'state_action_control' if cfg['mode']=='goal_free' else 'role_goal'
    require(pair['schema']=='coupled-goal-ACT-all14-original-GPU-pair-v1'
            and pair['status']=='both_original_GPU_children_exit0_natural_complete'
            and pair['original_supervisor_exit_code']==0 and pair['branches'][branch]==a['training_receipt'],
            'Actual matched role/state pair completion required')
    admission = json.loads(verify_ref(a['runtime_admission']).read_text())
    require(admission['schema']==ADMISSION and admission['execution_authorized'] is True
            and admission['robot_quality_promoted'] is False and admission['mode']==cfg['mode'], 'Separate root-authorized exploratory pilot required')
    for key in ('coupled_weights','coupled_contract','training_receipt','training_pair','original_training_wait','independent_training_drain'):
        require(admission[key]==a[key], 'Root admission evidence differs: '+key)
    wait = json.loads(verify_ref(a['original_training_wait']).read_text())
    require(wait['schema']==admission['training_wait_schema'] and wait['original_child_exit_code']==0
            and wait['child_reaped'] is True, 'Literal successful genuine training wait required')
    # The independently reviewed root admission names the actual terminal ABI;
    # the original raw drain is retained verbatim, never rewritten by this loader.
    drain = json.loads(verify_ref(a['independent_training_drain']).read_text())
    require(drain['schema']==admission['training_drain_schema']
            and admission['independent_training_drain_qualified'] is True, 'Bound independently qualified original drain required')


def load_manifest(path, expected):
    path = Path(path).resolve(strict=True)
    require(sha(path)==expected, 'Exact separately typed all14 manifest required')
    cfg = json.loads(path.read_text())
    require(cfg['schema']==SCHEMA and cfg['mode'] in ('parent','goal','goal_free'), 'Only matched parent/role/state branches permitted')
    require((cfg['native_chunk_generated'],cfg['prefix_executed'],cfg['motor_seed'])==(50,10,17), 'Frozen native50/prefix10/motor17 required')
    require(cfg['state_contract']==STATE and cfg['action_contract']==ACTION and cfg['actor_inputs']==INPUTS, 'Explicit current-state/native action boundary required')
    codec = 'drawer_normalized_opening_inverse_v1' if cfg['task']=='drawer_open_place' else 'physical_meters_identity_v1'
    require(cfg['gripper_codec']==codec and cfg['act_core_mode']=='eager' and cfg['act_preprocess_mode']=='native_every_action', 'Original ACT/native grip codec must remain unchanged')
    require(cfg['correction_compiler_backend'] in ('inductor','aot_eager') and type(cfg['require_compile_speedup']) is bool, 'Declared partial compiler policy required')
    require(0<cfg['worker_timeout_seconds']<=600 and 0<cfg['episode_wall_seconds']<=1800, 'Bounded native runtime required')
    root = Path(cfg['source_root']).resolve(strict=True)
    for relative, digest in cfg['source_sha256'].items():
        p = root/relative
        require(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root) and sha(p)==digest, 'Bound source changed: '+relative)
    required = {'experiments/coupled_goal_act_all14_runtime/'+name+'.py' for name in ('contracts','runtime','observer_worker','policy_shim')}
    required |= {'experiments/goal_latent_act_arm_prefix10/'+name+'.py' for name in ('runtime','contracts','observer_worker')}
    required |= {'experiments/goal_latent_act_sequence_runtime_full/'+name+'.py' for name in ('current_snapshot','terminal','semantics')}
    required |= {'experiments/goal_latent_act_sequence_runtime_30k/current_snapshot.py',
        'policy/relational_hierarchy/runtime.py','policy/relational_hierarchy/qwen_features.py',
        'policy/relational_hierarchy/qwen_contract_worker.py','policy/robosyn_sprint/qwen_planner.py',
        'policy/object_relative_hierarchy/canonical_context.py','policy/robosyn_sprint/act_drawer_adapter.py'}
    require(required<=set(cfg['source_sha256']), 'Exact new runtime and borrowed feature/ACT dependencies required')
    for ref in cfg['assets'].values(): verify_ref(ref)
    act = Path(cfg['act_root']).resolve(strict=True)
    require({'model.safetensors','config.json'}<=set(cfg['act_files']), 'Original ACT weights/config required')
    for name, ref in cfg['act_files'].items():
        require((act/name).resolve()==Path(ref['path']).resolve(), 'ACT inventory root differs')
        verify_ref(ref)
    if cfg['mode']=='parent':
        require(not any(k in cfg['assets'] for k in ('coupled_weights','coupled_contract','runtime_admission','training_receipt')), 'Parent is raw ACT50/prefix10')
    else:
        scientific = {'experiments/coupled_goal_act_all14/'+name+'.py' for name in ('common','units','model','contracts','inference','compiler')}
        require(scientific<=set(cfg['source_sha256']), 'New all14 inference/compiler source required')
        contract = json.loads(verify_ref(cfg['assets']['coupled_contract']).read_text())
        validate_training(cfg,contract)
        from experiments.goal_latent_act_sequence_runtime_full.semantics import validate_head
        validate_head(json.loads(verify_ref(cfg['assets']['highlevel_contract']).read_text()),cfg)
        if cfg['mode']=='goal':
            qwen = Path(cfg['qwen_root']).resolve(strict=True)
            require(qwen.is_dir() and not qwen.is_symlink(), 'Pinned local Qwen directory required')
            index = json.loads((qwen/'model.safetensors.index.json').read_text())
            require(set(index['weight_map'].values())<=set(cfg['qwen_files']), 'Complete current-observer Qwen shard inventory required')
    cfg['_manifest'] = path
    return cfg
