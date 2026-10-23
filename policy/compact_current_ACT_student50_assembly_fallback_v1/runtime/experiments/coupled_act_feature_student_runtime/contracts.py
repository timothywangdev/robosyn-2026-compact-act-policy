"""Student evidence is new; a teacher checkpoint cannot stand in for a student."""
import json
import math
from pathlib import Path
from experiments.coupled_act_feature_student.contracts import require, ref, verify, CAMERAS
from experiments.coupled_goal_act_all14_runtime.contracts import STATS, load_official, module

SCHEMA='coupled-ACT-current-map-student-native50-prefix10-v1'
ADMISSION='root_authorized_coupled_ACT_current_student_native_v1'
MOMENTS={'visual_mean':(3,512),'visual_scale':(3,512),'global_mean':(3,512),'global_scale':(3,512)}
HORIZONS={'click_bell':361,'drawer_open_place':900,'handle_basket':500,'item_assembly':500,
          'items_handover':350,'manipulate_pipette':1000,'mixer_operating':500,'sample_loading':500,
          'table_rearrangement':361,'water_pouring':500}


def validate_saved(saved,cfg,training):
    import numpy as np
    require(set(saved)=={'model','statistics','pooling','qpos_only','feature_plan','protocol','fixed_epoch'}, 'Only exact declared current student checkpoint fields')
    require(saved['pooling']=='spatial2x3' and saved['qpos_only']==cfg['student_qpos_only']
            and saved['fixed_epoch']==30 and saved['feature_plan']==cfg['assets']['student_feature_plan']
            and saved['protocol']==training['protocol'], 'Actual fixed selected student checkpoint metadata')
    require(set(saved['statistics'])==set(MOMENTS), 'Only optimizer800 current visual moments')
    for key,shape in MOMENTS.items():
        value=saved['statistics'][key]
        require(isinstance(value,np.ndarray) and value.shape==shape and value.dtype==np.float32 and np.isfinite(value).all(), 'Finite FP32 optimizer800 moments: '+key)
        if key.endswith('scale'):require(bool((value>=np.float32(.01)).all()), 'Frozen optimizer moment floor .01')


def sha(path): return ref(path)['sha256']
verify_ref=verify


def validate_teacher(contract, weights):
    from experiments.coupled_goal_act_all14.contracts import validate
    validate(contract)
    require(contract['state_action_only'] is False and contract['weights_sha256']==weights['sha256'], 'Actual frozen role teacher required')
    require(set(contract['tasks'])==set(HORIZONS), 'Actual all10 frozen teacher')
    for key,width in STATS.items():
        row=contract['statistics'][key]
        require(len(row)==width and all(math.isfinite(x) for x in row), 'Finite fixed teacher statistics')
        if key.endswith('scale'): require(all(x>0 for x in row), 'Positive fixed teacher scales')


def load_manifest(path, expected):
    path=Path(path).resolve(strict=True);require(sha(path)==expected,'Exact separate student deployment')
    cfg=json.loads(path.read_text())
    require(cfg['schema']==SCHEMA and cfg['mode'] in ('parent','student'), 'New student or matched raw ACT parent only')
    require((cfg['native_chunk_generated'],cfg['prefix_executed'],cfg['motor_seed'])==(50,10,17), 'Original50/10/motor17')
    require(cfg['task'] in HORIZONS and cfg['official_horizon']==HORIZONS[cfg['task']], 'Original official horizon')
    require(cfg['current_inputs']==['current_RGB_via_same_ACT_forward','current_qpos','current_FK']
            and cfg['future_inputs_allowed'] is False and cfg['Qwen_models_loaded']==0 and cfg['ACT_models_loaded']==1, 'Qwen-free causal one-ACT boundary')
    require(cfg['camera_order']==['observation.images.'+c for c in CAMERAS], 'Native high/right/left camera order')
    require(cfg['act_core_mode']=='eager' and cfg['act_preprocess_mode']=='native_every_action', 'Qualified original ACT execution')
    require(cfg['compile_policy']=='attempt_current_numeric_then_fastest_qualified', 'New student must attempt and measure compilation')
    require(cfg['gripper_codec']==('drawer_normalized_opening_inverse_v1' if cfg['task']=='drawer_open_place' else 'physical_meters_identity_v1'), 'Exact task native codec')
    require(0<cfg['episode_wall_seconds']<=1800, 'Bounded original episode')
    root=Path(cfg['source_root']).resolve(strict=True)
    required={'experiments/coupled_act_feature_student_runtime/'+n+'.py' for n in ('contracts','runtime','inference','policy','compiler')}
    required|={'experiments/coupled_act_feature_student/'+n+'.py' for n in ('hook','model','projected','contracts','compiler')}
    require(required<=set(cfg['source_sha256']), 'Exact new/borrowed source closure')
    for relative,digest in cfg['source_sha256'].items():
        source=root/relative
        require(source.is_file() and not source.is_symlink() and source.resolve().is_relative_to(root) and sha(source)==digest, 'Bound source changed: '+relative)
    for binding in cfg['assets'].values():verify(binding)
    act=Path(cfg['act_root']).resolve(strict=True)
    require({'model.safetensors','config.json'}<=set(cfg['act_files']), 'Actual released ACT complete binding')
    for name,binding in cfg['act_files'].items():
        require((act/name).resolve()==verify(binding), 'Actual ACT root and bytes differ')
    a=cfg['assets'];admission=json.loads(verify(a['runtime_admission']).read_text())
    require(admission['schema']==ADMISSION and admission['execution_authorized'] is True
            and admission['robot_quality_promoted'] is False and admission['mode']==cfg['mode']
            and admission['task']==cfg['task'], 'Fresh root-authorized exploratory student slot')
    for key in ('teacher_weights','teacher_contract','student_weights','student_training_receipt','student_cache_receipt','student_feature_plan',
                'student_original_training_wait','student_independent_training_drain'):
        if cfg['mode']=='student':require(admission[key]==a[key], 'Actual candidate evidence differs: '+key)
    if cfg['mode']=='student':
        teacher=json.loads(verify(a['teacher_contract']).read_text())
        validate_teacher(teacher,a['teacher_weights'])
        receipt=json.loads(verify(a['student_training_receipt']).read_text())
        declared={'actual_paired_ACT_feature_student_distill_and_policy_aware_CPU_v1':'fixed30_three_candidates_complete_exploratory_no_robot_promotion',
                  'actual_paired_ACT_feature_student_distill_and_policy_aware_GPU_v2':'fixed30_four_candidates_complete_exploratory_no_robot_promotion'}
        require(receipt['schema'] in declared and receipt['status']==declared[receipt['schema']]
                and receipt['robot_quality_promoted'] is False, 'Actual declared fixed student training required')
        selected=receipt['results'][cfg['student_candidate']]
        require(selected['weights']==a['student_weights'] and selected['optimizer_updates']==210, 'Exact selected fixed30/210updates candidate')
        require(receipt['cache']==a['student_cache_receipt'], 'Actual complete source cache differs')
        cache=json.loads(verify(a['student_cache_receipt']).read_text())
        require(cache['status']=='all1000_declared_current_features_complete' and cache['plan']==a['student_feature_plan'], 'Actual complete current cache/feature plan')
        feature_plan=json.loads(verify(a['student_feature_plan']).read_text())
        require(feature_plan['teacher_weights']==a['teacher_weights'] and feature_plan['teacher_contract']==a['teacher_contract'], 'Extraction teacher and frozen motor differ')
        require(all(feature_plan['teacher_statistics'][key]==teacher['statistics'][key] for key in STATS), 'Actual extraction/runtime teacher moments differ')
        require(receipt['teacher_weights']==a['teacher_weights'], 'Training frozen motor differs from native motor')
        protocol=json.loads(verify(receipt['protocol']).read_text())
        require(protocol['feature_plan']==a['student_feature_plan'], 'Training protocol/current feature plan differs')
        for name,digest in protocol['source_sha256'].items():
            require(cfg['source_sha256'].get(name)==digest, 'Actual trained student source differs: '+name)
        native_act=feature_plan['act'][cfg['task']]
        require(native_act['camera_order']==cfg['camera_order'], 'Extraction/runtime camera order differs')
        for name in ('config.json','model.safetensors'):
            require(all(native_act['files'][name][key]==cfg['act_files'][name][key] for key in ('bytes','sha256')), 'Extraction/native ACT bytes differ: '+name)
        wait=json.loads(verify(a['student_original_training_wait']).read_text())
        require(wait['schema']==admission['student_training_wait_schema'] and wait['original_child_exit_code']==0
                and wait['child_reaped'] is True, 'Literal genuine student training wait required')
        drain=json.loads(verify(a['student_independent_training_drain']).read_text())
        require(drain['schema']==admission['student_training_drain_schema'] and admission['independent_student_training_drain_qualified'] is True,
                'Exact independently qualified original student drain required')
        require(cfg['student_qpos_only'] is cfg['student_candidate'].endswith('_qpos_control'), 'Declared visual/qpos intervention')
    cfg['_manifest']=path
    return cfg
