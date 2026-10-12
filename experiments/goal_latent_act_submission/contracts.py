"""Portable packaging contracts; do not construct a model or waive science."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath

SCHEMA='goal-latent-act-portable-submission-v1'
TEMPLATE='goal-latent-act-portable-deployment-template-v1'
PARENT='goal-latent-act-native-prefix10-deployment-v1'
SCHEMA50='goal-latent-act-portable-submission-v2'
TEMPLATE50='goal-latent-act-portable-prefix50-deployment-template-v2'
PARENT50='goal-latent-act-native-prefix50-deployment-v2'
PROTOCOLS={PARENT:(SCHEMA,TEMPLATE,'goal_latent_act_runtime',10),
           PARENT50:(SCHEMA50,TEMPLATE50,'goal_latent_act_native50',50)}
TASKS={'click_bell','drawer_open_place','handle_basket','item_assembly','items_handover',
       'manipulate_pipette','mixer_operating','sample_loading','table_rearrangement','water_pouring'}


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for part in iter(lambda:stream.read(8<<20),b''):h.update(part)
    return h.hexdigest()


def portable(name):
    require(isinstance(name,str) and name and '\\'not in name,'Portable relative path required')
    p=PurePosixPath(name)
    require(not p.is_absolute() and '..'not in p.parts and str(p)==name and not any(x.startswith('.')for x in p.parts),
            'Unsafe/noncanonical portable path')
    return name


def child(root,name):
    p=Path(root)/portable(name)
    require(p.resolve().is_relative_to(Path(root).resolve()) and not any(x.is_symlink()for x in (p,*p.parents)if x.is_relative_to(root)),
            'Bundle path escaped or used symlink')
    return p


def reference(path):
    p=Path(path).absolute()
    require(p.is_file() and not p.is_symlink(),'Regular bound file required')
    return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))


def verify(row):
    require(set(row)=={'path','bytes','sha256'} and Path(row['path']).is_absolute(),'Exact absolute file reference required')
    require(reference(row['path'])==row,'Bound file content changed: '+row['path'])
    return Path(row['path'])


def load_native(root,path,expected):
    cfg=json.loads(Path(path).read_text());require(cfg['schema']in PROTOCOLS,'Explicit native deployment schema required')
    relative='experiments/'+PROTOCOLS[cfg['schema']][2]+'/contracts.py';code=Path(root)/relative
    require(cfg['source_sha256'].get(relative)==sha(code),'Native contract source must be pinned before import')
    spec=importlib.util.spec_from_file_location('packaged_native_goal_contracts',code)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.load_manifest(path,expected)


def scientific_bindings(cfg,receipts):
    """Repeat native constructor's JSON predicates before any copying/inference."""
    require(cfg['mode']=='goal','Scientific goal candidate only; parent/control are separate artifacts')
    a=cfg['assets'];high=json.loads(verify(a['highlevel_contract']).read_text())
    correction=json.loads(verify(a['correction_contract']).read_text());admit=json.loads(verify(a['robot_admission']).read_text())
    require(high['schema']=='cached-current-qwen-role-dual-eef-object-goal-v1' and high['role_zero_control']is False
        and high['weights_sha256']==a['highlevel_weights']['sha256'] and high['goal_horizon_frames']==10
        and high['context_version']=='qwen35-canonical-current-role-hidden-v2' and high['optimizer_updates']>0,
        'Actually trained factual high-level contract required')
    require(cfg['schema']in PROTOCOLS,'Explicit scientific runtime schema required')
    native50=cfg['schema']==PARENT50;prefix=50 if native50 else 10
    require(cfg['native_chunk_generated']==50 and cfg['prefix_executed']==prefix,'Declared runtime protocol differs')
    require(correction['schema']==('goal-latent-act-bounded-arm12-reliability-prefix-v2'if native50 else'goal-latent-act-bounded-arm12-reliability-v1') and correction['goal_free']is False
        and correction['weights_sha256']==a['correction_weights']['sha256'] and correction['optimizer_updates']>0
        and correction['native_chunk_generated']==50 and correction['prefix_executed']==prefix and correction['grippers_unchanged']is True
        and correction['use_future_labels_at_inference']is False
        and correction['highlevel_weights_sha256']==a['highlevel_weights']['sha256']
        and correction['highlevel_checkpoint_sha256']==a['highlevel_contract']['sha256']
        and correction['geometry_sha256']==a['geometry']['sha256'] and correction['urdf_sha256']==a['urdf']['sha256'],
        'Trained source-bound explicit-prefix arm-only correction required')
    require(admit['admit_robot_smoke']is True and admit['checkpoint_sha256']==a['correction_weights']['sha256'],
            'Existing measured offline admission required; not a robot qualification')
    hrow=receipts['highlevel'];crow=receipts['correction'];h=json.loads(verify(hrow).read_text());c=json.loads(verify(crow).read_text())
    require(h['status']=='completed_highlevel_training' and h['optimizer_updates']>0 and h['role_zero_control']is False
        and h['data_receipt_sha256']==high['data_receipt_sha256'],'Completed exact high-level training receipt required')
    require(c['optimizer_updates']==correction['optimizer_updates'] and c['goal_free']is False and c['grippers_unchanged']is True,
            'Exact completed correction training linkage required')
    if native50:
        require(correction['model_config']['execution_prefix']==50 and correction['discarded']==0
            and all(correction[k]==50 for k in ('gate_summary_prefix','residual_loss_prefix','gate_labels_prefix')),
            'Native50 summaries/loss/labels/discard contract differs')
        require(c['schema']==correction['schema'] and c['status']=='completed_fixed_budget_expanding_prefix_training'
            and all(c[k]==50 for k in ('execution_prefix','residual_loss_prefix','gate_labels_prefix','selection_prefix'))
            and c['gate_updates']==correction['gate_updates'] and c['gate_updates']>0
            and c['checkpoint_selection']==correction['selection']=='fixed_predeclared_final_epoch; MONITOR_reporting_only'
            and c['snapshot_sha256']==correction['snapshot_sha256'] and c['original_prediction_reload_bitexact']is True,
            'Exact completed fixed-budget native50 training required')
        require(crow==a['correction_training_receipt'] and admit['schema']=='goal-latent-act-prefix50-offline-smoke-admission-v1'
            and admit['checkpoint_contract_sha256']==a['correction_contract']['sha256']
            and admit['pair_original_exits_sha256']==a['correction_training_original_exits']['sha256']
            and admit['native_chunk_generated']==admit['prefix_executed']==50 and admit['grippers_unchanged']is True
            and admit['robot_success_claim']is False,'Native50 offline admission/original-exit proof differs')
        exits=json.loads(verify(a['correction_training_original_exits']).read_text())
        require(exits['schema']=='goal-latent-act-expanding-prefix50-matched-original-exits-v1'
            and exits['status']=='both_original_exit0_natural_CPU_completion' and len(exits['runs'])==2
            and {r['goal_free']for r in exits['runs']}=={True,False},'Exact matched training completion proof required')
        for run in exits['runs']:
            require(run['original_exit_code']==0 and run['original_child_reaped']is True and run['original_process_absent']is True
                and run['same_birth_present']is False and run['independently_observed_group_after_reap']==[]
                and run['natural_completion_no_signals_sent']is True,'Natural terminal trained-pair proof required')
            if run['goal_free']is False:require(run['training_receipt_sha256']==crow['sha256'],'Goal original-exit receipt differs')
    else:
        require(crow['sha256']==admit['training_receipt_sha256'] and c['status']=='completed_experimental_goal_ACT_correction_training',
                'Exact completed prefix10 correction receipt/admission linkage required')
    return high,correction


def training_source_name(name):
    """Canonical destination only; keep original absolute provenance bytes intact."""
    if Path(name).is_absolute():
        parent=Path('/home/wearaway/projects/research_projects/robosyn-hierarchical-vla')
        require(Path(name).is_relative_to(parent),'Unexpected absolute scientific source provenance')
        name=str(Path(name).relative_to(parent))
    elif '/'not in name:name='experiments/goal_latent_act/'+name
    return portable(name)


def materialize(root,template,scratch):
    cfg=copy.deepcopy(template)
    template_schema=cfg.pop('schema');parent_schema=cfg.pop('parent_schema')
    require(parent_schema in PROTOCOLS and template_schema==PROTOCOLS[parent_schema][1],'Explicit matching portable template only')
    cfg['schema']=parent_schema;cfg['source_root']=str(Path(root).resolve());cfg['scratch_root']=str(Path(scratch).resolve())
    cfg['qwen_python']=str(Path(root)/'.venvs/qwen35/bin/python')
    cfg['qwen_python_path']=[str(Path(root)/'dependencies/qwen-overlay'),str(Path(root))]
    for name in ('act_root','qwen_root'):cfg[name]=str(child(root,cfg[name]))
    for key in ('assets','act_files','qwen_files'):
        for row in cfg[key].values():row['path']=str(child(root,row['path']))
    return cfg


def json_write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
