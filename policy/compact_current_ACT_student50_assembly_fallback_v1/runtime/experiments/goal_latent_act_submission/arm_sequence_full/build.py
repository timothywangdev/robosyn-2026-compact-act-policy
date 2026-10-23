"""Small additive exact ARM12 stage sharing the existing public model/dependency blobs."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from experiments.goal_latent_act_submission.arm_sequence_full.contracts import *

PREFIXES=('upstream/','dependencies/','models/act/','models/Qwen3.5-9B/','assets/embodichain/','cache/','requirements/','licenses/')


def build(project,inventory,published_code,published_models,output):
    project=Path(project).resolve(strict=True);published_code=Path(published_code).resolve(strict=True)
    published_models=Path(published_models).resolve(strict=True);output=Path(output).absolute()
    require(not output.exists(),'Fresh ARM candidate root required')
    plan=json.loads(Path(inventory).read_text())
    require(plan['schema']=='goal-latent-act-full63000-arm12-additive-portable-plan-v1'
            and plan['robot_promotion_criterion_passed'] is False,'Actual nonpromoted additive inventory required')
    release_path=published_code/'releases/sequence-opening50-goal-v1.json'
    parent=json.loads(release_path.read_text());files={};origins={}
    def add(name,path,row,large=False):
        path=Path(path);require(path.is_file() and not path.is_symlink() and path.stat().st_size==row['bytes'],'Exact ordinary source required: '+name)
        if name in files:require(same(files[name],row),'Conflicting immutable destination: '+name);return
        target=child(output,name);target.parent.mkdir(parents=True,exist_ok=True)
        if large:
            require(stat.S_IMODE(path.stat().st_mode)==row.get('mode',stat.S_IMODE(path.stat().st_mode)),
                    'Hardlink must preserve existing blob mode; never chmod its parent')
            os.link(path,target)
        else:
            require(sha(path)==row['sha256'],'Small original source changed: '+name)
            shutil.copyfile(path,target);target.chmod(row.get('mode',stat.S_IMODE(path.stat().st_mode)))
        files[name]=dict(bytes=row['bytes'],sha256=row['sha256'],mode=stat.S_IMODE(target.stat().st_mode))
        origins[name]=dict(source=str(path),validation='previous_published_byte_bound_blob_type_size_inode_hardlink' if large else 'actual_full_SHA256_copy')
    require(shutil.disk_usage(output.parent).free>1<<30,'At least1GiB free for bounded new small files/static asset copies')
    output.mkdir(parents=True)
    for name,row in parent['files'].items():
        if name.startswith(PREFIXES) and '/assembled/' not in name:
            store=published_models if row['store']=='model' else published_code
            immutable_binary=Path(name).suffix.lower()in {'.so','.zip','.glb','.gltf','.obj','.stl','.ply','.png','.jpg','.jpeg','.pdf'}
            add(name,store/row['stored_path'],row,large=name.startswith(('models/','dependencies/wheels/','cache/torch/'))or immutable_binary)
    # Restore complete immutable geometry/material resources from the final
    # static source closure; generated simulator caches are never inherited.
    repaired=json.loads((published_code/'releases/trained-hybrid-cleanup-v2.json').read_text())
    for original_name,row in repaired['files'].items():
        name=original_name.replace('.cache/embodichain_data/','assets/embodichain/data/',1) if original_name.startswith('.cache/embodichain_data/') else original_name
        if name.startswith(('upstream/','assets/embodichain/')) and '/assembled/' not in name and name not in files:
            store=published_models if row['store']=='model' else published_code
            immutable_binary=Path(name).suffix.lower()in {'.so','.zip','.glb','.gltf','.obj','.stl','.ply','.png','.jpg','.jpeg','.pdf'}
            add(name,store/row['stored_path'],row,large=immutable_binary)
    for name,row in plan['source_inventory'].items():
        if name in files and not same(files[name],row):
            require(not name.startswith('upstream/'),'Frozen SDK/evaluator differs from original native actor')
            del files[name];del origins[name];child(output,name).unlink()
        add(name,row['path'],row)
    digest_paths={}
    weight_paths={
        'dabf15d49de48c60fae91905c85ee076d98daed9cf5571ed3cce6a03d5927bc4':'models/full63000/head50/highlevel.safetensors',
        '94b0da439d115851e6cc269113ac761236652f131161a4e11fa56b838f236210':'models/full63000/head50/checkpoint.json',
        'e22b614b23d638e4bfe5a792a84cfc934ac98a6880ed326f2d6aabfbac3522a3':'models/full63000/role_goal/arm-sequence.safetensors',
        '731f35f5032e88160663fa78f21ea581ac23c3ee3aff682ab02f846befe149ef':'models/full63000/role_goal/checkpoint.json',
        '08c9fffcda4a573516dba9e21c1e87d9dc1b657db72d7a28e67b5c20fbd84ccd':'models/full63000/state_action_control/arm-sequence.safetensors',
        'ebda0449de65110bf6d68da83dfd8984f373c27456d8e42784880e0116e030e9':'models/full63000/state_action_control/checkpoint.json',
    }
    for digest,row in plan['small_assets_and_scientific_proofs'].items():
        name=weight_paths.get(digest,'assets/training/by-sha/'+digest+'/'+Path(row['path']).name)
        add(name,row['path'],row);digest_paths[digest]=name
    oldpair=json.loads(Path(next(row['path']for row in plan['small_assets_and_scientific_proofs'].values()
        if 'runtime_asset:training_pair' in row['roles'])).read_text())
    def relocate(value):
        if isinstance(value,dict):
            if {'path','bytes','sha256'}<=set(value):
                require(value['sha256'] in digest_paths,'Uncaptured scientific transport ref')
                return dict(value,path=digest_paths[value['sha256']])
            return {k:relocate(v)for k,v in value.items()}
        if isinstance(value,list):return [relocate(v)for v in value]
        return value
    pair=relocate(oldpair)
    pair['historical_reference_aliases']={digest:dict(path=name,bytes=files[name]['bytes'],sha256=digest)
        for digest,name in digest_paths.items() if name.endswith('.json')}
    (output/'training').mkdir();json_write(output/'training/qualified-pair-template.json',pair)
    files['training/qualified-pair-template.json']=dict(reference(output/'training/qualified-pair-template.json'));files['training/qualified-pair-template.json'].pop('path')
    banks={mode:{}for mode in sorted(MODES)}
    for row in plan['original_configurations']:
        cfg=json.loads(Path(row['original_deployment']['path']).read_text());mode,task=row['mode'],row['task']
        cfg.update(schema=TEMPLATE,parent_schema=PARENT,source_root='.',scratch_root='runtime/scratch',
            act_root='models/act/'+task,qwen_root='models/Qwen3.5-9B',qwen_python='.venvs/qwen35/bin/python',qwen_python_path=['dependencies/qwen-overlay','.'])
        for value in cfg['assets'].values():value['path']=digest_paths[value['sha256']]
        for group,model_root in [('act_files',cfg['act_root']),('qwen_files',cfg['qwen_root'])]:
            for name,value in cfg[group].items():
                require(same(value,files[model_root+'/'+name]),'Exact original base weights/config required')
                value['path']=model_root+'/'+name
        name='deployments/'+mode+'/'+task+'.json';(output/name).parent.mkdir(parents=True,exist_ok=True);json_write(output/name,cfg)
        files[name]={k:v for k,v in reference(output/name).items()if k!='path'}
        native_path=Path(row['original_deployment']['path']).parent/(mode+'-'+task+'-command.json')
        native=json.loads(native_path.read_text());inner=list(native['inner_argv'])
        replacements={inner[0]:'{evaluator}',inner[inner.index('--config')+1]:'{config}',
            inner[inner.index('--goal_latent_manifest')+1]:'{manifest}',inner[inner.index('--goal_latent_manifest_sha256')+1]:'{manifest_sha}',
            inner[inner.index('--eval_result_dir')+1]:'{results}'}
        inner=[replacements.get(value,value)for value in inner]
        command_name='native-command-templates/'+mode+'/'+task+'.json';(output/command_name).parent.mkdir(parents=True,exist_ok=True)
        json_write(output/command_name,dict(schema='arm12_portable_native_command_template_v1',inner_argv=inner,original_command=reference(native_path)))
        files[command_name]={k:v for k,v in reference(output/command_name).items()if k!='path'}
        banks[mode][task]=dict(template=name,native_command_template=command_name,original_native_deployment=row['original_deployment'])
    for p in Path(__file__).parent.glob('*.py'):
        if p.name.startswith('test_'):continue
        add(str(p.relative_to(project)),p,reference(p))
    for name in ['experiments/simulator_cleanup_repair_v1/__init__.py',
                 'experiments/simulator_cleanup_repair_v1/native_bootstrap_v2.py',
                 'experiments/simulator_cleanup_repair_v1/constraint_teardown.py',
                 'experiments/goal_latent_act_submission/trained_hybrid/runtime_cache.py',
                 'experiments/goal_latent_act_submission/trained_hybrid/contracts_v2.py',
                 'experiments/goal_latent_act_submission/trained_hybrid/contracts.py',
                 'experiments/goal_latent_act_submission/trained_smol/contracts.py',
                 'experiments/robosyn_hybrid_fresh_evaluation/contracts.py',
                 'experiments/goal_latent_act_submission/cold_cpu.py']:
        add(name,project/name,reference(project/name))
    # The original qualified actor shim/native policy name is retained. An
    # organizer entry can delegate to it without modifying any action logic.
    shim=project/'experiments/goal_latent_act_arm_runtime_full/policy_shim.py'
    if 'policy/goal_latent_act/__init__.py' not in files:add('policy/goal_latent_act/__init__.py',shim,reference(shim))
    policy=output/'policy/goal_latent_act';policy.mkdir(exist_ok=True)
    yml='policy_name: goal_latent_act\nstate_obs_path: robot/qpos\nstrict_action_dim: true\nact_step: 50\n'
    (policy/'deploy_policy.yml').write_text(yml)
    (policy/'deploy_policy.py').write_text('from experiments.goal_latent_act_arm_runtime_full.runtime import get_model, eval, reset_model, close_model\n')
    (policy/'eval.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\nHERE="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"\nROOT="$(CDPATH= cd -- "$HERE/../.." && pwd)"\nexec "$ROOT/.venvs/sim/bin/python" -B "$ROOT/experiments/goal_latent_act_submission/arm_sequence_full/run_task.py" --root "$ROOT" "$@"\n')
    (policy/'eval.sh').chmod(0o755)
    for p in policy.iterdir():files[str(p.relative_to(output))]={k:v for k,v in reference(p).items()if k!='path'}
    visible=output/'policy/goal_latent_act_arm_release';visible.mkdir(parents=True)
    export='from experiments.goal_latent_act_submission.arm_sequence_full.entry_bridge import get_model, eval, reset_model, close_model\n'
    (visible/'__init__.py').write_text(export);(visible/'deploy_policy.py').write_text(export)
    (visible/'deploy_policy.yml').write_text('policy_name: goal_latent_act_arm_release\nstate_obs_path: robot/qpos\nstrict_action_dim: true\nact_step: 50\n')
    (visible/'eval.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\nHERE="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"\nROOT="$(CDPATH= cd -- "$HERE/../.." && pwd)"\nexec "${ROBOSYN_BOOTSTRAP_PYTHON:-python3}" -B "$ROOT/experiments/goal_latent_act_submission/arm_sequence_full/entry_cli.py" "$@" --root "$ROOT"\n')
    (visible/'eval.sh').chmod(0o755)
    for p in visible.iterdir():files[str(p.relative_to(output))]={k:v for k,v in reference(p).items()if k!='path'}
    # Shared runner metadata guards are copied unchanged, not weakened.
    for name in ['contracts.py','run_task.py']:
        p=project/'experiments/goal_latent_act_submission'/name;add(str(p.relative_to(project)),p,reference(p))
    spec=dict(schema=SCHEMA,parent_schema=PARENT,banks=banks,tasks=banks['goal'],files=files,
        environments=json.loads((project/'outputs/architecture-reassessment/sequence-opening-submission-prep-v1/portable-stage-v2/submission.json').read_text())['environments'],
        embodichain_data_root='assets/embodichain/data',primary_variant=None,offline_promotion_criterion_passed=False,
        RTX5090_qualified=False,cold_GPU_qualified=False,native_package_qualified=False,
        protocol=dict(generated=50,executed=50,refresh=50,discarded=0,motor_seed=17,ACT_core='eager',
            ACT_preprocessing='native_every_action',arms='bounded +/-0.15 radians handwise correction',grippers='bitexact original native ACT'),
        status='experimental_source_and_weight_stage_pending_CPU_relocation_and_native_package_proof',
        source_and_model_publication=dict(primary_variant=None,aggregate_score=None),
        provenance=dict(inventory=reference(inventory),unchanged_old_maps=plan['unchanged_old_release_maps']),
        validation_scope='Prior published large blob SHA inventories plus actual new-source/proof/weight hashing. No new cold environment installation claim.',
        models_constructed=0,GPU_launches=0)
    json_write(output/'submission.json',spec)
    json_write(output/'build-source-origins.json',origins)
    return dict(root=str(output),files=len(files),bytes=sum(row['bytes']for row in files.values()),models_constructed=0,GPU_launches=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(allow_abbrev=False)
    for name in ('project','inventory','published-code','published-models','output'):p.add_argument('--'+name,required=True)
    print(json.dumps(build(**vars(p.parse_args())),sort_keys=True))
