"""Inventory exact assets and copy small source only; never duplicate weights."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
LIMIT=1<<30


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(4<<20),b''):h.update(part)
    return h.hexdigest()


def ref(path):
    p=Path(path).resolve(strict=True)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))


def write(path,body):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(body,f,indent=2,sort_keys=True);f.write('\n')


def safe(name):
    p=Path(name)
    assert not p.is_absolute() and '..' not in p.parts and p.parts
    return str(p)


def prepare(campaign,base,output):
    campaign=Path(campaign).resolve(strict=True);base=Path(base).resolve(strict=True)
    output=Path(output).absolute();assert not output.exists()
    output.mkdir(parents=True);stage=output/'relocated-source';stage.mkdir()
    c=json.loads(campaign.read_text());old=json.loads((base/'submission.json').read_text())
    inventory={};aliases={};sources={};tasks=[];checked={}
    def add(name,row,category,branch=None):
        name=safe(name);path=Path(row['path'])
        assert path.is_file() and not path.is_symlink() and path.stat().st_size==row['bytes'],path
        ident=(row['sha256'],row['bytes'])
        if name in inventory:assert ident==(inventory[name]['sha256'],inventory[name]['bytes']),name
        else:inventory[name]=dict(path=str(path),bytes=row['bytes'],sha256=row['sha256'],category=category,
            content_hash_origin='preserved_sealed_input_reference; size_and_regular_file_checked_now',branches=[])
        if branch and branch not in inventory[name]['branches']:inventory[name]['branches'].append(branch)
        key=str(path)
        if key in aliases:assert aliases[key]==name,(key,aliases[key],name)
        else:aliases[key]=name
        return name
    # The earlier portable artifact is an inventory donor, never a qualification
    # of the new actor, new50/10 protocol or its installed policy entry.
    allowed=('upstream/','dependencies/','assets/embodichain/','cache/torch/','requirements/','licenses/')
    for name,row in old['files'].items():
        if name.startswith(allowed):add(name,dict(path=str(base/name),**row),'prior_pinned_offline_toolchain_or_static_asset')
    for slot in c['plans']:
        raw=Path(slot['deployment']['path']);cfg=json.loads(raw.read_text());mode,task=cfg['mode'],cfg['task']
        assert cfg['native_chunk_generated']==50 and cfg['prefix_executed']==10
        assert cfg['motor_seed']==17 and cfg['act_core_mode']=='eager'
        for name,digest in cfg['source_sha256'].items():
            path=Path(cfg['source_root'])/safe(name)
            if name in sources:assert sources[name]['sha256']==digest
            else:
                actual=ref(path);assert actual['sha256']==digest; sources[name]=actual
        template=deepcopy(cfg)
        template.update(schema='coupled_all14_portable_preparation_template_v1',
            parent_schema=cfg['schema'],source_root='.',scratch_root='runtime/scratch',
            act_root='models/act/'+task,qwen_root='models/Qwen3.5-9B',
            qwen_python='.venvs/qwen35/bin/python',qwen_python_path=['dependencies/qwen-overlay','.'])
        for name,row in cfg['act_files'].items():
            dest=add('models/act/'+task+'/'+name,row,'released_native_ACT',mode)
            template['act_files'][name]['path']=dest
        for name,row in cfg['qwen_files'].items():
            if mode=='goal':add('models/Qwen3.5-9B/'+name,row,'current_RGB_Qwen_foundation',mode)
            template['qwen_files'][name]['path']='models/Qwen3.5-9B/'+name
        for key,row in cfg['assets'].items():
            if key in ('coupled_contract','coupled_weights'):
                dest='models/coupled/'+mode+'/'+Path(row['path']).name
            elif key in ('highlevel_contract','highlevel_weights'):
                dest='models/command50_head/'+Path(row['path']).name
            elif key=='ACT_model_source':dest='upstream/lerobot-sim/src/lerobot/policies/act/modeling_act.py'
            else:dest='assets/by-sha/'+row['sha256']+'/'+Path(row['path']).name
            dest=add(dest,row,'model_or_preserved_provenance_or_native_boundary',mode)
            template['assets'][key]['path']=dest
        name='templates/'+mode+'-'+task+'.json';write(stage/name,template)
        plan=json.loads(Path(slot['plan']['path']).read_text())
        command=json.loads(Path(slot['command']['path']).read_text());argv=command['inner_argv']
        horizon=int(argv[argv.index('--max_steps')+1])
        tasks.append(dict(mode=mode,task=task,official_horizon=horizon,template=name,
            original_deployment=slot['deployment'],original_plan=slot['plan'],original_command=slot['command']))
    assert len(tasks)==30 and len({(r['mode'],r['task']) for r in tasks})==30
    copied=0
    for name,row in sources.items():
        dest=stage/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(row['path'],dest)
        assert sha(dest)==row['sha256'];copied+=dest.stat().st_size
    for source in Path(__file__).parent.glob('*.py'):
        name=str(source.relative_to(ROOT));dest=stage/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest);sources[name]=ref(source);copied+=source.stat().st_size
    policy=stage/'policy/coupled_all14_submission_preparation_v1';policy.mkdir(parents=True)
    (policy/'deploy_policy.py').write_text('from experiments.coupled_goal_act_all14_submission.policy_entry import get_model, eval, reset_model, close_model\n')
    (policy/'__init__.py').write_text('from .deploy_policy import get_model, eval, reset_model, close_model\n')
    (policy/'deploy_policy.yml').write_text('policy_name: coupled_all14_submission_preparation_v1\ntask_name: null\nsetting: random\nmodel_name: null\nseed: 0\nmax_episodes: 20\nmax_steps: 1000\nheadless: true\npytorch_device: cuda\nfilter_dataset_saving: true\neval_video_log: true\neval_video_obs_key: cam_high\nstate_obs_path: robot/qpos\nstrict_action_dim: true\nact_step: 10\n')
    (policy/'eval.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\nHERE="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"\nROOT="$(CDPATH= cd -- "$HERE/../.." && pwd)"\nexec "${PYTHON_BIN:-python}" -B "$ROOT/experiments/coupled_goal_act_all14_submission/entry_cli.py" --root "$ROOT" "$@"\n')
    (policy/'eval.sh').chmod(0o755)
    unique={}
    for row in inventory.values():unique[(row['sha256'],row['bytes'])]=row['bytes']
    variant_bytes={}
    for mode in ('parent','goal','goal_free'):
        rows=[v for v in inventory.values() if not v['branches'] or mode in v['branches']]
        variant_bytes[mode]=sum({(v['sha256'],v['bytes']):v['bytes'] for v in rows}.values())
    spec=dict(schema='coupled_all14_submission_preparation_inventory_v1',status='preparation_only_no_portable_runtime_or_robot_qualification',
        original_campaign=ref(campaign),prior_inventory_donor=ref(base/'submission.json'),
        tasks=tasks,source_files=sources,external_assets=inventory,original_path_aliases=aliases,
        environments=old['environments'],unique_payload_bytes=sum(unique.values()),variant_unique_payload_bytes=variant_bytes,
        embargo=dict(execution_authorized=False,primary_submission_selected=False,submission_ready=False,
            strict_offline_5percent_criterion_passed=False,all10_native_comparison_complete=False,
            installed_portable_runtime_qualified=False,RTX5090_qualified=False,aggregate_success_score=None),
        raw_provenance_policy='Original training/compiler/checkpoint JSON bytes retained; no rewritten original waits or invented completion.',
        portability_blocker='Frozen runtime compares absolute nested reference paths. New reviewed alias-aware loader is required; templates are not admitted native manifests.',
        source_only_stage=True,large_models_copied=False,small_source_bytes_copied=copied,
        toolchain_inventory_reuse_is_not_new_cold_installation_qualification=True)
    write(stage/'preparation.json',spec)
    disk=shutil.disk_usage(output)
    total=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    assert total<LIMIT and disk.free>8*(1<<30)
    write(output/'prepare-receipt.json',dict(schema='coupled_all14_submission_CPU_transport_preparation_v1',
        status=spec['status'],manifest=ref(stage/'preparation.json'),tasks=30,external_files=len(inventory),
        original_source_files=len(sources),new_bytes=total,bounded_new_bytes_limit=LIMIT,disk_free_bytes=disk.free,
        unique_payload_bytes=spec['unique_payload_bytes'],variant_unique_payload_bytes=variant_bytes,
        GPU_queries=0,GPU_launches=0,models_constructed=0,worlds_constructed=0))
    return dict(manifest=ref(stage/'preparation.json'),new_bytes=total,external_files=len(inventory))


if __name__=='__main__':
    p=argparse.ArgumentParser(allow_abbrev=False)
    for key in ('campaign','base','output'):p.add_argument('--'+key,required=True)
    print(json.dumps(prepare(**vars(p.parse_args()))))
