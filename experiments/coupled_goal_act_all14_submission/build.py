"""Ordinary hardlink fixture; original proof JSON files remain byte-identical."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
from .prepare import ROOT,ref,sha,safe,write
from .contracts import PACKAGE


def build(preparation,output):
    preparation=Path(preparation).resolve(strict=True);old=json.loads(preparation.read_text())
    root=Path(output).absolute();assert not root.exists();root.mkdir(parents=True)
    bindings=[];files={};source={};bytes_copied=0;linked=0
    def put(name,row,copy=False):
        nonlocal bytes_copied,linked
        name=safe(name);path=Path(row['path']);target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
        assert not path.is_symlink() and path.is_file() and path.stat().st_size==row['bytes']
        if name in files:assert files[name]['sha256']==row['sha256'] and files[name]['bytes']==row['bytes']
        else:
            if copy:shutil.copy2(path,target);bytes_copied+=row['bytes'];assert sha(target)==row['sha256']
            else:os.link(path,target);linked+=row['bytes']
            files[name]={k:row[k] for k in ('bytes','sha256')}
        bindings.append(dict(original={k:row[k] for k in ('path','bytes','sha256')},installed=name))
        return name
    for name,row in old['external_assets'].items():put(name,row)
    # Original source donor plus the current new transport implementation.
    for name,row in old['source_files'].items():
        if not name.startswith('experiments/coupled_goal_act_all14_submission/'):
            put(name,row,copy=True);source[name]=files[name]
    donor=Path(old['prior_inventory_donor']['path']).parent
    donor_spec=json.loads((donor/'submission.json').read_text())
    for name,row in donor_spec['files'].items():
        if name.startswith(('experiments/goal_latent_act_submission/','experiments/simulator_cleanup_repair_v1/')) and name.endswith('.py'):
            put(name,dict(path=str(donor/name),**row),copy=True);source[name]=files[name]
    for p in Path(__file__).parent.glob('*.py'):
        name=str(p.relative_to(ROOT));put(name,ref(p),copy=True);source[name]=files[name]
    tasks=[]
    for row in old['tasks']:
        mode,task=row['mode'],row['task'];raw=row['original_deployment']
        put('provenance/native-deployments/'+mode+'-'+task+'.json',raw,copy=True)
        cfg=json.loads(Path(raw['path']).read_text())
        # Several original refs name the same content via a different historical
        # path (e.g. donor ACT source versus native /tmp source). Preserve every
        # original tuple, not merely the first unique target inventory row.
        for group in ('assets','act_files','qwen_files'):
            for bound in cfg[group].values():
                put(old['original_path_aliases'][bound['path']],bound)
        if mode!='parent':
            attempt=json.loads(Path(cfg['assets']['compiler_attempt']['path']).read_text())
            extra=attempt['admission'];put('provenance/by-sha/'+extra['sha256']+'/'+Path(extra['path']).name,extra,copy=True)
        name='templates/'+mode+'-'+task+'.json'
        write(root/name,dict(schema='coupled_all14_byte_preserving_portable_template_v1',task=task,mode=mode,
            original_deployment=raw,act_root='models/act/'+task,qwen_root='models/Qwen3.5-9B'))
        files[name]={k:v for k,v in ref(root/name).items() if k!='path'}
        tasks.append(dict(task=task,mode=mode,template=name,template_sha256=files[name]['sha256'],official_horizon=row['official_horizon']))
    policy=root/'policy/coupled_all14_portable_v1';policy.mkdir(parents=True)
    (policy/'deploy_policy.py').write_text('from experiments.coupled_goal_act_all14_submission.policy import get_model, eval, reset_model, close_model\n')
    (policy/'__init__.py').write_text('from .deploy_policy import get_model, eval, reset_model, close_model\n')
    (policy/'deploy_policy.yml').write_text('policy_name: coupled_all14_portable_v1\ntask_name: null\nsetting: random\nmodel_name: coupled_all14_unpromoted\nseed: 0\nmax_episodes: 20\nmax_steps: 1000\nheadless: true\npytorch_device: cuda\nfilter_dataset_saving: true\neval_video_log: true\neval_video_obs_key: cam_high\nact_step: 10\nstate_obs_path: robot/qpos\nstrict_action_dim: true\n')
    (policy/'eval.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\nHERE="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"\nROOT="$(CDPATH= cd -- "$HERE/../.." && pwd)"\nexec "${PYTHON_BIN:-$ROOT/.venvs/sim/bin/python}" -B "$ROOT/experiments/coupled_goal_act_all14_submission/entry_cli_v2.py" --root "$ROOT" "$@"\n')
    (policy/'eval.sh').chmod(0o755)
    for p in policy.iterdir():files[str(p.relative_to(root))]={k:v for k,v in ref(p).items() if k!='path'}
    (root/'runtime').mkdir()
    spec=dict(schema=PACKAGE,status='CPU_prepared_no_native_or_5090_promotion',tasks=tasks,source_files=source,
        files=files,reference_bindings=bindings,environments=old['environments'],
        original_preparation=ref(preparation),robot_quality_promoted=False,offline_5percent_criterion_passed=False,
        submission_ready=False,RTX5090_qualified=False,installed_native_qualified=False,
        raw_original_JSON_rewritten=False,actual_small_bytes_copied=bytes_copied,
        immutable_hardlinked_logical_bytes=linked)
    write(root/'submission.json',spec)
    assert bytes_copied<1<<30 and shutil.disk_usage(root).free>8<<30
    print(json.dumps(dict(submission=ref(root/'submission.json'),source_files=len(source),
        actual_small_bytes_copied=bytes_copied,hardlinked_logical_bytes=linked,GPU_queries=0,models_constructed=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--preparation',required=True);p.add_argument('--output',required=True)
    build(**vars(p.parse_args()))
