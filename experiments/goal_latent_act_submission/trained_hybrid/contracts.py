"""Separately typed fixed all10 union; original scientific guards stay intact."""
import json
from pathlib import Path, PurePosixPath
from experiments.goal_latent_act_submission.contracts import require, sha, reference, json_write
from experiments.goal_latent_act_submission.trained_smol.contracts import validate_science as original_science, metadata_check
from experiments.robosyn_hybrid_fresh_evaluation.contracts import ORDER

SCHEMA = 'trained-smolvla-released-router-all10-portable-v1'
TASKS = set(ORDER)
V10_SHA = '128c7e237d1ed03390c83a493fc50352c9635abc823c2b21332a6f0b3780460d'
TRAINED_SHA = 'd92093ae8099ee104abd91d35f7c8c5150c4fc379df566c4660d55d88b1e3e83'
SELECTION_SHA = 'b1577a89548618bd13ab621cdc40c8946463139ccd58405b4afd51f05731e08b'


def child(root, name):
    require(isinstance(name,str) and name and '\\' not in name, 'Canonical relative path required')
    p = PurePosixPath(name)
    require(not p.is_absolute() and '..' not in p.parts and str(p)==name, 'Unsafe portable path')
    hidden = any(x.startswith('.') for x in p.parts)
    require(not hidden or name.startswith(('.cache/embodichain_data/', '.cache/torch/'))
            or (name.startswith('models/') and p.name=='.gitattributes'), 'Unbound hidden path')
    root = Path(root).resolve(strict=True); target = root/name
    require(target.resolve().is_relative_to(root) and not any(x.is_symlink() for x in (target,*target.parents)
            if x.is_relative_to(root)), 'Ordinary contained portable files required')
    return target


def stage_spec(root, full=False):
    root = Path(root).resolve(strict=True); spec = json.loads((root/'submission.json').read_text())
    require(spec['schema']==SCHEMA and set(spec['tasks'])==TASKS and spec['unsupported_tasks']==[], 'Fixed all10 typed artifact required')
    require(spec['aggregate_success_score'] is None and spec['primary_submission_selected'] is False, 'No invented score or selection')
    require(sha(root/'trained-submission.json')==TRAINED_SHA and sha(root/'submission-manifest.json')==V10_SHA
            and sha(root/'provenance/hybrid-frozen-selection.json')==SELECTION_SHA, 'Original three immutable contracts required')
    if full:
        for name,bound in spec['files'].items():
            p = child(root,name)
            require(p.is_file() and p.stat().st_size==bound['bytes'] and sha(p)==bound['sha256'], 'Exact hybrid payload differs: '+name)
    return spec


def validate_science(root,spec):
    from experiments.robosyn_hybrid_fresh_evaluation.contracts import route
    original = json.loads((Path(root)/'trained-submission.json').read_text())
    legacy = json.loads((Path(root)/'submission-manifest.json').read_text())
    selection = json.loads((Path(root)/'provenance/hybrid-frozen-selection.json').read_text())
    expected = {t:route(t,legacy) for t in TASKS}
    require(spec['tasks']==selection['task_choices']==expected, 'Fixed scientific actor routes changed')
    require(spec['trained_tasks']==original['trained_tasks'] and spec['environments']==original['environments'], 'Original models or environment contracts changed')
    require(sha(Path(root)/'scripts/run_task.py')=='5fcd5ceccaa90326f2c0170a3e546510df242496a3484dfb4f71c3f559105127', 'Released launcher changed')
    return original_science(root,original)
