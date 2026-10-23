"""Immutable infrastructure successor: complete static assets, isolated caches."""
import json
from pathlib import Path
from experiments.goal_latent_act_submission.trained_hybrid.contracts import (
    TASKS,V10_SHA,TRAINED_SHA,SELECTION_SHA,child,validate_science,metadata_check,
    require,sha,reference,json_write)
SCHEMA='trained-smolvla-released-router-all10-portable-v2'


def stage_spec(root,full=False):
    root=Path(root).resolve(strict=True);spec=json.loads((root/'submission.json').read_text())
    require(spec['schema']==SCHEMA and set(spec['tasks'])==TASKS and spec['unsupported_tasks']==[], 'Exact all10 infrastructure successor required')
    require(spec['aggregate_success_score'] is None and spec['primary_submission_selected'] is False,'No invented selection or score')
    require(spec['runtime_cache_policy']=='fresh_short_private_static_links_no_generated_cache', 'Explicit independent runtime cache required')
    require(sha(root/'trained-submission.json')==TRAINED_SHA and sha(root/'submission-manifest.json')==V10_SHA
        and sha(root/'provenance/hybrid-frozen-selection.json')==SELECTION_SHA,'Original immutable scientific contracts required')
    require(not any('/assembled/' in n or '/material-runs/' in n for n in spec['files']), 'Mutable generated files cannot be pinned static assets')
    if full:
        for name,row in spec['files'].items():
            p=child(root,name)
            require(p.is_file() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'Exact successor payload differs: '+name)
    return spec
