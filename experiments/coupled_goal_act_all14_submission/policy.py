"""Portable official entry requires a fresh source-bound root native admission."""
import json
from pathlib import Path
from .prepare import ref,sha
from .contracts import materialize,require
from experiments.coupled_goal_act_all14_runtime.policy_shim_eager import guarded


def _get_model(args):
    root=Path(args['coupled_all14_submission_root']).resolve(strict=True)
    template=Path(args['coupled_all14_submission_template']).resolve(strict=True)
    proof=Path(args['coupled_all14_portable_admission']).resolve(strict=True)
    require(sha(proof)==args['coupled_all14_portable_admission_sha256'],'Exact separate portable root admission required')
    a=json.loads(proof.read_text())
    require(a['schema']=='root_authorized_coupled_all14_portable_native_v1' and a['execution_authorized'] is True
        and a['robot_quality_promoted'] is False,'Only separately authorized unpromoted portable native pilot allowed')
    require(a['package']==ref(root/'submission.json') and a['template']==ref(template)
        and a['entry_source']==ref(Path(__file__)),'Portable package/template/entry identity differs')
    scratch=Path(args['coupled_all14_portable_scratch']).absolute()
    require(not scratch.exists() and not scratch.is_symlink(),'Fresh portable operational scratch required')
    scratch.mkdir(parents=True)
    path,cfg=materialize(root,template,scratch)
    require(args['task_name']==cfg['task'],'Organizer task differs from installed binding')
    from .runtime import get_model as native
    return native(dict(args,coupled_all14_manifest=str(path),coupled_all14_manifest_sha256=sha(path)))


def get_model(args):return guarded('get_model',_get_model,args)


def eval(*args,**kwargs):
    from .runtime import eval as function
    return guarded('eval',function,*args,**kwargs)


def reset_model(*args,**kwargs):
    from .runtime import reset_model as function
    return guarded('reset_model',function,*args,**kwargs)


def close_model(*args,**kwargs):
    from .runtime import close_model as function
    return guarded('close_model',function,*args,**kwargs)
