"""Official portable hooks require separate package-specific native admission."""
import json
from pathlib import Path
from .contracts import materialize,require,ref,sha
from experiments.coupled_goal_act_all14_runtime.policy_shim_eager import guarded

ADMISSION='root_authorized_compact_ACT_student50_portable_native_v1'

def _get_model(args):
    task=args['task_name'];mode=args.get('compact_student_mode','student')
    require(mode in ('parent','student'),'Explicit compact or raw ACT mode')
    root=Path(args.get('compact_student_submission_root',Path(__file__).resolve().parents[2])).resolve(strict=True)
    def installed(value):
        p=Path(value);return (p if p.is_absolute() else root/p).resolve(strict=True)
    template=installed(args.get('compact_student_submission_template','templates/'+mode+'-'+task+'.json'))
    proof=installed(args.get('compact_student_portable_admission','runtime/native-admission-'+task+'.json'))
    require(sha(proof)==args['compact_student_portable_admission_sha256'],'Exact separate portable admission')
    a=json.loads(proof.read_text())
    require(a['schema']==ADMISSION and a['execution_authorized'] is True and a['robot_quality_promoted'] is False,
            'Separate root portable execution admission required')
    require(a['package']==ref(root/'submission.json') and a['template']==ref(template)
            and a['entry_source']==ref(Path(__file__)),'Actual package/template/entry differs')
    scratch=Path(args['compact_student_portable_scratch']).absolute()
    require(not scratch.exists() and not scratch.is_symlink(),'Fresh portable scratch');scratch.mkdir(parents=True)
    path,cfg=materialize(root,template,scratch);require(args['task_name']==cfg['task'],'Organizer task differs')
    from .runtime import get_model as call
    return call(dict(args,coupled_student_manifest=str(path),coupled_student_manifest_sha256=sha(path)))

def get_model(args):return guarded('get_model',_get_model,args)
def eval(*args,**kwargs):
    from .runtime import eval as call
    return guarded('eval',call,*args,**kwargs)
def reset_model(*args,**kwargs):
    from .runtime import reset_model as call
    return guarded('reset_model',call,*args,**kwargs)
def close_model(*args,**kwargs):
    from .runtime import close_model as call
    return guarded('close_model',call,*args,**kwargs)
