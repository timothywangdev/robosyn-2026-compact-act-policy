"""Original Student evidence predicates plus a byte-bound asset transport view."""
from copy import deepcopy
import json
from pathlib import Path
import types
from experiments.coupled_act_feature_student_cadence50 import contracts as original
from experiments.coupled_goal_act_all14_submission.resolver import Resolver
from experiments.coupled_goal_act_all14_submission.prepare import ref,sha,write

require=original.require
STATS=original.STATS
PACKAGE='compact_current_ACT_student50_byte_preserving_portable_package_v1'
SCHEMA='compact_current_ACT_student50_portable_operational_deployment_v1'


def borrowed(function,overrides):
    return types.FunctionType(function.__code__,dict(function.__globals__,**overrides),function.__name__,function.__defaults__)


def materialize(root,template,scratch,*,operational_scratch=None):
    root=Path(root).resolve(strict=True);scratch=Path(scratch).resolve(strict=True)
    spec=json.loads((root/'submission.json').read_text())
    require(spec['schema']==PACKAGE and spec['status']=='CPU_prepared_no_native_or_5090_or_quality_promotion','Explicit unpromoted compact package')
    require(spec['robot_quality_promoted'] is False and spec['submission_ready'] is False,'No transported quality qualification')
    for name,row in spec['source_files'].items():
        path=root/name
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root) and sha(path)==row['sha256'],'Installed source changed: '+name)
    resolver=Resolver(root,spec['reference_bindings'])
    template=Path(template).resolve(strict=True)
    row=next((r for r in spec['tasks'] if root/r['template']==template),None)
    require(row is not None and sha(template)==row['template_sha256'],'Exact registered compact template')
    t=json.loads(template.read_text());raw=json.loads(resolver.verify_ref(t['original_deployment']).read_text())
    require((raw['task'],raw['mode'])==(row['task'],row['mode']),'Original mode/task changed')
    cfg=deepcopy(raw)
    cfg.update(source_root=str(root),act_root=str(root/t['act_root']),
        scratch_root=str(Path(operational_scratch).resolve(strict=True) if operational_scratch else scratch))
    logical=scratch/'historical-reference-view.json';write(logical,cfg)
    # The function code, all science predicates and raw nested references stay
    # unchanged. Only verifying an original reference resolves installed bytes.
    predicate=borrowed(original.load_manifest,dict(verify=resolver.verify_ref,verify_ref=resolver.verify_ref))
    checked=predicate(logical,sha(logical));checked.pop('_manifest',None)
    historical=deepcopy(checked['assets'])
    for group in ('assets','act_files'):
        checked[group]={name:resolver.operational_ref(value) for name,value in checked[group].items()}
    checked['schema']=SCHEMA
    checked['portable_binding']=dict(package=ref(root/'submission.json'),template=ref(template),
        original_deployment=t['original_deployment'],historical_assets=historical,
        raw_records_rewritten=False,only_operational_paths_rebound=True)
    target=scratch/'operational-deployment.json';write(target,checked)
    return target,checked


def load_manifest(path,expected):
    path=Path(path).resolve(strict=True);require(sha(path)==expected,'Exact compact operational manifest')
    cfg=json.loads(path.read_text());require(cfg['schema']==SCHEMA,'Separate portable operational schema required')
    root=Path(cfg['source_root']).resolve(strict=True);binding=cfg['portable_binding']
    require(binding['package']==ref(root/'submission.json') and binding['raw_records_rewritten'] is False
            and binding['only_operational_paths_rebound'] is True,'Byte-preserving transport identity changed')
    import tempfile
    with tempfile.TemporaryDirectory(prefix='compact-binding-',dir=cfg['scratch_root']) as temp:
        _,again=materialize(root,binding['template']['path'],temp,operational_scratch=cfg['scratch_root'])
        compare=deepcopy(cfg)
        for body in (again,compare):body.pop('portable_binding',None)
        require(again==compare,'Portable operational metadata changed')
    cfg['_manifest']=path;return cfg


def validate_saved(saved,cfg,training):
    # A checkpoint's historical feature_plan reference is evidence, not an
    # operational path. Keep the exact original saved-checkpoint predicate.
    view=deepcopy(cfg);view['assets']=deepcopy(cfg['portable_binding']['historical_assets'])
    return original.validate_saved(saved,view,training)
