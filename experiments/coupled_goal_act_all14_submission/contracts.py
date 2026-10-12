"""Frozen scientific predicates with an explicitly typed asset transport layer."""
import ast
from copy import deepcopy
import inspect
import json
from pathlib import Path
import types
from experiments.coupled_goal_act_all14_runtime import contracts as original
from experiments.coupled_goal_act_all14_runtime import runtime_eager as eager
from .resolver import Resolver
from .prepare import sha,ref,write

STATS=original.STATS
load_official=original.load_official
module=original.module
require=original.require
verify_ref=original.verify_ref
SCHEMA='coupled_all14_portable_operational_deployment_v1'
PACKAGE='coupled_all14_byte_preserving_portable_package_v1'


def borrowed(function,overrides):
    return types.FunctionType(function.__code__,dict(function.__globals__,**overrides),function.__name__,function.__defaults__)


class InventoryPath(ast.NodeTransformer):
    def __init__(self):self.changes=0
    def visit_Call(self,node):
        if ast.dump(node,include_attributes=False)==ast.dump(ast.parse("Path(ref['path']).resolve()",mode='eval').body,include_attributes=False):
            self.changes+=1
            return ast.copy_location(ast.parse('verify_ref(ref).resolve()',mode='eval').body,node)
        return self.generic_visit(node)


def native_predicate(resolver):
    # One operational path equality is resolved through the byte-bound table;
    # every training/inference/readiness predicate otherwise retains its code.
    tree=ast.parse(inspect.getsource(original.load_manifest));change=InventoryPath();tree=change.visit(tree)
    require(change.changes==1,'Frozen ACT inventory path boundary changed')
    namespace=dict(original.load_manifest.__globals__)
    namespace.update(verify_ref=resolver.verify_ref,
        validate_training=borrowed(original.validate_training,dict(verify_ref=resolver.verify_ref)))
    exec(compile(ast.fix_missing_locations(tree),'<portable-exact-native-predicates>','exec'),namespace)
    return namespace['load_manifest']


def materialize(root,template,scratch,*,operational_scratch=None):
    root=Path(root).resolve(strict=True);scratch=Path(scratch).resolve(strict=True)
    spec=json.loads((root/'submission.json').read_text())
    require(spec['schema']==PACKAGE and spec['status']=='CPU_prepared_no_native_or_5090_promotion','Typed unpromoted portable package required')
    require(spec['robot_quality_promoted'] is False and spec['offline_5percent_criterion_passed'] is False,'Promotion cannot be transported')
    for name,row in spec['source_files'].items():
        path=root/name
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root) and sha(path)==row['sha256'],'Installed frozen/new source changed: '+name)
    resolver=Resolver(root,spec['reference_bindings'])
    template=Path(template).resolve(strict=True)
    row=next((r for r in spec['tasks'] if root/r['template']==template),None)
    require(row is not None and sha(template)==row['template_sha256'],'Exact registered portable template required')
    t=json.loads(template.read_text());raw=json.loads(resolver.verify_ref(t['original_deployment']).read_text())
    require((raw['task'],raw['mode'])==(row['task'],row['mode']),'Original task/mode differs')
    cfg=deepcopy(raw)
    cfg.update(source_root=str(root),scratch_root=str(Path(operational_scratch).resolve(strict=True) if operational_scratch else scratch),act_root=str(root/t['act_root']),
        qwen_root=str(root/t['qwen_root']),qwen_python=str(root/'.venvs/qwen35/bin/python'),
        qwen_python_path=[str(root/'dependencies/qwen-overlay'),str(root)])
    logical=scratch/'historical-reference-view.json';write(logical,cfg)
    native=native_predicate(resolver)
    checked=borrowed(eager.load_eager_manifest,dict(load_manifest=native,verify_ref=resolver.verify_ref))(logical,sha(logical))
    checked.pop('_manifest',None)
    for group in ('assets','act_files','qwen_files'):
        if group=='qwen_files' and cfg['mode']!='goal':
            # The state/parent model never reads a Qwen asset; keep their exact
            # logical metadata, without requiring Qwen in a state-only package.
            checked[group]={}
            continue
        checked[group]={name:resolver.operational_ref(value) for name,value in checked[group].items()}
    checked['schema']=SCHEMA
    checked['portable_binding']=dict(package=ref(root/'submission.json'),template=ref(template),
        original_deployment=t['original_deployment'],raw_records_rewritten=False,only_operational_paths_rebound=True)
    target=scratch/'operational-deployment.json';write(target,checked)
    return target,checked


def load_manifest(path,expected):
    path=Path(path).resolve(strict=True);require(sha(path)==expected,'Exact operational deployment required')
    cfg=json.loads(path.read_text());require(cfg['schema']==SCHEMA,'New portable runtime cannot load a native/development manifest')
    binding=cfg['portable_binding'];root=Path(cfg['source_root']).resolve(strict=True)
    require(binding['package']==ref(root/'submission.json'),'Installed package identity differs')
    require(binding['raw_records_rewritten'] is False and binding['only_operational_paths_rebound'] is True,'Original records must remain immutable')
    # Every runtime/observer construction repeats the actual frozen predicates
    # through a fresh transport view, rather than trusting a CPU receipt flag.
    import tempfile
    with tempfile.TemporaryDirectory(prefix='portable-binding-',dir=cfg['scratch_root']) as temp:
        _,again=materialize(root,binding['template']['path'],temp,operational_scratch=cfg['scratch_root'])
        for key in ('portable_binding',):again.pop(key,None)
        compare=deepcopy(cfg);compare.pop('portable_binding',None)
        require(again==compare,'Materialized operational metadata changed')
    cfg['_manifest']=path
    return cfg
