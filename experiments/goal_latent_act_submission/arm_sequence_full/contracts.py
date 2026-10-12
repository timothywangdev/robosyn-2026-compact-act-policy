"""Portable paths around unchanged frozen ARM12 actor and scientific guards."""
import copy
import importlib
import json
from pathlib import Path
import sys
from experiments.goal_latent_act_submission.contracts import TASKS,require,sha,child,reference,json_write

SCHEMA='goal-latent-act-full63000-temporal-arm12-portable-submission-v1'
TEMPLATE='goal-latent-act-full63000-temporal-arm12-portable-template-v1'
PARENT='goal-latent-act-commanded50-full63000-temporal-arm12-deployment-v1'
MODES={'parent','goal','goal_free'}


def same(a,b):return all(a[k]==b[k] for k in ('bytes','sha256'))


def verify_stage(root,spec,full=True):
    require(spec['schema']==SCHEMA and spec['parent_schema']==PARENT,'Separately typed ARM12 portable candidate required')
    require(set(spec['banks'])==MODES and all(set(bank)==TASKS for bank in spec['banks'].values()),'All3/all10 explicit branches required')
    require(spec['primary_variant'] is None and spec['offline_promotion_criterion_passed'] is False
            and spec['RTX5090_qualified'] is False,'No automatic promotion or hardware qualification')
    require(spec['protocol']==dict(generated=50,executed=50,refresh=50,discarded=0,motor_seed=17,
        ACT_core='eager',ACT_preprocessing='native_every_action',arms='bounded +/-0.15 radians handwise correction',
        grippers='bitexact original native ACT'),'Frozen actor/protocol required')
    for name,row in spec['files'].items():
        path=child(root,name)
        require(path.is_file() and not path.is_symlink() and path.stat().st_size==row['bytes'],'Portable file type/size differs: '+name)
        if full:require(sha(path)==row['sha256'],'Portable file bytes differ: '+name)


def _absolute_refs(root,value):
    if isinstance(value,dict):
        if {'path','bytes','sha256'}<=set(value):
            return dict(value,path=str(child(root,value['path'])))
        return {k:_absolute_refs(root,v)for k,v in value.items()}
    if isinstance(value,list):return [_absolute_refs(root,v)for v in value]
    return value


def materialize(root,template,scratch):
    root=Path(root).resolve(strict=True);scratch=Path(scratch).resolve(strict=True)
    cfg=copy.deepcopy(template)
    require(cfg.pop('schema')==TEMPLATE and cfg.pop('parent_schema')==PARENT,'No older actor ABI promotion')
    cfg.update(schema=PARENT,source_root=str(root),scratch_root=str(scratch),
        act_root=str(child(root,cfg['act_root'])),qwen_root=str(child(root,cfg['qwen_root'])),
        qwen_python=str(root/'.venvs/qwen35/bin/python'),qwen_python_path=[str(root/'dependencies/qwen-overlay'),str(root)])
    for group in ('assets','act_files','qwen_files'):
        cfg[group]=_absolute_refs(root,cfg[group])
    if cfg['mode']!='parent':
        pair=_absolute_refs(root,json.loads(child(root,'training/qualified-pair-template.json').read_text()))
        # Original historical receipts and checkpoint contracts remain byte-identical.
        # Only this explicitly typed transport packet is regenerated with new refs.
        from experiments.goal_latent_act_arm_runtime_full.qualification import qualify
        transported=qualify(pair['original_pair'],pair['original_wait'],pair['supervisor'],pair['independent_terminal'],
            pair['prepared'],pair['head_contract'],{k:r['contract']for k,r in pair['branches'].items()},
            {k:r['training_receipt']for k,r in pair['branches'].items()},pair['historical_reference_aliases'])
        target=scratch/'qualified-training-pair.json';json_write(target,transported)
        cfg['assets']['training_pair']=reference(target)
        admission=json.loads(Path(cfg['assets']['robot_admission']['path']).read_text())
        require(admission['purpose']=='exploratory_native_comparison_after_failed_offline_promotion'
                and admission['offline_promotion_criterion_passed'] is False,'Original failed promotion remains explicit')
        admission['training_pair_sha256']=cfg['assets']['training_pair']['sha256']
        target=scratch/'exploratory-prerequisites.json';json_write(target,admission)
        cfg['assets']['robot_admission']=reference(target)
    return cfg


def load_native(root,manifest,expected):
    name='experiments.goal_latent_act_arm_runtime_full.contracts'
    module=importlib.import_module(name)
    require(Path(module.__file__).resolve()==child(root,'experiments/goal_latent_act_arm_runtime_full/contracts.py').resolve(),
            'Imported ARM guard must be from this reconstructed source root')
    return module.load_manifest(manifest,expected)
