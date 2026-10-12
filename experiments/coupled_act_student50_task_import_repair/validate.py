"""Typed read-only qualification of the separately installed routing overlay."""
import json
from pathlib import Path
from .install import inventory,ref,sha,PTH


def validate(proof,qualified,root):
    root=Path(root).resolve(strict=True)
    def read(binding):
        path=Path(binding['path']);assert path.is_absolute() and not path.is_symlink() and ref(path)==binding
        return json.loads(path.read_text())
    receipt=read(proof);cpu=read(qualified)
    assert receipt['schema']=='Student50_official_nested_task_import_routing_repair_v1' and receipt['installed'] is True
    assert receipt['package']==cpu['package']==ref(root/'submission.json')
    assert receipt['official_source_inventory']==inventory(root) and receipt['relocatable_sys_prefix_routing'] is True
    for name in ('finder_source','installer_source','installed_finder','installed_pth'):
        source=receipt[name];assert ref(source['path'])==source
    assert receipt['installed_finder']['sha256']==receipt['finder_source']['sha256']
    assert Path(receipt['installed_finder']['path']).is_relative_to(root/'.venvs/sim')
    assert Path(receipt['installed_pth']['path']).is_relative_to(root/'.venvs/sim')
    assert Path(receipt['installed_pth']['path']).read_bytes()==PTH and receipt['new_bytes']==1254
    assert cpu['schema']=='actual_installed_Student50_official_all10_task_and_evaluator_CPU_import_v1'
    assert cpu['installed_route'] is True and cpu['temporary_route'] is False and cpu['CUDA_initialized'] is False
    assert cpu['models']==cpu['worlds']==cpu['GPU_queries']==cpu['GPU_launches']==0
    assert len(cpu['task_sources'])==10 and cpu['native_qualified'] is False
    for row in cpu['task_sources'].values():assert ref(row['path'])==row
    assert ref(root/'upstream/EmbodiChain/embodichain_tasks/embodichain_tasks/__init__.py')==cpu['task_package']
    assert ref(root/'upstream/EmbodiChain/embodichain_tasks/configs/__init__.py')==cpu['task_configs']
    return receipt,cpu
