"""Relocated routing, exact official bindings and unknown-name refusal fixtures."""
import argparse,importlib.util,json,sys,tempfile
from pathlib import Path
from .finder import OfficialTaskFinder
from .install import inventory,ref,write,PTH


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.root).resolve(strict=True)
    rows=inventory(root);cases=['every_archived_task_dependency_SHA_matches_original191e']
    old=sys.prefix
    try:
        with tempfile.TemporaryDirectory(prefix='rstaskcpu-') as folder:
            base=Path(folder);(base/'.venvs/sim').mkdir(parents=True)
            task=base/'upstream/EmbodiChain/embodichain_tasks'
            for name in ('embodichain_tasks','configs'):
                (task/name).mkdir(parents=True);(task/name/'__init__.py').write_text('# CPU-only routing fixture; never executed\n')
            sys.prefix=str(base/'.venvs/sim');finder=OfficialTaskFinder()
            for name,sub in [('embodichain_tasks','embodichain_tasks'),('embodichain_tasks.configs','configs')]:
                spec=finder.find_spec(name);assert spec.origin==str(task/sub/'__init__.py') and spec.submodule_search_locations==[str(task/sub)]
                assert '/home/' not in spec.origin;cases.append('relocated_'+name+'_same_nested_layout')
            assert finder.find_spec('unrelated_module') is None and finder.find_spec('embodichain.lab.sim') is None;cases.append('unrelated_imports_unmodified')
            source=task/'configs/__init__.py';source.unlink();source.symlink_to(task/'embodichain_tasks/__init__.py')
            try:finder.find_spec('embodichain_tasks.configs')
            except ImportError:cases.append('symlink_route_rejected')
            else:raise AssertionError('Symlink must reject')
            assert PTH==b'import robosyn_official_task_paths; robosyn_official_task_paths.install()\n';cases.append('site_startup_import_contains_no_original_home_paths')
    finally:sys.prefix=old
    assert 'torch' not in sys.modules
    write(a.output,dict(schema='official_nested_task_import_repair_relocation_actual_CPU_tests_v1',cases=cases,
        official_archived_source_count=len(rows),models=0,worlds=0,GPU_queries=0,Torch_imported=False))
    print(json.dumps(ref(a.output)))

if __name__=='__main__':main()
