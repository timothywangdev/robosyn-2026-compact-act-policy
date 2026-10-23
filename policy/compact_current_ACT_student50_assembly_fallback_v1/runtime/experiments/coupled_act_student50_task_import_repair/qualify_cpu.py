"""Import the actual installed official evaluator and all tasks without worlds."""
import argparse,hashlib,importlib,json,os
from pathlib import Path
import runpy
import sys

TASKS=('click_bell','drawer_open_place','handle_basket','item_assembly','items_handover','manipulate_pipette','mixer_operating','sample_loading','table_rearrangement','water_pouring')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):
    p=Path(p).resolve(strict=True);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--temporary-route',action='store_true');a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    root=Path(a.root).resolve(strict=True);out=Path(a.output).resolve();assert not out.exists()
    assert Path(sys.prefix).resolve(strict=True)==root/'.venvs/sim'
    paths=[root,root/'upstream/lerobot-sim/src',root/'upstream/EmbodiChain',root/'upstream/RoboSynChallenge']
    sys.path[:0]=list(map(str,paths));os.environ['EMBODICHAIN_ROOT']=str(root/'upstream/EmbodiChain')
    if a.temporary_route:
        source=Path(__file__).with_name('finder.py');spec=importlib.util.spec_from_file_location('_actual_official_task_route',source)
        finder=importlib.util.module_from_spec(spec);spec.loader.exec_module(finder);finder.install()
    else:
        import robosyn_official_task_paths as finder
        assert ref(finder.__file__)['sha256']==sha(Path(__file__).with_name('finder.py'))
    import robosynchallenge
    import embodichain_tasks
    from embodichain_tasks.tableware.base_agent_env import BaseAgentEnv
    import embodichain_tasks.configs as task_configs
    assert Path(embodichain_tasks.__file__).resolve()==root/'upstream/EmbodiChain/embodichain_tasks/embodichain_tasks/__init__.py'
    assert Path(task_configs.__file__).resolve()==root/'upstream/EmbodiChain/embodichain_tasks/configs/__init__.py'
    modules={}
    for task in TASKS:
        module=importlib.import_module('robosynchallenge.tasks.'+task+'.'+task);path=Path(module.__file__).resolve()
        assert path==root/('upstream/RoboSynChallenge/robosynchallenge/tasks/'+task+'/'+task+'.py')
        modules[task]=ref(path)
    # The literal evaluator FILE import previously failed at robosynchallenge
    # initialization. Import its functions now, without calling main/get_model.
    evaluator=root/'upstream/RoboSynChallenge/scripts/eval_policy.py'
    scope=runpy.run_path(str(evaluator),run_name='_actual_installed_official_evaluator_FILE')
    assert callable(scope['load_policy_adapter']) and callable(scope['main'])
    import torch
    from embodichain.lab.sim import sim_manager
    assert not torch.cuda.is_initialized() and sim_manager.dexsim.get_world_num()==0
    value=dict(schema='actual_installed_Student50_official_all10_task_and_evaluator_CPU_import_v1',
        task_sources=modules,task_package=ref(embodichain_tasks.__file__),task_configs=ref(task_configs.__file__),
        evaluator=ref(evaluator),temporary_route=a.temporary_route,installed_route=not a.temporary_route,
        CUDA_initialized=False,worlds=0,models=0,GPU_queries=0,GPU_launches=0,package=ref(root/'submission.json'),
        native_qualified=False,actor_or_weight_change=False)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(ref(out)))

if __name__=='__main__':main()
