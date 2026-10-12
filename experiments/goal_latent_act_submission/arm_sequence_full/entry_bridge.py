"""Visible organizer lifecycle for one explicit experimental ARM12 variant."""
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def prepare_native(root,args):
    root=Path(root).resolve(strict=True)
    sys.path[:0]=[str(root),str(root/'upstream/lerobot-sim/src'),str(root/'upstream/EmbodiChain'),str(root/'upstream/RoboSynChallenge')]
    from experiments.goal_latent_act_submission.arm_sequence_full.contracts import verify_stage,materialize,load_native,json_write,sha,child,require
    from experiments.goal_latent_act_submission.arm_sequence_full.run_task import metadata_check,environment
    spec=json.loads((root/'submission.json').read_text());verify_stage(root,spec);metadata_check(root,spec)
    mode=args['release_mode'];task=args['task_name'];require(mode in spec['banks'] and task in spec['banks'][mode],'Explicit mode/task required; no selected primary')
    output=Path(args['eval_result_dir']).absolute()/'visible-arm-binding';require(not output.exists(),'Fresh organizer binding required');output.mkdir(parents=True)
    scratch=output/'scratch';scratch.mkdir()
    cfg=materialize(root,json.loads(child(root,spec['banks'][mode][task]['template']).read_text()),scratch)
    manifest=output/'deployment.json';json_write(manifest,cfg);load_native(root,manifest,sha(manifest))
    env,cache=environment(root,spec)
    os.environ.clear();os.environ.update(env)
    paths=env['PYTHONPATH'].split(os.pathsep)
    for path in paths:
        while path in sys.path:sys.path.remove(path)
    sys.path[:0]=paths
    # Direct organizer imports need the same narrow, source-bound teardown
    # primitive as the native bootstrap; the original SDK remains unchanged.
    from experiments.goal_latent_act_submission.arm_sequence_full.run_task import HOOK
    hook=child(root,'experiments/simulator_cleanup_repair_v1/constraint_teardown.py')
    require(sha(hook)==HOOK,'Qualified direct cleanup hook differs')
    module_spec=importlib.util.spec_from_file_location('_portable_arm_direct_constraint_teardown',hook)
    cleanup=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(cleanup)
    cleanup.install(root/'upstream/EmbodiChain/embodichain/lab/sim/sim_manager.py')
    native=dict(args,goal_latent_manifest=str(manifest),goal_latent_manifest_sha256=sha(manifest),act_step=50)
    return root,native


def bind(args):
    root=Path(args['materialized_root']).resolve(strict=True)
    if Path(sys.prefix).resolve()!=(root/'.venvs/sim').resolve():
        raise ValueError('Contained isolated SIM interpreter required; eval.sh performs setup')
    return prepare_native(root,args)


def get_model(args):
    root,native=bind(args)
    actor=importlib.import_module('experiments.goal_latent_act_arm_runtime_full.runtime')
    if Path(actor.__file__).resolve()!=root/'experiments/goal_latent_act_arm_runtime_full/runtime.py':
        raise ValueError('Cached actor from another source tree refused')
    return actor,actor.get_model(native)


def eval(env,model,obs):return model[0].eval(env,model[1],obs)
def reset_model(model):return model[0].reset_model(model[1])
def close_model(model):return model[0].close_model(model[1])
