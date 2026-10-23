"""Run the unchanged evaluator with a narrowly source-bound cleanup-order hook."""
import ast
import hashlib
import os
import json
from pathlib import Path
import runpy
import sys

EVALUATOR_SHA256 = '7f8c3205190451954caefba7f4c4d3ab7a03db1da8c9479746aa055928235dc5'


def inner_argv(arguments):
    values = list(arguments)
    check_only = values[:1] == ['--check-only']
    if check_only: values.pop(0)
    if not values or values.pop(0) != '--' or not values:
        raise ValueError('Expected -- followed by the exact original evaluator argv')
    evaluator = Path(values[0])
    if not evaluator.is_absolute() or evaluator.is_symlink():
        raise ValueError('Original evaluator must be an absolute ordinary file')
    evaluator = evaluator.resolve(strict=True)
    if not str(evaluator).endswith('/upstream/RoboSynChallenge/scripts/eval_policy.py'):
        raise ValueError('Original evaluator layout differs')
    if hashlib.sha256(evaluator.read_bytes()).hexdigest() != EVALUATOR_SHA256:
        raise ValueError('Original evaluator bytes differ')
    project = evaluator.parents[3]
    sdk = project/'upstream/EmbodiChain/embodichain/lab/sim/sim_manager.py'
    from constraint_teardown import SDK_SHA256
    if not sdk.is_file() or sdk.is_symlink() or hashlib.sha256(sdk.read_bytes()).hexdigest() != SDK_SHA256:
        raise ValueError('Original SDK bytes/layout differ')
    return values, sdk, check_only



def evaluator_path_prelude(evaluator):
    """Execute the pinned evaluator's own path setup before importing its SDK.

    Python script execution sets sys.path[0] first. The original evaluator then
    prepends its repository, policy and environment-selected SDK source paths.
    Execute those exact AST statements, not a substitute PYTHONPATH override.
    """
    path = Path(evaluator)
    tree = ast.parse(path.read_text(), filename=str(path))
    start = next(i for i, node in enumerate(tree.body)
                 if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SCRIPT_DIR'
                                                        for t in node.targets))
    end = next(i for i, node in enumerate(tree.body[start:], start)
               if isinstance(node, ast.Import) and any(a.name == 'robosynchallenge' for a in node.names))
    statements = tree.body[start:end]
    namespace = dict(__file__=str(path), os=os, sys=sys)
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(path), 'exec'), namespace)
    return dict(original_evaluator_path_prelude=True,
                paths=[namespace['REPO_ROOT'], namespace['POLICY_DIR'], namespace['EMBODICHAIN_ROOT']])

def main():
    values, sdk, check_only = inner_argv(sys.argv[1:])
    # Match Python's original script search path before any native import.
    # Leave cwd, environment and every evaluator argument unchanged.
    from constraint_teardown import install
    sys.argv = values
    sys.path[0] = str(Path(values[0]).parent)
    prelude = evaluator_path_prelude(values[0])
    binding = install(sdk)
    if check_only:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise ValueError('CPU import qualification requires CUDA hidden')
        import torch
        from embodichain.lab.sim import sim_manager
        if torch.cuda.is_initialized() or sim_manager.dexsim.get_world_num() != 0:
            raise ValueError('CPU import qualification initialized CUDA or constructed a world')
        print(json.dumps(dict(schema='prospective_native_cleanup_actual_CPU_startup_v2',
            inner_argv=values, evaluator_sha256=EVALUATOR_SHA256,
            installed_SDK=str(Path(sim_manager.__file__).resolve()), binding=binding, prelude=prelude,
            CUDA_initialized=False, native_world_count=0, models_constructed=0, GPU_launches=0)))
        return
    print('[teardown-bootstrap] '+json.dumps(binding, sort_keys=True), flush=True)
    # No exit translation, exception suppression, os._exit or deferred-cleanup skip.
    runpy.run_path(values[0], run_name='__main__')


if __name__ == '__main__': main()
