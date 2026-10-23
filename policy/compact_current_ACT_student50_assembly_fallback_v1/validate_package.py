"""Exercise the actual submitted folder after discarding every ancestor source."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--artifact-root', type=Path)
    parser.add_argument('--official-evaluator', type=Path,
                        help='Copy an actual official eval_policy.py for command construction checks')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((source / 'runtime-files.json').read_text())
    actual = {p.relative_to(source / 'runtime').as_posix()
              for p in (source / 'runtime').rglob('*') if p.is_file()
              and '__pycache__' not in p.parts}
    inventory = dict(manifest['files'], **manifest.get('supplemental_files', {}))
    assert actual == set(inventory), 'Runtime inventory differs'
    for name, row in inventory.items():
        path = source / 'runtime' / name
        assert not path.is_symlink()
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], name
    receipt = dict(schema='Student50_policy_only_relocation_checks_v1',
                   source_commit=manifest['source_commit'],
                   HF_revision=manifest['HF_revision'],
                   byte_identical_original_runtime_files=len(manifest['files']),
                   supplemental_metadata_files=len(manifest.get('supplemental_files', {})),
                   checks=[],
                   models_constructed=0, GPU_queries=0, worlds_constructed=0,
                   native_evaluation_performed=False, cold_install_performed=False)
    started = time.monotonic()

    def save():
        (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')

    def run(name, command, cwd, env):
        before = time.monotonic()
        result = subprocess.run(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                text=True, capture_output=True, timeout=300)
        (output / (name + '.stdout.txt')).write_text(result.stdout)
        (output / (name + '.stderr.txt')).write_text(result.stderr)
        receipt['checks'].append(dict(name=name, command=command,
                                     exit_code=result.returncode,
                                     seconds=time.monotonic() - before))
        save()
        assert result.returncode == 0, (name, result.stderr[-4000:])
        return result

    with tempfile.TemporaryDirectory(prefix='rsc-policy-only-') as temporary:
        prepared = Path(temporary)
        policy = prepared / 'policy' / source.name
        shutil.copytree(source, policy, ignore=shutil.ignore_patterns(
            'artifact', '__pycache__', '*.pyc'))
        assert not (prepared / 'experiments').exists()
        assert not (prepared / 'release').exists()
        assert not (prepared / 'setup.sh').exists()
        (prepared / 'scripts').mkdir()
        if args.official_evaluator:
            evaluator_source = args.official_evaluator.resolve(strict=True)
            shutil.copy2(evaluator_source, prepared / 'scripts/eval_policy.py')
            receipt['official_evaluator_source'] = dict(
                path=str(evaluator_source), sha256=sha(evaluator_source))
        else:
            # Check-only must not execute this evaluator-location fixture.
            (prepared / 'scripts/eval_policy.py').write_text(
                "raise RuntimeError('Command construction must not execute the evaluator fixture')\n")
        env = dict(os.environ)
        for name in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE',
                     'ROBOSYN_STUDENT50_PACKAGE_ROOT', 'PYTHON_BIN', 'PYTHON311'):
            env.pop(name, None)
        env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
                   CUDA_VISIBLE_DEVICES='', PYTHON311=args.python)
        run('policy-local-setup-help', ['bash', str(policy / 'setup.sh'), '--help'],
            prepared, env)
        run('policy-local-eval-help', ['bash', str(policy / 'eval.sh'), '--help'],
            prepared, env)
        # Isolated interpreter: no source checkout, user packages, or inherited
        # PYTHONPATH can make this policy-only import succeed accidentally.
        code = '''import importlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);sys.path.insert(0,str(root))
hooks=importlib.import_module('policy.'+sys.argv[2]+'.deploy_policy')
assert all(callable(getattr(hooks,n)) for n in ('get_model','eval','reset_model','close_model'))
runtime=root/'policy'/sys.argv[2]/'runtime'
assert Path(hooks._policy.__file__).is_relative_to(runtime)
assert hooks._policy.route('item_assembly')=='plain_ACT50_Assembly'
assert all(hooks._policy.route(t)=='unchanged_public_Student50' for t in hooks._policy.TASKS if t!='item_assembly')
for name in hooks._policy.PINS:
    assert Path(hooks._policy.pinned_module(name).__file__).is_relative_to(runtime)
cache=importlib.import_module('experiments.goal_latent_act_submission.trained_hybrid.runtime_cache')
assert Path(cache.__file__).is_relative_to(runtime)
assert 'torch' not in sys.modules
print(json.dumps(dict(four_hooks=True,all10_task_routes=True,pinned_module_hashes=True,private_runtime_import=True,torch_imported=False)))
'''
        run('isolated-four-hooks-and-source-pins',
            [args.python, '-I', '-B', '-c', code, str(prepared), source.name], prepared, env)
        code = '''import importlib,json,sys,types
from pathlib import Path
root=Path(sys.argv[1]);sys.path.insert(0,str(root))
hooks=importlib.import_module('policy.'+sys.argv[2]+'.deploy_policy')
fixture=root/'fixture-artifact';actor=fixture/'upstream/lerobot-sim/src';actor.mkdir(parents=True)
hooks._policy.get_model=lambda value:value
args=dict(task_name='item_assembly',compact_student_submission_root=str(fixture))
received=hooks.get_model(args)
assert received==args and received is not args and sys.path[0]==str(actor)
wrong=root/'wrong-act.py';wrong.write_text('# Deliberately incompatible already-imported module')
sys.modules['lerobot.policies.act.modeling_act']=types.SimpleNamespace(__file__=str(wrong))
try:hooks.get_model(args)
except ValueError as error:assert 'already imported a different LeRobot ACT' in str(error)
else:raise AssertionError('Preloaded incompatible actor source accepted')
assert 'torch' not in sys.modules
print(json.dumps(dict(actor_source_path_selected=True,organizer_args_preserved=True,incompatible_preloaded_actor_rejected=True,models_constructed=0)))
'''
        run('direct-model-source-path-and-collision',
            [args.python, '-I', '-B', '-c', code, str(prepared), source.name], prepared, env)
        run('original-binder-contract-tests',
            [args.python, '-I', '-B', '-c',
             "import runpy,sys;sys.path.insert(0,sys.argv[1]);runpy.run_module('experiments.coupled_act_student50_release.test_cpu',run_name='__main__')",
             str(policy / 'runtime')], prepared, env)
        if args.artifact_root:
            artifact = args.artifact_root.resolve(strict=True)
            env.update(ROBOSYN_STUDENT50_PACKAGE_ROOT=str(artifact), PYTHON_BIN=args.python)
            tasks = json.loads((policy / 'runtime' / 'task-checkpoints.json').read_text())
            for task, row in tasks.items():
                target = prepared / ('check-' + task)
                result = run('check-only-' + task,
                    ['bash', str(policy / 'eval.sh'), '--task', task,
                     '--seed', '1701', '--episodes', '1', '--output', str(target),
                     '--check-only'], prepared, env)
                command = json.loads(result.stdout)
                argv = command['argv']
                assert command['task'] == task
                assert command['official_horizon'] == row['official_horizon']
                assert argv[argv.index('--act_step') + 1] == ('10' if task == 'item_assembly' else '50')
                assert command['models_constructed'] == command['worlds_constructed'] == command['GPU_queries'] == 0
                assert Path(command['public_YAML']['path']) == policy / 'deploy_policy.yml'
                assert command['launch_mode'] == 'organizer_evaluator_and_environment'
                assert argv[:3] == [args.python, '-B', str(prepared / 'scripts/eval_policy.py')]
                assert not command['archived_bootstrap_used']
                assert not command['archived_python_environment_used']
                assert command['policy_actor_source_root'] == str(artifact / 'upstream/lerobot-sim/src')
                shutil.copy2(target / 'command.json', output / ('command-' + task + '.json'))
    receipt.update(status='PASS', seconds=time.monotonic() - started,
                   all10_actual_shell_check_only=bool(args.artifact_root))
    save()
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
