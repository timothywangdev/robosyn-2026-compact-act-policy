"""Run retained policy files with the organizer's evaluator and interpreter."""
import argparse
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
RUNTIME = HERE / 'runtime'
sys.path.insert(0, str(RUNTIME))
from experiments.coupled_act_student50_assembly_fallback_release.entry_cli import (
    command, native_environment)
from experiments.coupled_act_student50_release.binder import ref, write


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--root', default=os.environ.get(
        'ROBOSYN_STUDENT50_PACKAGE_ROOT', str(RUNTIME / 'artifact')))
    parser.add_argument('--evaluator-root', default=os.environ.get('ROBOSYN_EVALUATOR_ROOT'))
    parser.add_argument('--standalone', action='store_true',
                        help='Explicitly use the archived evaluator and installed artifact environment')
    parser.add_argument('--task', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--episodes', required=True, type=int)
    parser.add_argument('--expert-seed-attempts', type=int, default=100)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    root = Path(args.root).absolute()
    if not root.is_dir() or not (root / 'submission.json').is_file():
        parser.error('Run policy-local setup.sh --download-only --root /tmp/rscp-policy first')
    root = root.resolve(strict=True)
    if args.standalone and args.evaluator_root:
        parser.error('--standalone and --evaluator-root cannot be combined')
    official = Path(args.evaluator_root).absolute() if args.evaluator_root else HERE.parents[1]
    evaluator = official / 'scripts' / 'eval_policy.py'
    if not args.standalone and (not evaluator.is_file() or evaluator.is_symlink()):
        parser.error('Place the policy directory in the official evaluation checkout, '
                     'set ROBOSYN_EVALUATOR_ROOT, or explicitly use --standalone')
    official = official.resolve(strict=True)
    receipt, spec = command(root, RUNTIME, args.task, args.seed, args.episodes,
                            args.output, args.expert_seed_attempts)
    if args.standalone:
        env = native_environment(root, RUNTIME, os.environ)
        cwd = root / 'upstream' / 'RoboSynChallenge'
        receipt.update(launch_mode='explicit_archived_standalone')
    else:
        # Preserve the qualified actor controls while using the organizer's
        # interpreter, source checkout, and installed dependencies.
        original = receipt['argv']
        options = original[original.index('--config'):]
        options[options.index('--config') + 1] = str(HERE / 'deploy_policy.yml')
        receipt.update(argv=[sys.executable, '-B', str(evaluator)] + options,
                       launch_mode='organizer_evaluator_and_environment',
                       official_evaluator=ref(evaluator),
                       public_YAML=ref(HERE / 'deploy_policy.yml'),
                       public_fourhooks=ref(HERE / 'deploy_policy.py'),
                       archived_bootstrap_used=False,
                       archived_python_environment_used=False,
                       policy_actor_source_root=str(root / 'upstream/lerobot-sim/src'),
                       official_environment_dependencies_preserved=True)
        env = dict(os.environ)
        paths = [str(official), str(RUNTIME), str(root / 'upstream/lerobot-sim/src'), str(root)]
        if env.get('PYTHONPATH'):
            paths.append(env['PYTHONPATH'])
        env.update(PYTHONPATH=os.pathsep.join(paths),
                   ROBOSYN_STUDENT50_PACKAGE_ROOT=str(root),
                   EMBODICHAIN_SIM_EXIT_PROCESS='0',
                   PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        cwd = official
    write(Path(args.output) / 'command.json', receipt)
    if args.check_only:
        print(json.dumps(receipt))
        return
    from experiments.goal_latent_act_submission.trained_hybrid.runtime_cache import prepare
    env, cache = prepare(root, spec, env, scratch_parent='/tmp')
    write(Path(args.output) / 'private-runtime-cache-proof.json', cache)
    os.chdir(cwd)
    os.execvpe(receipt['argv'][0], receipt['argv'], env)


if __name__ == '__main__':
    main()
