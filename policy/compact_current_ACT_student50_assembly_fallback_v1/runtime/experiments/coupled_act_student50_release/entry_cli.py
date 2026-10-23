"""Automatic release binding then the unchanged portable official entry."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.coupled_act_student50_release.binder import bind_all


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--root', default=str(Path(__file__).resolve().parents[2] / 'artifact'))
    p.add_argument('--task', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--episodes', type=int, required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--expert-seed-attempts', type=int, default=100)
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    root = Path(a.root).resolve(strict=True)
    if Path(a.output).exists() or a.episodes <= 0 or not 0 <= a.seed < 2**32:
        raise ValueError('Fresh output and explicit valid organizer controls required')
    temp = Path(tempfile.mkdtemp(prefix='rsc50-', dir='/tmp'))
    binding = bind_all(root, temp / 'bindings')
    if a.task not in binding['tasks']:
        raise ValueError('Exact selected task required')
    row = binding['tasks'][a.task]
    argv = [str(root / '.venvs/sim/bin/python'), '-B',
        str(root / 'experiments/coupled_act_feature_student_cadence50_submission/entry_cli.py'),
        '--root', str(root), '--task', a.task, '--mode', 'student', '--seed', str(a.seed),
        '--episodes', str(a.episodes), '--output', str(Path(a.output).absolute()),
        '--expert-seed-attempts', str(a.expert_seed_attempts),
        '--admission', row['admission']['path'], '--admission-sha256', row['admission']['sha256']]
    if a.check_only:
        print(json.dumps(dict(schema='Student50_automatic_organizer_entry_command_CPU_v1',
            argv=argv, binding=binding, session_authentication_used=False,
            GPU_queries=0, models_constructed=0, worlds_constructed=0)))
        return
    # The selected offline installation provides this interpreter. No arbitrary
    # host environment is substituted and no scientific entry/model is edited.
    if not Path(argv[0]).is_file():
        raise ValueError('Run the pinned offline CPython3.11 installation first')
    os.execvpe(argv[0], argv, os.environ.copy())


if __name__ == '__main__':
    main()
