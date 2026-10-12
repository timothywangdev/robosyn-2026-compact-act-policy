"""Run the fixed matched CPU pair and capture actual original process exits."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import time
from experiments.goal_latent_act.prepare import sha, write_json


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, birth=int(fields[19]), group=int(fields[2]), session=int(fields[3]))
    except FileNotFoundError:
        return None


def group_members(group):
    found = []
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            current = identity(int(path.name))
            if current and current['group'] == group:
                found.append(current)
    return found


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--snapshot-sha256', required=True)
    p.add_argument('--highlevel', type=Path, required=True)
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--urdf', type=Path, required=True)
    p.add_argument('--python', required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    report = dict(schema='goal-latent-act-expanding-prefix50-matched-original-exits-v1',
                  status='running_CPU_pair', runs=[], start_UTC=datetime.now(timezone.utc).isoformat(),
                  supervisor_identity=identity(os.getpid()), source_sha256=sha(__file__), GPU_launched=False)
    write_json(a.output/'receipt.json', report)
    for name, goal_free in [('goal', False), ('goal_free', True)]:
        destination = a.output/name
        argv = [a.python, '-B', '-m', 'experiments.goal_latent_act_full_data.train_expanding',
                '--snapshot', str(a.snapshot), '--snapshot-sha256', a.snapshot_sha256,
                '--highlevel', str(a.highlevel), '--highlevel-sha256',
                'd700006da32d600e3505e58d52ccff4c3bd8c09627ebe50c75cce451c5b5bade',
                '--output', str(destination), '--execution-prefix', '50', '--epochs', '30',
                '--gate-epochs', '20', '--threads', '8', '--wall-seconds', '1800',
                '--geometry', str(a.geometry), '--urdf', str(a.urdf)]
        if goal_free:
            argv.append('--goal-free')
        started = time.monotonic()
        with (a.output/f'{name}.stdout.log').open('w') as stdout, (a.output/f'{name}.stderr.log').open('w') as stderr:
            child = subprocess.Popen(argv, stdout=stdout, stderr=stderr, start_new_session=True)
            owned = identity(child.pid)
            if not owned or owned['group'] != child.pid or owned['session'] != child.pid:
                raise RuntimeError('Fresh CPU child identity unavailable')
            run = dict(goal_free=goal_free, output=str(destination), argv=argv, identity=owned,
                       original_exit_code=None, status='actual_CPU_child_running')
            report['runs'].append(run)
            write_json(a.output/'receipt.json', report)
            print(f'{name} actual CPU PID={child.pid} birth={owned["birth"]}', flush=True)
            # Child's unchanged SIGALRM bounds the entire operation at 1800 seconds.
            original_exit = child.wait()
        current = identity(child.pid)
        remaining = group_members(child.pid)
        run.update(original_exit_code=original_exit, original_child_reaped=True,
                   original_process_absent=current is None,
                   same_birth_present=current is not None and current['birth'] == owned['birth'],
                   independently_observed_group_after_reap=remaining,
                   natural_completion_no_signals_sent=True, elapsed_seconds=time.monotonic()-started,
                   status='captured_original_exit')
        if (destination/'receipt.json').exists():
            run['training_receipt_sha256'] = sha(destination/'receipt.json')
        write_json(a.output/'receipt.json', report)
        if original_exit != 0 or current is not None or remaining:
            report['status'] = 'failed_preserved_no_retry'
            write_json(a.output/'receipt.json', report)
            raise RuntimeError(f'{name} actual exit/drain failed')
    report.update(status='both_original_exit0_natural_CPU_completion', completed_UTC=datetime.now(timezone.utc).isoformat())
    write_json(a.output/'receipt.json', report)
    print(report['status'], flush=True)


if __name__ == '__main__':
    main()
