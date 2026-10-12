"""Concrete preparation CLI; execution remains blocked until portable review."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--root', default=str(Path(__file__).resolve().parents[2]))
    p.add_argument('--task', required=True)
    p.add_argument('--mode', choices=('parent', 'goal', 'goal_free'), required=True)
    p.add_argument('--setting', default='random')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--episodes', type=int, required=True)
    p.add_argument('--expert-seed-attempts', type=int, default=100)
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    root = Path(a.root).resolve(strict=True)
    manifest = json.loads((root/'preparation.json').read_text())
    rows = [r for r in manifest['tasks'] if (r['task'],r['mode']) == (a.task,a.mode)]
    assert len(rows)==1 and a.setting=='random' and 0<=a.seed<2**32
    assert a.episodes>0 and a.expert_seed_attempts>0
    if not a.check_only:
        raise RuntimeError('Preparation only: this entry has no portable inference admission')
    row=rows[0]
    print(json.dumps(dict(status='CPU_preparation_command_plan_only',
        task=a.task,mode=a.mode,setting=a.setting,organizer_scene_seed=a.seed,
        organizer_episodes=a.episodes,expert_attempts=a.expert_seed_attempts,
        official_horizon=row['official_horizon'],native_generated=50,prefix_executed=10,
        template=row['template'],models_constructed=0,GPU_queries=0,worlds_constructed=0,
        portable_runtime_binding_qualified=False,submission_ready=False)))
    assert 'torch' not in sys.modules


if __name__=='__main__':
    main()
