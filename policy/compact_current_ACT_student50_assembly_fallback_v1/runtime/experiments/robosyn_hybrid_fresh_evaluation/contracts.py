"""CPU-only immutable candidate choices; dependency admission remains separate."""
import hashlib
import importlib.util
import json
from pathlib import Path

SCHEMA = 'robosyn_trained_released_hybrid_fresh200_v1'
ORDER = ('click_bell', 'drawer_open_place', 'item_assembly', 'items_handover', 'table_rearrangement',
         'mixer_operating', 'sample_loading', 'water_pouring', 'manipulate_pipette', 'handle_basket')
HORIZONS = dict(zip(ORDER, (361, 900, 500, 350, 361, 500, 500, 500, 1000, 500)))
SEED = 2026101201
GPU = 'GPU-d308805c-10d7-4893-1cb4-f788b16b3634'
DEADLINE = '2026-10-12T06:00:00+00:00'
V10_SHA = '128c7e237d1ed03390c83a493fc50352c9635abc823c2b21332a6f0b3780460d'
TRAINED_SHA = 'd92093ae8099ee104abd91d35f7c8c5150c4fc379df566c4660d55d88b1e3e83'
V10_RUNNER_SHA = '5fcd5ceccaa90326f2c0170a3e546510df242496a3484dfb4f71c3f559105127'
V8_SHA = '9cccdf88b8df19dcc4b73b1c86d7bcd5efbe77de2bf857ac28a10905460c9fd3'


def require(value, message):
    if not value:
        raise ValueError(message)


def ref(path):
    p = Path(path).absolute()
    require(p.is_file() and not p.is_symlink(), 'Bound ordinary regular file required: '+str(p))
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=h.hexdigest())


def verify(binding):
    require(set(binding) == {'path', 'bytes', 'sha256'} and ref(binding['path']) == binding,
            'Exact path/size/hash changed')
    return Path(binding['path'])


def module(binding, name):
    p = verify(binding)
    spec = importlib.util.spec_from_file_location(name, p)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def write(path, value):
    p = Path(path)
    require(not p.exists(), 'Fresh evidence path required: '+str(p))
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        __import__('os').fsync(f.fileno())


def route(task, v10):
    require(task in ORDER, 'All10 fixed task set only')
    original = v10['tasks'][task]
    if task in ('drawer_open_place', 'items_handover'):
        return dict(kind='trained', variant='drawer-expert' if task == 'drawer_open_place' else 'handover-difference',
                    policy='trained_smol_fallback', motor='smolvla', maximum_gpu_processes=3 if task == 'items_handover' else 2,
                    generated=50, executed=10, discarded=40, high_level_refresh=50 if task == 'items_handover' else None)
    return dict(kind='released', variant='released-router-v10', policy=original['policy'], motor=original['motor'],
                checkpoint=original['checkpoint'], maximum_gpu_processes=1 if original['motor'] == 'act' else 2,
                original_options=original['options'], original_environment=original['env'],
                ACT_step=10 if original['motor'] == 'act' else None)


def blocks():
    return [dict(index=i, episodes=5, seed_generator=SEED+1009*i) for i in range(4)]


def controls(argv, *, task, seed, output):
    """Change evaluation controls/output only, preserving every native actor knob."""
    argv = list(argv)
    allowed = {'--max_steps': str(HORIZONS[task]), '--act_step': '10', '--eval_video_log': 'True',
               '--eval_result_dir': str(Path(output)/'official-results')}
    for key, value in allowed.items():
        require(argv.count(key) <= 1, 'Ambiguous native control: '+key)
        if key in argv:
            argv[argv.index(key)+1] = value
        else:
            argv += [key, value]
    for key, value in (('--task_name', task), ('--seed', str(seed)), ('--max_episodes', '5'),
                       ('--eval_expert_check', 'True'), ('--eval_max_seed_attempts_per_episodes', '100')):
        require(argv.count(key) == 1 and argv[argv.index(key)+1] == value,
                'Original fixed fresh protocol differs: '+key)
    return argv
