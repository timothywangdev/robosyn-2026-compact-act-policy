"""Policy-local import shim for organizer copies retaining only policy/."""
from pathlib import Path
import hashlib
import json
import os
import sys

_RUNTIME = Path(__file__).resolve().parent / 'runtime'
# Keep the byte-pinned scientific modules in their original relative layout.
sys.path.insert(0, str(_RUNTIME))
from experiments.coupled_act_student50_assembly_fallback import policy as _policy

_INITIALIZER_SHA256 = 'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
_INITIALIZER_BYTES = 46830571


def _bind_torch_cache(root):
    home = root / 'cache/torch'
    initializer = home / 'hub/checkpoints/resnet18-f37072fd.pth'
    if (initializer.is_symlink() or not initializer.is_file()
            or not initializer.resolve(strict=True).is_relative_to(root)
            or initializer.stat().st_size != _INITIALIZER_BYTES
            or hashlib.sha256(initializer.read_bytes()).hexdigest() != _INITIALIZER_SHA256):
        raise ValueError('Pinned packaged ResNet initializer is missing or differs')
    os.environ['TORCH_HOME'] = str(home)
    # Respect the same package path even if the organizer imported torch.hub
    # earlier and explicitly selected another global cache directory.
    hub = sys.modules.get('torch.hub')
    if hub is not None:
        hub.set_dir(str(home / 'hub'))
    print('[RoboSyn model cache] ' + json.dumps(dict(
        TORCH_HOME=str(home), initializer=str(initializer),
        sha256=_INITIALIZER_SHA256, bytes=_INITIALIZER_BYTES)), flush=True)


def get_model(args):
    default = os.environ.get('ROBOSYN_STUDENT50_PACKAGE_ROOT', str(_RUNTIME / 'artifact'))
    root = Path(args.get('compact_student_submission_root', default)).resolve(strict=True)
    _bind_torch_cache(root)
    if args.get('task_name') == 'item_assembly':
        from .constraint_lifetime import install
        install()
    actor_source = root / 'upstream/lerobot-sim/src'
    # This is participant actor implementation source; installed organizer
    # dependencies and the official evaluator remain in the supplied environment.
    sys.path.insert(0, str(actor_source))
    already_loaded = sys.modules.get('lerobot.policies.act.modeling_act')
    if already_loaded is not None:
        actual = hashlib.sha256(Path(already_loaded.__file__).read_bytes()).hexdigest()
        if actual != '48ba3a2620198154b7b63948d14ebf76fa50da53efba194aedd76d9392c955ae':
            raise ValueError('Organizer interpreter already imported a different LeRobot ACT implementation; '
                             'start a fresh interpreter with the policy actor source path')
    return _policy.get_model(dict(args, compact_student_submission_root=str(root)))


def eval(*args, **kwargs):
    return _policy.eval(*args, **kwargs)


def reset_model(*args, **kwargs):
    return _policy.reset_model(*args, **kwargs)


def close_model(*args, **kwargs):
    return _policy.close_model(*args, **kwargs)
