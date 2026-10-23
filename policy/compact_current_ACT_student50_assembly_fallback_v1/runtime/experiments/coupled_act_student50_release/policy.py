"""Four official hooks; only verified local execution metadata is added."""
import importlib
import os
from pathlib import Path
import sys
import tempfile
from .binder import bind_all, injected_args


def _original():
    return importlib.import_module('experiments.coupled_act_feature_student_cadence50_submission.policy')


def get_model(args):
    default = Path(__file__).resolve().parents[2] / 'artifact'
    root = Path(args.get('compact_student_submission_root',
                         os.environ.get('ROBOSYN_STUDENT50_PACKAGE_ROOT', str(default)))).resolve(strict=True)
    folder = Path(tempfile.mkdtemp(prefix='rsc50-', dir='/tmp'))
    binding = bind_all(root, folder / 'bindings')
    sys.path.insert(0, str(root))
    return _original().get_model(injected_args(args, root, binding, folder / 'actor-scratch'))


def eval(*args, **kwargs):
    return _original().eval(*args, **kwargs)


def reset_model(*args, **kwargs):
    return _original().reset_model(*args, **kwargs)


def close_model(*args, **kwargs):
    return _original().close_model(*args, **kwargs)
