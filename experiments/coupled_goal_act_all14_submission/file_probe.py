"""Actual relocated official-policy FILE import, without model construction."""
import argparse
import importlib
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--root', required=True)
    a = p.parse_args()
    root = Path(a.root).resolve(strict=True)
    sys.path.insert(0, str(root))
    name = 'policy.coupled_all14_submission_preparation_v1'
    policy = importlib.import_module(name)
    for function in ('get_model', 'eval', 'reset_model', 'close_model'):
        assert callable(getattr(policy, function)), function
    # This is an intentional negative inference test, before Torch is imported.
    try:
        policy.get_model({})
    except RuntimeError as error:
        assert 'preparation only' in str(error)
    else:
        raise AssertionError('Unqualified preparation must reject model construction')
    modules = {}
    for key, value in tuple(sys.modules.items()):
        if key.startswith(('experiments.', 'policy.')) and getattr(value, '__file__', None):
            path = Path(value.__file__).resolve()
            assert path.is_relative_to(root), (key, path)
            modules[key] = str(path.relative_to(root))
    assert 'torch' not in sys.modules and 'transformers' not in sys.modules
    print(json.dumps(dict(status='actual_relocated_official_FILE_import_only_PASS',
        modules=modules, models_constructed=0, GPU_queries=0, GPU_launches=0,
        native_world_count=0, inference_rejected_before_Torch_import=True)))


if __name__ == '__main__':
    main()
