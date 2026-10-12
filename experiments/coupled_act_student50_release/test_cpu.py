"""Metadata fail-closed cases; no model, world or CUDA dependency."""
import json
from pathlib import Path
import tempfile
from . import binder


def main():
    cases = []
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        (root / 'submission.json').write_text('{}')
        try:
            binder.verify_package(root)
        except ValueError:
            cases.append('wrong_selected_manifest_rejected_before_import_or_model')
        else:
            raise AssertionError('Changed manifest accepted')
        actual = root / 'actual.json'
        actual.write_text('{}')
        (root / 'submission.json').unlink()
        (root / 'submission.json').symlink_to(actual)
        try:
            binder.verify_package(root)
        except ValueError:
            cases.append('symlink_manifest_rejected')
        else:
            raise AssertionError('Symlink manifest accepted')
        binding = dict(tasks={'click_bell': dict(admission={'path': '/tmp/admission', 'sha256': 'a'*64},
                                                 template={'path': '/tmp/template'})})
        source = dict(task_name='click_bell', seed=2026102701, act_step=50)
        injected = binder.injected_args(source, root, binding, root / 'scratch')
        assert source == dict(task_name='click_bell', seed=2026102701, act_step=50)
        assert injected['seed'] == source['seed'] and injected['act_step'] == 50
        assert injected['compact_student_portable_admission_sha256'] == 'a'*64
        cases.append('argument_copy_preserves_organizer_seed_and_action_cadence')
        (root / 'scratch').mkdir()
        try:
            binder.injected_args(source, root, binding, root / 'scratch')
        except ValueError:
            cases.append('stale_actor_scratch_rejected')
        else:
            raise AssertionError('Stale scratch accepted')
        try:
            binder.injected_args(dict(task_name='unknown'), root, binding, root / 'new')
        except ValueError:
            cases.append('unsupported_task_rejected')
        else:
            raise AssertionError('Unknown task accepted')
    print(json.dumps(dict(schema='Student50_release_metadata_meaningful_CPU_cases_v1',
                          cases=cases, models_constructed=0, GPU_queries=0, worlds=0)))


if __name__ == '__main__':
    main()
