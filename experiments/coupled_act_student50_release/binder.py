"""Verify the selected artifact and generate automatic installed-path bindings."""
import hashlib
import json
import os
from pathlib import Path
import time

PACKAGE_SHA = '191e3a69625175b96c685304e19d5362522e16338943cd6c3065448cabb89515'
ADMISSION_SCHEMA = 'root_authorized_compact_ACT_student50_portable_native_v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def ref(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def verify_package(root):
    root = Path(root).resolve(strict=True)
    manifest = root / 'submission.json'
    if manifest.is_symlink() or sha(manifest) != PACKAGE_SHA:
        raise ValueError('Exact selected Student50 manifest required')
    spec = json.loads(manifest.read_text())
    if (spec['schema'] != 'compact_current_ACT_student50_byte_preserving_portable_package_v1'
            or spec['selected_candidate'] != 'distill_visual'
            or spec['organizer_action_contract']['generated_chunk'] != 50
            or spec['organizer_action_contract']['internally_enqueued_prefix'] != 50
            or spec['Qwen_model_payload'] or len(spec['tasks']) != 10):
        raise ValueError('Selected all10 execute50 artifact required')
    started = time.monotonic()
    for name, expected in spec['files'].items():
        relative = Path(name)
        path = root / relative
        if (relative.is_absolute() or '..' in relative.parts or path.is_symlink()
                or not path.is_file() or not path.resolve().is_relative_to(root)
                or path.stat().st_size != expected['bytes'] or sha(path) != expected['sha256']):
            raise ValueError('Selected artifact file differs: ' + name)
    for name, expected in spec['source_files'].items():
        if spec['files'][name] != expected:
            raise ValueError('Source inventory and artifact inventory differ')
    if len(spec['files']) != 2608 or len(spec['source_files']) != 247:
        raise ValueError('Selected source/dependency closure changed')
    return spec, dict(package=ref(manifest), files_SHA_verified=2608,
        source_files_SHA_verified=247, bytes_SHA_verified=sum(v['bytes'] for v in spec['files'].values()),
        verification_seconds=time.monotonic() - started,
        GPU_queries=0, models_constructed=0, worlds_constructed=0)


def bind_all(root, output):
    """Execution metadata derives from verified bytes, without session credentials."""
    root = Path(root).resolve(strict=True)
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError('Fresh local binding directory required')
    spec, verification = verify_package(root)
    output.mkdir(parents=True)
    entry = root / 'experiments/coupled_act_feature_student_cadence50_submission/policy.py'
    rows = {}
    for task in spec['tasks']:
        template = root / task['template']
        if sha(template) != task['template_sha256'] or task['mode'] != 'student':
            raise ValueError('Exact selected per-task template required')
        admission = dict(schema=ADMISSION_SCHEMA, execution_authorized=True,
            robot_quality_promoted=False, package=verification['package'],
            template=ref(template), entry_source=ref(entry),
            execution_contract_generated_from_verified_release_bytes=True,
            organizer_or_root_session_token_required=False,
            hardware_or_quality_qualification_claimed=False)
        path = output / ('native-admission-' + task['task'] + '.json')
        write(path, admission)
        rows[task['task']] = dict(admission=ref(path), template=ref(template),
            official_horizon=task['official_horizon'], generated=50, executed=50)
    receipt = dict(schema='Student50_automatic_organizer_installed_binding_v1',
        status='exact_all10_artifact_verified_execution_metadata_generated_no_model',
        verification=verification, tasks=rows, binding_source=ref(__file__),
        session_authentication_used=False, GPU_queries=0, GPU_launches=0,
        models_constructed=0, worlds_constructed=0, submission_ready=False)
    write(output / 'receipt.json', receipt)
    return receipt


def injected_args(args, root, binding, scratch):
    task = args['task_name']
    if task not in binding['tasks']:
        raise ValueError('Supported all10 task required')
    row = binding['tasks'][task]
    scratch = Path(scratch).absolute()
    if scratch.exists() or scratch.is_symlink():
        raise ValueError('Fresh per-model actor scratch required')
    return dict(args, compact_student_mode='student',
        compact_student_submission_root=str(Path(root).resolve(strict=True)),
        compact_student_submission_template=row['template']['path'],
        compact_student_portable_admission=row['admission']['path'],
        compact_student_portable_admission_sha256=row['admission']['sha256'],
        compact_student_portable_scratch=str(scratch))
