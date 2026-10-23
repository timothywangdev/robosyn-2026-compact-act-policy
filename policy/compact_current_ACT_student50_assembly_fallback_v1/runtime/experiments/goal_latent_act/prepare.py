"""One bounded CPU pass binding all70000 actual current roles to physics labels."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import numpy as np

VERSION = 'goal-latent-act-full70000-packed-role-labels-v1'
TEACHER_SHA = '220509d25d8a9073edd865d5cafd31fd1dbeff294ce75d2b924b895cd6315e72'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 ** 2), b''): h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    tmp = Path(str(path) + '.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--project-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--wall-seconds', type=int, default=600)
    a = p.parse_args()
    if not 1 <= a.wall_seconds <= 900: raise ValueError('Bounded CPU preparation')
    root = a.project_root.resolve(strict=True); a.output.mkdir(parents=True, exist_ok=False)
    report = dict(schema=VERSION, status='started', pid=os.getpid(), GPU_launched=False,
                  source_sha256={'prepare.py': sha(__file__)}, verified_feature_rows=0)
    started = time.monotonic()
    def expired(*_): raise TimeoutError('Role packing wall budget')
    signal.signal(signal.SIGALRM, expired); signal.alarm(a.wall_seconds)
    try:
        teacher = root / 'outputs/architecture-reassessment/object-relative-full-teacher-data-v1/data'
        coverage_path = root / 'outputs/architecture-reassessment/packed-context-full-completion-v1/coverage-review.json'
        coverage = json.loads(coverage_path.read_text())
        if coverage['status'] != 'all70000_context_index_coverage_verified': raise ValueError('Full context completion required')
        if sha(teacher / 'report.json') != TEACHER_SHA: raise ValueError('Pinned full physics teacher changed')
        tr = json.loads((teacher / 'report.json').read_text())
        for name in ('samples.jsonl', 'training-only-labels.npz'):
            if sha(teacher / name) != tr['artifact_sha256'][name]: raise ValueError('Teacher artifact changed: ' + name)
        rows = [json.loads(line) for line in (teacher / 'samples.jsonl').open()]
        if len(rows) != 70000 or len({r['clip_id'] for r in rows}) != 70000: raise ValueError('Exact unique full teacher required')
        clips = {r['clip_id']: i for i, r in enumerate(rows)}
        tasks = sorted(coverage['tasks']); counts = Counter(); demos = Counter()
        for i, r in enumerate(rows):
            expected = 'fit' if 0 <= r['episode'] < 900 else 'monitor' if 900 <= r['episode'] < 1000 else None
            if r['split'] != expected or r['labels_row'] != i or r['future_label_frame'] != r['frame'] + 10:
                raise ValueError('Teacher split/index/future target mismatch')
            counts[r['task'], r['split']] += 1; demos[r['task'], r['episode']] += 1
        if any(counts[t, 'fit'] != 6300 or counts[t, 'monitor'] != 700 for t in tasks) or any(n != 7 for n in demos.values()):
            raise ValueError('Allten exact episode and seven-anchor coverage required')
        with np.load(teacher / 'training-only-labels.npz', allow_pickle=False) as archive:
            labels = {k: archive[k].copy() for k in archive.files}
        if list(labels['clip_id']) != [r['clip_id'] for r in rows]: raise ValueError('Exact teacher label-row identities required')
        roles = np.lib.format.open_memmap(a.output / 'roles.npy', mode='w+', dtype=np.float16, shape=(70000, 3, 4096))
        seen = set(); receipts = []
        for scope in coverage['inputs']:
            cache = root / scope['local_root']
            if sha(cache / 'manifest.json') != scope['manifest_sha256'] or sha(cache / 'features.jsonl') != scope['features_jsonl_sha256']:
                raise ValueError('Exact cached role scope/index changed')
            manifest = json.loads((cache / 'manifest.json').read_text())
            if manifest['status'] != 'complete_for_declared_source_scope' or manifest['current_only'] is not True or manifest['batch_size'] != 1:
                raise ValueError('Actual complete current-only cache required')
            if manifest['context_version'] != 'qwen35-canonical-current-role-hidden-v2': raise ValueError('Current context version changed')
            n = 0
            for line in (cache / 'features.jsonl').open():
                f = json.loads(line); clip = f['clip_id']; audit = f['audit']
                if clip in seen or clip not in clips: raise ValueError('Duplicate or unmatched role clip')
                i = clips[clip]; row = rows[i]
                if (audit['model_id'] != 'Qwen/Qwen3.5-9B' or audit['model_revision'] != 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'
                    or audit['generated_tokens'] != 0 or audit['batch_size'] != 1
                    or audit['instruction'] != row['instruction'] or not np.array_equal(audit['qpos'], row['current_qpos'])
                    or audit['state_evidence_sha256'] != row['request']['state_evidence_sha256']):
                    raise ValueError('Role current observation/instruction/state binding changed')
                for camera in row['current_rgb_sources']:
                    if (audit['frames'][camera]['frame_id'] != row['request']['frame_ids'][camera]
                        or audit['frames'][camera]['presented_rgb_sha256'] != f['storage']['presented_rgb_sha256'][camera]):
                        raise ValueError('Role/teacher current frame or retained pixel identity changed')
                if f['storage']['selected_frame'] != row['frame'] or f['storage']['only_selected_current_output'] is not True:
                    raise ValueError('Role current-frame causality changed')
                path = (cache / f['feature_path']).resolve(strict=True)
                if not path.is_relative_to(cache.resolve()) or sha(path) != f['feature_sha256']:
                    raise ValueError('Role payload hash/path changed')
                with np.load(path, allow_pickle=False) as value:
                    if value.files != ['features']: raise ValueError('Exact compact role payload required')
                    feature = value['features']
                    if feature.shape != (3, 4096) or feature.dtype != np.float16 or not np.isfinite(feature).all():
                        raise ValueError('Exact finite role dimensions/dtype required')
                    roles[i] = feature
                seen.add(clip); n += 1
                if len(seen) % 5000 == 0:
                    report.update(verified_feature_rows=len(seen), elapsed_seconds=time.monotonic() - started)
                    write_json(a.output / 'receipt.json', report)
            if n != scope['records']: raise ValueError('Scope row count changed')
            receipts.append(dict(scope=scope['scope'], manifest_sha256=scope['manifest_sha256'],
                                 features_jsonl_sha256=scope['features_jsonl_sha256'], records=n))
        if len(seen) != 70000: raise ValueError('Full actual role coverage required')
        roles.flush(); del roles
        qpos = np.asarray([r['current_qpos'] for r in rows], dtype=np.float32)
        fit = np.asarray([r['split'] == 'fit' for r in rows], dtype=bool)
        task_index = np.asarray([tasks.index(r['task']) for r in rows], dtype=np.int64)
        ee = labels['eef_now']; pose = np.concatenate((ee[..., :3, 3], ee[..., :3, :2].transpose(0, 1, 3, 2).reshape(70000, 2, 6)), -1).reshape(70000, 18)
        stats = dict(schema='goal-latent-current-fit-only-normalizer-v1', qpos_mean=qpos[fit].mean(0).tolist(),
                     qpos_scale=np.maximum(qpos[fit].std(0), .001).tolist(), eef_mean=pose[fit].mean(0).tolist(),
                     eef_scale=np.maximum(pose[fit].std(0), .001).tolist(), rows=63000, fit_episodes=[0, 899],
                     state_units='arm radians; measured normalized gripper opening; arm-major14',
                     eef_units='arena/world meters and rigid rotation; source-derived measured-current FK',
                     historical_collector_lineage_certified=False)
        labels.update(qpos=qpos, fit=fit, task_index=task_index)
        np.savez(a.output / 'labels.npz', **labels)
        write_json(a.output / 'statistics.json', stats)
        write_json(a.output / 'rows.json', [{'clip_id': r['clip_id'], 'task': r['task'], 'episode': r['episode'],
                                           'frame': r['frame'], 'split': r['split'], 'revision': r['revision'],
                                           'parquet_sha256': r['parquet_sha256']} for r in rows])
        report.update(status='all70000_actual_roles_teacher_bound', verified_feature_rows=70000,
                      fit=63000, monitor=7000, fit_demonstrations=9000, monitor_demonstrations=1000,
                      tasks=tasks, scopes=receipts, coverage_sha256=sha(coverage_path),
                      teacher_report_sha256=TEACHER_SHA, teacher_artifact_sha256=tr['artifact_sha256'],
                      artifacts={name: dict(sha256=sha(a.output / name), bytes=(a.output / name).stat().st_size)
                                 for name in ('roles.npy', 'labels.npz', 'statistics.json', 'rows.json')},
                      object_supervision_by_task={t: int(labels['object_valid'][task_index == j, 0].sum()) for j, t in enumerate(tasks)},
                      allten_eef_goal_supervision=True,
                      raw_media_reopened=False,
                      current_RGB_binding='exact current frame IDs, state/instruction/pixels, completed source-index provenance; original raw-media hashes inherited',
                      Bell_historical_unit_lineage_certified=False)
    except BaseException as error:
        report.update(status='failed_no_retry', error=repr(error)); raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        write_json(a.output / 'receipt.json', report); signal.alarm(0)
    print(json.dumps({'status': report['status'], 'rows': report['verified_feature_rows'], 'elapsed_seconds': report['elapsed_seconds']}))


if __name__ == '__main__': main()
