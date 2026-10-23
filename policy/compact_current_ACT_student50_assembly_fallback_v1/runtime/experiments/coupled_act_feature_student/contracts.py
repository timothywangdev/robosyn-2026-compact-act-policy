"""Pure metadata contracts; episode partitions never inspect targets/outcomes."""
import hashlib
import json
from pathlib import Path

CAMERAS = ('cam_high', 'cam_right_wrist', 'cam_left_wrist')
ROLE_QUERIES = ('left_arm', 'right_arm', 'object_destination')
PLAN = 'conditional-balanced1000-ACT-layer4-current-student-plan-v1'
FIELDS = {'visual': (3, 512, 2, 3), 'visual_global': (3, 512), 'qpos_features': (14,),
          'projected_roles': (96,), 'predicted_goal_delta': (18,)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ref(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def verify(binding):
    require(ref(binding['path']) == binding, 'Exact source/reference mismatch')
    return Path(binding['path'])


def partition(episode):
    require(type(episode) is int and 0 <= episode < 900, 'Only declared FIT episodes0:900')
    return 'optimizer' if episode % 10 < 8 else ('diagnostic' if episode % 10 == 8 else 'gate')


def choose_episodes(task, revision, count=100, seed=1701):
    require(count in (100, 500), 'Fixed1000 or5000 balanced anchor pilot only')
    result = {}
    for name, size in (('optimizer', count * 8 // 10), ('diagnostic', count // 10), ('gate', count // 10)):
        pool = [e for e in range(900) if partition(e) == name]
        pool.sort(key=lambda e: hashlib.sha256(f'{seed}/{task}/{revision}/{e}'.encode()).digest())
        result[name] = sorted(pool[:size])
    return result


def anchor_index(task, revision, episode, seed=1701):
    return int.from_bytes(hashlib.sha256(f'anchor/{seed}/{task}/{revision}/{episode}'.encode()).digest()[:8], 'big') % 7


def records(path):
    """Stream the existing263MB JSON index with bounded1MiB parser buffer."""
    decoder = json.JSONDecoder()
    with Path(path).open() as stream:
        buffer = ''; ended = False; first = True
        while True:
            if not ended and len(buffer) < (1 << 20):
                chunk = stream.read(1 << 20); ended = not chunk; buffer += chunk
            buffer = buffer.lstrip()
            if first:
                require(buffer.startswith('['), 'Source index must be a JSON list')
                buffer = buffer[1:]; first = False; continue
            if buffer.startswith(','):
                buffer = buffer[1:]; continue
            if buffer.startswith(']'):
                require(not buffer[1:].strip() and not stream.read().strip(), 'Unexpected index trailer')
                return
            try:
                row, length = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                require(not ended and len(buffer) < (4 << 20), 'Malformed/oversize row')
                chunk = stream.read(1 << 20); ended = not chunk; buffer += chunk; continue
            require(isinstance(row, dict), 'Source row must be object')
            yield row
            buffer = buffer[length:]


def validate_feature_arrays(arrays, rows):
    import numpy as np
    require(set(arrays) == set(FIELDS), 'Only current visual/state and frozen predicted teacher targets')
    for key, shape in FIELDS.items():
        value = arrays[key]
        require(value.shape == (len(rows), *shape) and value.dtype == np.float32
                and np.isfinite(value).all(), 'Finite FP32 feature field required: ' + key)
    require(all(row['split'] == 'fit' and row['partition'] == partition(row['episode']) for row in rows),
            'Episode split or future/MONITOR leakage')
    keys = [(r['task'], r['episode'], r['frame']) for r in rows]
    require(len(set(keys)) == len(keys), 'Duplicate feature anchors')
