"""Versioned, task-agnostic current context for the shared ten-task observer.

The data boundary owns raw-to-canonical conversion and its evidence. This module
requires that declaration; it never guesses a gripper unit or joint ordering.
Historical handover role-feature caches are deliberately a different identity.
"""
import hashlib
import json
import math
import re
from pathlib import Path

from policy.relational_hierarchy.qwen_contract_worker import (
    MODEL_ID, MODEL_REVISION, TRANSFORMERS_VERSION, prepare_request,
)

STATE_SCHEMA = 'cobotmagic-current-qpos14-arm-major-normalized-opening-v1'
CONTEXT_VERSION = 'qwen35-canonical-current-role-hidden-v2'
FEATURE_DIM = 4096
MARKERS = ('[LEFT_ARM_QUERY]', '[RIGHT_ARM_QUERY]', '[OBJECT_DESTINATION_QUERY]')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def role_prompt(instruction, qpos):
    if (type(instruction) is not str or not instruction.strip()
            or len(instruction) > 4096 or any(marker in instruction for marker in MARKERS)):
        raise ValueError('Bounded original instruction without reserved markers required')
    if (len(qpos) != 14 or any(type(x) not in (float, int) or not math.isfinite(x) for x in qpos)):
        raise ValueError('Finite canonical current14D qpos required')
    return (
        'Use only the three labeled CURRENT RGB images, original instruction, and measured current state. '
        'Represent the next short executable effects for the left arm, right arm, and object/destination relation. '
        'Do not infer contact, successful grasp, completion, or visibility from joint state alone. '
        'The state has the same canonical ordering for every task: indices0..5 are left arm angles in radians, '
        'index6 is left measured normalized gripper opening, indices7..12 are right arm angles in radians, '
        'and index13 is right measured normalized gripper opening. '
        'Opening zero is near-closed and one near-open; small measured simulation overshoot may occur. '
        'These are current observations, not future commands or object pose labels. '
        f'Original instruction: {instruction}\nCanonical current qpos (14D): {json.dumps(qpos)}\n'
        f'Left arm next executable effect:\n{MARKERS[0]}\n'
        f'Right arm next executable effect:\n{MARKERS[1]}\n'
        f'Object relative to destination next desired relation:\n{MARKERS[2]}'
    )


def prepare_canonical_current(request, input_root, max_image_edge=384):
    """Validate three current RGB + canonical q only; return content and audit.

    ``state_evidence_sha256`` binds an externally reviewed conversion receipt.
    Its presence is not a claim that this function independently audited units.
    Task/revision/episode identity belongs to the caller's sealed row manifest;
    no task-ID embedding, target label or validity mask enters the model prompt.
    """
    expected = {'request_id', 'path', 'instruction', 'frame_ids', 'state_schema', 'state_evidence_sha256'}
    if type(request) is not dict or set(request) != expected:
        raise ValueError('Exact canonical current request fields required; no labels/futures')
    if request['state_schema'] != STATE_SCHEMA:
        raise ValueError('Explicit audited canonical current state schema required')
    if not isinstance(request['state_evidence_sha256'], str) or not re.fullmatch('[0-9a-f]{64}', request['state_evidence_sha256']):
        raise ValueError('Pinned canonical conversion evidence SHA256 required')
    legacy = {key: request[key] for key in ('request_id', 'path', 'frame_ids')}
    legacy['task'] = request['instruction']
    content, original = prepare_request(legacy, input_root, max_image_edge)
    prompt = role_prompt(request['instruction'], original['qpos'])
    audit = {key: original[key] for key in ('request_id', 'input_path', 'frame_ids', 'frames', 'qpos')}
    audit.update(instruction=request['instruction'], state_schema=STATE_SCHEMA,
        state_evidence_sha256=request['state_evidence_sha256'], context_version=CONTEXT_VERSION,
        model_id=MODEL_ID, model_revision=MODEL_REVISION, transformers=TRANSFORMERS_VERSION,
        feature_shape=[3, FEATURE_DIM], prompt=prompt,
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        input_npz_sha256=digest(original['input_path']), max_image_edge=max_image_edge,
        context_source_sha256=digest(__file__),
        extractor_source_sha256=digest(Path(__file__).parents[1]/'relational_hierarchy/qwen_features.py'),
        validator_source_sha256=digest(Path(__file__).parents[1]/'relational_hierarchy/qwen_contract_worker.py'))
    audit['context_cache_key'] = context_cache_key(audit)
    return content[:-1] + [{'type': 'text', 'text': prompt}], audit


def context_cache_key(audit):
    fields = ('context_version', 'state_schema', 'state_evidence_sha256', 'model_id',
              'model_revision', 'transformers', 'input_npz_sha256', 'prompt_sha256', 'max_image_edge',
              'context_source_sha256', 'extractor_source_sha256', 'validator_source_sha256')
    payload = {key: audit[key] for key in fields}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def extract_canonical_context(extractor, request, input_root, max_image_edge=384):
    """Exactly ONE current prefill; keep cache/runtime batchsize identical.

    ``extractor`` is the existing qualified FrozenQwenFeatures instance. This
    wrapper changes only the declared prompt; it never generates text or feeds
    a future clip. The new prompt still needs an actual GPU activation gate.
    """
    import numpy as np
    content, audit = prepare_canonical_current(request, input_root, max_image_edge)
    features, positions, token_audit = extractor.extract([content])
    if features.shape != (1, 3, FEATURE_DIM) or features.dtype != np.float16 or not np.isfinite(features).all():
        raise ValueError('Exact batch1 finite FP16 role feature contract required')
    if len(positions) != 1 or len(positions[0]) != 3 or token_audit.get('generated_tokens') != 0:
        raise ValueError('Exactly three role endpoints and zero generated tokens required')
    audit.update(batch_size=1, generated_tokens=0, role_token_positions=positions[0], token_audit=token_audit)
    return features[0], audit
