"""Portable checks for the two original task-specific trained Smol artifacts."""
import importlib.util
import json
from pathlib import Path
import sys

from experiments.goal_latent_act_submission.contracts import child as canonical_child, json_write, reference, require, sha, verify

SCHEMA = 'trained-smolvla-two-task-portable-v1'
TASK_VARIANT = {'items_handover': 'handover-difference', 'drawer_open_place': 'drawer-expert'}


def child(root, name):
    # One exact pinned public model metadata filename; no general hidden-path exemption.
    if name == 'models/SmolVLM2-500M-Video-Instruct/.gitattributes':
        base = canonical_child(root, 'models/SmolVLM2-500M-Video-Instruct')
        path = base / '.gitattributes'
        require(not path.is_symlink() and path.resolve().is_relative_to(Path(root).resolve()), 'Pinned metadata path escaped root')
        return path
    return canonical_child(root, name)


def supported(task):
    require(task in TASK_VARIANT, 'Only trained Handover and Drawer are admitted; other eight tasks need an independently bound policy')
    return TASK_VARIANT[task]


def stage_spec(root, full=False):
    root = Path(root).resolve(strict=True)
    spec = json.loads((root / 'submission.json').read_text())
    require(spec['schema'] == SCHEMA and set(spec['trained_tasks']) == set(TASK_VARIANT), 'Exact two-task trained artifact required')
    require(spec['unsupported_tasks'] and spec['aggregate_success_score'] is None, 'No invented all-task coverage or hybrid result')
    if full:
        for name, bound in spec['files'].items():
            path = child(root, name)
            require(not path.is_symlink() and path.stat().st_size == bound['bytes'] and sha(path) == bound['sha256'],
                    'Exact immutable packaged file differs: ' + name)
    return spec


def validate_science(root, spec):
    root = Path(root).resolve(strict=True)
    source = json.loads((root / 'provenance/historical-source-receipt.json').read_text())
    for name, row in source['files'].items():
        require(sha(child(root, name)) == row['sha256'], 'Historical source differs: ' + name)
    for task, row in spec['trained_tasks'].items():
        supported(task)
        checkpoint = child(root, row['checkpoint'])
        proof = json.loads((checkpoint / 'training-provenance.json').read_text())
        require(proof['task'] == task and proof['completed_step'] == proof['steps'] == 5000 and proof['wiring_pilot'] is False,
                'Original completed task-specific scientific checkpoint required')
        policy_cfg = json.loads((checkpoint / 'policy/config.json').read_text())
        require(policy_cfg['type'] == 'smolvla' and policy_cfg['chunk_size'] == policy_cfg['n_action_steps'] == 50
                and policy_cfg['input_features']['observation.state']['shape'] == [14]
                and policy_cfg['output_features']['action']['shape'] == [14], 'Native saved real14/chunk50 config required')
        for name, digest in proof['source_sha256'].items():
            require(sha(child(root, name)) == digest, 'Original scientific source differs: ' + name)
        if task == 'items_handover':
            module_spec = importlib.util.spec_from_file_location('portable_historical_predictive_proof',
                root / 'policy/predictive_hierarchy/artifact_proof.py')
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            cfg = json.loads((checkpoint / 'adapter-config.json').read_text())
            module.validate_training_proof(checkpoint, cfg, proof)
            require(cfg['flow_adapter_mode'] == 'difference' and proof['all_base_state_unchanged'] is True,
                    'Exact trained difference adapter with frozen base required')
            require(sha(root / 'scripts/train_predictive_smol.py') == proof['trainer_sha256'], 'Original trainer differs')
        else:
            require(proof['schema'] == 'native-smol-real14-v1' and proof['mode'] == 'native'
                    and proof['training_scope'] == 'expert_only' and proof['real_action_loss'] is True
                    and proof['frozen_unchanged'] is True and proof['bridge_constructed'] is False,
                    'Original native expert-only real-action training required')
            require(sha(checkpoint / 'policy/model.safetensors') == proof['model_safetensors_sha256']
                    and sha(checkpoint / 'policy/config.json') == proof['policy_config_sha256'], 'Original Drawer model differs')
            for name, digest in proof['released_processor_sha256'].items():
                require(sha(checkpoint / 'policy' / name) == digest, 'Released Drawer processor differs')
    return True


def metadata_check(root, spec):
    import subprocess
    for name in ('sim', 'smol', 'qwen35'):
        python = Path(root) / '.venvs' / name / 'bin/python'
        program = "import importlib.metadata as m,json;print(json.dumps({d.metadata['Name'].lower().replace('_','-'):d.version for d in m.distributions()}))"
        r = subprocess.run([str(python), '-I', '-B', '-c', program], capture_output=True, text=True, timeout=60)
        require(r.returncode == 0, 'Cold metadata interpreter failed: ' + name)
        actual = json.loads(r.stdout)
        for package, version in spec['environments'][name]['packages'].items():
            require(actual.get(package.lower().replace('_', '-')) == version, 'Exact historical package missing: ' + name + '/' + package)
