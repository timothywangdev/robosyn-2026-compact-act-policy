"""One current-image Qwen prefill -> learned physical dual-arm goal."""
import argparse
import contextlib
import json
import sys
from pathlib import Path
import numpy as np
from .contracts import load_manifest, module, require, sha, verify_ref


class Backend:
    def __init__(self, cfg):
        import torch
        from experiments.goal_latent_act_arm_runtime_full.semantics import HEAD_SCHEMA, validate_head
        from policy.relational_hierarchy.qwen_features import FrozenQwenFeatures
        self.cfg, self.torch = cfg, torch
        qwen = Path(cfg['qwen_root']).resolve(strict=True)
        index = json.loads((qwen / 'model.safetensors.index.json').read_text())
        require(set(index['weight_map'].values()) <= set(cfg['qwen_files']), 'Complete pinned Qwen shards required')
        for relative, ref in cfg['qwen_files'].items():
            require((qwen / relative).resolve() == Path(ref['path']).resolve(), 'Qwen inventory root differs')
            verify_ref(ref)
        require(torch.cuda.is_available(), 'Explicit CUDA observer required')
        self.extractor = FrozenQwenFeatures(qwen)
        contract = json.loads(verify_ref(cfg['assets']['highlevel_contract']).read_text())
        require(contract['schema'] == HEAD_SCHEMA, 'Actually trained factual high-level model required')
        require(contract['weights_sha256'] == cfg['assets']['highlevel_weights']['sha256'], 'Trained high-level identity differs')
        validate_head(contract, cfg)
        from experiments.goal_latent_act_command_goal_training_full.contracts import load_head
        self.goal, loaded_contract = load_head(
            verify_ref(cfg['assets']['highlevel_contract']).parent, cfg['assets']['highlevel_weights']['sha256'])
        require(loaded_contract == contract, 'Full63000 strict head loader contract differs')
        self.goal.eval().requires_grad_(False)
        self.stats = {k: torch.tensor(contract['statistics'][k], dtype=torch.float32) for k in ('qpos_mean','qpos_scale','eef_mean','eef_scale')}
        self.geometry = module('goal_latent_current_geometry', verify_ref(cfg['assets']['geometry']))
        self.chain = self.geometry.read_chain(verify_ref(cfg['assets']['urdf']))

    def extract(self, request, input_root):
        from policy.object_relative_hierarchy.canonical_context import extract_canonical_context
        require(request['instruction'] == self.cfg['instruction'], 'Original instruction differs')
        require(request['state_evidence_sha256'] == self.cfg['assets']['state_evidence']['sha256'], 'Measured-state binding differs')
        roles, proof = extract_canonical_context(self.extractor, request, input_root, 384)
        q = np.asarray(proof['qpos'], dtype=np.float32)
        now = np.stack([self.geometry.fk_link(self.chain, q[axes].astype(np.float64), side) @ self.geometry.tcp()
                        for side, axes in (('left', slice(0,6)), ('right', slice(7,13)))]).astype(np.float32)
        with self.torch.inference_mode():
            result = self.goal(roles=self.torch.from_numpy(roles[None].astype(np.float32)),
                               qpos=self.torch.from_numpy(q[None]), eef_now=self.torch.from_numpy(now[None]), **self.stats)
        goal = result['eef_goal_pose'][0].numpy().copy()
        require(np.isfinite(goal).all(), 'Nonfinite physical goal')
        require(roles.shape == (3,4096) and roles.dtype == np.float16 and np.isfinite(roles).all(), 'Frozen current role FP16 storage ABI required')
        return dict(goal=goal, current_eef=now, qpos=q, current_roles=roles.astype(np.float32)), proof


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    for name in ('manifest','manifest-sha256','input-root','output-root'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        cfg = load_manifest(a.manifest, a.manifest_sha256); backend = Backend(cfg)
    inputs, output = Path(a.input_root).resolve(strict=True), Path(a.output_root).resolve(strict=True)
    print(json.dumps(dict(ok=True,event='ready',kind='goal_latent_arm_prefix10_frozen_Qwen',highlevel_sha256=cfg['assets']['highlevel_weights']['sha256'])), flush=True)
    for line in sys.stdin:
        try:
            require(len(line) <= 65536, 'Bounded RPC required')
            r = json.loads(line); require(set(r) == {'cmd','request'} and r['cmd'] == 'extract', 'Exact current extraction RPC')
            with contextlib.redirect_stdout(sys.stderr):
                arrays, proof = backend.extract(r['request'], inputs)
            key = r['request']['request_id']; require(key and all(c.isalnum() or c in '-_' for c in key), 'Bounded safe current identity')
            path = output / (key+'.npz')
            with path.open('xb') as stream:
                np.savez(stream, **arrays, input_sha256=np.asarray(proof['input_npz_sha256']))
            print(json.dumps(dict(ok=True,request_id=key,path=path.name,sha256=sha(path),input_sha256=proof['input_npz_sha256'],generated_tokens=0)), flush=True)
        except BaseException as e:
            print(json.dumps(dict(ok=False,error=str(e))), flush=True)
            raise


if __name__ == '__main__':
    main()
