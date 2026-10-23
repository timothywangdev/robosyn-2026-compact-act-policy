"""Unchanged frozen current-camera feature/goal equations, new manifest boundary."""
import argparse
import contextlib
import json
from pathlib import Path
import sys
import numpy as np
from .contracts import load_manifest, require, sha
from experiments.goal_latent_act_arm_prefix10.observer_worker import Backend


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    for name in ('manifest','manifest-sha256','input-root','output-root'): p.add_argument('--'+name,required=True)
    a = p.parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        cfg = load_manifest(a.manifest,a.manifest_sha256)
        require(cfg['mode']=='goal', 'Only role branch may start Qwen')
        backend = Backend(cfg)
    inputs, output = Path(a.input_root).resolve(strict=True),Path(a.output_root).resolve(strict=True)
    print(json.dumps(dict(ok=True,event='ready',kind='coupled_all14_current_frozen_Qwen',
        highlevel_sha256=cfg['assets']['highlevel_weights']['sha256'])),flush=True)
    for line in sys.stdin:
        try:
            require(len(line)<=65536, 'Bounded current RPC required')
            r = json.loads(line)
            require(set(r)=={'cmd','request'} and r['cmd']=='extract', 'Exact current extraction RPC required')
            with contextlib.redirect_stdout(sys.stderr): arrays,proof=backend.extract(r['request'],inputs)
            key = r['request']['request_id']
            require(key and all(c.isalnum() or c in '-_' for c in key), 'Safe current identity required')
            path = output/(key+'.npz')
            with path.open('xb') as stream: np.savez(stream,**arrays,input_sha256=np.asarray(proof['input_npz_sha256']))
            print(json.dumps(dict(ok=True,request_id=key,path=path.name,sha256=sha(path),
                input_sha256=proof['input_npz_sha256'],generated_tokens=0)),flush=True)
        except BaseException as error:
            print(json.dumps(dict(ok=False,error=str(error))),flush=True)
            raise


if __name__=='__main__': main()
