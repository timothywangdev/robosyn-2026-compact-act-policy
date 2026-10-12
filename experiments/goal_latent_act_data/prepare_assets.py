"""Fetch only exact ACT70-pinned public releases; CPU/network only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
QUALIFIER=ROOT/'outputs/architecture-reassessment/act-goal-contingency-actual-v1/completed-v1/originals/evidence/qualifier_plan.json'
LOCAL=ROOT/'outputs/architecture-reassessment/physicalrsi-local3090-preparation-v1'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024**2),b''):h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    output=a.output.resolve();output.mkdir(parents=True,exist_ok=False)
    qualified=json.loads(QUALIFIER.read_text());baseline=json.loads((ROOT/'docs/baseline-manifest.json').read_text())
    releases={m['repo_id'].split('/')[-1]:m for m in baseline['model_inventory'] if '/ACT_sim_' in m['repo_id']}
    expected_bytes=sum(t['model']['files']['model.safetensors']['bytes'] for t in qualified['tasks'])
    if shutil.disk_usage(output).free<expected_bytes+15*1024**3:raise RuntimeError('Exact model download and15GiB reserve required')
    started=time.monotonic()
    def one(task):
        name='ACT_sim_'+task['task'];release=releases[name];existing=LOCAL/'models'/name
        root=existing if all((existing/k).is_file() and sha(existing/k)==v['sha256'] for k,v in task['model']['files'].items()) else output/name
        reused=root==existing
        if not reused:root.mkdir(exist_ok=False)
        for filename,identity in task['model']['files'].items():
            if reused:continue
            partial=root/(filename+'.partial');url='https://huggingface.co/'+release['repo_id']+'/resolve/'+release['revision']+'/'+filename
            result=subprocess.run(['curl','--fail','--location','--silent','--show-error','--connect-timeout','15','--max-time','240','--output',str(partial),url],capture_output=True,text=True,timeout=250)
            (root/(filename+'.download.stderr')).write_text(result.stderr)
            if result.returncode:raise RuntimeError('Pinned public ACT download failed '+name+'/'+filename+'; preserved, no retry')
            if partial.stat().st_size!=identity['bytes'] or sha(partial)!=identity['sha256']:raise ValueError('Exact ACT70 release bytes differ '+name+'/'+filename)
            partial.rename(root/filename)
        return dict(task=task['task'],root=str(root),release_revision=release['revision'],files=task['model']['files'],reused_local_verified=reused)
    with ThreadPoolExecutor(max_workers=3) as pool:models=list(pool.map(one,qualified['tasks']))
    report=dict(schema='goal-latent-act-local-released-assets-v1',status='all10_exact_ACT70_release_files_verified',models=models,
                qualifier_plan_sha256=sha(QUALIFIER),baseline_manifest_sha256=sha(ROOT/'docs/baseline-manifest.json'),elapsed_seconds=time.monotonic()-started,
                GPU_initialized=False,download_connections_max=3,lerobot_source_root=str(LOCAL/'upstream/lerobot-sim/src'))
    (output/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(status=report['status'],seconds=report['elapsed_seconds'],models=10)),flush=True)

if __name__=='__main__':main()
