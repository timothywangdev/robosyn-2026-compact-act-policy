"""Public immutable artifact download and automatic offline-install metadata."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.coupled_act_student50_release.binder import PACKAGE_SHA, ref, verify_package, write
from experiments.coupled_act_student50_release import pointer_transport_v1 as transport
TRANSPORT_SHA = 'e1138bad9e093809840ddaa16fcf9447ed1b747802e742ab51d67fd48b0878d5'
TRANSPORT_BYTES = 23126

HF_REPO = '0x00ff00ff/robosyn-2026-compact-act-policy'


def download_file(root, name, row, revision):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe artifact path')
    target = root / relative
    if target.is_symlink() or not target.resolve().is_relative_to(root):
        raise ValueError('Artifact symlink or path escape')
    if target.is_file() and ref(target)['sha256'] == row['sha256'] and target.stat().st_size == row['bytes']:
        return 'existing_verified'
    if target.exists():
        raise ValueError('Existing artifact file differs: ' + name)
    target.parent.mkdir(parents=True, exist_ok=True)
    url = 'https://huggingface.co/' + HF_REPO + '/resolve/' + revision + '/' + name
    temporary = target.with_name(target.name + '.download-part')
    if temporary.exists() or temporary.is_symlink():
        raise ValueError('Incomplete prior download must be inspected before retry')
    digest = hashlib.sha256(); size = 0
    try:
        with urllib.request.urlopen(url, timeout=180) as response, temporary.open('xb') as stream:
            while True:
                block = response.read(8 << 20)
                if not block: break
                stream.write(block); digest.update(block); size += len(block)
        if digest.hexdigest() != row['sha256'] or size != row['bytes']:
            raise ValueError('Public artifact byte verification failed: ' + name)
        os.replace(temporary, target)
    except BaseException:
        # Keep a real partial file as failure evidence; never treat it as verified.
        raise
    return 'downloaded_verified'


def install_metadata(root, output, installer):
    _, verification = verify_package(root)
    output.mkdir(parents=True, exist_ok=False)
    admission = dict(schema='root_authorized_Student50_portable_fresh_CP311_CPU_install_v2',
        execution_authorized=True,package=verification['package'],installer=ref(installer),
        resource_caps=dict(wall_seconds=600,owned_tree_RSS_bytes=2<<30,incremental_disk_bytes=14<<30,
            free_disk_floor_bytes=8<<30,threads=2),
        generated_from_exact_public_release_bytes=True,session_authentication_required=False)
    path=output/'installation-metadata.json'; write(path,admission)
    return path


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--revision',required=True)
    parser.add_argument('--root',default=str(Path(__file__).resolve().parents[2]/'artifact'))
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--download-only',action='store_true')
    parser.add_argument('--offline-check-only',action='store_true')
    args=parser.parse_args()
    if sys.version_info[:2] != (3,11): raise ValueError('The frozen offline wheels require CPython 3.11')
    if not re.fullmatch(r'[0-9a-f]{40}',args.revision) or not 1<=args.workers<=4:
        raise ValueError('Explicit immutable HF revision and 1-4 workers required')
    root=Path(args.root).absolute()
    if root.is_symlink(): raise ValueError('Artifact root symlink forbidden')
    root.mkdir(parents=True,exist_ok=True)
    if not args.offline_check_only:
        manifest=root/'submission.json'
        if not manifest.exists():
            url='https://huggingface.co/'+HF_REPO+'/resolve/'+args.revision+'/submission.json'
            with urllib.request.urlopen(url,timeout=60) as response: raw=response.read(4<<20)
            if hashlib.sha256(raw).hexdigest()!=PACKAGE_SHA: raise ValueError('Exact selected release manifest required')
            with manifest.open('xb') as stream: stream.write(raw)
        if manifest.is_symlink() or ref(manifest)['sha256']!=PACKAGE_SHA: raise ValueError('Selected manifest differs')
        spec=json.loads(manifest.read_text())
        download_file(root,transport.MANIFEST,dict(bytes=TRANSPORT_BYTES,sha256=TRANSPORT_SHA),args.revision)
        manifest_raw=(root/transport.MANIFEST).read_bytes()
        entries=json.loads(manifest_raw)['entries']
        if len(entries)!=45:raise ValueError('Exact45 frozen pointer fixtures required')
        def fetch(item):
            name,row=item
            download_file(root,row['transport_path'],row['transport'],args.revision)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(fetch,entries.items()))
            regular=[item for item in spec['files'].items() if item[0] not in entries]
            list(pool.map(lambda item: download_file(root,item[0],item[1],args.revision),regular))
        if transport.restore(root,manifest_raw,lambda name: transport.safe_path(root,name).read_bytes(),spec)!=45:
            raise ValueError('Frozen pointer reconstitution incomplete')
    verify_package(root)
    if args.download_only: return
    installer=Path(__file__).resolve().parents[2]/'release/cold_install_cpu_v2.py'
    expected='50302820aa048fa13dabcd0e9d2a6ee8c8c13ac5fcdb930d11643fbb696461c2'
    if ref(installer)['sha256']!=expected: raise ValueError('Exact proven offline installer source required')
    temporary=Path(tempfile.mkdtemp(prefix='student50-setup-'))
    metadata=install_metadata(root,temporary/'metadata',installer)
    argv=[sys.executable,'-B',str(installer),'--root',str(root),'--output',str(temporary/'installer'),
        '--admission',str(metadata),'--admission-sha256',ref(metadata)['sha256']]
    if args.offline_check_only: argv.append('--check-only')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',HF_DEBUG='0')
    result=subprocess.run(argv,env=env,stdin=subprocess.DEVNULL)
    raise SystemExit(result.returncode)


if __name__=='__main__': main()
