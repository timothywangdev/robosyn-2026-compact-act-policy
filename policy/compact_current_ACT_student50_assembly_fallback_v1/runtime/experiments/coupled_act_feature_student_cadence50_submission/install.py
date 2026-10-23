"""Relative offline installation recipe; --check-only does not install packages."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--root',required=True);p.add_argument('--check-only',action='store_true');a=p.parse_args()
    root=Path(a.root).resolve(strict=True);spec=json.loads((root/'submission.json').read_text())
    assert spec['cold_installation_qualified'] is False and not spec['Qwen_model_payload']
    lock=root/spec['environments']['sim']['lock'];assert lock.is_file()
    for line in lock.read_text().splitlines():
        if line and not line.startswith('#'):assert '==' in line and '--hash=sha256:' in line and '/home/' not in line and '/tmp/' not in line
    wheels=root/'dependencies/wheels';assert wheels.is_dir()
    argv=[str(root/'.venvs/sim/bin/python'),'-m','pip','install','--no-index','--find-links',str(wheels),'--require-hashes','--no-deps','-r',str(lock)]
    if a.check_only:
        print(json.dumps(dict(schema='compact_student_relative_offline_install_recipe_CPU_v1',root=str(root),sim_lock_relative=str(lock.relative_to(root)),wheel_files=len(list(wheels.glob('*.whl'))),argv=argv,
            installation_performed=False,cold_installation_qualified=False,GPU_queries=0,models_constructed=0)));return
    assert sys.version_info[:2]==(3,11),'Pinned CPython3.11 ABI required'
    assert not (root/'.venvs/sim').exists(),'Fresh isolated installation only'
    subprocess.run([sys.executable,'-m','venv',str(root/'.venvs/sim')],check=True)
    subprocess.run(argv,check=True)
    # Upstream packages are imported from the explicitly rooted PYTHONPATH in
    # the official entry; no editable install with historical absolute paths.
    print(json.dumps(dict(status='offline_locked_native_packages_installed',native_worlds_qualified=False,GPU_queries=0)))
if __name__=='__main__':main()
