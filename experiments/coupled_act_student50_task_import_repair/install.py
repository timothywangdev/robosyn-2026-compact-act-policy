"""Install two source-pinned routing files; preserve every original package byte."""
import argparse,hashlib,json,os
from pathlib import Path
import sys
import sysconfig

PACKAGE='191e3a69625175b96c685304e19d5362522e16338943cd6c3065448cabb89515'
PTH=b'import robosyn_official_task_paths; robosyn_official_task_paths.install()\n'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path):
    p=Path(path).resolve(strict=True);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def write(path,value):
    p=Path(path);assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2)+'\n')

def inventory(root):
    manifest=root/'submission.json';assert not manifest.is_symlink() and sha(manifest)==PACKAGE
    spec=json.loads(manifest.read_text());prefix='upstream/EmbodiChain/embodichain_tasks/'
    rows={n:r for n,r in spec['files'].items() if n.startswith(prefix)}
    assert rows and prefix+'embodichain_tasks/tableware/base_agent_env.py' in rows
    for name,row in rows.items():
        source=root/name;assert not source.is_symlink() and source.is_file() and source.resolve().is_relative_to(root)
        assert source.stat().st_size==row['bytes'] and sha(source)==row['sha256']
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--install',action='store_true');a=p.parse_args()
    root=Path(a.root).resolve(strict=True);rows=inventory(root);site=Path(sysconfig.get_paths()['purelib']).resolve(strict=True)
    assert Path(sys.prefix).resolve(strict=True)==root/'.venvs/sim' and site.is_relative_to(root/'.venvs/sim')
    source=Path(__file__).with_name('finder.py');body=source.read_bytes();dest=site/'robosyn_official_task_paths.py';pth=site/'robosyn_official_task_paths.pth'
    assert not dest.exists() and not pth.exists() and not dest.is_symlink() and not pth.is_symlink()
    receipt=dict(schema='Student50_official_nested_task_import_routing_repair_v1',package=ref(root/'submission.json'),
        official_source_inventory=rows,official_source_bytes=sum(r['bytes'] for r in rows.values()),
        finder_source=ref(source),installer_source=ref(__file__),installed=False,
        untouched=['submission.json and all2608 archived files','all actors/checkpoints/locks','all original failures/proofs'],
        mapping=['embodichain_tasks -> upstream/EmbodiChain/embodichain_tasks/embodichain_tasks',
                 'embodichain_tasks.configs -> upstream/EmbodiChain/embodichain_tasks/configs'],
        relocatable_sys_prefix_routing=True,models=0,worlds=0,GPU_queries=0,GPU_launches=0,native_qualified=False)
    if a.install:
        with dest.open('xb') as f:f.write(body)
        with pth.open('xb') as f:f.write(PTH)
        receipt.update(installed=True,installed_finder=ref(dest),installed_pth=ref(pth),new_bytes=len(body)+len(PTH))
    write(a.output,receipt);print(json.dumps(ref(a.output)))

if __name__=='__main__':main()
