"""Fresh CPython3.11 offline install; unchanged full lock, no GPU/model calls."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import argparse,hashlib,importlib.metadata,json,re,subprocess,sys
from pathlib import Path

def ref(p):
 p=Path(p).resolve();h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def inspect(root):
 assert sys.version_info[:2]==(3,11),'Frozen binary wheels require CPython3.11, not3.12'
 spec=json.loads((root/'submission.json').read_text());assert spec['schema']=='compact_current_ACT_student50_byte_preserving_portable_package_v1'
 lock=root/spec['environments']['sim']['lock'];lines=lock.read_text().splitlines()
 build=('setuptools','wheel','Cython','numpy','pip','toml')
 selected=[line for line in lines if line.split('==')[0] in build];assert len(selected)==len(build)
 for path in (root/'dependencies/wheels').iterdir():
  row=spec['files'][str(path.relative_to(root))];assert ref(path)['sha256']==row['sha256'] and path.stat().st_size==row['bytes']
 return spec,lock,selected
def main():
 p=argparse.ArgumentParser(allow_abbrev=False)
 for key in ('root','output'):p.add_argument('--'+key,required=True)
 p.add_argument('--check-only',action='store_true');p.add_argument('--admission');p.add_argument('--admission-sha256');a=p.parse_args()
 root=Path(a.root).resolve(strict=True);out=Path(a.output).absolute();assert not out.exists();out.mkdir(parents=True)
 spec,lock,build=inspect(root)
 venv=root/'.venvs/sim';assert not venv.exists() and not venv.is_symlink(),'Genuinely fresh environment required'
 recipe=dict(schema='Student50_portable_fresh_CP311_offline_install_recipe_v2',package=ref(root/'submission.json'),full_sim_lock=ref(lock),
   build_requirements=build,full_frozen_lock_modified=False,source_distributions=['nuitka','toppra','fvcore','iopath','evdev'],
   offline_distribution_count=len(list((root/'dependencies/wheels').iterdir())),interpreter=sys.executable,Python_ABI='cp311',
   proposed_resource_caps=dict(wall_seconds=600,owned_tree_RSS_bytes=2<<30,incremental_disk_bytes=14<<30,free_disk_floor_bytes=8<<30,threads=2),
   no_build_isolation_only_after_actual_bootstrap_install=True,GPU_queries=0,GPU_launches=0,models_constructed=0,worlds=0,cold_installation_qualified=False)
 write(out/'recipe.json',recipe)
 if a.check_only:print(json.dumps(ref(out/'recipe.json')));return
 assert a.admission and a.admission_sha256 and ref(a.admission)['sha256']==a.admission_sha256
 admission=json.loads(Path(a.admission).read_text())
 assert admission['schema']=='root_authorized_Student50_portable_fresh_CP311_CPU_install_v2' and admission['execution_authorized'] is True
 assert admission['package']==recipe['package'] and admission['installer']==ref(__file__) and admission['resource_caps']==recipe['proposed_resource_caps']
 env=os.environ.copy()
 for key in ('PYTHONPATH','PYTHONHOME','PYTHONUSERBASE','PYTHONSTARTUP'):env.pop(key,None)
 tmp=out/'tmp';tmp.mkdir();env.update(CUDA_VISIBLE_DEVICES='',PIP_NO_INDEX='1',PIP_NO_CACHE_DIR='1',PIP_FIND_LINKS=str(root/'dependencies/wheels'),
  OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CMAKE_BUILD_PARALLEL_LEVEL='2',MAKEFLAGS='-j2',MAX_JOBS='2',TORCHINDUCTOR_COMPILE_THREADS='1',
  PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',TMPDIR=str(tmp))
 subprocess.run([sys.executable,'-m','venv',str(venv)],env=env,check=True)
 build_lock=out/'build-bootstrap.lock';build_lock.write_text('\n'.join(build)+'\n')
 py=str(venv/'bin/python');common=[py,'-m','pip','install','--no-index','--no-cache-dir','--find-links',str(root/'dependencies/wheels'),'--require-hashes','--no-deps']
 steps=[('build-bootstrap',common+['-r',str(build_lock)]),
        ('full-unchanged-lock',common+['--no-build-isolation','-r',str(lock)])]
 for name,argv in steps:
  with (out/(name+'.stdout.log')).open('xb') as stdout,(out/(name+'.stderr.log')).open('xb') as stderr:
   code=subprocess.run(argv,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr).returncode
  write(out/(name+'-exit.json'),dict(argv=argv,actual_exit_code=code,no_cache=True,no_index=True))
  assert code==0,(name,code)
 # Version metadata validation imports no installed model or CUDA module.
 check="import importlib.metadata,json,sys; from pathlib import Path; from packaging.version import Version; rows={}; "
 check+="\nfor line in Path(sys.argv[1]).read_text().splitlines():\n if line and not line.startswith('#'):\n  name,rest=line.split('==',1);version=rest.split()[0];actual=importlib.metadata.version(name);assert Version(actual)==Version(version),(name,version,actual);rows[name]=actual\n"
 check+="print(json.dumps(dict(installed_packages=rows,Python_ABI='cp311',GPU_queries=0,models_constructed=0,worlds=0)))"
 result=subprocess.run([py,'-I','-B','-c',check,str(lock)],cwd=root,env=env,check=True,text=True,capture_output=True)
 validated=json.loads(result.stdout);write(out/'installed-version-metadata.json',validated)
 write(out/'receipt.json',dict(schema='Student50_portable_actual_fresh_CP311_offline_install_v2',status='actual_fresh_full_frozen_lock_install_and_versions_PASS_native_pending',
  package=recipe['package'],recipe=ref(out/'recipe.json'),full_lock_byte_unchanged=ref(lock)==recipe['full_sim_lock'],
  installed_version_metadata=ref(out/'installed-version-metadata.json'),cold_installation_qualified=True,installed_native_qualified=False,
  CUDA_or_Torch_models_imported_by_installer=False,GPU_queries=0,GPU_launches=0,models_constructed=0,worlds=0,submission_ready=False))
 print(json.dumps(ref(out/'receipt.json')))
if __name__=='__main__':main()
