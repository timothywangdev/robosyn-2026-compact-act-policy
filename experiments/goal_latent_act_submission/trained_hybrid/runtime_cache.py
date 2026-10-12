"""Short fresh mutable cache, built only after original scientific/asset guards."""
import json
import os
from pathlib import Path
import shutil
import tempfile
from experiments.goal_latent_act_submission.trained_hybrid.contracts_v2 import require,sha,json_write
from experiments.goal_latent_act_submission.cold_cpu import identity


def prepare(root,spec,env,*,scratch_parent='/tmp'):
    root=Path(root).resolve(strict=True);parent=Path(scratch_parent).resolve(strict=True)
    require(parent.is_dir() and len(str(parent))<=32,'Short existing scratch parent required')
    selected={n:r for n,r in spec['files'].items() if n.startswith('assets/embodichain/data/')}
    require(selected and not any('/assembled/' in n for n in selected),'Only static pinned data may populate runtime cache')
    cache=Path(tempfile.mkdtemp(prefix='rh-',dir=parent));os.chmod(cache,0o700)
    links=copies=0
    for name,row in sorted(selected.items()):
        source=root/name;relative=Path(name).relative_to('assets/embodichain/data');target=cache/'data'/relative
        require(not source.is_symlink() and source.is_file() and source.stat().st_size==row['bytes'] and sha(source)==row['sha256'],'Static cache source differs: '+name)
        target.parent.mkdir(parents=True,exist_ok=True)
        if source.stat().st_dev==target.parent.stat().st_dev:os.link(source,target);links+=1
        else:shutil.copy2(source,target);copies+=1
    for name in ('data/assembled','mat','xdg'):(cache/name).mkdir(mode=0o700,parents=True)
    receipt=dict(schema='trained_hybrid_private_runtime_cache_v1',owner=identity(os.getpid()),path=str(cache),
        static_members=len(selected),ordinary_hardlinks=links,ordinary_copies=copies,
        generated_assembled_files_initially=0,private_mode='0700',actor_and_checkpoint_changed=False,
        cleanup='Retain until owned native processes drain; operator may remove only this exact receipt-bound fresh cache')
    json_write(cache/'owner.json',receipt)
    result=dict(env);result.update(EMBODICHAIN_DATA_ROOT=str(cache/'data'),EMBODICHAIN_MATERIAL_CACHE_DIR=str(cache/'mat'),XDG_RUNTIME_DIR=str(cache/'xdg'))
    return result,receipt
