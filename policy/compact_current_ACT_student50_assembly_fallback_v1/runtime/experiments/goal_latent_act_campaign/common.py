"""Small CPU-only exact-reference contracts for the native campaign."""
from pathlib import Path
import hashlib
import importlib.util
import json

SCHEMA = 'goal_latent_act_native_campaign_v1'
ORDER = ('click_bell','drawer_open_place','item_assembly','items_handover','table_rearrangement',
         'mixer_operating','sample_loading','water_pouring','manipulate_pipette','handle_basket')
MODES = ('goal','parent','goal_free')
GPU = 'GPU-d308805c-10d7-4893-1cb4-f788b16b3634'
STOP = '2026-10-12T06:00:00+00:00'


def require(condition, message):
    if not condition: raise ValueError(message)


def ref(path):
    path = Path(path).resolve(strict=True); h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(8<<20), b''): h.update(part)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=h.hexdigest())


def verify(row):
    require(set(row)=={'path','bytes','sha256'}, 'Exact path/bytes/SHA reference required')
    require(Path(row['path']).is_absolute() and ref(row['path'])==row, 'Admitted file bytes changed')
    return Path(row['path'])


def supervisor(row):
    path = verify(row)
    spec = importlib.util.spec_from_file_location('goal_act_admitted_supervisor',path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    require(all(callable(getattr(module,n,None))for n in ('validate_plan','run_plan','group_members','gpu_snapshot','identity')),
            'Explicit reviewed native supervisor API required')
    return module


def fresh(path):
    path = Path(path)
    require(path.is_absolute() and path==path.resolve() and not path.exists() and not path.is_symlink(),
            'Fresh absolute output required')
    return path


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
