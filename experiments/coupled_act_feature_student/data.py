"""Read only declared prepared features; never expose future labels to student."""
import json
from pathlib import Path
import numpy as np
from .contracts import validate_feature_arrays,verify,require


def load(root,expected):
    path=Path(root)/'receipt.json';require(verify(expected)==path.resolve(),'Exact feature cache required')
    r=json.loads(path.read_text());require(r['status']=='all1000_declared_current_features_complete','Actual complete extraction required')
    plan=json.loads(verify(r['plan']).read_text());rows=plan['rows']
    with np.load(verify(r['arrays']),allow_pickle=False) as z:arrays={key:z[key] for key in z.files}
    validate_feature_arrays(arrays,rows)
    optimizer=np.array([r['partition']=='optimizer' for r in rows]);gate=np.array([r['partition']=='gate' for r in rows]);diagnostic=np.array([r['partition']=='diagnostic' for r in rows])
    require((optimizer.sum(),gate.sum(),diagnostic.sum())==(800,100,100),'Fixed800/100/100 source split')
    return plan,arrays,dict(optimizer=optimizer,gate=gate,diagnostic=diagnostic)


def moments(arrays,optimizer):
    require(int(optimizer.sum())==800,'Optimizer800 normalization only')
    visual=arrays['visual'][optimizer];global_visual=arrays['visual_global'][optimizer]
    return dict(visual_mean=visual.mean((0,3,4)),visual_scale=np.maximum(visual.std((0,3,4)),.01),
        global_mean=global_visual.mean(0),global_scale=np.maximum(global_visual.std(0),.01))


def batch(arrays,ids,statistics,device='cpu'):
    import torch
    result={key:torch.from_numpy(value[ids]).to(device) for key,value in arrays.items()}
    mean=torch.as_tensor(statistics['visual_mean'],dtype=torch.float32,device=device)[None,:,:,None,None]
    scale=torch.as_tensor(statistics['visual_scale'],dtype=torch.float32,device=device)[None,:,:,None,None]
    result['visual']=(result['visual']-mean)/scale
    result['visual_global']=(result['visual_global']-torch.as_tensor(statistics['global_mean'],device=device))/torch.as_tensor(statistics['global_scale'],device=device)
    return result
