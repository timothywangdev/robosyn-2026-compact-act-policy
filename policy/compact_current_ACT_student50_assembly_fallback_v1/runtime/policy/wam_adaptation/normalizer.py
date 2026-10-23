"""Unclipped fit-only affine normalization; grippers retain fixed[-1,1]."""
import numpy as np
VALID=np.r_[np.arange(10),np.arange(34,44)]
GRIP=np.array([9,43])
def fit(rows):
    x=np.asarray(rows,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=80 or not np.isfinite(x).all():raise ValueError('Finite[N,80]')
    mean=x.mean(0);scale=x.std(0);scale=np.where(scale>1e-6,scale,1)
    mean[GRIP]=0;scale[GRIP]=1
    unused=np.ones(80,bool);unused[VALID]=False;mean[unused]=0;scale[unused]=1
    return mean.astype(np.float32),scale.astype(np.float32)
def transform(x,mean,scale,inverse=False):
    x=np.asarray(x,dtype=np.float32)
    if x.shape[-1]!=80 or not np.isfinite(x).all() or np.any(scale<=0):raise ValueError('Normalizer inputs')
    return x*scale+mean if inverse else (x-mean)/scale
