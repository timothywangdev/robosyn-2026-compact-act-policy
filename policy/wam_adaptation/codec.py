"""Absolute command EEF override; immutable measured observations remain separate."""
import numpy as np
from .normalizer import VALID

def eef80(rows,chain,command=False):
 rows=np.asarray(rows,dtype=np.float64)
 if rows.ndim!=2 or rows.shape[1]!=14 or not np.isfinite(rows).all():raise ValueError('Finite[N,14]')
 out=np.zeros((len(rows),80),np.float32)
 for j,k in ((0,0),(7,34)):
  T=np.broadcast_to(np.eye(4),(len(rows),4,4)).copy();idx=0
  for origin,axis in chain:
   T=T@origin
   if axis is not None:
    axis=np.asarray(axis)/np.linalg.norm(axis);x,y,z=axis;K=np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])
    angle=rows[:,j+idx];R=np.broadcast_to(np.eye(4),T.shape).copy();R[:,:3,:3]=np.eye(3)+np.sin(angle)[:,None,None]*K+(1-np.cos(angle))[:,None,None]*(K@K);T=T@R;idx+=1
  tcp=np.diag([-1.,-1.,1.,1.]);tcp[2,3]=.143;T=T@tcp
  out[:,k:k+3]=T[:,:3,3];out[:,k+3:k+9]=T[:,:3,:2].transpose(0,2,1).reshape(-1,6)
  opening=np.clip(rows[:,j+6]/.05 if command else rows[:,j+6],0,1);out[:,k+9]=2*opening-1
 return out
