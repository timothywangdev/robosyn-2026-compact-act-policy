"""CPU canonical streaming primitives. No foundation/simulator imports."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
from .multitask_data import CanonicalTaskData,TASK_REVISIONS,valid_starts,digest,balanced_stream
from .codec import eef80
from .normalizer import VALID,GRIP,transform

SCHEMA='wam-task-balanced-inputs-v1'
def clip_id(task,revision,episode,frame):return f'{task}/{revision}/{episode:06d}/{frame:06d}'

def fit_normalizer(datasets,chain,episodes=range(900)):
 """Equal task weights, actual fit command rows only; fixed grip[-1,1]."""
 moments=[];sources={}
 for task,data in sorted(datasets.items()):
  total=np.zeros(80);squares=np.zeros(80);count=0;sources[task]=[]
  for ep in episodes:
   row=data.read_episode(ep);x=eef80(row['commands'],chain,command=True).astype(np.float64)
   total+=x.sum(0);squares+=(x*x).sum(0);count+=len(x);sources[task].append({'episode':ep,'parquet_sha256':row['parquet_sha256']})
  if not count:raise ValueError('No real fit command rows')
  moments.append((total/count,squares/count))
 mean=np.mean([x[0] for x in moments],axis=0);second=np.mean([x[1] for x in moments],axis=0)
 scale=np.sqrt(np.maximum(second-mean*mean,0));scale=np.where(scale>1e-6,scale,1)
 unused=np.ones(80,bool);unused[VALID]=False;mean[unused]=0;scale[unused]=1;mean[GRIP]=0;scale[GRIP]=1
 return mean.astype(np.float32),scale.astype(np.float32),sources

def decode_frames(path,frames,timeout=90):
 frames=sorted(set(frames));info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height','-of','json',str(path)],timeout=15))['streams'][0]
 w,h=info['width'],info['height'];expr='select='+ '+'.join(f'eq(n\\,{n})' for n in frames)
 raw=subprocess.check_output(['ffmpeg','-v','error','-threads','1','-i',str(path),'-vf',expr,'-vsync','0','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],timeout=timeout)
 if len(raw)!=len(frames)*w*h*3:raise ValueError('Decoded video frame count/size differs')
 values=np.frombuffer(raw,np.uint8).reshape(len(frames),h,w,3)
 return dict(zip(frames,values))

def episode_samples(data,episode,chain,mean,scale,assemble,decoder=decode_frames):
 """At most one episode RGB decoded; clean future video stays labels only."""
 from PIL import Image
 row=data.read_episode(episode);starts=valid_starts(len(row['qpos']))
 if not starts:raise ValueError('No complete future32 clip')
 selected=sorted({s+4*k for s in starts for k in range(9)})
 paths={cam:data.video_path(episode,cam) for cam in data.camera_keys};hashes={cam:digest(p) for cam,p in paths.items()}
 decoded={cam:decoder(p,selected) for cam,p in paths.items()}
 for frame in starts:
  action=transform(eef80(row['commands'][frame+1:frame+33],chain,True),mean,scale)
  proprio=transform(eef80(row['qpos'][frame:frame+1],chain)[0],mean,scale)
  mask=np.zeros((32,80),bool);mask[:,VALID]=True;pmask=np.zeros((1,80),bool);pmask[:,np.r_[VALID,np.arange(10,19),np.arange(44,53)]]=True
  videos=[assemble({label:Image.fromarray(decoded[cam][n]) for label,cam in zip(('head','left','right'),data.camera_keys)},['head','left','right'],384,320) for n in range(frame,frame+33,4)]
  sample={'action':action,'proprio':proprio,'action_mask':mask,'proprio_mask':pmask,'video_mask':np.ones(9,bool),'video':videos,'prompt':row['instruction']}
  record={'clip_id':clip_id(data.task,data.revision,episode,frame),'task':data.task,'revision':data.revision,'episode':episode,'frame':frame,'instruction':row['instruction'],'current_qpos':row['qpos'][frame].tolist(),'normalized_proprio':proprio.tolist(),'proprio_mask':pmask.tolist(),'parquet_sha256':row['parquet_sha256'],'source_video_sha256':hashes,'command_frames':list(range(frame+1,frame+33)),'video_frames':list(range(frame,frame+33,4)),'action_sha256':hashlib.sha256(action.tobytes()).hexdigest(),'forecast50_supported':frame+50<len(row['qpos']),'episode_length':len(row['qpos']),'bell_historical_collector_certified':data.historical_collector_certified,'supervision_convention':data.supervision_convention}
  yield record,sample

def balanced_cache_order(records,steps,seed=1701):
 lengths={};lookup={}
 for i,r in enumerate(records):
  if r['episode']>=900:raise ValueError('Monitor sample in fit cache')
  identity=(r['task'],r['revision'],r['episode'],r['frame'])
  if identity in lookup:raise ValueError('Duplicate canonical cache identity')
  lookup[identity]=i;lengths.setdefault(r['task'],{})[r['episode']]=r['episode_length']
 expected={(t,TASK_REVISIONS[t],ep,f) for t,rows in lengths.items() for ep,n in rows.items() for f in valid_starts(n)}
 if set(lookup)!=expected:raise ValueError('Incomplete seven-start episode cache')
 return [lookup[(c.task,c.revision,c.episode,c.frame)] for c in balanced_stream(lengths,steps,seed)]

def collate_prepared(items):
 """Pinned native cache tensors already have singleton batch dimension.

Concatenate that dimension only; identical scalar runtime settings must agree.
Prompt context remains a separate row per example, never shared across tasks.
"""
 import torch
 if not items or any(set(x)!=set(items[0]) for x in items):raise ValueError('Prepared batch keys mismatch')
 result={}
 for key in items[0]:
  values=[x[key] for x in items]
  if isinstance(values[0],torch.Tensor):
   if any(not isinstance(v,torch.Tensor) or v.ndim<1 or v.shape[0]!=1 or v.shape[1:]!=values[0].shape[1:] or v.dtype!=values[0].dtype for v in values):raise ValueError('Pinned singleton cache tensor shape/dtype mismatch '+key)
   result[key]=torch.cat(values,dim=0)
  else:
   if any(type(v)!=type(values[0]) or v!=values[0] for v in values):raise ValueError('Prepared runtime scalar mismatch '+key)
   result[key]=values[0]
 return result

def checkpoint_exposure(records,order,optimizer_step,batch_size):
 count=optimizer_step*batch_size
 if not 0<=count<=len(order):raise ValueError('Checkpoint sample exposure beyond order')
 ids=order[:count];counts={task:0 for task in sorted({r['task'] for r in records})}
 for i in ids:counts[records[i]['task']]+=1
 return {'last_step':optimizer_step,'samples_seen':count,'unique_clips_seen':len(set(ids)),'samples_seen_by_task':counts}
