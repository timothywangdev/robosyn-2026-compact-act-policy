"""CPU canonical task-local data for a future, separately qualified WAM run.

No simulator/model imports. Bell state units require a sealed explicit evidence
artifact; field names and observed ranges alone do not establish units.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np

TASK_REVISIONS = {
 'click_bell':'dd1f0de495cfec3240beaf2714f239168d5b3fae',
 'drawer_open_place':'6fdf6ee1ed74bbbbc0216d507c8a19c8994a15c3',
 'handle_basket':'c40819caabfea75e0724307845ddf5ee7f15ae8a',
 'item_assembly':'a2f2a4cecaa4f7f08bff309a3eac85c574bb2a86',
 'items_handover':'20a88bf0bc3fe3a1ae58ad97ed60a06c3431684a',
 'manipulate_pipette':'168885aec78256de14e0c130ea12a6ba43e714a7',
 'mixer_operating':'fe5428c644932747b1109da98c6dc79a1caabd51',
 'sample_loading':'67208f9ba05f2f29a77b9c5e7b64acf26c3b3c44',
 'table_rearrangement':'ca14fdfbb9d39914f09b343e072618462bea97b8',
 'water_pouring':'932a287657983c41465d6c9baee5dd478ef61b90',
}
CAMERAS = ('cam_high','cam_left_wrist','cam_right_wrist')
GRIP_AXES = (6,13)
CLAMP_TASKS = frozenset(TASK_REVISIONS)-{'drawer_open_place','table_rearrangement','click_bell'}


def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()


def valid_starts(length, count=7):
 """Unique starts with all32 next commands/video frames present; no padding."""
 if type(length) is not int or length<0 or type(count) is not int or count<1:
  raise ValueError('Integer length/count required')
 if length<33:return []
 return sorted(set(np.linspace(0,length-33,count).astype(int).tolist()))


def physical_commands(task, rows):
 """Inverse recorder units, then public physical clamp; never infer from range."""
 if task not in TASK_REVISIONS:raise ValueError('Unpinned task')
 result=np.asarray(rows,dtype=np.float64).copy()
 if result.ndim!=2 or result.shape[1]!=14 or not np.isfinite(result).all():raise ValueError('Finite[N,14] commands required')
 grip=result[:,GRIP_AXES]
 if task=='drawer_open_place':
  # DrawerOpenPlace._postprocess_action normalizes only recorded EEF commands.
  result[:,GRIP_AXES]=.05*np.clip(grip,0,1)
 elif task=='click_bell':
  if not np.all(grip==0):raise ValueError('Bell zero-gripper contract violated; audit new snapshot')
  result[:,GRIP_AXES]=0
 else:
  # Seven expert paths pass literal0..1 to physical clamp; table already metres.
  result[:,GRIP_AXES]=np.clip(grip,0,.05)
 return result


@dataclass(frozen=True)
class ClipIdentity:
 task:str
 revision:str
 episode:int
 frame:int


class CanonicalTaskData:
 def __init__(self, task, root, *, bell_state_evidence=None, allow_bell_source_convention=False):
  if task not in TASK_REVISIONS:raise ValueError('Unpinned task')
  self.task=task;self.revision=TASK_REVISIONS[task];self.root=Path(root).resolve(strict=True)
  if self.root.name!=self.revision:raise ValueError('Snapshot revision directory differs')
  self.info=json.loads((self.root/'meta/info.json').read_text())
  fps=float(self.info['fps'])
  if not np.isfinite(fps) or fps!=25:raise ValueError('Qualified25Hz convention required')
  self.state_key='observation.qpos' if task=='click_bell' else 'observation.state'
  self.camera_keys={name:(name+'.color' if task=='click_bell' else 'observation.images.'+name) for name in CAMERAS}
  if self.state_key not in self.info['features'] or any(key not in self.info['features'] for key in self.camera_keys.values()):raise ValueError('Pinned task feature schema differs')
  self.state_units='normalized_opening'
  self.unit_evidence_sha256=None
  self.historical_collector_certified=None
  self.supervision_convention='post_step_same_row_command'
  if task=='click_bell':
   if bell_state_evidence is None:raise ValueError('Bell state gripper units unverified; explicit evidence required')
   p=Path(bell_state_evidence);v=json.loads(p.read_text())
   identity=v.get('task')==task and v.get('revision')==self.revision and bool(v.get('source_evidence'))
   verified=identity and v.get('verified') is True and v.get('state_gripper_units') in ('physical_m','normalized_opening') and v.get('observation_alignment')=='post_step_same_row_command'
   declared=identity and allow_bell_source_convention is True and v.get('status')=='source_backed_convention_historical_lineage_unverified' and v.get('verified') is False and v.get('recommended_state_gripper_units')=='normalized_opening' and v.get('recommended_observation_alignment')=='post_step_same_row_command' and v.get('actual_data',{}).get('all_parquets_match_pinned_inventory') is True and v.get('commands',{}).get('all_grippers_exact_zero') is True
   if not (verified or declared):raise ValueError('Invalid Bell state unit evidence or undeclared historical convention')
   self.state_units=v['state_gripper_units'] if verified else 'normalized_opening'
   self.historical_collector_certified=bool(verified)
   self.supervision_convention='post_step_same_row_command' if verified else 'bell-source-consistent-next-command-v1'
   self.unit_evidence_sha256=digest(p)
  self.instructions={}
  for line in (self.root/'meta/tasks.jsonl').read_text().splitlines():
   row=json.loads(line);i=row['task_index'];text=row['task']
   if i in self.instructions or not isinstance(text,str) or not text:raise ValueError('Invalid task-index instruction')
   self.instructions[i]=text

 def read_episode(self, episode):
  import pyarrow.parquet as pq
  if type(episode) is not int or not 0<=episode<1000:raise ValueError('Task-local episode0..999 required')
  path=self.root/f'data/chunk-000/episode_{episode:06d}.parquet'
  # A metadata-only snapshot is never enough for constructing training samples.
  table=pq.read_table(path,columns=[self.state_key,'action','episode_index','frame_index','task_index','timestamp'])
  q=np.asarray(table[self.state_key].to_pylist(),dtype=np.float64);a=np.asarray(table['action'].to_pylist(),dtype=np.float64)
  n=len(table)
  if q.shape!=(n,14) or a.shape!=(n,14) or not np.isfinite(q).all() or not np.isfinite(a).all():raise ValueError('Actual finite14D rows required')
  if table['episode_index'].to_pylist()!=[episode]*n or table['frame_index'].to_pylist()!=list(range(n)):raise ValueError('Episode/frame namespace drift')
  times=np.asarray(table['timestamp'].to_pylist(),dtype=np.float64)
  if not np.isfinite(times).all() or not np.allclose(times,np.arange(n)/25,atol=1e-5):raise ValueError('Frame timestamps differ from25Hz')
  task_indices=table['task_index'].to_pylist()
  if len(set(task_indices))!=1 or task_indices[0] not in self.instructions:raise ValueError('Episode instruction identity differs')
  if self.state_units=='physical_m':q[:,GRIP_AXES]/=.05
  # State observations are normalized opening; codec performs declared0..1 clamp.
  return {'qpos':q,'commands':physical_commands(self.task,a),'instruction':self.instructions[task_indices[0]],'parquet_sha256':digest(path),'episode':episode,'task':self.task,'revision':self.revision}

 def video_path(self, episode, camera):
  if camera not in CAMERAS or type(episode) is not int or not 0<=episode<1000:raise ValueError('Camera/episode identity required')
  template=self.info['video_path'];path=self.root/template.format(episode_chunk=episode//1000,episode_index=episode,video_key=self.camera_keys[camera])
  resolved=path.resolve(strict=True)
  if not resolved.is_relative_to(self.root):raise ValueError('Video path escaped snapshot')
  return resolved

 def clip(self, episode, frame):
  row=self.read_episode(episode);n=len(row['qpos'])
  if type(frame) is not int or not 0<=frame<=n-33:raise ValueError('All32 future commands must be present')
  return {'identity':ClipIdentity(self.task,self.revision,episode,frame),'current_qpos':row['qpos'][frame].copy(),'future_commands':row['commands'][frame+1:frame+33].copy(),'command_frames':list(range(frame+1,frame+33)),'video_frames':list(range(frame,frame+33,4)),'instruction':row['instruction'],'parquet_sha256':row['parquet_sha256'],'forecast50_supported':frame+50<n}


def balanced_stream(episode_lengths, steps, seed=1701):
 """Round-robin tasks; equal update counts, episode coverage before repeats.

Lengths are actual-row-derived mappings task->{task-local episode: length}.
No weighting by trajectory length, no future state or labels used for selection.
"""
 if type(steps) is not int or steps<1 or not episode_lengths:raise ValueError('Positive steps and actual episode lengths required')
 rng=np.random.default_rng(seed);tasks=sorted(episode_lengths);orders={};starts={};positions={t:0 for t in tasks}
 for t in tasks:
  if t not in TASK_REVISIONS:raise ValueError('Unpinned task')
  eligible=[]
  for ep,n in sorted(episode_lengths[t].items()):
   if type(ep) is not int or not 0<=ep<900:raise ValueError('Fit episodes0..899 only')
   windows=valid_starts(n)
   if windows:eligible.append(ep);starts[t,ep]=rng.permutation(windows).tolist()
  if not eligible:raise ValueError('Task has no actual valid episodes')
  orders[t]=rng.permutation(eligible).tolist()
 for step in range(steps):
  task=tasks[step%len(tasks)];position=positions[task];order=orders[task];ep=order[position%len(order)];windows=starts[task,ep];frame=int(windows[(position//len(order))%len(windows)]);positions[task]+=1
  yield ClipIdentity(task,TASK_REVISIONS[task],ep,frame)
