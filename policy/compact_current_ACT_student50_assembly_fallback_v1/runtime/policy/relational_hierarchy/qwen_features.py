"""Frozen current-only Qwen prefill features; never generates or parses subgoals.

Three appended role-query endpoints see all current cameras/task/qpos. Features
are latent representations, not certificates of grounding or planned effects.
Offline and online use exactly the same preparation, token positions and forward.
"""
import argparse,contextlib,hashlib,json,os,signal,sys,time
from pathlib import Path
import numpy as np
import torch
from .qwen_contract_worker import prepare_request,MODEL_ID,MODEL_REVISION,TRANSFORMERS_VERSION,MAX_LINE_BYTES

FEATURE_VERSION='qwen35-role-hidden-v1'
FEATURE_DIM=4096
ROLES=('left_arm','right_arm','object_destination')
MARKERS=('[LEFT_ARM_QUERY]','[RIGHT_ARM_QUERY]','[OBJECT_DESTINATION_QUERY]')

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def role_prompt(task,qpos):
 if any(marker in task for marker in MARKERS):raise ValueError('Task contains reserved role query marker')
 return (
  'Use only the three labeled CURRENT RGB images, task, and measured joint state. '
  'Represent the next short executable effects for the left arm, right arm, and object/destination relation. '
  'Do not assume contact, successful grasp, completion or visibility from joint state alone. '
  'The first seven joint values refer to the left arm and last seven to the right arm. '
  'For this handover/assembly input, indices6 and13 are measured normalized gripper opening fractions: '
  'zero is near-closed, one near-open, with possible small simulation overshoot. Other joints are rotational qpos. '
  f'Task: {task}\nCurrent qpos (14D): {json.dumps(qpos)}\n'
  f'Left arm next executable effect:\n{MARKERS[0]}\n'
  f'Right arm next executable effect:\n{MARKERS[1]}\n'
  f'Object relative to destination next desired relation:\n{MARKERS[2]}'
 )

def prepare_current(request,input_root,max_image_edge=384):
 content,audit=prepare_request(request,input_root,max_image_edge)
 # Reuse strict image/state validation, discard every byte of the JSON prompt.
 if content[-1].get('type')!='text':raise ValueError('Unexpected validated content layout')
 prompt=role_prompt(request['task'],audit['qpos'])
 content=content[:-1]+[{'type':'text','text':prompt}]
 audit={k:v for k,v in audit.items() if k not in ('prompt','contract_version','feature_dim','requested_coordinate_system')}
 audit.update(prompt=prompt,feature_version=FEATURE_VERSION,feature_dim=FEATURE_DIM,input_npz_sha256=digest(audit['input_path']))
 return content,audit

def marker_positions(input_ids,attention_mask,tokenizer):
 if input_ids.ndim!=2 or attention_mask.shape!=input_ids.shape:raise ValueError('Expected batched token IDs/mask')
 patterns=[tokenizer.encode(marker,add_special_tokens=False) for marker in MARKERS]
 if any(not p for p in patterns):raise ValueError('Empty marker tokenization')
 result=[]
 for ids,mask in zip(input_ids.detach().cpu().tolist(),attention_mask.detach().cpu().tolist()):
  row=[]
  for pattern in patterns:
   matches=[i+len(pattern)-1 for i in range(len(ids)-len(pattern)+1) if ids[i:i+len(pattern)]==pattern and all(mask[i:i+len(pattern)])]
   if len(matches)!=1:raise ValueError('Role marker must match exactly once in actual nonpadding token IDs')
   row.append(matches[0])
  if row!=sorted(row) or len(set(row))!=3:raise ValueError('Role query endpoints must be distinct and ordered')
  result.append(row)
 return torch.tensor(result,dtype=torch.long,device=input_ids.device)

def gather_roles(hidden,positions):
 if hidden.ndim!=3 or hidden.shape[-1]!=FEATURE_DIM or positions.shape!=(hidden.shape[0],3):raise ValueError('Expected final Qwen hidden[B,L,4096] and3query endpoints')
 if bool((positions<0).any()) or bool((positions>=hidden.shape[1]).any()):raise ValueError('Query position outside sequence')
 value=hidden[torch.arange(hidden.shape[0],device=hidden.device)[:,None],positions]
 if not bool(torch.isfinite(value).all()) or not bool((value!=0).any()):raise ValueError('Nonfinite/allzero frozen foundation features')
 return value.detach()

class FrozenQwenFeatures:
 def __init__(self,model_path):
  os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
  import transformers
  from policy.robosyn_sprint.qwen_planner import load_local_model
  if transformers.__version__!=TRANSFORMERS_VERSION:raise ValueError('Unexpected pinned Transformers version')
  if not torch.cuda.is_available():raise RuntimeError('Explicit GPU extractor launch required; no CPU model fallback')
  root=Path(model_path).resolve(strict=True);config=json.loads((root/'config.json').read_text())
  if config.get('model_type')!='qwen3_5' or config['text_config']['hidden_size']!=FEATURE_DIM:raise ValueError('Expected pinned Qwen3.5-9B4096D architecture')
  load_started=time.monotonic()
  self.processor,self.model,architecture=load_local_model(str(root),torch,transformers)
  self.processor.tokenizer.padding_side='left';self.model.eval();self.model.requires_grad_(False)
  torch.cuda.synchronize();load_seconds=time.monotonic()-load_started
  torch.cuda.reset_peak_memory_stats()
  self.metadata={'load_seconds':load_seconds,'allocated_bytes_after_load':torch.cuda.memory_allocated(),'reserved_bytes_after_load':torch.cuda.memory_reserved(),'model_id':MODEL_ID,'declared_revision':MODEL_REVISION,'weight_identity':'launch coordinator must verify pinned weight inventory','config_sha256':digest(root/'config.json'),'architecture':architecture,'transformers':transformers.__version__,'torch':torch.__version__,'model_dtype':'bfloat16','cache_dtype':'float16','use_cache':False,'autoregressive_generation':False,'hidden_states':'final underlying multimodal model only; no LMhead/all-layer states'}
 def extract(self,contents):
  messages=[[{'role':'user','content':content}] for content in contents]
  with torch.inference_mode():
   inputs=self.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=False,return_dict=True,return_tensors='pt',padding=True,enable_thinking=False).to(self.model.device)
   positions=marker_positions(inputs['input_ids'],inputs['attention_mask'],self.processor.tokenizer)
   # Query suffix must follow the final current-image placeholder.
   for ids,pos in zip(inputs['input_ids'],positions):
    image_indices=(ids==self.model.config.image_token_id).nonzero().flatten()
    if len(image_indices)==0 or int(image_indices.max())>=int(pos.min()):raise ValueError('Role queries must follow all current image tokens')
   output=self.model.model(**inputs,use_cache=False,return_dict=True)
   features=gather_roles(output.last_hidden_state,positions)
   arrays=features.float().cpu().numpy().astype(np.float16)
   if not np.isfinite(arrays).all():raise ValueError('FP16 feature cache overflow')
   return arrays,positions.cpu().tolist(),{'input_tokens':inputs['attention_mask'].sum(1).cpu().tolist(),'image_grid_thw':inputs['image_grid_thw'].cpu().tolist(),'input_ids_sha256':[hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest() for ids in inputs['input_ids']],'generated_tokens':0}

def process_batch(requests,extractor,input_root,output_root,sink,audit,max_image_edge,online=False):
 prepared=[prepare_current(request,input_root,max_image_edge) for request in requests]
 started=time.monotonic();arrays,positions,token_audit=extractor.extract([x[0] for x in prepared])
 if arrays.shape!=(len(requests),3,FEATURE_DIM):raise ValueError('Feature batch shape mismatch')
 for i,(_,proof) in enumerate(prepared):
  key=hashlib.sha256((proof['request_id']+proof['input_npz_sha256']).encode()).hexdigest()
  relative=Path('features')/(key+'.npz');path=output_root/relative
  with path.open('xb') as f:np.savez_compressed(f,features=arrays[i])
  record={'ok':True,'request_id':proof['request_id'],'feature_version':FEATURE_VERSION,'feature_dim':FEATURE_DIM,'feature_path':str(relative),'feature_sha256':digest(path),'shape':[3,FEATURE_DIM],'dtype':'float16','nnz':int(np.count_nonzero(arrays[i])),'valid_slots':[True]*3,'validity_meaning':'finite latent query present; not semantic grounding','input_npz_sha256':proof['input_npz_sha256'],'frame_ids':proof['frame_ids'],'role_token_positions':positions[i]}
  sink.write(json.dumps(record)+'\n');sink.flush()
  audit.write(json.dumps({**record,**proof,'token_audit':token_audit,'batch_seconds':time.monotonic()-started})+'\n');audit.flush()
  if online:print(json.dumps(record),flush=True)

def main(argv=None):
 p=argparse.ArgumentParser(allow_abbrev=False);mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--requests',type=Path);mode.add_argument('--worker',action='store_true');p.add_argument('--model-path',required=True);p.add_argument('--input-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--batch-size',type=int,default=1);p.add_argument('--max-image-edge',type=int,default=384);p.add_argument('--max-requests',type=int,default=20000);p.add_argument('--timeout-seconds',type=int,default=3600);a=p.parse_args(argv)
 if not 1<=a.batch_size<=8 or a.worker and a.batch_size!=1 or not 1<=a.max_requests<=100000 or not 1<=a.timeout_seconds<=86400:p.error('Invalid batch/request/lifetime bounds')
 if not 32<=a.max_image_edge<=2048:p.error('Invalid image budget')
 if a.output_root.exists() or a.output_root.is_symlink():raise ValueError('Fresh output directory required')
 a.output_root.mkdir(parents=True,exist_ok=False);(a.output_root/'features').mkdir()
 def deadline(*_):raise TimeoutError('Frozen feature extractor lifetime cap')
 started=time.monotonic()
 previous=signal.signal(signal.SIGALRM,deadline);signal.alarm(a.timeout_seconds);count=0;seen=set();batch=[];manifest={'status':'running','feature_version':FEATURE_VERSION,'feature_dim':FEATURE_DIM,'roles':ROLES,'cache_dtype':'float16','max_image_edge':a.max_image_edge,'batch_size':a.batch_size,'model_id':MODEL_ID,'model_revision':MODEL_REVISION,'extractor_sha256':digest(__file__),'validator_sha256':digest(Path(__file__).with_name('qwen_contract_worker.py')),'current_only':True,'not_a_grounding_certificate':True}
 try:
  with contextlib.ExitStack() as stack:
   sink=stack.enter_context((a.output_root/'features.jsonl').open('x'));audit=stack.enter_context((a.output_root/'audit.jsonl').open('x'));source=sys.stdin if a.worker else stack.enter_context(a.requests.open())
   with contextlib.redirect_stdout(sys.stderr):extractor=FrozenQwenFeatures(a.model_path)
   manifest['model']=extractor.metadata
   if (a.input_root/'index.json').is_file():manifest['request_index_sha256']=digest(a.input_root/'index.json')
   if a.worker:print(json.dumps({'ok':True,'event':'ready','feature_version':FEATURE_VERSION,'output_root':str(a.output_root.resolve())}),flush=True)
   while True:
    line=source.readline(MAX_LINE_BYTES+1)
    if not line:break
    if len(line.encode())>MAX_LINE_BYTES:raise ValueError('Oversized request line')
    request=json.loads(line);key=request.get('request_id') if isinstance(request,dict) else None
    if key in seen:raise ValueError('Duplicate request ID')
    seen.add(key);count+=1
    if count>a.max_requests:raise ValueError('Request count cap')
    batch.append(request)
    if len(batch)==a.batch_size:process_batch(batch,extractor,a.input_root,a.output_root,sink,audit,a.max_image_edge,a.worker);batch=[]
   if batch:process_batch(batch,extractor,a.input_root,a.output_root,sink,audit,a.max_image_edge)
  manifest.update(status='complete',requests=count,features_jsonl_sha256=digest(a.output_root/'features.jsonl'),audit_sha256=digest(a.output_root/'audit.jsonl'))
 except BaseException as error:
  manifest.update(status='failed',requests_seen=count,error_type=type(error).__name__,error=str(error));raise
 finally:
  manifest['elapsed_seconds']=time.monotonic()-started
  if 'extractor' in locals():manifest.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),memory_scope='extractor process; reset after model load; retains model resident weights; not complete simulator stack')
  (a.output_root/'manifest.json').write_text(json.dumps(manifest,indent=2));signal.alarm(0);signal.signal(signal.SIGALRM,previous)
if __name__=='__main__':main()
