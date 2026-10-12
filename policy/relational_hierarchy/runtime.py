"""Competition entrypoints for an actually trained dense hierarchy checkpoint.

Two isolated local workers, causal refresh50, RPC10, explicit prefix10/50.
Official observation/physical-action codecs and timing boundary are delegated.
No release-checkpoint fallback, generated JSON goal, or hidden state input.
"""
import atexit,hashlib,json,math,os,selectors,subprocess,tempfile,time
from pathlib import Path
import numpy as np
from .artifact_proof import validate_training_proof

ROOT=Path(__file__).resolve().parents[2]
VERSION='qwen35-role-hidden-v1'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def boolean(value):
 if type(value)==bool:return value
 if type(value)==str and value.lower() in ('true','false'):return value.lower()=='true'
 raise ValueError('Expected explicit true/false')

class RpcWorker:
 def __init__(self,command,env,timeout=180):
  if not math.isfinite(timeout) or timeout<=0:raise ValueError('Finite positive worker timeout required')
  self.timeout=timeout;self.buffer=bytearray();self.proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,env=env)
  try:self.ready=self.receive()
  except BaseException:self.close();raise
  if self.ready.get('ok') is not True or self.ready.get('event')!='ready':self.close();raise RuntimeError('Invalid worker startup response')
 def receive(self):
  deadline=time.monotonic()+self.timeout
  with selectors.DefaultSelector() as selector:
   selector.register(self.proc.stdout,selectors.EVENT_READ)
   while b'\n' not in self.buffer:
    remaining=deadline-time.monotonic()
    if remaining<=0 or not selector.select(remaining):raise TimeoutError('Hierarchy worker response deadline')
    block=os.read(self.proc.stdout.fileno(),65536)
    if not block:raise RuntimeError('Hierarchy worker exited before response')
    self.buffer.extend(block)
    if len(self.buffer)>1048576:raise RuntimeError('Oversized hierarchy worker response')
  line,_,rest=self.buffer.partition(b'\n');self.buffer=bytearray(rest);reply=json.loads(line)
  if not isinstance(reply,dict) or reply.get('ok') is not True:raise RuntimeError(reply.get('error','Invalid worker response') if isinstance(reply,dict) else 'Invalid worker response')
  return reply
 def rpc(self,request):
  try:self.proc.stdin.write((json.dumps(request)+'\n').encode());self.proc.stdin.flush();return self.receive()
  except BaseException:self.close();raise
 def close(self):
  if getattr(self,'proc',None) is None:return
  if self.proc.stdin and not self.proc.stdin.closed:
   try:self.proc.stdin.close()
   except BrokenPipeError:pass
  try:self.proc.wait(timeout=5)
  except subprocess.TimeoutExpired:
   self.proc.terminate()
   try:self.proc.wait(timeout=5)
   except subprocess.TimeoutExpired:self.proc.kill();self.proc.wait(timeout=5)
  if self.proc.stdout and not self.proc.stdout.closed:self.proc.stdout.close()

class HierarchyModel:
 def __init__(self,args):
  from policy.smolvla import deploy_policy as official
  self.official=official;self.checkpoint=Path(args['checkpoint_path']).resolve(strict=True)
  self.cfg=json.loads((self.checkpoint/'bridge-config.json').read_text());self.proof=json.loads((self.checkpoint/'training-provenance.json').read_text());self.mode=self.cfg['mode']
  if self.cfg.get('schema')!=VERSION or self.cfg.get('feature_kind')!='dense-qwen' or self.cfg.get('goal_dim')!=4096:raise ValueError('Expected dense trained hierarchy artifact')
  validate_training_proof(self.checkpoint,self.cfg,self.proof)
  if args.get('task_name') and args['task_name']!=self.proof['task']:raise ValueError('Task does not match trained checkpoint')
  self.allow_wiring=boolean(args.get('relational_allow_wiring_pilot',False))
  if self.proof.get('wiring_pilot') and not self.allow_wiring:raise ValueError('Wiring pilot requires explicit integration-test flag')
  self.prefix=int(args.get('relational_execution_prefix',50))
  if self.prefix not in (10,50):raise ValueError('Supported declared execution prefixes10/50 only')
  self.device=args.get('pytorch_device','cuda');self.executed=0;self.last_refresh=-50;self.episode_index=0;self.feature=None;self.motor=None;self.qwen=None
  seed=args.get('motor_random_seed',os.environ.get('SMOLVLA_RANDOM_SEED'))
  self.seed=int(seed) if seed is not None else None
  if self.seed is not None and not 0<=self.seed<2**32:raise ValueError('Episode motor seed outsideuint32')
  self.temp=tempfile.TemporaryDirectory(prefix='relational-runtime-');self.cache=Path(self.temp.name);self.inputs=self.cache/'inputs';self.inputs.mkdir()
  # Reuse official image-key and gripper-auto resolution without spawning its worker.
  self.codec=object.__new__(official.SmolVLAWorkerClient);self.codec.checkpoint_dir=self.checkpoint/'policy';self.codec.gripper_indices=(6,13)
  self.image_keys=self.codec._resolve_image_keys();self.rescale_gripper=self.codec._resolve_gripper_rescale(args.get('smolvla_rescale_gripper','auto'))
  common=os.environ.copy();common.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES=str(args.get('gpu_id',0)),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4');common.pop('SMOLVLA_EXECUTION_PREFIX',None);common.pop('PYTHONHOME',None)
  modern=Path(args.get('lerobot_root') or os.environ['SMOLVLA_LEROBOT_ROOT']).resolve();motor_env=dict(common,PYTHONPATH=str(ROOT)+os.pathsep+str(modern/'src'))
  motor_python=args.get('smolvla_python') or os.environ['SMOLVLA_PYTHON'];qwen_python=args.get('qwen_python') or os.environ.get('QWEN_PYTHON')
  timeout=float(args.get('relational_worker_timeout_seconds',180))
  try:
   if self.mode!='native':
    provenance=self.cfg['feature_provenance']
    if provenance['extractor_sha256']!=digest(Path(__file__).with_name('qwen_features.py')):raise ValueError('Runtime extractor differs from trained cached feature source')
    if provenance['validator_sha256']!=digest(Path(__file__).with_name('qwen_contract_worker.py')):raise ValueError('Runtime image validator differs from trained cached feature source')
    if provenance.get('model_id')!='Qwen/Qwen3.5-9B' or provenance.get('model_revision')!='c202236235762e1c871ad0ccb60c8ee5ba337b9a' or provenance.get('current_only') is not True:raise ValueError('Dense foundation model/current-only pin mismatch')
    if not qwen_python:raise ValueError('Separate pinned Qwen interpreter required')
    model_path=args.get('qwen_model_path') or os.environ['QWEN_LOCAL_MODEL'];self.feature_root=self.cache/'qwen-features'
    command=[qwen_python,'-m','policy.relational_hierarchy.qwen_features','--worker','--model-path',str(model_path),'--input-root',str(self.inputs),'--output-root',str(self.feature_root),'--batch-size','1','--max-image-edge',str(provenance['max_image_edge']),'--max-requests','100000','--timeout-seconds','86400']
    self.qwen=RpcWorker(command,dict(common,PYTHONPATH=str(ROOT)),timeout)
   command=[motor_python,'-m','policy.relational_hierarchy.smol_worker','--checkpoint',str(self.checkpoint),'--device',self.device,'--lerobot-root',str(modern),'--native-worker',str(ROOT/'upstream/RoboSynChallenge/policy/smolvla/smolvla_worker.py'),'--execution-prefix',str(self.prefix)]
   if self.allow_wiring:command+=['--allow-wiring-pilot']
   self.motor=RpcWorker(command,motor_env,timeout)
   if self.motor.ready.get('queue_prefix')!=self.prefix or self.motor.ready.get('trained_chunk_size')!=50:raise ValueError('Actual motor prefix/chunk mismatch')
   print('[relational runtime] '+json.dumps({'event':'loaded','checkpoint':str(self.checkpoint),'mode':self.mode,'execution_prefix':self.prefix,'trained_chunk_size':50,'rpc_steps':10,'high_level_refresh':50,'rescale_gripper':self.rescale_gripper,'completed_training_step':self.proof['completed_step'],'wiring_pilot':self.proof.get('wiring_pilot',False)}),flush=True)
  except BaseException:self.close();raise
  atexit.register(self.close)
 def refresh(self,obs,instruction):
  arrays={'qpos':self.official._extract_joint(obs,'qpos')}
  for name in ('cam_high','cam_left_wrist','cam_right_wrist'):arrays[name]=self.official._extract_image(obs,name)
  key=f'episode-{self.episode_index}-action-{self.executed}';path=self.inputs/(key+'.npz');np.savez_compressed(path,**arrays)
  request={'request_id':key,'path':path.name,'task':instruction,'frame_ids':{name:key+'-'+name for name in ('cam_high','cam_left_wrist','cam_right_wrist')}}
  response=self.qwen.rpc(request)
  if response.get('feature_version')!=VERSION or response.get('shape')!=[3,4096] or response.get('request_id')!=key:raise ValueError('Dense current-feature response mismatch')
  feature=(self.feature_root/response['feature_path']).resolve(strict=True)
  if not feature.is_relative_to(self.feature_root.resolve()) or digest(feature)!=response['feature_sha256']:raise ValueError('Feature path/hash mismatch')
  self.feature={'feature_path':str(feature),'feature_sha256':response['feature_sha256']};self.last_refresh=self.executed
 def infer(self,obs,instruction):
  if self.qwen and self.executed-self.last_refresh>=50:self.refresh(obs,instruction)
  encoded=self.official.encode_obs(obs,image_keys=self.image_keys);path=self.cache/'motor-observation.npz';np.savez_compressed(path,**encoded,task=np.asarray(instruction))
  request={'cmd':'infer','obs_path':str(path),'n_action_steps':10,**(self.feature or {})};reply=self.motor.rpc(request)
  actions=np.asarray(reply['actions'],dtype=np.float32)
  if actions.shape!=(10,14) or not np.isfinite(actions).all():raise ValueError('Expected finite10x14 motor actions')
  return actions
 def reset(self):
  self.executed=0;self.last_refresh=-50;self.feature=None;request={'cmd':'reset'}
  if self.seed is not None:request['random_seed']=(self.seed+self.episode_index)%2**32
  reply=self.motor.rpc(request);print('[relational episode] '+json.dumps({'episode_index':self.episode_index,**reply}),flush=True);self.episode_index+=1
 def close(self):
  try:
   if self.qwen:self.qwen.close()
  finally:
   try:
    if self.motor:self.motor.close()
   finally:
    if getattr(self,'temp',None):self.temp.cleanup()

def get_model(args):return HierarchyModel(args)
def reset_model(model):return model.reset()
def close_model(model):return model.close()
def eval(env,model,obs):
 from policy.inference_timing import start_inference,finish_inference
 instruction=getattr(env,'_current_instruction',None)
 if not instruction:raise ValueError('Official instruction required')
 times=[];started=start_inference(model.device)
 try:
  actions=model.infer(obs,instruction);converted=[model.official.encode_action(a,env,rescale_gripper=model.rescale_gripper,gripper_indices=(6,13),gripper_source_range=(0.,1.)) for a in actions];finish_inference(started,times,model.device)
  final,info,truncated=obs,{},False
  for action in converted:
   final,reward,terminated,truncated,info=env.step(action);model.executed+=1
   if bool(model.official._to_numpy(terminated).reshape(-1)[0]) or bool(model.official._to_numpy(truncated).reshape(-1)[0]) or bool(model.official._to_numpy(env.get_wrapper_attr('is_task_success')()).reshape(-1)[0]):break
  return final,info,truncated,times
 except BaseException:model.close();raise
