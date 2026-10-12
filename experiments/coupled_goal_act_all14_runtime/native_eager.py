"""CPU FILE preflight or same-PID native bootstrap; no queued/automatic retries."""
import argparse
import json
import os
from pathlib import Path
import sys
if __package__ in (None,''):
    own = Path(__file__).resolve().parent
    sys.path[:] = [value for value in sys.path if Path(value or os.getcwd()).resolve()!=own]
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    __package__ = 'experiments.coupled_goal_act_all14_runtime'
from .contracts import load_manifest, module, require, verify_ref
from .runtime_eager import load_eager_manifest as load_manifest
from experiments.goal_latent_act_campaign.common import ref,write
from experiments.goal_latent_act_sequence_runtime_full.protocol import BOOTSTRAP_SHA,HOOK_SHA,SDK_SHA,EVALUATOR_SHA

COMMAND = 'coupled_all14_native50_prefix10_original_command_v1'
GENERATOR = 2026102401


def execute(command_ref,output,check_only):
    command = json.loads(verify_ref(command_ref).read_text())
    require(command['schema']==COMMAND and command['bindings']['wrapper']==ref(__file__), 'Exact new standalone command/wrapper required')
    for binding in command['bindings'].values(): verify_ref(binding)
    for name,expected in (('bootstrap',BOOTSTRAP_SHA),('hook',HOOK_SHA),('SDK',SDK_SHA),('evaluator',EVALUATOR_SHA)):
        require(command['bindings'][name]['sha256']==expected, 'Qualified native execution source differs: '+name)
    cfg = load_manifest(command['bindings']['deployment']['path'],command['bindings']['deployment']['sha256'])
    require((cfg['mode'],cfg['task'])==(command['mode'],command['task']), 'Exact new native task/mode required')
    inner = command['inner_argv']
    require(inner[0]==command['bindings']['evaluator']['path'], 'Pinned original evaluator required')
    for key,value in (('--seed',GENERATOR),('--act_step',10),('--max_episodes',1),('--max_steps',command['official_horizon']),
        ('--eval_expert_check',True),('--eval_max_seed_attempts_per_episodes',100),('--eval_video_log',True),
        ('--coupled_all14_manifest',command['bindings']['deployment']['path']),
        ('--coupled_all14_manifest_sha256',command['bindings']['deployment']['sha256'])):
        require(inner.count(key)==1 and inner[inner.index(key)+1]==str(value), 'Frozen native flag changed: '+key)
    gym = json.loads(verify_ref(command['bindings']['gym']).read_text())
    require(command['official_horizon']==gym['max_episode_steps'], 'Full task-specific official horizon required')
    out = Path(output).absolute(); require(not out.exists(), 'Fresh native/CPU output required'); out.mkdir(parents=True)
    if check_only:
        import queue
        import sysconfig
        import torch
        require(os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'CPU FILE check must hide CUDA')
        require(Path(queue.__file__).resolve()==Path(sysconfig.get_paths()['stdlib'])/'queue.py', 'FILE invocation shadowed stdlib queue')
        bridge = module('_coupled_actual_CPU_bridge',verify_ref(command['bindings']['policy']))
        require(all(callable(getattr(bridge,name,None)) for name in ('get_model','eval','reset_model','close_model')), 'Native factory lifecycle import failed')
        require(not torch.cuda.is_initialized(), 'CPU FILE preflight initialized CUDA')
        write(out/'actual-FILE-preflight.json',dict(schema='coupled_all14_actual_FILE_CPU_native_preflight_v1',
            command=command_ref,CUDA_initialized=False,models_constructed=0,worlds_constructed=0,GPU_queries=0))
    env = json.loads(verify_ref(command['bindings']['native_environment']).read_text())
    for key in ('VPP2_GPU_SUPERVISOR_OWNER','CUDA_VISIBLE_DEVICES'):
        if key in os.environ: env[key]=os.environ[key]
    require(env['EMBODICHAIN_SIM_EXIT_PROCESS']=='0', 'SDK must return to the unchanged official metric writer')
    require(check_only or env.get('CUDA_VISIBLE_DEVICES')==command['GPU_uuid'], 'Exact admitted physical device required')
    native = Path(command['native_root'])
    installer = module('_coupled_static_cache_installer',verify_ref(command['bindings']['cache_installer']))
    cache = module('_coupled_private_short_cache',verify_ref(command['bindings']['cache_implementation']))
    env,cache_proof = cache.prepare(native,installer.stage_spec(native,full=False),env,scratch_parent='/tmp')
    argv = [str(native/'.venvs/sim/bin/python'),'-B',str(verify_ref(command['bindings']['bootstrap']))]
    if check_only: argv.append('--check-only')
    argv += ['--',*inner]
    write(out/'actual-native-command.json',dict(schema=COMMAND,command=command_ref,argv=argv,inner_argv=inner,
        same_PID_exec=True,actual_private_cache=cache_proof,mode=cfg['mode'],generator=GENERATOR,motor_seed=17,
        native_generated=50,native_executed=10,ACT_compiled=False,correction_partial_compile=False,
        correction_runtime_mode=cfg['correction_runtime_mode'],
        preprocessing='native_every_action',GPU_launches=0 if check_only else None))
    os.chdir(command['cwd']); os.execvpe(argv[0],argv,env)


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--command-record',required=True); p.add_argument('--command-sha256',required=True)
    p.add_argument('--output',required=True); p.add_argument('--check-only',action='store_true')
    args = p.parse_args(); binding=ref(args.command_record)
    require(binding['sha256']==args.command_sha256, 'Exact frozen native command required')
    execute(binding,args.output,args.check_only)


if __name__=='__main__': main()
