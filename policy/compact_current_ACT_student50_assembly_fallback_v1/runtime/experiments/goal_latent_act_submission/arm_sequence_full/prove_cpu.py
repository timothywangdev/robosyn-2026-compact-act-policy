"""Real relocated science/FILE imports and all30 commands; no model forward.

Runs with the already qualified SIM interpreter. This explicitly proves CPU
environment reuse, not a fresh offline installation or native package rollout.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def process_identity(pid):
    try:
        raw=Path('/proc/'+str(pid)+'/stat').read_text(); fields=raw[raw.rindex(')')+2:].split()
    except FileNotFoundError:return None
    return dict(pid=pid,birth_ticks=int(fields[19]),group=int(fields[2]),session=int(fields[3]))


def cpu_wait(process,seconds):
    """Capture the actual CPU child wait; only signal its still-exact identity."""
    owner=process_identity(process.pid);signals=[]
    try:code=process.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        for name,wait in [('terminate',5),('kill',5)]:
            if process.poll()is not None:break
            if process_identity(process.pid)!=owner:raise RuntimeError('CPU child identity changed; refusing signal')
            getattr(process,name)();signals.append(name)
            try:process.wait(timeout=wait)
            except subprocess.TimeoutExpired:continue
        code=process.wait(timeout=5)
    return dict(owner=owner,original_child_exit_code=code,original_child_reaped=True,
        original_child_absent=process_identity(process.pid)is None,signals_sent=signals)


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for name in ('root','original-packet','existing-environment-root','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();root=Path(a.root).resolve(strict=True);old=Path(a.original_packet).resolve(strict=True)
    existing=Path(a.existing_environment_root).resolve(strict=True);output=Path(a.output).absolute()
    if output.exists():raise ValueError('Fresh CPU proof root required')
    output.mkdir(parents=True);started=time.monotonic();sys.path[:0]=[str(root)]
    from experiments.goal_latent_act_submission.arm_sequence_full.contracts import require,verify_stage,materialize,load_native,child,sha,reference,json_write,TASKS
    from experiments.goal_latent_act_submission.arm_sequence_full.run_task import command,metadata_check,environment
    import torch
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),'CPU hidden CUDA required')
    spec=json.loads((root/'submission.json').read_text());verify_stage(root,spec,full=True)
    metadata_check(existing,spec)
    from experiments.goal_latent_act_submission.arm_sequence_full.entry_cli import native_argv
    records=[];file_imports=[]
    for mode in ('parent','goal','goal_free'):
      for task in sorted(TASKS):
        target=output/(mode+'-'+task);target.mkdir();scratch=target/'scratch';scratch.mkdir()
        cfg=materialize(root,json.loads(child(root,spec['banks'][mode][task]['template']).read_text()),scratch)
        manifest=target/'deployment.json';json_write(manifest,cfg)
        # All30 relocation/qualify predicates execute. The unchanged runtime
        # manifest guard hashes all weights for one real representative permode;
        # the one whole-package SHA pass above already checks every ACT task.
        if task=='click_bell':load_native(root,manifest,sha(manifest))
        argv=command(root,cfg,manifest,task,'random',2026101803,1,100,target,python=existing/'.venvs/sim/bin/python')
        original=json.loads((old/(mode+'-'+task+'-command.json')).read_text())
        inner=argv[argv.index('--')+1:];prior=original['inner_argv']
        def semantic(values):
            return [v for i,v in enumerate(values) if i!=0 and not(i and values[i-1] in
                ('--config','--goal_latent_manifest','--goal_latent_manifest_sha256','--eval_result_dir'))]
        require(semantic(inner)==semantic(prior),'Original ALL30 native flag/order mismatch: '+mode+'/'+task)
        require(sha(Path(inner[inner.index('--config')+1]))==original['bindings']['policy_config']['sha256'],'Actual native YAML bytes differ')
        require(sha(root/'upstream/RoboSynChallenge/scripts/eval_policy.py')==original['bindings']['evaluator']['sha256']
                and sha(root/'upstream/EmbodiChain/embodichain/lab/sim/sim_manager.py')==original['bindings']['SDK']['sha256'],
                'Original evaluator/SDK bytes changed')
        row=dict(task=task,mode=mode,manifest=reference(manifest),argv=argv,
            original_command=reference(old/(mode+'-'+task+'-command.json')),native_option_order_and_values_equal=True,
            path_relocation_only=['evaluator','config','manifest','manifest_sha256','results'],
            original_policy_YAML_bitexact=True,original_evaluator_SDK_bitexact=True,
            unchanged_actor_sources=cfg['source_sha256'],compiled=False,preprocessing='native_every_action',models_constructed=0)
        records.append(row);json_write(target/'command-CPU-proof.json',row)
        entry_argv=native_argv(root,mode,task,'random',2026101803,1,100,target,False)
        require(entry_argv[1:5]==['-B',str(root/'experiments/goal_latent_act_submission/arm_sequence_full/run_task.py'),
            '--root',str(root)] and entry_argv[entry_argv.index('--mode')+1]==mode
            and entry_argv[entry_argv.index('--task')+1]==task,'Visible organizer route differs')
        row['visible_organizer_native_argv']=entry_argv
        json_write(target/'command-CPU-proof.json',row)
        if task=='click_bell':
            # Actual native bootstrap invocation uses the same cwd/environment
            # as execution, with CUDA hidden. No manual extra SDK path is added.
            env,cache=environment(root,spec);env['CUDA_VISIBLE_DEVICES']=''
            check=argv[:3]+['--check-only']+argv[3:]
            with(target/'FILE-stdout.log').open('xb')as stdout,(target/'FILE-stderr.log').open('xb')as stderr:
                process=subprocess.Popen(check,cwd=root/'upstream/RoboSynChallenge',env=env,
                    stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
                wait=cpu_wait(process,120);code=wait['original_child_exit_code']
            json_write(target/'original-CPU-child-wait.json',wait)
            require(code==0,'Original CPU FILE native bootstrap failed; preserved logs/no retry: '+mode)
            require(wait['original_child_absent'] and not wait['signals_sent'],'Original CPU child did not naturally reap')
            parsed=[]
            for line in (target/'FILE-stdout.log').read_text().splitlines():
                try:value=json.loads(line)
                except ValueError:continue
                if isinstance(value,dict):parsed.append(value)
            require(any(v.get('CUDA_initialized') is False and v.get('native_world_count')==0 for v in parsed),
                    'Actual SDK prelude/hook install no-CUDA/world0 proof missing')
            file_imports.append(dict(mode=mode,task=task,original_child_exit_code=code,original_child_reaped=True,actual_original_wait=wait,
                original_child_absent=True,actual_native_bootstrap=check,SDK_prelude_proofs=parsed,
                stdout=reference(target/'FILE-stdout.log'),stderr=reference(target/'FILE-stderr.log'),models_constructed=0))
            print(json.dumps(dict(mode=mode,status='actual_relocated_science_and_FILE_CPU_import0_no_model')),flush=True)
    from experiments.goal_latent_act_submission.arm_sequence_full import entry_bridge
    require(all(callable(getattr(entry_bridge,name,None))for name in ('get_model','eval','reset_model','close_model')),'Visible lifecycle export missing')
    import experiments.goal_latent_act_arm_runtime_full.runtime as actor
    require(Path(actor.__file__).resolve()==root/'experiments/goal_latent_act_arm_runtime_full/runtime.py','Runtime import root differs')
    require(not torch.cuda.is_initialized(),'CPU proof initialized CUDA')
    result=dict(schema='full63000_arm12_portable_CPU_relocation_and_ALL30_native_routing_v1',
        status='actual_relocated_science_FILE_imports0_and_ALL30_native_argument_parity_environment_reused_no_model',
        submission=reference(root/'submission.json'),existing_environment_root=str(existing),
        environment_validation='Exact prior SIM/Qwen/overlay metadata checked in already installed contained old environment; no fresh install claim',
        records=records,actual_FILE_imports=file_imports,visible_entry_lifecycle_exports=True,
        visible_entry_model_construction_exercised=False,model_inference_exercised=False,
        visible_organizer_ALL30_routing=True,
        original_actor_source_import=reference(Path(actor.__file__)),seconds=time.monotonic()-started,
        CUDA_initialized=False,models_constructed=0,GPU_queries=0,GPU_launches=0,external_uploads=0,
        primary_variant=None,offline_promotion_criterion_passed=False,cold_GPU_qualified=False,RTX5090_qualified=False)
    json_write(output/'receipt.json',result)
    print(json.dumps(dict(status=result['status'],records=len(records),FILE_imports=len(file_imports),seconds=result['seconds'])),flush=True)


if __name__=='__main__':main()
