"""Typed native command adapter around the unchanged proven ownership ledger."""
import argparse
import fcntl
import json
import os
from pathlib import Path
from .contracts import load_manifest,module,require,verify_ref
from .native import COMMAND,GENERATOR
from experiments.goal_latent_act_campaign.common import ref,write

SCHEMA='coupled_ACT_student_local3090_native32GiB_supervisor_v1'
UUID='GPU-8e45cdd0-bbd6-9b4f-de12-e7125a11bed9'
GIB=1<<30


def validate_command(command):
    require(command['schema']==COMMAND and command['GPU_uuid']==UUID,'Exact typed native command/local UUID required')
    for binding in command['bindings'].values():verify_ref(binding)
    cfg=load_manifest(command['bindings']['deployment']['path'],command['bindings']['deployment']['sha256'])
    require((cfg['task'],cfg['mode'])==(command['task'],command['mode']),'Bound task/control differs')
    inner=command['inner_argv']
    for key,value in (('--seed',GENERATOR),('--act_step',10),('--max_episodes',1),('--max_steps',command['official_horizon']),
        ('--eval_expert_check',True),('--eval_max_seed_attempts_per_episodes',100),('--eval_video_log',True),
        ('--coupled_student_manifest',command['bindings']['deployment']['path']),
        ('--coupled_student_manifest_sha256',command['bindings']['deployment']['sha256'])):
        require(inner.count(key)==1 and inner[inner.index(key)+1]==str(value),'Native protocol changed: '+key)
    gym=json.loads(verify_ref(command['bindings']['gym']).read_text())
    require(gym['max_episode_steps']==command['official_horizon'],'Official full horizon changed')
    return cfg


def validate_plan(plan):
    require(plan['schema']==SCHEMA and plan['gpu_uuid']==UUID and plan['admission_memory_floor_gib']==32,'Exact native32GiB/local3090 admission required')
    require(plan['whole_GPU_cap_bytes']==24*GIB-256*(1<<20) and plan['owned_tree_RSS_cap_bytes']==32*GIB,'Prospective local24GiB/256MiB margin and32GiB RSS caps required')
    output=Path(plan['output_root'])
    require(output.is_absolute() and output==output.resolve() and not output.exists(),'Fresh supervisor output required')
    require(len(plan['jobs'])==1,'Each original task uses one native command; no automatic retry')
    job=plan['jobs'][0];target=Path(job['output_root']);argv=job['argv']
    require(target==output/'native' and not target.exists() and not target.is_symlink(),'Fresh private native child output required')
    command=json.loads(verify_ref(job['command']).read_text());cfg=validate_command(command)
    expected=[str(Path(command['native_root'])/'.venvs/sim/bin/python'),'-B',command['bindings']['wrapper']['path'],
        '--command-record',job['command']['path'],'--command-sha256',job['command']['sha256'],'--output',str(target)]
    require(argv==expected and Path(job['cwd']).is_absolute() and Path(job['cwd']).is_dir(),'Exact same-PID bootstrap command required')
    require(job['max_owned_gpu_processes']==1 and job['wall_seconds']==1800,'Bound role/native owner process count and full task wall cap required')
    require(job['name']==cfg['mode']+'-'+cfg['task'],'Exact native slot identity required')
    verify_ref(plan['borrowed_guard']);require(plan['borrowed_guard']['path'].endswith('/experiments/coupled_goal_act_all14/supervisor.py'),'Exact frozen proven guard required')
    return plan


def owned_tree_RSS(guard,owner):
    rows={}
    for path in Path('/proc').iterdir():
        if path.name.isdecimal():
            try:rows[int(path.name)]=guard.identity(int(path.name))
            except (FileNotFoundError,ProcessLookupError):pass
    selected={owner['pid']};changed=True
    while changed:
        previous=len(selected)
        selected.update(pid for pid,row in rows.items() if row['parent_pid'] in selected and row['birth_ticks']>=owner['birth_ticks'])
        changed=len(selected)!=previous
    total=0
    for pid in selected:
        try:
            before=guard.identity(pid)
            for line in Path('/proc',str(pid),'status').read_text().splitlines():
                if line.startswith('VmRSS:'):total+=int(line.split()[1])*1024;break
            after=guard.identity(pid)
            require(guard.same_identity(before,after),'Owned RSS identity changed during inspection')
        except (FileNotFoundError,ProcessLookupError):pass
    return total


def run_plan(plan):
    validate_plan(plan)
    guard=module('_coupled_exact_original_native_ownership_guard',verify_ref(plan['borrowed_guard']))
    # New instance only: the immutable guard file and its original module are
    # untouched. All execution/identity/drain functions retain original bytes.
    guard.validate_plan=validate_plan;guard.SCHEMA=SCHEMA
    owner=guard.identity(os.getpid());peaks=dict(whole_GPU_bytes=0,owned_tree_RSS_bytes=0)
    def gpu_reader(uuid):
        value=guard.gpu_snapshot(uuid);used=value['total_bytes']-value['free_bytes']
        peaks['whole_GPU_bytes']=max(peaks['whole_GPU_bytes'],used)
        require(used<=plan['whole_GPU_cap_bytes'],'Whole physicalGPU memory margin breached')
        return value
    def memory_reader():
        value=guard.memory_snapshot();rss=owned_tree_RSS(guard,owner)
        peaks['owned_tree_RSS_bytes']=max(peaks['owned_tree_RSS_bytes'],rss)
        require(rss<=plan['owned_tree_RSS_cap_bytes'],'Aggregate owned original tree RSS cap breached')
        return value
    with Path('/tmp/robosyn-local3090-experiment.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=guard.run_plan(plan,memory_reader=memory_reader,gpu_reader=gpu_reader)
    write(Path(plan['output_root'])/'native-adapter-receipt.json',dict(schema=SCHEMA,adapter=ref(__file__),borrowed_guard=plan['borrowed_guard'],
        original_guard_receipt=ref(Path(plan['output_root'])/'receipt.json'),original_guard_exit_code=result['exit_code'],
        unchanged_original_ownership_ledger=True,measured_peaks=peaks,robot_quality_promoted=False))
    return result


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False);parser.add_argument('--plan',required=True)
    parser.add_argument('--root-admission');parser.add_argument('--root-admission-sha256')
    modes=parser.add_mutually_exclusive_group(required=True);modes.add_argument('--validate-only',action='store_true');modes.add_argument('--execute',action='store_true')
    args=parser.parse_args();plan=validate_plan(json.loads(Path(args.plan).read_text()))
    if args.validate_only:print(json.dumps(dict(status='CPU_exact_typed_native_plan_valid',GPU_queries=0)));return
    require(args.root_admission and args.root_admission_sha256,'Separate actual root GPU slot admission required')
    admission_ref=ref(args.root_admission)
    require(admission_ref['sha256']==args.root_admission_sha256,'Exact root slot admission bytes required')
    admission=json.loads(verify_ref(admission_ref).read_text())
    require(admission['schema']=='root_authorized_coupled_ACT_student_local_native_slot_v1' and admission['execution_authorized'] is True
        and admission['plan']==ref(args.plan) and admission['runtime_supervisor']==ref(__file__), 'Root must authorize this exact immutable native slot/guard')
    require(admission['predecessor_genuine_wait_consumed'] is True and admission['predecessor_processes_drained'] is True
        and admission['independent_CPU_review_qualified'] is True and len(admission['predecessor_proofs'])>0,
        'Independent CPU review and actual preceding wait/drain required')
    for binding in admission['predecessor_proofs']:verify_ref(binding)
    verify_ref(admission['independent_CPU_review'])
    raise SystemExit(run_plan(plan)['exit_code'])


if __name__=='__main__':main()
