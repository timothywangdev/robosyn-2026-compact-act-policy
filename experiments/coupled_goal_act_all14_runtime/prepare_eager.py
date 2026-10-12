"""Additive eager candidate from preserved compiled template, no science edits."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
from .prepare import prepare,ROOT,NATIVE
from .contracts import require
from .runtime_eager import EAGER,load_eager_manifest
from .supervisor import validate_plan
from experiments.goal_latent_act_campaign.common import ref,write


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    for key in ('output','run-root','training-root','role-admission','state-admission','wait','drain','pair','gpu-uuid',
        'compiler-attempt','compiler-attempt-original-wait','compiler-attempt-terminal-drain'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args();output=Path(args.output).absolute();run_root=Path(args.run_root).absolute()
    require(not output.exists() and not run_root.exists(),'Fresh eager preparation and future original results required');output.mkdir(parents=True)
    base=prepare(output/'preserved-compiled-template',output/'never-executed-template-results',args.training_root,args.role_admission,
        args.state_admission,args.wait,args.drain,args.pair,args.gpu_uuid)
    source=Path(base['source_root']);policy=source/'policy/coupled_goal_act_all14_eager_v1';policy.mkdir(parents=True)
    shutil.copy2(ROOT/'experiments/coupled_goal_act_all14_runtime/policy_shim_eager.py',policy/'__init__.py')
    config=policy/'deploy_policy.yml';config.write_text('policy_name: coupled_goal_act_all14_eager_v1\nstate_obs_path: robot/qpos\nstrict_action_dim: true\nact_step: 10\n')
    protocol=json.loads(Path(base['protocol']['path']).read_text())
    protocol.update(correction_runtime_mode=EAGER,correction_partial_compile=False,
        compilation='actual measured compiler attempt preserved; original eager ACT and eager small GRU actor; no compiled speedup claim')
    write(output/'protocol.json',protocol);rows=[]
    for original in base['plans']:
        task,mode=original['task'],original['mode'];cfg=json.loads(Path(original['deployment']['path']).read_text())
        cfg.update(correction_runtime_mode=EAGER,correction_partial_compile=False,require_compile_speedup=False)
        if mode!='parent':cfg['assets'].update(compiler_attempt=ref(args.compiler_attempt),
            compiler_attempt_original_wait=ref(args.compiler_attempt_original_wait),compiler_attempt_terminal_drain=ref(args.compiler_attempt_terminal_drain))
        deployment=output/f'{mode}-{task}-deployment.json';write(deployment,cfg);load_eager_manifest(deployment,ref(deployment)['sha256'])
        command=json.loads(Path(original['command']['path']).read_text());target=run_root/mode/task/'native'
        inner=command['inner_argv']
        for flag,value in (('--config',str(config)),('--policy_name','coupled_goal_act_all14_eager_v1'),
            ('--coupled_all14_manifest',str(deployment)),('--coupled_all14_manifest_sha256',ref(deployment)['sha256']),
            ('--eval_result_dir',str(target/'evaluator-results'))):inner[inner.index(flag)+1]=value
        command['bindings'].update(deployment=ref(deployment),wrapper=ref(source/'experiments/coupled_goal_act_all14_runtime/native_eager.py'),
            policy=ref(policy/'__init__.py'),policy_config=ref(config),protocol=ref(output/'protocol.json'))
        command_path=output/f'{mode}-{task}-command.json';write(command_path,command)
        plan=json.loads(Path(original['plan']['path']).read_text());plan['output_root']=str(target.parent)
        job=plan['jobs'][0];job.update(output_root=str(target),command=ref(command_path),argv=[str(NATIVE/'.venvs/sim/bin/python'),'-B',
            command['bindings']['wrapper']['path'],'--command-record',str(command_path),'--command-sha256',ref(command_path)['sha256'],'--output',str(target)])
        validate_plan(plan);plan_path=output/f'{mode}-{task}-plan.json';write(plan_path,plan)
        row=deepcopy(original);row.update(command=ref(command_path),deployment=ref(deployment),plan=ref(plan_path),output=str(target),original_argv=job['argv']);rows.append(row)
    campaign=deepcopy(base);campaign.update(plans=rows,protocol=ref(output/'protocol.json'),correction_runtime_mode=EAGER,
        compiler_attempt_preserved=ref(args.compiler_attempt),compiled_at_runtime=False,template=ref(output/'preserved-compiled-template/campaign.json'))
    write(output/'campaign.json',campaign)
    print(json.dumps(dict(status='CPU_exact_all30_explicit_eager_candidate_pending_independent_review_and_root_admission',campaign=ref(output/'campaign.json'),GPU_launches=0)))


if __name__=='__main__':main()
