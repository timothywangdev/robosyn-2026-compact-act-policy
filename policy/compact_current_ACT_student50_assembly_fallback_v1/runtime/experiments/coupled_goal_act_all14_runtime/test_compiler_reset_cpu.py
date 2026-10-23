"""Two real actor phases execute the frozen compiler with a bounded reset."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
from .contracts import require
from experiments.goal_latent_act_campaign.common import ref,write


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False);parser.add_argument('--output',required=True);args=parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU compiler reset regression must hide CUDA before Torch import')
    import torch
    from .compiler_assay_v2 import reset_compiler_branch
    from .test_cpu import fixture,context,native
    from experiments.coupled_goal_act_all14.inference import correct_native50
    from experiments.coupled_goal_act_all14.compiler import compile_and_compare
    torch.set_num_threads(2);branches=[]
    for mode,control in (('goal',False),('goal_free',True)):
        reset=reset_compiler_branch();actor=fixture(mode);captured={}
        hook=actor.correction.register_forward_pre_hook(lambda _module,_args,kwargs:captured.update(kwargs),with_kwargs=True)
        current={key:torch.from_numpy(value[None]) for key,value in context().items()}
        with torch.inference_mode():correct_native50(actor.correction,native(0),current,actor.stats,
            task=actor.cfg['task'],codec=actor.cfg['gripper_codec'],state_action_only=control)
        hook.remove();require(captured.pop('state_action_only')==control,'Actual compiler input mode differs')
        compiled,report=compile_and_compare(actor.correction,captured,state_action_only=control,backend='aot_eager',warm_calls=3)
        require(all(value['graphs']>0 and value['calls']>=4 for value in report['graph_and_kernel_calls'].values()),
            'Reset CPU phase never executed the real four compiled paths')
        branches.append(dict(mode=mode,reset=reset,actual_frozen_compiler=report,synthetic_small_actor_only=True))
        del compiled,actor
    require(not torch.cuda.is_initialized(),'CPU reset fixture initialized CUDA')
    out=Path(args.output).absolute();require(not out.exists(),'Fresh reset evidence required');out.mkdir(parents=True)
    result=dict(schema='coupled_all14_two_original_actor_Dynamo_reset_CPU_v1',status='both_CPU_actor_phases_actual_fourpath_compile_PASS',
        branches=branches,recompile_limit_raised=False,checkpoint_or_frozen_compiler_source_changed=False,
        CUDA_initialized=False,GPU_queries=0,GPU_launches=0,ACT_models_constructed=0,Qwen_models_constructed=0,
        native_worlds_constructed=0,trained_GPU_equivalence_qualified=False)
    write(out/'receipt.json',result);print(json.dumps(ref(out/'receipt.json')))


if __name__=='__main__':main()
