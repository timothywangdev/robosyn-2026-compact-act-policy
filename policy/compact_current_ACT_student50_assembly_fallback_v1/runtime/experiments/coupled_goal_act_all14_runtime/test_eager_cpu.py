"""Same real small-actor/queue equations with no compiler invocation or fallback."""
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
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Eager CPU fixture must hide CUDA before Torch import')
    import torch
    from .test_cpu import fixture,context,native,batch
    from .runtime_eager import Runtime,EAGER
    from experiments.coupled_goal_act_all14.inference import correct_native50
    torch.set_num_threads(2);cases=[];original_compile=torch.compile
    def forbidden(*_args,**_kwargs):raise AssertionError('Explicit eager runtime unexpectedly invoked compiler')
    torch.compile=forbidden
    try:
        for mode in ('goal','goal_free'):
            for task in ('item_assembly','drawer_open_place'):
                actor=fixture(mode,task);actor.__class__=Runtime;actor.compiler_report=None
                actor.cfg.update(correction_runtime_mode=EAGER,assets={'compiler_attempt':{'explicit_CPU_fixture':True}})
                actor.reset();expected=None
                with contextlib.redirect_stdout(io.StringIO()):
                    for step in range(31):
                        if step%10==0:
                            current={key:torch.from_numpy(value[None]) for key,value in context(step).items()}
                            with torch.inference_mode():expected,_=correct_native50(actor.correction,native(step),current,actor.stats,
                                task=task,codec=actor.cfg['gripper_codec'],state_action_only=mode=='goal_free')
                        actual=actor.select_action(batch(step))
                        require(torch.equal(actual,expected[:,step%10]),'Eager wrapper changed scientific actor/queue commands')
                require(actor.compiler_report['compiled'] is False and actor.goal_calls==[0.,10.,20.,30.],
                    'Eager metadata or current-feature cadence changed')
                cases.append(dict(mode=mode,task=task,actual_commands=31,every_command_bitexact=True,compiler_invocations=0,
                    current_feature_steps=actor.goal_calls,rejected_whole_hand_native_unchanged=True))
    finally:torch.compile=original_compile
    require(not torch.cuda.is_initialized(),'CPU eager fixture initialized CUDA')
    output=Path(args.output).absolute();require(not output.exists(),'Fresh eager CPU proof required');output.mkdir(parents=True)
    report=dict(schema='coupled_all14_explicit_eager_actual_actor_queue_CPU_v1',status='four_real_small_actor_queue_cases_bitexact_no_compiler_PASS',
        cases=cases,synthetic_small_actor_weights=True,ACT_models_constructed=0,Qwen_models_constructed=0,native_worlds_constructed=0,
        GPU_queries=0,GPU_launches=0,CUDA_initialized=False,compiler_mode_promoted=False,robot_quality_promoted=False)
    write(output/'receipt.json',report);print(json.dumps(ref(output/'receipt.json')))


if __name__=='__main__':main()
