"""Exact queue50 replay and source-only cadence boundary fixtures."""
from collections import deque
import json
import time
import types
import torch
from .contracts import load_manifest
from .runtime import CoreRuntime,Runtime,OriginalRuntime,PrecisionObservedRuntime

def main():
    assert not torch.cuda.is_initialized()
    assert sorted(load_manifest.cadence_only_AST_changes)==['declared_generation50_execution50_motor17']
    assert set(CoreRuntime.__init__.cadence_only_AST_changes)=={'model_act_step50','logged_native_executed50'}
    assert set(CoreRuntime.select_action.cadence_only_AST_changes)=={'queue_first50','logged_native_executed50','replan_tag_prefix50'}
    assert CoreRuntime.current is OriginalRuntime.current and CoreRuntime.qualify_compiler is OriginalRuntime.qualify_compiler
    assert Runtime.select_action is PrecisionObservedRuntime.select_action
    native=torch.arange(700,dtype=torch.float32).reshape(1,50,14)/1000;count=[0]
    class Base:
        def normalize_inputs(self,b):return b
        def model(self,b):count[0]+=1;return native,None
        def unnormalize_outputs(self,b):return b
    actor=object.__new__(CoreRuntime);actor.torch=torch;actor.base=Base();actor.base.config=types.SimpleNamespace(image_features=[])
    actor.sync=lambda:None;actor.cfg={'episode_wall_seconds':60,'mode':'student'};actor.episode=0;actor.executed=0;actor.started=time.monotonic()
    actor.capture=None;actor.student=None;actor._action_queue=deque()
    actual=torch.cat([actor.select_action({}) for _ in range(50)])
    assert torch.equal(actual,native[0]) and count[0]==1 and actor.executed==50 and not actor._action_queue
    actor.select_action({});assert count[0]==2 and actor.executed==51 and len(actor._action_queue)==49
    # reset is identical and must discard all remaining49 stale commands.
    actor.base.reset=lambda:None;actor.reset();assert not actor._action_queue and actor.executed==0 and actor.episode==1
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(schema='current_ACT_student_executor50_CPU_fixtures_v1',cases=['contract_declares50_50_17','only_two_init_cadence_changes','only_three_action_cadence_changes','current_features_and_compiler_inherited','precision_mixin_retained','exact50_order_one_generation','51st_action_replans','reset_discards_remaining49'],models_constructed=0,GPU_queries=0,CUDA_initialized=False,worlds=0)))

if __name__=='__main__':main()
