"""Explicit compiled affine/head paths; GRU remains its unchanged native kernel.

This is partial compilation, never a claim that Dynamo supports fullgraph GRU.
Actual whole-model calls, graph/kernel counts, output parity and warm latency are
required. Failure is surfaced; callers must not silently deploy the eager copy.
"""
import copy
import time
import torch
from .common import require


def compile_and_compare(model,inputs,*,state_action_only,backend='inductor',warm_calls=5):
    require(backend in ('inductor','aot_eager') and warm_calls>=3,'Explicit compiled backend/bounded warm calls required')
    from torch._dynamo.backends.registry import lookup_backend
    implementation=lookup_backend(backend);compiled=copy.deepcopy(model).eval().requires_grad_(False)
    counts={name:dict(graphs=0,calls=0) for name in ('roles','initial','output','reliability')}
    def backend_for(name):
        def run(graph,example_inputs):
            counts[name]['graphs']+=1;function=implementation(graph,example_inputs)
            def actual(*args):counts[name]['calls']+=1;return function(*args)
            return actual
        return run
    for name in counts:
        setattr(compiled,name,torch.compile(getattr(compiled,name),backend=backend_for(name),fullgraph=True,dynamic=False))
    def sync():
        if inputs['native_base'].is_cuda:torch.cuda.synchronize()
    with torch.inference_mode():
        sync();start=time.perf_counter();eager=model(**inputs,state_action_only=state_action_only);sync();eager_first=time.perf_counter()-start
        start=time.perf_counter();result=compiled(**inputs,state_action_only=state_action_only);sync();cold=time.perf_counter()-start
        errors={}
        for key in eager:
            if eager[key].dtype==torch.bool:require(torch.equal(eager[key],result[key]),'Compiled acceptance/limits changed: '+key)
            else:
                difference=(eager[key]-result[key]).double()
                errors[key]=dict(max_abs=float(difference.abs().max()),RMSE=float(difference.square().mean().sqrt()))
                # FP32 bound fixed prospectively before any trained-weight run.
                require(errors[key]['max_abs']<=1e-5 and errors[key]['RMSE']<=2e-6,'Compiled FP32 output parity failed: '+key)
        times={}
        for name,actor in (('eager',model),('compiled',compiled)):
            values=[]
            for _ in range(warm_calls):
                sync();start=time.perf_counter();actor(**inputs,state_action_only=state_action_only);sync();values.append(time.perf_counter()-start)
            times[name]=values
    require(all(v['graphs']>0 and v['calls']>=warm_calls+1 for v in counts.values()),'Compiled wrappers did not execute actual whole-model path')
    report=dict(backend=backend,scope='roles projection/context initial/14-output/reliability only; unchanged native GRU kernel',
        full_GRU_compiled=False,graph_and_kernel_calls=counts,parity=errors,
        eager_first_seconds=eager_first,compiled_cold_seconds=cold,warm_seconds=times,
        actual_warm_speedup=sum(times['eager'])/sum(times['compiled']),
        acceleration_qualified=backend=='inductor' and sum(times['compiled'])<sum(times['eager']),
        robot_promotion=False)
    return compiled,report
