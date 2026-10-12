"""Validation outside pure tensor student; fixed conservative compile attempt."""
import time
import torch
from torch import nn
from .contracts import require


def validate(model,visual,qpos_features,visual_global=None):
    require(visual.ndim==5 and visual.shape[1:]==(3,512,2,3) and qpos_features.shape==(len(visual),14),'Current spatial512x2x3/normalizedqpos14 required')
    require(visual.dtype==qpos_features.dtype==torch.float32 and bool(torch.isfinite(visual).all()) and bool(torch.isfinite(qpos_features).all()),'Finite FP32 student input')
    if model.pooling=='global':
        require(visual_global is not None and visual_global.shape==(len(visual),3,512) and visual_global.dtype==torch.float32 and bool(torch.isfinite(visual_global).all()),'Exact raw-map global average required')


class PureStudent(nn.Module):
    def __init__(self,student):
        super().__init__();self.student=student
    def forward(self,visual,qpos_features,visual_global=None):
        features=visual_global if self.student.pooling=='global' else visual
        if self.student.qpos_only:features=features*0
        hidden=self.student.hidden(torch.cat((features.flatten(1),qpos_features),-1))
        return dict(projected_roles=torch.tanh(self.student.roles(hidden)),predicted_goal_delta=self.student.goal(hidden))


def validated(model,call,inputs):
    validate(model,**inputs)
    return call(**inputs)


def qualify(model,inputs,*,backend='inductor',benchmark=True,postprocess=None):
    require(not model.training and inputs and all(x['visual'].shape[0]==1 for x in inputs),'Actual trained eval model/B1 current inputs only')
    require(backend in ('inductor','aot_eager') and type(benchmark) is bool,'Fixed compile backend qualification')
    if benchmark:require(backend=='inductor' and all(x['visual'].device.type=='cuda' for x in inputs) and callable(postprocess),'GPU student+actual frozen tail full-call wall latency only')
    pure=PureStudent(model);torch._dynamo.reset();start=time.perf_counter()
    options=dict(backend=backend,fullgraph=True,dynamic=False)
    if backend=='inductor':options['mode']='reduce-overhead'
    compiled=torch.compile(pure,**options);parity=[]
    with torch.inference_mode():
        for index,x in enumerate(inputs):
            eager=validated(model,pure,x);candidate=validated(model,compiled,x)
            diff={k:float((eager[k]-candidate[k]).abs().max()) for k in eager}
            require(all(torch.allclose(eager[k],candidate[k],atol=2e-5,rtol=2e-5) for k in eager),'Compiled actual student numeric parity failed')
            tail_diff=None
            if postprocess is not None:
                eager_tail=postprocess(eager,index);compiled_tail=postprocess(candidate,index)
                require(set(eager_tail)==set(compiled_tail),'Actual tail output fields differ')
                tail_diff={}
                for key,value in eager_tail.items():
                    other=compiled_tail[key];require(value.shape==other.shape and value.dtype==other.dtype,'Tail dtype/shape differs')
                    if value.dtype==torch.bool:require(torch.equal(value,other),'Compiled Student changes actual hard gate/Boolean output');tail_diff[key]=0
                    else:require(torch.allclose(value,other,atol=2e-6,rtol=1e-6),'Actual frozen tail numeric parity failed');tail_diff[key]=float((value-other).abs().max())
            parity.append(dict(index=index,maximum_absolute_difference=diff,actual_tail_maximum_difference=tail_diff))
        if benchmark:torch.cuda.synchronize()
        cold=time.perf_counter()-start;timing={}
        if benchmark:
            for label,call in (('eager',pure),('compiled',compiled)):
                def whole(index):
                    value=validated(model,call,inputs[index])
                    return postprocess(value,index) if postprocess is not None else value
                for i in range(5):whole(i%len(inputs))
                torch.cuda.synchronize();samples=[]
                for i in range(20):
                    torch.cuda.synchronize();start=time.perf_counter();whole(i%len(inputs));torch.cuda.synchronize();samples.append(time.perf_counter()-start)
                ordered=sorted(samples);timing[label]=dict(samples_seconds=samples,median_seconds=(ordered[9]+ordered[10])/2,mean_seconds=sum(samples)/len(samples))
    speedup=timing['eager']['median_seconds']/timing['compiled']['median_seconds'] if benchmark else None
    accelerated=bool(benchmark and speedup>=1.05)
    report=dict(schema='ACT_feature_student_validated_pureTensor_compile_qualification_v1',backend=backend,mode='reduce-overhead' if backend=='inductor' else None,
        fullgraph=True,dynamic=False,validation_outside_compiled_graph=True,numerical=parity,cold_seconds=cold,timing=timing,
        latency_includes_identical_input_validation_and_full_MLP=True,latency_includes_actual_frozen_tail=postprocess is not None,
        actual_tail_numeric_and_Boolean_parity_required=postprocess is not None,median_speedup=speedup,acceleration_qualified=accelerated,
        selected_inference_mode='compiled' if accelerated else 'eager',CPU_AOT_is_not_GPU_speed_evidence=not benchmark)
    return (compiled if accelerated else pure),report
