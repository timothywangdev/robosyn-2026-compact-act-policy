"""Conditional fixed CPU student/control training; execution needs root evidence."""
import argparse,json,os,time
from pathlib import Path
from .contracts import require,ref,verify


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for key in ('cache','cache-receipt-sha256','admission','admission-sha256','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--check-only',action='store_true');a=p.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','This bounded trainer is CPU-only')
    admission_ref=ref(a.admission);require(admission_ref['sha256']==a.admission_sha256,'Exact root student admission')
    admission=json.loads(Path(a.admission).read_text())
    require(admission['schema']=='root_authorized_conditional_ACT_feature_student_CPU_training_v1'
        and admission['execution_authorized'] is True and admission['native_teacher_benefit_qualified'] is True,
        'Actual native teacher benefit and separate root CPU training admission required')
    verify(admission['matched_native_teacher_benefit']);require(admission['CPU_tree_cap_bytes']==3*(1<<30)
        and admission['threads']==4 and admission['wall_seconds']==600,'Fixed bounded CPU training budget')
    from .data import load,moments,batch
    import numpy as np,torch
    from .model import Student
    from .extract import resource_gate
    torch.set_num_threads(4)
    cache=ref(Path(a.cache)/'receipt.json');require(cache['sha256']==a.cache_receipt_sha256 and cache==admission['cache'],'Actual cache differs')
    plan,arrays,masks=load(a.cache,cache);statistics=moments(arrays,masks['optimizer'])
    for relative,digest in plan['source_sha256'].items():require(ref(Path(__file__).resolve().parents[2]/relative)['sha256']==digest,'Student source changed')
    if a.check_only:print(json.dumps(dict(status='CPU_actual_student_cache_and_root_plan_valid',GPU_queries=0,optimizer_updates=0)));return
    out=Path(a.output);require(not out.exists(),'Fresh fixed paired student output');out.mkdir(parents=True)
    start=time.monotonic();results={};indices=np.flatnonzero(masks['optimizer'])
    for mode,pooling,state_only in (('spatial_visual','spatial2x3',False),('spatial_qpos_control','spatial2x3',True),('global_visual','global',False)):
        torch.manual_seed(1701);model=Student(pooling,state_only);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        rng=np.random.default_rng(1701);updates=0
        for epoch in range(30):
            order=rng.permutation(indices)
            for first in range(0,len(order),128):
                resource_gate(start,600,3*(1<<30));values=batch(arrays,order[first:first+128],statistics)
                pred=model(values['visual'],values['qpos_features'],values['visual_global'])
                loss=sum((pred[key]-values[key]).square().mean() for key in ('projected_roles','predicted_goal_delta'))
                require(bool(torch.isfinite(loss)),'Finite teacher-distillation objective')
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();updates+=1
        model.eval();metrics={}
        with torch.inference_mode():
            for split in ('gate','diagnostic'):
                values=batch(arrays,np.flatnonzero(masks[split]),statistics);pred=model(values['visual'],values['qpos_features'],values['visual_global'])
                metrics[split]={key:float((pred[key]-values[key]).square().mean().sqrt()) for key in ('projected_roles','predicted_goal_delta')}
        target=out/(mode+'.pt');torch.save(dict(model=model.state_dict(),pooling=pooling,qpos_only=state_only,statistics=statistics,plan=ref(Path(a.cache)/'plan.json') if (Path(a.cache)/'plan.json').exists() else admission['plan'],epoch=30),target)
        results[mode]=dict(weights=ref(target),fixed_epoch=30,optimizer_updates=updates,metrics=metrics)
    require(not torch.cuda.is_initialized(),'CPU trainer initialized CUDA')
    (out/'receipt.json').write_text(json.dumps(dict(schema='actual_conditional_ACT_feature_student_CPU_training_v1',status='fixed30_three_CPU_candidates_complete_no_robot_promotion',admission=admission_ref,cache=cache,results=results,elapsed_seconds=time.monotonic()-start,GPU_queries=0,GPU_launches=0,robot_quality_promoted=False),indent=2)+'\n')


if __name__=='__main__':main()
