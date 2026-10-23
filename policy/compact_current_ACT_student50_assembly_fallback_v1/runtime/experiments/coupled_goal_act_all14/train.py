"""One explicitly admitted matched branch; fixed optimizer FIT and separate FIT gate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
import torch
from safetensors.torch import save_file
from experiments.goal_latent_act_arm_sequence_full.fk import CommandFK
from .common import MODEL,require,ref,write,sha,validate_protocol
from .data import load,tensors,batch
from .model import CoupledGoalACT
from .objective import loss,gate_labels
from .evaluate import evaluate
from .contracts import load_checkpoint


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for k in ('prepared','prepared-sha256','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--state-action-only',action='store_true');p.add_argument('--execute',action='store_true')
    a=p.parse_args();require(a.execute,'Explicit root-admitted GPU execute required')
    started=time.monotonic();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    report=dict(schema=MODEL,status='original_GPU_preflight',pid=os.getpid(),state_action_only=a.state_action_only,
                optimizer_updates=0,gate_updates=0,selection='fixed_final_epoch_no_MONITOR_selection',robot_launches=0)
    def expired(*_):raise TimeoutError('Fixed all14 branch 1200-second inclusive cap')
    signal.signal(signal.SIGALRM,expired);signal.alarm(1200)
    try:
        torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(1701);np.random.seed(1701)
        r,arrays,features,tasks=load(a.prepared,a.prepared_sha256)
        protocol=validate_protocol(json.loads(Path(r['protocol']['path']).read_text()))
        fit=np.flatnonzero(arrays['optimizer_fit']);gate=np.flatnonzero(arrays['gate_fit']);diag=np.flatnonzero(arrays['diagnostic_fit']);monitor=np.flatnonzero(~arrays['fit'])
        require((len(fit),len(gate),len(diag),len(monitor))==(50400,6300,6300,7000),'Fixed independent partitions required')
        require(torch.cuda.is_available() and torch.__version__=='2.7.1+cu128','Pinned CUDA Torch2.7.1 required')
        require(torch.cuda.get_device_name()=='NVIDIA GeForce RTX 3090','Prospective measured local3090 branch only')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.cuda.set_per_process_memory_fraction(8*(1<<30)/torch.cuda.get_device_properties(0).total_memory)
        device=torch.device('cuda');d=tensors(arrays,features,device);del arrays,features
        stats={k:torch.tensor(r['statistics'][k],dtype=torch.float32,device=device) for k in ('qpos_mean','qpos_scale','base_mean','base_scale','current_eef_mean','current_eef_scale','eef_scale')}
        fk=CommandFK(r['geometry'],r['urdf']).to(device);model=CoupledGoalACT(**protocol['model']).to(device)
        x=batch(d,fit[:32],stats,tasks)
        with torch.no_grad():initial=model(**x,state_action_only=a.state_action_only)
        require(torch.equal(initial['native_actions'],x['native_base']),'Zero-init gated all14 must preserve literal released ACT')
        report.update(status='actual_GPU_training_running',source_pins=r['source_pins'],prepared=ref(Path(a.prepared)/'receipt.json'),
            protocol=r['protocol'],statistics=r['statistics'],tasks=tasks,parameters=sum(v.numel() for v in model.parameters()),
            counts=protocol['partitions'],source_counts=protocol['source_counts'],device_name=torch.cuda.get_device_name(),
            CUDA_initialized=torch.cuda.is_initialized(),precision='FP32; TF32 disabled',zero_initial_gated_native_parent_equal=True)
        write(out/'receipt.json',report)
        parameters=[v for name,v in model.named_parameters() if not name.startswith('reliability.')]
        optimizer=torch.optim.AdamW(parameters,lr=3e-4,weight_decay=.001);rng=np.random.default_rng(1701)
        for epoch in range(30):
            model.train();order=rng.permutation(fit);total=0.
            for start in range(0,len(order),128):
                ids=order[start:start+128];i=torch.as_tensor(ids,device=device);optimizer.zero_grad(set_to_none=True)
                result=model(**batch(d,ids,stats,tasks),state_action_only=a.state_action_only,gated=False)
                value=loss(result,d['physical_actions'][i],d['valid'][i],fk)
                require(bool(torch.isfinite(value)),'Finite physical all14 objective required');value.backward()
                require(all(v.grad is None or bool(torch.isfinite(v.grad).all()) for v in parameters),'Finite arm/opening gradients required')
                torch.nn.utils.clip_grad_norm_(parameters,5.);optimizer.step();report['optimizer_updates']+=1;total+=float(value.detach())*len(ids)
                if report['optimizer_updates']==100:report['first100_updates_elapsed_seconds']=time.monotonic()-started
            report.setdefault('history',[]).append(dict(epoch=epoch+1,FIT_loss=total/len(fit),FIT_order_sha256=hashlib.sha256(order.tobytes()).hexdigest()))
            report.update(epochs_completed=epoch+1,elapsed_seconds=time.monotonic()-started);write(out/'receipt.json',report)
            print(json.dumps(dict(epoch=epoch+1,updates=report['optimizer_updates'],loss=total/len(fit),elapsed_seconds=report['elapsed_seconds'])),flush=True)
        for parameter in parameters:parameter.requires_grad_(False)
        optimizer=torch.optim.AdamW(model.reliability.parameters(),lr=1e-3,weight_decay=.001);positive=torch.zeros(2,device=device)
        for epoch in range(20):
            order=rng.permutation(gate);total=0.
            for start in range(0,len(order),128):
                ids=order[start:start+128];i=torch.as_tensor(ids,device=device);optimizer.zero_grad(set_to_none=True)
                result=model(**batch(d,ids,stats,tasks),state_action_only=a.state_action_only,gated=False)
                with torch.no_grad():label=gate_labels(result,d['physical_actions'][i],d['valid'][i],fk)
                if epoch==0:positive+=label.sum(0)
                value=torch.nn.functional.binary_cross_entropy_with_logits(result['reliability_logits'],label)
                require(bool(torch.isfinite(value)),'Finite independent FIT gate loss required');value.backward();optimizer.step()
                report['gate_updates']+=1;total+=float(value.detach())*len(ids)
            report.setdefault('gate_history',[]).append(dict(epoch=epoch+1,FIT_BCE=total/len(gate),FIT_order_sha256=hashlib.sha256(order.tobytes()).hexdigest()))
        require((report['optimizer_updates'],report['gate_updates'])==(11820,1000),'Exact fixed matched updates required')
        weights=out/'coupled-all14.safetensors';save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(weights))
        with weights.open('rb') as f:os.fsync(f.fileno())
        c=dict(schema=MODEL,model_config=model.config,weights_sha256=sha(weights),state_action_only=a.state_action_only,
            control_zero_channels=['roles','predicted_goal_delta'] if a.state_action_only else [],statistics=r['statistics'],tasks=tasks,
            prepared=report['prepared'],highlevel_contract=r['highlevel_contract'],highlevel_weights=r['highlevel_weights'],
            geometry=r['geometry'],urdf=r['urdf'],source_pins=r['source_pins'],protocol=protocol,coordinate_frame='robot_root_meters',
            command_goal_offset=50,native_chunk_generated=50,prefix_executed=10,native_grippers_bitexact=False,rejected_hand_all7_bitexact=True,
            hand_atomic_all7=True,crosshand_mixed_acceptance_possible=True,selected_epoch=30,gate_epochs=20,optimizer_updates=11820,
            gate_updates=1000,batch_size=128,seed=1701,gate_threshold=.5,source_counts=protocol['source_counts'],partitions=protocol['partitions'],
            MONITOR_selection=False,use_future_labels_at_inference=False,
            actor_inputs=['current_roles','current_qpos','current_root_FK','predicted_command50_goal','native_ACT50','explicit_source_grip_codec'])
        write(out/'checkpoint.json',c);loaded,_=load_checkpoint(out,device=device)
        with torch.no_grad():before=model(**x,state_action_only=a.state_action_only);after=loaded(**x,state_action_only=a.state_action_only)
        require(all(torch.equal(before[k],after[k]) for k in before),'Saved/reloaded coupled predictions changed')
        report.update(status='completed_fixed30_coupled_all14_20_independent_FIT_gate',checkpoint=c,reload_predictions_bitexact=True,
            positive_gate_FIT_hand_rows=positive.long().cpu().tolist(),diagnostic_FIT=evaluate(loaded,d,diag,stats,fk,tasks,a.state_action_only),
            final_gate_FIT=evaluate(loaded,d,gate,stats,fk,tasks,a.state_action_only),
            final_monitor=evaluate(loaded,d,monitor,stats,fk,tasks,a.state_action_only),
            role_shuffle_monitor=evaluate(loaded,d,monitor,stats,fk,tasks,a.state_action_only,shuffle=True) if not a.state_action_only else None,
            peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:
        report.update(status='failed_original_preserved_no_retry',failure=dict(type=type(error).__name__,message=str(error)));raise
    finally:
        signal.alarm(0);report['elapsed_seconds']=time.monotonic()-started;write(out/'receipt.json',report)


if __name__=='__main__':main()
