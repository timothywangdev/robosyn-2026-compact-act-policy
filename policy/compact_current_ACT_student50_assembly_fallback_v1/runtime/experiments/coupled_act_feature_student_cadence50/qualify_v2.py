"""CPU original waits for all10 typed executor50 slots and actual FILE probes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .contracts import load_manifest,validate_saved,require,ref,verify
from .supervisor import validate_plan
from experiments.goal_latent_act_campaign.common import write
from experiments.goal_latent_act_submission.cold_cpu import identity

def family(group,session):
    rows=[]
    for path in Path("/proc").iterdir():
        if path.name.isdecimal():
            try:row=identity(int(path.name))
            except (FileNotFoundError,ProcessLookupError):continue
            if row and (row["group"]==group or row["session"]==session):rows.append(row)
    return rows


def metadata(campaign,output):
    import torch
    body=json.loads(verify(campaign).read_text());require(len(body['plans'])==10 and body['execution_authorized'] is False,'Separate all10 CPU-only executor50 packet')
    rows=[];saved=None
    for row in body['plans']:
        cfg=load_manifest(row['deployment']['path'],row['deployment']['sha256']);plan=json.loads(verify(row['plan']).read_text());validate_plan(plan)
        require(cfg['prefix_executed']==50 and cfg['executor_variant']=='same_trained_current_ACT_student_execute50_v1'
                and cfg['original_prefix10_quality_qualification_reused'] is False,'Executor50 variant explicit')
        if saved is None:saved=torch.load(verify(cfg['assets']['student_weights']),map_location='cpu',weights_only=False)
        training=json.loads(verify(cfg['assets']['student_training_receipt']).read_text());validate_saved(saved,cfg,training)
        rows.append(dict(task=cfg['task'],mode=cfg['mode'],official_horizon=cfg['official_horizon'],generated=50,executed=50,motor=17,
            actual_trained_checkpoint_metadata_validated=True,native50_cadence_qualified=False))
    require(not torch.cuda.is_initialized(),'CPU metadata initialized CUDA')
    write(output,dict(schema='actual_current_ACT_student_executor50_all10_CPU_metadata_v1',rows=rows,campaign=campaign,
        actual_saved_checkpoint_loaded_only=True,ACT_Qwen_models_constructed=0,GPU_queries=0,worlds=0,CUDA_initialized=False))


def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--campaign',required=True);p.add_argument('--output',required=True);p.add_argument('--child',action='store_true');a=p.parse_args()
    campaign=ref(a.campaign)
    if a.child:metadata(campaign,Path(a.output));return
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU qualification must hide CUDA')
    out=Path(a.output).absolute();require(not out.exists(),'Fresh genuine CPU proof');out.mkdir(parents=True)
    cfg=json.loads(verify(campaign).read_text());source=Path(cfg['source_root']);env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONPATH=str(source))
    rows=[]
    def child(name,argv):
        start=time.monotonic()
        with (out/(name+'.stdout.log')).open('xb') as stdout,(out/(name+'.stderr.log')).open('xb') as stderr:
            process=subprocess.Popen(argv,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
            owner=identity(process.pid);code=process.wait(timeout=300)
        live=family(owner['group'],owner['session']);absent=identity(process.pid) is None
        body=dict(name=name,argv=argv,owner=owner,original_child_exit_code=code,child_reaped=True,exact_identity_absent=absent,
            live_owned_groups_sessions=live,signals_sent=[],seconds=time.monotonic()-start,
            stdout=ref(out/(name+'.stdout.log')),stderr=ref(out/(name+'.stderr.log')))
        rows.append(body);write(out/(name+'-original-wait.json'),body)
        require(code==0 and absent and not live,'Original CPU child failed: '+name)
    child('all10-metadata',[sys.executable,'-B','-m','experiments.coupled_act_feature_student_cadence50.qualify_v2','--campaign',campaign['path'],'--output',str(out/'all10.json'),'--child'])
    child('queue50-fixtures',[sys.executable,'-B','-m','experiments.coupled_act_feature_student_cadence50.test_cpu'])
    for task in ('click_bell','drawer_open_place'):
        row=next(r for r in cfg['plans'] if r['task']==task);cmd=json.loads(verify(row['command']).read_text())
        child(task+'-FILE',[sys.executable,'-B',cmd['bindings']['wrapper']['path'],'--command-record',row['command']['path'],
              '--command-sha256',row['command']['sha256'],'--output',str(out/(task+'-FILE')),'--check-only'])
    write(out/'receipt.json',dict(schema='current_ACT_student_executor50_genuine_CPU_preparation_v1',status='all10_actual_typed_metadata_queue50_two_FILE_originals_PASS_no_native_qualification',
        campaign=campaign,actual_original_children=rows,source_sha256=cfg['source_sha256'],GPU_queries=0,GPU_launches=0,native_worlds=0,
        ACT_Qwen_models_constructed=0,optimizer_updates=0,generated50_executed50=True,source_mathematical_change=False,
        original_prefix10_robot_qualification_reused=False,native_executor50_qualified=False,robot_quality_promoted=False))
    print(json.dumps(dict(receipt=ref(out/'receipt.json'))))

if __name__=='__main__':main()
