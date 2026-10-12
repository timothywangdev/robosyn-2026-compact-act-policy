"""Build exact hash-pinned local CPU argv only AFTER final corpus qualification."""
import argparse
import json
from pathlib import Path
from experiments.goal_latent_act_campaign.common import require,ref,write
from . import data

def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for k in ('snapshot','campaign','corpus-admission','protocol','source-seal','geometry','urdf','python','training-output','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args()
    data.verify_terminal(a.campaign,data.sha(a.campaign),a.corpus_admission,data.sha(a.corpus_admission))
    snapshot=json.loads(Path(a.snapshot).read_text());require((snapshot['fit'],snapshot['monitor'])==(63000,7000),'Final actual70k source snapshot required')
    seal=json.loads(Path(a.source_seal).read_text());require(seal['source_pins']==data.pins(),'Reviewed full scientific closure changed')
    require(not Path(a.training_output).exists() and not Path(a.output).exists(),'Fresh training and request outputs required')
    require(data.sha(a.geometry)==data.GEOMETRY_SHA and data.sha(a.urdf)==data.URDF_SHA,'Exact root-frame physical FK required')
    argv=[str(Path(a.python).absolute()),'-B','-m','experiments.goal_latent_act_command_goal_training_full.run']
    for name in ('snapshot','campaign','corpus-admission','protocol','source-seal'):
        value=getattr(a,name.replace('-','_'));argv.extend(['--'+name,str(Path(value).resolve()),'--'+name+'-sha256',data.sha(value)])
    for name in ('geometry','urdf','python'):argv.extend(['--'+name,str(Path(getattr(a,name)).absolute())])
    argv.extend(['--output',str(Path(a.training_output).absolute())])
    write(a.output,dict(schema='goal-latent-act-commanded50-full63000-CPU-exact-request-v1',status='qualified_corpus_ready_for_root_CPU_launch',
        argv=argv,campaign=ref(a.campaign),corpus_admission=ref(a.corpus_admission),snapshot=ref(a.snapshot),protocol=ref(a.protocol),source_seal=ref(a.source_seal),
        source_FIT=63000,source_MONITOR=7000,optimizer_FIT=50400,diagnostic_FIT=6300,gate_FIT=6300,training_started=False,GPU_queries=0,GPU_launches=0))
    print(str(Path(a.output).absolute()))

if __name__=='__main__':main()
