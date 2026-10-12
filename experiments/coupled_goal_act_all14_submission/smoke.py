"""Genuine original CPU FILE/CLI children and immutable transport receipts."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from .prepare import ref,sha,write,safe


def identity(pid):
    p=Path('/proc')/str(pid)
    try:
        raw=(p/'stat').read_text();tail=raw[raw.rfind(')')+2:].split()
        return dict(pid=pid,birth=tail[19],group=int(tail[2]),session=int(tail[3]))
    except (FileNotFoundError,ProcessLookupError):return None


def family(group,session):
    result=[]
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            row=identity(int(path.name))
            if row and (row['group']==group or row['session']==session):result.append(row)
    return result


def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--stage',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.stage).resolve(strict=True);output=Path(a.output).absolute();assert not output.exists();output.mkdir(parents=True)
    spec=json.loads((root/'preparation.json').read_text());assert not spec['embargo']['execution_authorized']
    for name,row in spec['source_files'].items():assert sha(root/safe(name))==row['sha256']
    rows=[]
    for task in spec['tasks']:
        template=json.loads((root/task['template']).read_text());original=json.loads(Path(task['original_deployment']['path']).read_text())
        changed={'schema','parent_schema','source_root','scratch_root','act_root','qwen_root','qwen_python','qwen_python_path','assets','act_files','qwen_files'}
        assert {k:v for k,v in template.items() if k not in changed}=={k:v for k,v in original.items() if k not in changed}
        for group in ('assets','act_files','qwen_files'):
            assert set(template[group])==set(original[group])
            for key,value in template[group].items():
                assert (value['bytes'],value['sha256'])==(original[group][key]['bytes'],original[group][key]['sha256'])
                safe(value['path'])
        assert template['schema']=='coupled_all14_portable_preparation_template_v1'
        assert template['native_chunk_generated']==50 and template['prefix_executed']==10
        rows.append(dict(task=task['task'],mode=task['mode'],official_horizon=task['official_horizon'],transport_only_delta=True))
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
    for key in ('PYTHONPATH','PYTHONHOME','PYTHONSTARTUP'):env.pop(key,None)
    children=[]
    def run(name,argv,expected):
        out=output/(name+'.stdout.log');err=output/(name+'.stderr.log')
        with out.open('xb') as stdout,err.open('xb') as stderr:
            child=subprocess.Popen(argv,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
            owner=identity(child.pid);code=child.wait(timeout=90)
        assert code==expected,(name,code,err.read_text()[-4000:])
        live=family(owner['group'],owner['session']);assert identity(child.pid) is None and not live
        children.append(dict(name=name,argv=argv,owner=owner,original_child_exit_code=code,child_reaped=True,
            exact_identity_absent=True,owned_groups_sessions_absent=True,signals_sent=[],stdout=ref(out),stderr=ref(err)))
    py=str(Path(sys.executable).absolute());probe=root/'experiments/coupled_goal_act_all14_submission/file_probe.py'
    run('relocated-official-FILE',[py,'-I','-B',str(probe),'--root',str(root)],0)
    cli=root/'experiments/coupled_goal_act_all14_submission/entry_cli.py'
    for mode in ('parent','goal','goal_free'):
        run('entry-'+mode,[py,'-I','-B',str(cli),'--root',str(root),'--task','click_bell','--mode',mode,'--seed','1204258515','--episodes','1','--check-only'],0)
    run('unqualified-inference-rejected',[py,'-I','-B',str(cli),'--root',str(root),'--task','click_bell','--mode','goal','--seed','1204258515','--episodes','1'],1)
    write(output/'receipt.json',dict(schema='coupled_all14_submission_original_CPU_FILE_transport_smoke_v1',
        status='relocated_source_import_and_all30_transport_metadata_only_PASS_inference_blocked',
        preparation=ref(root/'preparation.json'),all30_transport_checks=rows,actual_original_children=children,
        models_constructed=0,GPU_queries=0,GPU_launches=0,native_world_count=0,
        original_raw_provenance_rewritten=False,portable_model_load_qualified=False,
        RTX5090_qualified=False,submission_ready=False))
    print(json.dumps(dict(receipt=ref(output/'receipt.json'),original_children=len(children),metadata_cases=len(rows),GPU_launches=0)))


if __name__=='__main__':main()
