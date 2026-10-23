"""Own genuine CPU subprocesses for installed tests/installer/official entries."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from experiments.coupled_goal_act_all14_submission.prepare import ref,write
from experiments.coupled_goal_act_all14_submission.smoke import identity,family

def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.root).resolve(strict=True);out=Path(a.output).absolute();assert not out.exists();out.mkdir(parents=True)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    for key in ('PYTHONPATH','PYTHONHOME','PYTHONSTARTUP'):env.pop(key,None)
    py=str(Path(sys.executable).absolute());rows=[]
    def run(name,argv):
        with (out/(name+'.stdout.log')).open('xb') as stdout,(out/(name+'.stderr.log')).open('xb') as stderr:
            child=subprocess.Popen(argv,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
            owner=identity(child.pid);code=child.wait(timeout=180)
        absent=identity(child.pid) is None;live=family(owner['group'],owner['session'])
        row=dict(name=name,argv=argv,owner=owner,original_child_exit_code=code,child_reaped=True,exact_identity_absent=absent,
            live_owned_groups_sessions=live,signals_sent=[],stdout=ref(out/(name+'.stdout.log')),stderr=ref(out/(name+'.stderr.log')))
        rows.append(row);write(out/(name+'-original-wait.json'),row)
        assert code==0 and absent and not live,(name,code,(out/(name+'.stderr.log')).read_text()[-4000:])
    code='import sys,runpy;sys.path.insert(0,sys.argv[1]);runpy.run_module("experiments.coupled_act_feature_student_cadence50_submission.test_cpu",run_name="__main__")'
    run('installed-ten-tests',[py,'-I','-B','-c',code,str(root)])
    run('offline-install-recipe',[py,'-I','-B',str(root/'experiments/coupled_act_feature_student_cadence50_submission/install.py'),'--root',str(root),'--check-only'])
    for mode,task in (('student','drawer_open_place'),('student','click_bell')):
        run('entry-'+task,[py,'-I','-B',str(root/'experiments/coupled_act_feature_student_cadence50_submission/entry_cli.py'),
            '--root',str(root),'--task',task,'--mode',mode,'--seed','1204258515','--episodes','1','--output',str(out/('entry-'+task)),'--check-only'])
    write(out/'receipt.json',dict(schema='compact_student_portable_actual_original_CPU_smoke_v1',
        status='installed_ten_tests_offline_recipe_and_two_official_commands_PASS_no_model',package=ref(root/'submission.json'),
        actual_original_children=rows,models_constructed=0,GPU_queries=0,GPU_launches=0,native_world_count=0,
        installation_performed=False,installed_native_qualified=False,RTX5090_qualified=False,submission_ready=False))
    print(json.dumps(dict(receipt=ref(out/'receipt.json'),original_children=len(rows))))

if __name__=='__main__':main()
