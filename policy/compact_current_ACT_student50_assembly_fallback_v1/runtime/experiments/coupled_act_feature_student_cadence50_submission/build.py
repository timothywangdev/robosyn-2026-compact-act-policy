"""Bounded immutable portable fixture: hardlink weights, copy small proofs/source."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
from .contracts import PACKAGE
from experiments.coupled_goal_act_all14_submission.prepare import ref,sha,safe,write

ROOT=Path(__file__).resolve().parents[2]


def build(campaign,donor,output):
    campaign=ref(campaign);c=json.loads(Path(campaign['path']).read_text());donor=Path(donor).resolve(strict=True)
    d=json.loads((donor/'submission.json').read_text());root=Path(output).absolute();assert not root.exists();root.mkdir(parents=True)
    bindings=[];files={};sources={};small=0;linked=0;keys=set()
    def put(name,row,copy=False):
        nonlocal small,linked
        name=safe(name);src=Path(row['path']);dst=root/name
        assert src.is_file() and not src.is_symlink() and src.stat().st_size==row['bytes'],src
        dst.parent.mkdir(parents=True,exist_ok=True)
        if name not in files:
            if copy:shutil.copy2(src,dst);small+=row['bytes'];assert sha(dst)==row['sha256']
            else:os.link(src,dst);linked+=row['bytes']
            files[name]={k:row[k] for k in ('bytes','sha256')}
        else:assert files[name]=={k:row[k] for k in ('bytes','sha256')}
        original={k:row[k] for k in ('path','bytes','sha256')};key=tuple(original.values())
        if key not in keys:bindings.append(dict(original=original,installed=name));keys.add(key)
        return name
    # Reuse the already pinned native toolchain/static inventory, never Qwen
    # model shards, learned teacher planner heads, or another Python environment.
    for name,row in d['files'].items():
        if name.startswith(('upstream/','dependencies/','assets/embodichain/','cache/torch/','licenses/')) or name=='requirements/sim.lock':
            put(name,dict(path=str(donor/name),**row))
    for name,digest in c['source_sha256'].items():
        row=ref(Path(c['source_root'])/safe(name));assert row['sha256']==digest;put(name,row,True);sources[name]=files[name]
    # Portable dependency code is copied literally from its qualified donor.
    for name,row in d['source_files'].items():
        if name.startswith(('experiments/coupled_goal_act_all14_submission/', 'experiments/goal_latent_act_submission/','experiments/simulator_cleanup_repair_v1/')):
            if name not in files:put(name,dict(path=str(donor/name),**row),True);sources[name]=files[name]
    for p in Path(__file__).parent.glob('*.py'):
        name=str(p.relative_to(ROOT));put(name,ref(p),True);sources[name]=files[name]
    api_proofs={}
    for label,p in {'tutorial':ROOT/'upstream/RoboSynChallenge/docs/tutorials/policy/your_own_policy.md',
                    'evaluator':Path('/tmp/ra5/upstream/RoboSynChallenge/scripts/eval_policy.py'),
                    'native_ACT_adapter':Path('/tmp/ra5/upstream/RoboSynChallenge/policy/act/deploy_policy.py')}.items():
        row=ref(p);name='provenance/official-policy-API/'+label+Path(p).suffix;put(name,row,True);api_proofs[label]=dict(original=row,installed=name)
    tasks=[]
    for slot in c['plans']:
        raw=slot['deployment'];cfg=json.loads(Path(raw['path']).read_text());task,mode=cfg['task'],cfg['mode']
        assert (cfg['native_chunk_generated'],cfg['prefix_executed'],cfg['motor_seed'])==(50,50,17)
        assert cfg['Qwen_models_loaded']==0 and cfg['ACT_models_loaded']==1
        put('provenance/native-deployments/'+mode+'-'+task+'.json',raw,True)
        for name,row in cfg['act_files'].items():put('models/act/'+task+'/'+name,row)
        for key,row in cfg['assets'].items():
            if key in ('teacher_weights','teacher_contract'):dest='models/frozen-coupled-role/'+Path(row['path']).name
            elif key=='student_weights':dest='models/current-student/'+Path(row['path']).name
            elif key=='ACT_model_source':dest='upstream/lerobot-sim/src/lerobot/policies/act/modeling_act.py'
            else:dest='provenance/by-sha/'+row['sha256']+'/'+Path(row['path']).name
            put(dest,row,copy=Path(row['path']).suffix=='.json')
        if mode=='student':
            training=json.loads(Path(cfg['assets']['student_training_receipt']['path']).read_text())
            row=training['protocol'];put('provenance/by-sha/'+row['sha256']+'/'+Path(row['path']).name,row,True)
        template='templates/'+mode+'-'+task+'.json'
        write(root/template,dict(schema='compact_student50_byte_preserving_template_v1',task=task,mode=mode,
            original_deployment=raw,act_root='models/act/'+task))
        files[template]={k:v for k,v in ref(root/template).items() if k!='path'}
        tasks.append(dict(task=task,mode=mode,official_horizon=cfg['official_horizon'],template=template,template_sha256=files[template]['sha256']))
    # Preserve selected training/prospective selection/parity/compile provenance.
    for key in ('candidate_selection','student_GPU_parity','student_GPU_parity_wait'):
        row=c[key];put('provenance/by-sha/'+row['sha256']+'/'+Path(row['path']).name,row,True)
    compile_root=ROOT/'outputs/architecture-reassessment/coupled-act-feature-student-inference-compile-actual-v4'
    for name in ('actual-child/receipt.json','root-original-wait-and-drain.json','original-owned/root-owner-receipt.json'):
        row=ref(compile_root/name);put('provenance/by-sha/'+row['sha256']+'/'+Path(row['path']).name,row,True)
    policy=root/'policy/compact_current_ACT_student50_v1';policy.mkdir(parents=True)
    (policy/'deploy_policy.py').write_text('from experiments.coupled_act_feature_student_cadence50_submission.policy import get_model, eval, reset_model, close_model\n')
    (policy/'__init__.py').write_text('from .deploy_policy import get_model, eval, reset_model, close_model\n')
    (policy/'deploy_policy.yml').write_text('policy_name: compact_current_ACT_student50_v1\ntask_name: null\nsetting: random\nmodel_name: compact_student_unpromoted\nseed: 0\nmax_episodes: 20\nmax_steps: 1000\nheadless: true\npytorch_device: cuda\nfilter_dataset_saving: true\neval_video_log: true\nact_step: 50\nstate_obs_path: robot/qpos\nstrict_action_dim: true\n')
    (policy/'eval.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\nHERE="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"\nROOT="$(CDPATH= cd -- "$HERE/../.." && pwd)"\nexec "${PYTHON_BIN:-$ROOT/.venvs/sim/bin/python}" -B "$ROOT/experiments/coupled_act_feature_student_cadence50_submission/entry_cli.py" --root "$ROOT" "$@"\n')
    (policy/'eval.sh').chmod(0o755)
    for p in policy.iterdir():files[str(p.relative_to(root))]={k:v for k,v in ref(p).items() if k!='path'}
    (root/'runtime').mkdir()
    spec=dict(schema=PACKAGE,status='CPU_prepared_no_native_or_5090_or_quality_promotion',tasks=tasks,source_files=sources,
        files=files,reference_bindings=bindings,environments={'sim':d['environments']['sim']},original_campaign=campaign,
        generated=50,executed=50,mode='student_only_all10',selected_candidate=c['candidate']['student_candidate'],selected_weights=c['candidate']['student_weights'],
        robot_quality_promoted=False,submission_ready=False,installed_native_qualified=False,RTX5090_qualified=False,
        raw_original_JSON_rewritten=False,checkpoint_bytes_rewritten=False,Qwen_model_payload=False,Qwen_inference=False,RPC_inference=False,
        second_CNN_forward=False,future_labels_at_inference=False,actual_small_bytes_copied=small,immutable_hardlinked_logical_bytes=linked,
        retained_nested_report_only_absolute_refs='Immutable provenance identifiers; only frozen inference lookup closure requires installed aliases.',
        cold_installation_qualified=False)
    spec['official_policy_API_sources']=api_proofs
    spec['model_parameter_scope']=dict(student_head=2392434,frozen_coupled_role_checkpoint_total=180528,
        motor_tail_used=49424,frozen_ACT_backbone_and_native50_model_also_required=True,
        Qwen_parameters_at_inference=0,student_head_is_not_total_deployment_parameter_count=True)
    spec['organizer_action_contract']=dict(policy_exports=['get_model','eval','reset_model','close_model'],
        eval_returns=['observation','info','truncated','inference_times'],select_action_shape=[1,14],
        generated_chunk=50,internally_enqueued_prefix=50,model_act_step=50,
        organizer_global_act_step_not_used_by_official_native_ACT_eval=True,
        organizer_evaluator_modifications_required=False,CPU_action_loop_fixture_only=True,
        actual_installed_organizer_native_model_run_qualified=False)
    assert len(tasks)==10 and small<1<<30 and shutil.disk_usage(root).free>8<<30
    write(root/'submission.json',spec);print(json.dumps(dict(submission=ref(root/'submission.json'),sources=len(sources),files=len(files),small_bytes_copied=small,hardlinked_bytes=linked)))

if __name__=='__main__':
    p=argparse.ArgumentParser(allow_abbrev=False)
    for key in ('campaign','donor','output'):p.add_argument('--'+key,required=True)
    build(**vars(p.parse_args()))
