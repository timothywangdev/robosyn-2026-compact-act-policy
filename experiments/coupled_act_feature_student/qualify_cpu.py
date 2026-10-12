"""CPU source/metadata/fixture qualification, without extraction or training."""
import argparse,ast,hashlib,json,os,struct,sys
from pathlib import Path
from .contracts import ref,verify,require,PLAN,CAMERAS,partition


def qualify(plan,expected,output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Pure CPU qualification hides CUDA')
    plan_ref=ref(plan);require(plan_ref['sha256']==expected,'Exact prospective plan');p=json.loads(Path(plan).read_text())
    require(p['schema']==PLAN and p['execution_authorized'] is False and len(p['rows'])==1000,'Only prospective1000 packet')
    root=Path(__file__).resolve().parents[2]
    for relative,digest in p['source_sha256'].items():require(ref(root/relative)['sha256']==digest,'New source changed')
    require(p['source_rows']['sha256']=='d597e5e95c055a11b2968455a932b47318e1be23ee583479c78b1fc3ca706031','Original70k source rows differs')
    counts={};seen=set();headers=[]
    for row in p['rows']:
        require(row['partition']==partition(row['episode']) and row['split']=='fit' and row['frame']>=0
            and row['current_timestamp_seconds']==row['frame']/25 and row['camera_frame_indices']=={c:row['frame'] for c in CAMERAS},'Frame/time/episode identity differs')
        key=(row['task'],row['episode']);require(key not in seen,'One selected current anchor per distinctFIT episode');seen.add(key)
        counts[row['task'],row['partition']]=counts.get((row['task'],row['partition']),0)+1
        require(ref(row['file']['path'])==row['file'],'Original accepted cache bytes changed')
        require(Path(row['parquet']['path']).is_file() and all(Path(v['path']).is_file() for v in row['videos'].values()),'Source path reconstruction differs')
    require(all(counts[t,s]==n for t in p['tasks'] for s,n in (('optimizer',80),('gate',10),('diagnostic',10))),'All10 balanced80/10/10')
    for task,binding in p['act'].items():
        with verify(binding['files']['model.safetensors']).open('rb') as stream:
            n=struct.unpack('<Q',stream.read(8))[0];require(n<1<<20,'Bounded safetensors header');h=json.loads(stream.read(n))
        backbone={k:v for k,v in h.items() if k.startswith('model.backbone.')};require(len(backbone)==100,'Exact ResNet18 frozenBN100 tensors')
        require(all(h['normalize_inputs.buffer_observation_images_'+c+'.'+name]['shape']==[3,1,1]
            for c in CAMERAS for name in ('mean','std')),'Exact native image buffers')
        headers.append(dict(task=task,backbone_tensor_count=100,header_only_read=True,weights_loaded=False))
    # Mechanically compare the copied actor tail, not only its outputs.
    original=ast.parse((root/'experiments/coupled_goal_act_all14/model.py').read_text())
    fn=next(n for n in ast.walk(original) if isinstance(n,ast.FunctionDef) and n.name=='forward')
    start=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='parent' for t in n.targets))
    class Substitute(ast.NodeTransformer):
        def visit_Name(self,n):
            return ast.copy_location(ast.Name(id='actor',ctx=n.ctx),n) if n.id=='self' else n
        def visit_Assign(self,n):
            if len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='role' and any(isinstance(x,ast.Attribute) and x.attr=='roles' for x in ast.walk(n)):return ast.parse('role = projected_roles.detach()').body[0]
            return self.generic_visit(n)
    expected_tail=[Substitute().visit(n) for n in fn.body[start:]]
    new=ast.parse((root/'experiments/coupled_act_feature_student/projected.py').read_text());new_fn=next(n for n in new.body if isinstance(n,ast.FunctionDef));new_start=next(i for i,n in enumerate(new_fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='parent' for t in n.targets))
    require([ast.dump(n) for n in expected_tail]==[ast.dump(n) for n in new_fn.body[new_start:]],'Scientific tail differs beyond projected role substitution')
    import contextlib,io,unittest,torch
    from .test_cpu import Tests
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests));require(result.wasSuccessful(),'Actual synthetic CPU fixture failed')
    require(not torch.cuda.is_initialized(),'CPU qualification initialized CUDA')
    output=Path(output);require(not output.exists(),'Fresh CPU receipt');output.mkdir(parents=True)
    report=dict(schema='conditional_ACT_feature_student_plan_and_actual_synthetic_CPU_qualification_v1',status='metadata_frame_units_header_split_and_nine_CPU_tests_PASS',
        plan=plan_ref,source_sha256=p['source_sha256'],all1000_original_cache_byte_refs_verified=True,source_video_and_parquet_paths_exist=True,
        original_full_pipeline25Hz_frame_and_pixel_hash_provenance_bound=True,new_video_parquet_and_pixel_decode_reverification_pending=True,
        ACT_header_inventory=headers,unchanged_coupled_tail_AST_except_explicit_projectedrole=True,
        tests=result.testsRun,test_log=stream.getvalue(),actual_new_small_synthetic_CPU_models_constructed=True,
        pretrained_model_loads=0,RGB_frames_decoded=0,new_teacher_targets_materialized=0,student_optimizer_updates=0,
        GPU_queries=0,GPU_launches=0,worlds_constructed=0,CUDA_initialized=False,
        CPU_vs_GPU_feature_parity_qualified=False,source_qualified_backbone_constructor_load_PENDING=True,
        future_extraction_requires_root_admission=True,native_teacher_5percent_offline_criterion_still_failed=True,
        student_robot_quality_promoted=False,novelty_claim=False)
    target=output/'receipt.json';target.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(ref(target)))


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for key in ('plan','plan-sha256','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();qualify(a.plan,a.plan_sha256,a.output)


if __name__=='__main__':main()
