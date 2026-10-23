"""Bounded metadata-only1000-anchor plan; reads no RGB/model tensors/targets."""
import argparse
import json
from pathlib import Path
from .contracts import PLAN,CAMERAS,ROLE_QUERIES,choose_episodes,anchor_index,partition,records,ref,verify,require

ROOT=Path(__file__).resolve().parents[2]


def prepare(output):
    output=Path(output).absolute();require(not output.exists(),'Fresh metadata plan output')
    campaign=ROOT/'outputs/architecture-reassessment/coupled-goal-act-all14-runtime-preparation-v1/CPU-exact30-eager-v2/campaign.json'
    p=json.loads(campaign.read_text());cfgs={row['task']:json.loads(Path(row['deployment']['path']).read_text())
        for row in p['plans'] if row['mode']=='goal'}
    tasks=list(cfgs);require(len(tasks)==10,'Exact all10 ACT domains')
    contract=json.loads(Path(cfgs[tasks[0]]['assets']['coupled_contract']['path']).read_text())
    all14=ROOT/'outputs/architecture-reassessment/coupled-goal-act-all14-preparation-v1/CPU-prepared-v3/receipt.json'
    a=json.loads(all14.read_text());source=json.loads(verify(a['source_data']).read_text());index=verify(source['artifacts']['rows.json'])
    revisions={t:json.loads((Path(cfgs[t]['act_root'])/'config.json').read_text()) for t in tasks}
    # Revision is obtained from the immutable row identities, never modelconfig.
    revision_by_task={};chosen={};selected={};all_rows=0
    for row in records(index):
        row=dict(row,source_row_index=all_rows)
        all_rows+=1;task=row['task']
        if task not in revision_by_task:
            revision_by_task[task]=row['revision'];chosen[task]=choose_episodes(task,row['revision'])
        require(row['revision']==revision_by_task[task],'Task revision changed')
        if row['split']!='fit':continue
        part=partition(row['episode'])
        if row['episode'] not in chosen[task][part]:continue
        key=(task,row['episode']);selected.setdefault(key,[]).append(row)
    require(all_rows==70000 and len(selected)==1000,'Exact source70k and selected1000 distinctFIT episodes')
    rows=[]
    for task in tasks:
        for part in ('optimizer','diagnostic','gate'):
            for episode in chosen[task][part]:
                anchors=sorted(selected[(task,episode)],key=lambda r:r['frame']);require(len(anchors)==7,'Seven immutable anchors required')
                row=dict(anchors[anchor_index(task,revision_by_task[task],episode)])
                require(row['camera_order']==['observation.images.'+c for c in CAMERAS]
                    and set(row['pixels_sha256'])==set(CAMERAS),'Explicit pixel/camera alignment required')
                parquet=ROOT/'datasets/official-cache'/task/row['revision']/'data/chunk-000'/f'episode_{episode:06d}.parquet'
                # Store exact paths+known original hashes; actual decode admission
                # verifies parquet/video bytes and timestamp contents before use.
                videos={c:dict(path=str(ROOT/'datasets/official-cache'/task/row['revision']/'videos/chunk-000'/(c+'.color' if task=='click_bell' else 'observation.images.'+c)/f'episode_{episode:06d}.mp4'),
                    sha256=row['source_video_sha256'][c]) for c in CAMERAS}
                row.update(partition=part,current_timestamp_seconds=row['frame']/25,
                    camera_frame_indices={c:row['frame'] for c in CAMERAS},parquet=dict(path=str(parquet),sha256=row['parquet_sha256']),videos=videos)
                rows.append(row)
    own={str(path.relative_to(ROOT)):ref(path)['sha256'] for path in sorted(Path(__file__).parent.glob('*.py'))}
    act={t:dict(root=cfgs[t]['act_root'],files=cfgs[t]['act_files'],camera_order=[k for k in revisions[t]['input_features'] if k.startswith('observation.images.')],
        image_key_map={'observation.images.'+c:c for c in CAMERAS},config=revisions[t],revision=revision_by_task[t]) for t in tasks}
    for task,binding in act.items():
        require(binding['config']['vision_backbone']=='resnet18'
            and list(binding['config']['input_features'])[:3]!=[], 'Native ResNet18 config required')
        verify(binding['files']['config.json']);verify(binding['files']['model.safetensors'])
    report=dict(schema=PLAN,status='metadata_only_balanced1000_current_anchor_plan_no_extraction',execution_authorized=False,
        tasks=tasks,rows=rows,counts=dict(optimizer=800,diagnostic=100,gate=100,MONITOR=0),
        per_task=dict(optimizer=80,diagnostic=10,gate=10),one_anchor_per_distinct_episode=True,
        pilot_source_rows=1000,full63000_optimizer_coverage_claimed=False,seed=1701,
        ACT_camera_order=list(CAMERAS),teacher_query_order=list(ROLE_QUERIES),teacher_query_not_camera_axis=True,
        act=act,teacher_contract=cfgs[tasks[0]]['assets']['coupled_contract'],teacher_weights=cfgs[tasks[0]]['assets']['coupled_weights'],
        frozen_current_head_contract=cfgs[tasks[0]]['assets']['highlevel_contract'],
        frozen_current_head_weights=cfgs[tasks[0]]['assets']['highlevel_weights'],source_current_predictions=a['source_features'],
        source_data=a['source_data'],source_rows=source['artifacts']['rows.json'],teacher_statistics=contract['statistics'],
        native_campaign=ref(campaign),source_sha256=own,
        ACT_source=ref('/tmp/ra5/upstream/lerobot-sim/src/lerobot/policies/act/modeling_act.py'),
        ACT_normalizer_source=ref('/tmp/ra5/upstream/lerobot-sim/src/lerobot/policies/normalize.py'),
        canonical_dataset_source=ref(ROOT/'policy/wam_adaptation/multitask_data.py'),
        old_cache_producer=ref(ROOT/'experiments/goal_latent_act_full_data/produce_batch_v2.py'),
        extraction=dict(device='cpu',backbone_only=True,fullACT_constructed=False,Qwen_constructed=False,
            pooling='spatial2x3_primary_exactglobal_ablation',batch_size=7,Torch_threads=2,decoder_workers=2,
            ffmpeg_threads=1,host_tree_peak_limit_bytes=3*(1<<30),output_limit_bytes=1<<30,wall_seconds=1800,
            native_normalize_inputs_image_buffers_once=True,source_pixels_reverified=True,
            raw_future_frames_read_as_student_inputs=False),
        teacher_targets=dict(projected_roles='tanh(frozen role_goal.roles(current query_features_FP32)).flatten96',
            predicted_goal_delta='(eef_features(frozen current predicted_goal50)-eef_features(current_FK))/frozen coupled eef_scale18',
            no_ground_truth_future_targets=True,no_extra_role_or_goal_mean_std=True),
        training=dict(epochs=30,batch_size=128,AdamW_lr=.001,weight_decay=.0001,seed=1701,device='cpu',
            visual_input_mean_std='optimizer800 only; channel-wise bycamera over spatialcells; floor0.01',
            roles_loss_scale='identity; projected teacher activations unaltered',goal_loss_scale='identity; frozen normalized eef_scale already applied',
            objective='mean projectedroles96 squared error + mean predicted_goal_delta18 squared error',
            matched_control='same spatial model and same initialization/order; visual inputs zeroed',
            global_ablation='same3x512 rawmapglobalaverage, not mean of adaptive2x3cells',MONITOR_selection=False,
            checkpoint='fixed epoch30; no best-checkpoint selection'),
        gates=dict(teacher_native_benefit_before_execution=True,student_roles_RMSE_max=.10,
            teacher_physical_goal_translation_RMSE_max_m=.01,teacher_rotation6D_RMSE_max=.05,
            executed10_per_task_arm_FK_error_relative_to_teacher_max=1.01,
            executed10_opening_error_absolute_margin_m=.0001,per_hand_acceptance_agreement_min=.99,
            false_accepts_relative_to_teacher_max=0,rejected_native7_bitexact=True,
            visual_normalized_target_MSE_gain_over_qpos_control_min=.05,
            original_teacher_5percent_promotion_not_reclassified=True,
            embedding_MSE_alone_is_not_robot_gain=True),
        inference_runtime='new projected_forward ABI to frozen coupledGRU/reliability, ACT50/native10 unchanged; no hook repeats backbone',
        GPU_queries=0,GPU_launches=0,model_loads=0,images_decoded=0,student_training=False,robot_quality_promoted=False)
    output.mkdir(parents=True);path=output/'plan.json';path.write_text(json.dumps(report,indent=2)+'\n')
    return ref(path)


def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(prepare(a.output)))


if __name__=='__main__':main()
