"""Actual GPU pair and genuine parent waits are mandatory before deployment."""
import json
from pathlib import Path
from .contracts import verify_ref,require
from .semantics import PAIR_SCHEMA,same_bytes


def qualify(pair_ref,wait_ref,supervisor_ref,terminal_ref,prepared_ref,head_ref,branch_refs,receipt_refs,reference_aliases=None):
    pair=json.loads(verify_ref(pair_ref).read_text());wait=json.loads(verify_ref(wait_ref).read_text())
    sup=json.loads(verify_ref(supervisor_ref).read_text());terminal=json.loads(verify_ref(terminal_ref).read_text())
    prepared=json.loads(verify_ref(prepared_ref).read_text());head=json.loads(verify_ref(head_ref).read_text())
    require(pair['schema']=='goal-latent-act-full63000-temporal-arm12-original-GPU-pair-v1'
            and pair['status']=='both_original_GPU_children_exit0_natural_complete'
            and pair['original_supervisor_exit_code']==0 and same_bytes(pair['supervisor_receipt'],supervisor_ref),
            'Genuine original completed GPU pair required')
    require(wait['schema']=='goal-latent-act-full63000-arm-pair-original-parent-wait-v1'
            and wait['original_controller_exit_code']==0 and wait['original_controller_reaped'] is True
            and wait['original_controller_absent'] is True and wait['independently_observed_group_after_reap']==[]
            and wait['signals']==[] and wait['success'] is True
            and same_bytes(wait['scientific_pair_receipt'],pair_ref)
            and same_bytes(wait['supervisor_receipt'],supervisor_ref),
            'Separately captured genuine original controller0/reap/natural drain required')
    require(sup['exit_code']==0 and sup['all_owned_groups_drained'] is True
            and sup['controller']==wait['owner']
            and [j['name'] for j in sup['jobs']]==['role_goal','state_action_control'],
            'Exactly two completed original GPU children required')
    for job in sup['jobs']:
        require(job['original_child_exit_code']==0 and job['child_reaped'] is True
                and job['signals']==[] and job['drain']['drained'] is True
                and job['drain']['natural'] is True and job['drain']['after_reap']==[],
                'Original GPU child0/reap/natural drain required')
    require(terminal['schema']=='goal-latent-act-full63000-temporal-arm-pair-independent-terminal-v1'
            and terminal['original_outer_exit_code']==0 and same_bytes(terminal['original_outer_wait'],wait_ref)
            and same_bytes(terminal['supervisor_receipt'],supervisor_ref)
            and same_bytes(terminal['scientific_pair_receipt'],pair_ref)
            and terminal['all_recorded_groups_sessions_live_members']==[]
            and terminal['fresh_GPU_snapshot']['compute_pids']==[]
            and terminal['independent_shared_flock_acquired_and_released'] is True and terminal['signals_sent']==[],
            'Independent entire-family/GPU/flock terminal proof required')
    require(all(r['exact_recorded_identity_absent'] is True for r in terminal['identities']),
            'Actual original identities must all be absent')
    require(prepared['schema']=='goal-latent-act-full63000-temporal-arm12-preparation-v1'
            and prepared['status']=='CPU_full_source_current_only_FK_parity_verified'
            and prepared['source_counts']==dict(FIT=63000,MONITOR=7000)
            and same_bytes(prepared['highlevel_contract'],head_ref), 'Actual full-source arm preparation required')
    # Explicit content-preserving aliases support a transported packet without
    # rewriting any historical receipt bytes or following its former host paths.
    aliases=reference_aliases or {}
    for digest,value in aliases.items():
        require(digest==value['sha256'],'Alias key must be exact original SHA256');verify_ref(value)
    def historical(value):
        actual=aliases.get(value['sha256'],value)
        require(same_bytes(actual,value),'Transported historical receipt changed')
        return json.loads(verify_ref(actual).read_text())
    original=historical(prepared['original_training']);original_wait=historical(prepared['original_outer_wait'])
    require(original['schema']=='goal-latent-act-commanded50-full63000-original-CPU-pilot-v1'
            and original['status']=='all_original_exit0_natural_CPU_pilot_complete'
            and [r['name'] for r in original['runs']]==['prepared','head50','sequence_prepared','role_goal','state_action_control']
            and (original['source_FIT'],original['source_MONITOR'],original['optimizer_FIT'],original['diagnostic_FIT'],original['gate_FIT'])==(63000,7000,50400,6300,6300)
            and original['MONITOR_selection'] is False and original['GPU_launches']==0,
            'Actual original five-child full-source CPU science required')
    require(original_wait['schema']=='goal-latent-act-full63000-controller-original-wait-v1'
            and original_wait['owner']==original['controller']
            and same_bytes(original_wait['scientific_receipt'],prepared['original_training'])
            and original_wait['original_controller_exit_code']==0 and original_wait['original_controller_reaped'] is True
            and original_wait['original_process_absent'] is True and original_wait['independently_observed_group_after_reap']==[]
            and original_wait['natural_completion_no_signals_sent'] is True and original_wait['signals_sent']==[],
            'Genuine original CPU controller wait/reap/natural drain required')
    for row in original['runs']:
        require(row['original_exit_code']==0 and row['original_child_reaped'] is True and row['original_process_absent'] is True
                and row['signals_sent']==[] and row['natural_completion_no_signals_sent'] is True
                and row['independently_observed_group_after_reap']==[],'Original scientific CPU child failed or lacks drain')
        historical(row['child_receipt'])
    head_receipt=historical(original['runs'][1]['child_receipt'])
    require(head_receipt['checkpoint']==head and head_receipt['reload_prediction_bitexact'] is True,
            'Actually trained/reloaded fixed50 head contract required')
    from experiments.goal_latent_act_arm_sequence_full.contracts import validate
    branches={};reports={}
    for mode,name in (('goal','role_goal'),('goal_free','state_action_control')):
        c=json.loads(verify_ref(branch_refs[mode]).read_text());validate(c)
        r=json.loads(verify_ref(receipt_refs[mode]).read_text());reports[mode]=r
        require(same_bytes(pair['branches'][name],receipt_refs[mode])
                and same_bytes(terminal['branches'][name]['checkpoint'],branch_refs[mode])
                and same_bytes(terminal['branches'][name]['receipt'],receipt_refs[mode])
                and terminal['branches'][name]['weights']['sha256']==c['weights_sha256']
                and r['checkpoint']==c and r['reload_predictions_bitexact'] is True
                and r['status']=='completed_fixed30_temporal_arm20_physical_FIT_gate'
                and (r['optimizer_updates'],r['gate_updates'])==(11820,1000)
                and c['state_action_only']==(mode=='goal_free')
                and same_bytes(c['prepared'],prepared_ref), 'Actual fixed30+20 matched branch completion required')
        branches[mode]=dict(contract=branch_refs[mode],weights_sha256=c['weights_sha256'],training_receipt=receipt_refs[mode])
    for key in ('history','gate_history'):
        require([r['FIT_order_sha256'] for r in reports['goal'][key]]==[r['FIT_order_sha256'] for r in reports['goal_free'][key]],
                'Actual paired optimization-FIT orders differ')
    require(reports['goal']['checkpoint']['statistics']==reports['goal_free']['checkpoint']['statistics'],
            'Shared optimizer-FIT moments required')
    return dict(schema=PAIR_SCHEMA,original_pair=pair_ref,original_wait=wait_ref,supervisor=supervisor_ref,
        independent_terminal=terminal_ref,prepared=prepared_ref,head_contract=head_ref,branches=branches,
        source_pins=prepared['source_pins'],historical_reference_aliases=aliases,genuine_original_GPU_pair_and_outer_exit0_natural=True,
        diagnostic_FIT_criterion_met=pair['diagnostic_FIT_criterion_met'],
        robot_deployment_authorized=False,robot_success_claim=False)


def validate_bound_training(cfg):
    pair=json.loads(verify_ref(cfg['assets']['training_pair']).read_text())
    expected=qualify(pair['original_pair'],pair['original_wait'],pair['supervisor'],pair['independent_terminal'],
        pair['prepared'],pair['head_contract'],{k:r['contract'] for k,r in pair['branches'].items()},
        {k:r['training_receipt'] for k,r in pair['branches'].items()},pair['historical_reference_aliases'])
    require(pair==expected,'Actual typed arm training qualification packet differs')
    require(same_bytes(pair['head_contract'],cfg['assets']['highlevel_contract'])
            and same_bytes(pair['branches'][cfg['mode']]['contract'],cfg['assets']['arm_contract'])
            and pair['branches'][cfg['mode']]['weights_sha256']==cfg['assets']['arm_weights']['sha256'],
            'Deployment differs from genuine matched arm/head pair')
    require(all(cfg['source_sha256'].get(k)==v for k,v in pair['source_pins'].items()),'Actual scientific source closure differs')
    return pair
