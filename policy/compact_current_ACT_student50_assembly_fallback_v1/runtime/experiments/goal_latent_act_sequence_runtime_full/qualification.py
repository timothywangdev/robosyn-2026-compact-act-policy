"""Qualify actual original CPU receipts; metadata fixtures confer no admission."""
import json
from pathlib import Path
from experiments.goal_latent_act_campaign.common import ref, verify, require
from .semantics import PAIR_SCHEMA

TRAINING_SCHEMA = 'goal-latent-act-commanded50-full63000-original-CPU-pilot-v1'
RUNS = ('prepared','head50','sequence_prepared','role_goal','state_action_control')


def same_bytes(left, right):
    return all(left[k] == right[k] for k in ('sha256','bytes'))


def qualify(original_ref, child_refs, head_ref, branch_refs, controller_wait_ref):
    original = json.loads(verify(original_ref).read_text())
    require(original['schema']==TRAINING_SCHEMA
            and original['status']=='all_original_exit0_natural_CPU_pilot_complete'
            and [r['name'] for r in original['runs']]==list(RUNS)
            and set(child_refs)==set(RUNS), 'Actual five original completed full63000 CPU children required')
    require((original['source_FIT'],original['source_MONITOR'],original['optimizer_FIT'],
             original['diagnostic_FIT'],original['gate_FIT'])==(63000,7000,50400,6300,6300)
            and original['MONITOR_selection'] is False and original['GPU_launches']==0,
            'Exact independently partitioned full63000 training scope required')
    wait=json.loads(verify(controller_wait_ref).read_text())
    require(wait['schema']=='goal-latent-act-full63000-controller-original-wait-v1'
            and wait['owner']==original['controller'] and wait['owner']['pid']==wait['owner']['group']==wait['owner']['session']
            and same_bytes(wait['scientific_receipt'],original_ref)
            and wait['original_controller_exit_code']==0 and wait['original_controller_reaped'] is True
            and wait['original_process_absent'] is True and wait['independently_observed_group_after_reap']==[]
            and wait['natural_completion_no_signals_sent'] is True and wait['signals_sent']==[],
            'Actual separately captured original controller0/reap/natural drain required')
    children={}
    for row in original['runs']:
        require(row['original_exit_code']==0 and row['original_child_reaped'] is True
                and row['original_process_absent'] is True and row['signals_sent']==[]
                and row['natural_completion_no_signals_sent'] is True
                and row['independently_observed_group_after_reap']==[], 'Original natural child0/reap/drain required')
        reference=child_refs[row['name']]
        require(same_bytes(row['child_receipt'],reference), 'Transported child receipt differs from original bytes')
        children[row['name']]=json.loads(verify(reference).read_text())
    head=json.loads(verify(head_ref).read_text())
    from experiments.goal_latent_act_command_goal_training_full.contracts import validate_head
    from experiments.goal_latent_act_sequence_contact_training_full.contracts import validate
    validate_head(head)
    require(children['head50']['status']=='completed_fixed50_epoch_commanded_head'
            and children['head50']['checkpoint']==head
            and children['head50']['reload_prediction_bitexact'] is True,
            'Actual fixed50 trained and reloaded head contract required')
    branches={}
    for mode, name in (('goal','role_goal'),('goal_free','state_action_control')):
        contract_ref=branch_refs[mode];contract=json.loads(verify(contract_ref).read_text());validate(contract)
        child=children[name]
        require(child['status']=='completed_fixed30_full63000_sequence20_FIT_gate'
                and child['checkpoint']==contract and child['reload_prediction_bitexact'] is True
                and child['optimizer_updates']==11820 and child['gate_updates']==1000,
                'Actual fixed30+20 trained/reloaded sequence required')
        require(contract['state_action_only']==(mode=='goal_free')
                and contract['highlevel_weights_sha256']==head['weights_sha256']
                and contract['highlevel_contract_sha256']==head_ref['sha256'], 'Exact matched full head/branch required')
        branches[mode]=dict(weights_sha256=contract['weights_sha256'],contract=contract_ref,training_receipt=child_refs[name])
    factual, control=children['role_goal'],children['state_action_control']
    for key in ('history','gate_history'):
        require([r['FIT_order_sha256'] for r in factual[key]]==[r['FIT_order_sha256'] for r in control[key]],
                'Matched role/control actual FIT orders differ')
    require(factual['checkpoint']['statistics']==control['checkpoint']['statistics']
            and factual['checkpoint']['prepared_receipt_sha256']==child_refs['sequence_prepared']['sha256']
            and control['checkpoint']['prepared_receipt_sha256']==child_refs['sequence_prepared']['sha256'],
            'Matched shared optimization FIT moments/preparation required')
    return dict(schema=PAIR_SCHEMA,original_CPU_receipt=original_ref,child_receipts=child_refs,
        head=dict(weights_sha256=head['weights_sha256'],contract=head_ref,training_receipt=child_refs['head50']),
        branches=branches,source_pins=original['source_pins'],source_FIT=63000,source_MONITOR=7000,
        original_five_children_exit0_natural_drained=True,controller_original_wait=controller_wait_ref,
        original_outer_controller_exit0_captured=True,
        offline_diagnostic=original['result'],robot_deployment_authorized=False,robot_success_claim=False,
        root_serial_launch_certificate_still_required=True)


def validate_bound_training(cfg):
    pair_ref=cfg['assets']['training_pair'];pair=json.loads(verify(pair_ref).read_text())
    expected=qualify(pair['original_CPU_receipt'],pair['child_receipts'],pair['head']['contract'],
        {m:r['contract'] for m,r in pair['branches'].items()},pair['controller_original_wait'])
    require(pair==expected, 'Exact actual full63000 training qualification seal differs')
    require(same_bytes(pair['head']['contract'],cfg['assets']['highlevel_contract'])
            and pair['head']['weights_sha256']==cfg['assets']['highlevel_weights']['sha256']
            and same_bytes(pair['branches'][cfg['mode']]['contract'],cfg['assets']['contact_contract'])
            and pair['branches'][cfg['mode']]['weights_sha256']==cfg['assets']['contact_weights']['sha256'],
            'Deployment head/branch differs from completed matched training pair')
    require(all(cfg['source_sha256'].get(k)==v for k,v in pair['source_pins'].items()),
            'Completed original scientific source pins differ')
    return pair
