"""Exact optimizer-FIT feature equations; native grippers are never converted."""
from .contracts import require


def correct_native50(model,native,current,stats,*,state_action_only):
    import torch
    from experiments.goal_latent_act_full_data.correction_prefix import eef_features
    from experiments.goal_latent_act_arm_sequence_full.model import ARM,GRIP
    require(native.shape==(1,50,14) and native.dtype==torch.float32
            and bool(torch.isfinite(native).all()),'Finite original FP32 native50 plan required')
    expected={'current_roles':(1,3,4096),'qpos':(1,14),'goal':(1,2,4,4),'current_eef':(1,2,4,4)}
    require(set(current)==set(expected),'Only current observation and predicted command50 goal permitted')
    for key,shape in expected.items():
        x=current[key]
        require(x.shape==shape and x.dtype==native.dtype and x.device==native.device
                and bool(torch.isfinite(x).all()),'Finite current feature required: '+key)
    now=eef_features(current['current_eef'])
    result=model(roles=current['current_roles'],
        qpos_features=(current['qpos']-stats['qpos_mean'])/stats['qpos_scale'],
        current_eef_features=(now-stats['current_eef_mean'])/stats['current_eef_scale'],
        goal_delta=(eef_features(current['goal'])-now)/stats['eef_scale'],
        native_base_features=(native-stats['base_mean'])/stats['base_scale'],native_base=native,
        state_action_only=state_action_only,gated=True,threshold=.5)
    action=result['native_actions']
    require(action.shape==native.shape and action.dtype==native.dtype and action.device==native.device
            and bool(torch.isfinite(action).all()),'Finite native temporal-arm result required')
    require(result['accepted'].shape==(1,2) and result['accepted'].dtype==torch.bool,
            'Actual two-hand fixed gate required')
    require(result['delta_arm_rad'].shape==(1,50,12)
            and bool((result['delta_arm_rad'].abs()<=.15).all()),'Declared arm delta bound changed')
    require(torch.equal(action[...,list(GRIP)],native[...,list(GRIP)]),'Native grippers must remain bitexact')
    arms=action[...,list(ARM)].reshape(1,50,2,6)
    original=native[...,list(ARM)].reshape_as(arms)
    reject=(~result['accepted'])[:,None,:,None].expand_as(arms)
    require(torch.equal(arms[reject],original[reject]),'Rejected hands must preserve literal native arm coordinates')
    return action,result
