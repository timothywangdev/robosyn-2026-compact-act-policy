"""Causal current-only feature boundary; full-hand rejection preserves raw ACT."""
from .common import require


def correct_native50(model,native,current,stats,*,task,codec,state_action_only):
    import torch
    from experiments.goal_latent_act_full_data.correction_prefix import eef_features
    from experiments.goal_latent_act_contact_runtime.gripper_boundary import DRAWER,IDENTITY,TASKS
    from .units import native_to_physical
    require(task in TASKS and codec==(DRAWER if task=='drawer_open_place' else IDENTITY),'Exact source task/codec required')
    require(native.shape==(1,50,14) and native.dtype==torch.float32 and bool(torch.isfinite(native).all()),'Finite actual native50 required')
    expected={'current_roles':(1,3,4096),'qpos':(1,14),'goal':(1,2,4,4),'current_eef':(1,2,4,4)}
    require(set(current)==set(expected),'Future state/phase/episode fields are forbidden at actor boundary')
    for key,shape in expected.items():
        x=current[key]
        require(x.shape==shape and x.dtype==native.dtype and x.device==native.device and bool(torch.isfinite(x).all()),'Finite current field required: '+key)
    scale=torch.full((1,),.05 if task=='drawer_open_place' else 1.,dtype=native.dtype,device=native.device)
    physical=native_to_physical(native,scale);now=eef_features(current['current_eef'])
    result=model(roles=current['current_roles'],qpos_features=(current['qpos']-stats['qpos_mean'])/stats['qpos_scale'],
        current_eef_features=(now-stats['current_eef_mean'])/stats['current_eef_scale'],
        goal_delta=(eef_features(current['goal'])-now)/stats['eef_scale'],
        native_base_features=(physical-stats['base_mean'])/stats['base_scale'],native_base=native,grip_scale=scale,
        state_action_only=state_action_only,gated=True,threshold=.5)
    action=result['native_actions'];require(action.shape==native.shape and bool(torch.isfinite(action).all()),'Finite corrected native50 required')
    rejected=(~result['accepted'])[:,None,:,None].expand(1,50,2,7)
    require(torch.equal(action.reshape(1,50,2,7)[rejected],native.reshape(1,50,2,7)[rejected]),'Rejected full hand differs from exact native ACT')
    return action,result
