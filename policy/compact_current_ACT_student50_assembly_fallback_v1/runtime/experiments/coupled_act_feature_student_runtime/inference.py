"""Current ACT-map student features and the unchanged projected motor ABI."""
from experiments.coupled_act_feature_student.contracts import require
from experiments.coupled_goal_act_all14.units import native_to_physical


def student_inputs(visual, visual_global, qpos, statistics, teacher_statistics):
    import torch
    b = len(qpos)
    for value, shape in ((visual, (b,3,512,2,3)), (visual_global, (b,3,512)), (qpos, (b,14))):
        require(value.shape == shape and value.dtype == torch.float32 and bool(torch.isfinite(value).all()),
                'Finite same-forward current student features required')
        require(value.device == qpos.device, 'Student inputs must share the ACT device')
    return dict(visual=(visual-statistics['visual_mean'][None,:,:,None,None])/statistics['visual_scale'][None,:,:,None,None],
                visual_global=(visual_global-statistics['global_mean'])/statistics['global_scale'],
                qpos_features=(qpos-teacher_statistics['qpos_mean'])/teacher_statistics['qpos_scale'])


def motor_inputs(native, current, statistics, *, task, codec):
    import torch
    from experiments.goal_latent_act_contact_runtime.gripper_boundary import DRAWER, IDENTITY, TASKS
    from experiments.goal_latent_act_full_data.correction_prefix import eef_features
    require(task in TASKS and codec == (DRAWER if task == 'drawer_open_place' else IDENTITY), 'Exact source task codec')
    require(set(current) == {'qpos','current_eef'}, 'Only measured current qpos/FK permitted; future fields forbidden')
    b = len(native)
    for value,shape in ((native,(b,50,14)),(current['qpos'],(b,14)),(current['current_eef'],(b,2,4,4))):
        require(value.shape == shape and value.dtype == torch.float32 and value.device == native.device
                and bool(torch.isfinite(value).all()), 'Finite native50/current-only FP32 inputs required')
    scale = torch.full((b,), .05 if task == 'drawer_open_place' else 1., dtype=native.dtype, device=native.device)
    physical = native_to_physical(native,scale)
    return dict(qpos_features=(current['qpos']-statistics['qpos_mean'])/statistics['qpos_scale'],
        current_eef_features=(eef_features(current['current_eef'])-statistics['current_eef_mean'])/statistics['current_eef_scale'],
        native_base_features=(physical-statistics['base_mean'])/statistics['base_scale'],
        native_base=native,grip_scale=scale,state_action_only=False,gated=True,threshold=.5)


def correct_with_student(student, actor, inputs, motor, *, student_call=None, projected_call=None):
    import torch
    from experiments.coupled_act_feature_student.projected import projected_forward
    require(set(inputs)=={'visual','visual_global','qpos_features'}, 'Exact current student ABI')
    require(set(motor)=={'qpos_features','current_eef_features','native_base_features','native_base','grip_scale',
                        'state_action_only','gated','threshold'}, 'Exact measured motor ABI')
    require(torch.equal(inputs['qpos_features'],motor['qpos_features']), 'Student and motor measured state normalization differs')
    predicted = (student_call or student)(**inputs)
    require(set(predicted)=={'projected_roles','predicted_goal_delta'}, 'Only current predicted semantic role/goal outputs')
    call = projected_call or (lambda **kwargs: projected_forward(actor,**kwargs))
    result = call(projected_roles=predicted['projected_roles'],goal_delta=predicted['predicted_goal_delta'],**motor)
    native = motor['native_base']; action = result['native_actions']
    require(action.shape==native.shape and action.dtype==native.dtype and bool(torch.isfinite(action).all()), 'Finite native50 output')
    rejected=(~result['accepted'])[:,None,:,None].expand(-1,50,2,7)
    require(torch.equal(action.reshape(-1,50,2,7)[rejected],native.reshape(-1,50,2,7)[rejected]), 'Rejected full native7 differs')
    return action,result,predicted
