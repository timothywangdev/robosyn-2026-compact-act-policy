"""Actual current Student compile attempt, including frozen motor decisions."""
from experiments.coupled_act_feature_student.contracts import require


def qualify_current(student, actor, inputs, motor, *, backend='inductor', benchmark=True):
    from torch._dynamo.utils import counters
    from experiments.coupled_act_feature_student.compiler import qualify,validated
    from experiments.coupled_act_feature_student.projected import projected_forward
    before={group:dict(rows) for group,rows in counters.items()}
    def tail(predicted,index):
        require(index==0, 'Exact first-current motor identity')
        return projected_forward(actor,projected_roles=predicted['projected_roles'],goal_delta=predicted['predicted_goal_delta'],**motor)
    selected,report=qualify(student,[inputs],backend=backend,benchmark=benchmark,postprocess=tail)
    report=dict(report)
    report['qualification_scope']='one_actual_first_current_context_per_native_task_not_all10_promotion'
    report['teacher_motor_execution_mode']='qualified_eager_fixed_projected_tail'
    report['native_ACT_execution_mode']='qualified_eager_FP32'
    report['compiler_counter_delta']={group:{key:value-before.get(group,{}).get(key,0) for key,value in rows.items()
        if value-before.get(group,{}).get(key,0)} for group,rows in counters.items()}
    report['Qwen_models_loaded']=0
    def student_call(**values):return validated(student,selected,values)
    return student_call,None,report
