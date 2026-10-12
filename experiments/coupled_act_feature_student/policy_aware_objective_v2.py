"""Actual gated executed10 motor loss plus fixed tenth-weight proposal aid."""
from .policy_aware_objective import objective as original_objective,WEIGHTS as BASE_WEIGHTS

WEIGHTS={key:weight for key,weight in BASE_WEIGHTS.items() if key in ('role_distillation','goal_distillation')}
for prefix,multiplier in (('gated',1.),('proposal',.1)):
    WEIGHTS.update({prefix+'_'+key:multiplier*weight for key,weight in BASE_WEIGHTS.items() if key not in ('role_distillation','goal_distillation')})


def objective(student,result,teacher,target,valid,fk):
    # Same physical first10 losses/units/FK/masks. Actual accepted output is
    # primary; a rejected hand has zero gated-policy derivative, so the fixed
    # auxiliary proposal term preserves a supervised gradient there.
    actual=dict(result,ungated_physical_actions=result['physical_actions'])
    _,gated=original_objective(student,actual,teacher,target,valid,fk)
    _,proposal=original_objective(student,result,teacher,target,valid,fk)
    parts={key:gated[key] for key in ('role_distillation','goal_distillation')}
    for prefix,values in (('gated',gated),('proposal',proposal)):
        parts.update({prefix+'_'+key:value for key,value in values.items() if key not in ('role_distillation','goal_distillation')})
    return sum(WEIGHTS[key]*value for key,value in parts.items()),parts
