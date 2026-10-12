"""Official hooks for separately source-bound executor50 candidate."""
from experiments.coupled_goal_act_all14_runtime.policy_shim_eager import guarded

def get_model(args):
    from .runtime import get_model as call
    return guarded('get_model',call,args)
def eval(*args,**kwargs):
    from .runtime import eval as call
    return guarded('eval',call,*args,**kwargs)
def reset_model(*args,**kwargs):
    from .runtime import reset_model as call
    return guarded('reset_model',call,*args,**kwargs)
def close_model(*args,**kwargs):
    from .runtime import close_model as call
    return guarded('close_model',call,*args,**kwargs)
