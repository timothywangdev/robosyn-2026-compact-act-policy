"""Native lifecycle exports with flushed exception evidence before SDK cleanup."""
import sys
from experiments.coupled_goal_act_all14_runtime import runtime_eager as implementation

EXCEPTION_MARKER='COUPLED_ALL14_POLICY_EXCEPTION_CAPTURED'


def guarded(name,function,*args,**kwargs):
    try:return function(*args,**kwargs)
    except BaseException as error:
        print(EXCEPTION_MARKER+' '+name+' '+type(error).__name__,file=sys.stderr,flush=True)
        raise


def get_model(*args,**kwargs):return guarded('get_model',implementation.get_model,*args,**kwargs)
def eval(*args,**kwargs):return guarded('eval',implementation.eval,*args,**kwargs)
def reset_model(*args,**kwargs):return guarded('reset_model',implementation.reset_model,*args,**kwargs)
def close_model(*args,**kwargs):return guarded('close_model',implementation.close_model,*args,**kwargs)
