"""Same causal student/tail/native motor; only execute the complete native50."""
from experiments.coupled_act_feature_student_runtime.runtime import Runtime as OriginalRuntime
from experiments.coupled_act_feature_student_runtime.runtime import eval,reset_model,close_model
from experiments.coupled_act_feature_student_runtime.runtime_v2 import PrecisionObservedRuntime
from .derive import derive
from .contracts import load_manifest

class CoreRuntime(OriginalRuntime):
    __init__=derive(OriginalRuntime.__init__,'init',{'load_manifest':load_manifest})
    select_action=derive(OriginalRuntime.select_action,'action')

class Runtime(PrecisionObservedRuntime,CoreRuntime):pass

def get_model(args):return Runtime(args)
