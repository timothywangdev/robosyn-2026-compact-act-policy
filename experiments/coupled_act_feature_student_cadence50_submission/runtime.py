"""Exact native actor methods with only typed manifest/checkpoint transport."""
from experiments.coupled_act_feature_student_cadence50.runtime import CoreRuntime as OriginalRuntime
from experiments.coupled_act_feature_student_cadence50.runtime import Runtime as OriginalPrecisionRuntime
from experiments.coupled_act_feature_student_runtime.runtime import eval,reset_model,close_model
from .contracts import borrowed,load_manifest,validate_saved

class Runtime(OriginalPrecisionRuntime):
    __init__=borrowed(OriginalRuntime.__init__,dict(load_manifest=load_manifest,validate_saved=validate_saved))

def get_model(args):return Runtime(args)
