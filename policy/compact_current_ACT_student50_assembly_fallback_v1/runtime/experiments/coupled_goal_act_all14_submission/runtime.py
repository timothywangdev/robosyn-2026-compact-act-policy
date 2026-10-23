"""Frozen eager coupled actor behind separately reviewed portable bindings."""
from .runtime_native import Runtime as NativeRuntime,eval,reset_model,close_model
from experiments.coupled_goal_act_all14_runtime.runtime_eager import Runtime as FrozenEager


class Runtime(NativeRuntime):
    qualify_compiler=FrozenEager.qualify_compiler


def get_model(args):return Runtime(args)
