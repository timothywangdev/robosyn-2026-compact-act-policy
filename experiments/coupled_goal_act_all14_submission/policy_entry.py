"""Official policy exports with a fail-closed preparation boundary."""
from experiments.coupled_goal_act_all14_runtime.policy_shim_eager import (
    eval, reset_model, close_model,
)

POLICY = 'coupled_all14_submission_preparation_v1'


def get_model(usr_args):
    # The model implementation remains frozen. Absolute-reference relocation
    # needs a separate reviewed binding loader and installed-entry qualification.
    raise RuntimeError('Coupled all14 submission is preparation only: portable provenance binding and installed native qualification pending')
