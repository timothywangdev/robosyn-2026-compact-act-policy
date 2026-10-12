"""Same qualified native terminal retention hook; no physics or scoring change."""
import numpy as np
from .contracts import require

def step_preserving_terminal(env,action):
    """Keep the real terminal observation/counter until the evaluator resets.

    BaseEnv's existing replay hook only controls its reset AFTER physics,
    observations, rewards, terminal checks and the after-step hook. Restore it
    immediately, including exceptions. No simulation/config source is patched.
    """
    native=env.unwrapped
    existed='_replay_no_auto_reset' in native.__dict__
    previous=getattr(native,'_replay_no_auto_reset',False)
    require(type(previous)is bool,'Unexpected native replay-reset hook type')
    native._replay_no_auto_reset=True
    try:
        obs,reward,terminated,truncated,info=env.step(action)
        info=dict(info)
        elapsed=info.get('elapsed_steps')
        if hasattr(elapsed,'clone'):info['elapsed_steps']=elapsed.clone()
        elif isinstance(elapsed,np.ndarray):info['elapsed_steps']=elapsed.copy()
        return obs,reward,terminated,truncated,info
    finally:
        if existed:native._replay_no_auto_reset=previous
        else:delattr(native,'_replay_no_auto_reset')


class TerminalEnvironment:
    def __init__(self,env): self.env=env
    def __getattr__(self,name): return getattr(self.env,name)
    def step(self,action): return step_preserving_terminal(self.env,action)
