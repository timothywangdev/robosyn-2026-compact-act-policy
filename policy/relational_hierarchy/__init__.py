"""Learned hierarchy with lazy competition entrypoints and model exports."""

__all__ = ["BridgeOutput", "RelationalGoalBridge", "verified_contrastive_loss", "get_model", "eval", "reset_model", "close_model"]

def __getattr__(name):
    if name in ("BridgeOutput", "RelationalGoalBridge", "verified_contrastive_loss"):
        from . import model
        return getattr(model, name)
    raise AttributeError(name)

def get_model(args):
    from .runtime import get_model as implementation
    return implementation(args)

def eval(env, model, obs):
    from .runtime import eval as implementation
    return implementation(env, model, obs)

def reset_model(model):
    return model.reset()

def close_model(model):
    return model.close()
