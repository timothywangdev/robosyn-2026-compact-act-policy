"""Opt-in drawer dataset-label inverse inside the official ACT inference timing boundary."""
import json
import math
from importlib import import_module


def select_adapter(usr_args, motor_policy):
    enabled = usr_args.get('act_drawer_rescale_gripper', False)
    if not isinstance(enabled, bool):
        raise ValueError('act_drawer_rescale_gripper must be an explicit boolean')
    if enabled and (motor_policy != 'act' or usr_args.get('task_name') != 'drawer_open_place'):
        raise ValueError('ACT drawer gripper rescaling requires act + drawer_open_place')
    module = 'policy.robosyn_sprint.act_drawer_adapter' if enabled else f'policy.{motor_policy}.deploy_policy'
    return import_module(module)


def get_model(usr_args):
    # Selection validation also applies when this module is imported directly.
    if usr_args.get('motor_policy') != 'act' or usr_args.get('task_name') != 'drawer_open_place' or usr_args.get('act_drawer_rescale_gripper') is not True:
        raise ValueError('This adapter requires explicit ACT drawer rescaling')
    return import_module('policy.act.deploy_policy').get_model(usr_args)


class _ActionBoundary:
    """Forward official model attributes; decode select_action inside its timer."""
    def __init__(self, model, env):
        self._model = model
        self._env = env

    def __getattr__(self, name):
        return getattr(self._model, name)

    def select_action(self, batch):
        import torch
        action = self._model.select_action(batch)
        space = self._env.unwrapped.single_action_space
        if tuple(space.shape) != (14,) or action.ndim not in (1, 2) or action.shape[-1] != 14:
            raise ValueError('Drawer ACT inverse requires [14] or [B,14] joint actions')
        result = action.clone()
        limits = []
        for index in (6, 13):
            low, high = float(space.low[index]), float(space.high[index])
            if not (math.isfinite(low) and math.isfinite(high) and high > low):
                raise ValueError('Invalid drawer gripper joint limits')
            if not torch.isfinite(action[..., index]).all():
                raise ValueError('Nonfinite drawer gripper prediction')
            result[..., index] = low + action[..., index].clamp(0, 1) * (high - low)
            limits.append({'index': index, 'lower': low, 'upper': high})
        if not getattr(self._model, '_drawer_codec_logged', False):
            print('[ACT drawer codec] ' + json.dumps({'variant': 'drawer-normalized-gripper-inverse-v1', 'gripper_limits': limits, 'arms': 'unchanged', 'first_original_grippers': action[..., [6, 13]].detach().cpu().tolist(), 'first_converted_grippers': result[..., [6, 13]].detach().cpu().tolist()}), flush=True)
            self._model._drawer_codec_logged = True
        return result


def eval(env, model, obs):
    return import_module('policy.act.deploy_policy').eval(env, _ActionBoundary(model, env), obs)


def reset_model(model):
    return import_module('policy.act.deploy_policy').reset_model(model)
