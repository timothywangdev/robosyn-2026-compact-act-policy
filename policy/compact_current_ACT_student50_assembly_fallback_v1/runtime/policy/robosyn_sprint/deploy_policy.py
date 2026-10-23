"""Delegate to official motors, with separately declared task-specific variants."""

from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass
class SprintModel:
    adapter: Any
    motor_model: Any
    motor_policy: str
    planner: Any = None
    planner_interval_actions: int = 50
    executed_actions: int = 0
    last_plan_action: int = -50
    command: str | None = None
    planner_mode: str = "subgoal"
    planner_calls: int = 0
    planner_valid_responses: int = 0
    refresh_count: int = 0
    motor_random_seed: int | None = None
    motor_episode_index: int = 0


def get_model(usr_args):
    mode = usr_args.get("planner_mode", "subgoal")
    if mode not in ("subgoal", "refresh"):
        raise ValueError("planner_mode must be subgoal or refresh.")
    motor_policy = usr_args.get("motor_policy", "smolvla")
    if motor_policy not in ("act", "dp", "smolvla"):
        raise ValueError("motor_policy must be 'smolvla', 'act', or 'dp'.")
    from .act_drawer_adapter import select_adapter
    adapter = select_adapter(usr_args, motor_policy)
    motor_args = dict(usr_args)
    motor_seed = None
    if motor_policy == "smolvla":
        import os
        value = os.environ.get("SMOLVLA_RANDOM_SEED")
        if value is not None:
            motor_seed = int(value)
            if not 0 <= motor_seed < 2**32:
                raise ValueError("SMOLVLA_RANDOM_SEED must be within [0, 2**32-1].")
    if motor_policy == "smolvla":
        from pathlib import Path

        checkpoint = motor_args.get("checkpoint_path")
        if not checkpoint:
            raise ValueError("checkpoint_path is required.")
        checkpoint = Path(checkpoint).resolve()
        if (checkpoint / "pretrained_model").is_dir():
            checkpoint = checkpoint / "pretrained_model"
        motor_args["checkpoint_dir"] = str(checkpoint)
    # Official adapters retain responsibility for checkpoint/device validation.
    interval = int(usr_args.get("planner_interval_actions", 50))
    if interval <= 0:
        raise ValueError("planner_interval_actions must be positive.")
    if motor_policy == "smolvla":
        from .qwen_planner import create_planner
        planner = create_planner(usr_args)
    else:
        planner = None
    try:
        motor_model = adapter.get_model(motor_args)
    except Exception:
        if planner is not None:
            planner.close()
        raise
    if planner is not None:
        import atexit
        atexit.register(planner.close)
    return SprintModel(adapter, motor_model, motor_policy, planner, interval,
                       last_plan_action=-interval, planner_mode=mode, motor_random_seed=motor_seed)


def eval(env, model, obs):
    try:
        return _eval(env, model, obs)
    except Exception:
        close_model(model)
        raise


def close_model(model):
    """Release both local workers after a fatal evaluator error."""
    try:
        if model.planner is not None:
            model.planner.close()
    finally:
        close = getattr(model.adapter, "close_model", None)
        if close is not None:
            close(model.motor_model)
        else:
            close = getattr(model.motor_model, "close", None)
            if close is not None:
                close()


def _eval(env, model, obs):
    if model.motor_policy == "smolvla":
        # Upstream SmolVLA returns three values; the pinned evaluator requires
        # four. Preserve its motor path and add the official timing boundary.
        from policy.inference_timing import finish_inference, start_inference

        adapter, motor = model.adapter, model.motor_model
        instruction = getattr(env, "_current_instruction", None)
        if not instruction:
            raise ValueError("The evaluator must supply the official task instruction.")
        times = []
        started_at = start_inference(motor.device)
        if model.planner is not None and (
            model.executed_actions - model.last_plan_action >= model.planner_interval_actions
        ):
            import json
            fallback = False
            refresh, reason = False, ""
            model.planner_calls += 1
            try:
                if model.planner_mode == "refresh":
                    decision = model.planner.monitor(obs, instruction)
                    from .qwen_planner import validate_refresh
                    decision = validate_refresh(json.dumps(decision))
                    refresh, reason = decision["refresh"], decision["reason"]
                    command = instruction
                else:
                    command = model.planner.plan(obs, instruction)
                    if not isinstance(command, str) or not command.strip():
                        raise ValueError("Planner returned an empty/non-string instruction.")
                    command = command.strip()
                model.planner_valid_responses += 1
            except Exception as exc:
                fallback = True
                print(f"[sprint planner] fallback to original instruction: {exc}", flush=True)
                command = instruction
                refresh, reason = False, "monitor unavailable" if model.planner_mode == "refresh" else ""
            if model.planner_mode == "refresh":
                if refresh:
                    motor.reset()  # Pinned Smol reset clears only its pending action queue.
                    model.refresh_count += 1
            elif model.command is not None and command != model.command:
                motor.reset()
            print('[sprint telemetry] ' + json.dumps({
                'event': 'planner_call', 'executed_actions': model.executed_actions,
                'mode': model.planner_mode, 'refresh': refresh, 'reason': reason,
                'fallback': fallback, 'instruction': command,
                'original_instruction': instruction, 'planner_calls': model.planner_calls,
                'valid_responses': model.planner_valid_responses, 'refresh_count': model.refresh_count,
            }), flush=True)
            model.command = command
            model.last_plan_action = model.executed_actions
        encoded_obs = adapter.encode_obs(obs, image_keys=motor.image_keys)
        if model.executed_actions == 0:
            import json
            from .observation_identity import observation_identity
            print('[sprint initial observation] ' + json.dumps(observation_identity(encoded_obs)), flush=True)
        actions = motor.infer(encoded_obs, task=instruction if model.planner_mode == "refresh" else model.command or instruction)
        motor.maybe_print_action_debug(actions)
        action_tensors = [
            adapter.encode_action(
                action, env, rescale_gripper=motor.rescale_gripper,
                gripper_indices=motor.gripper_indices,
                gripper_source_range=motor.gripper_source_range,
            ) for action in actions
        ]
        finish_inference(started_at, times, motor.device)
        final_obs, info, truncated = obs, {}, False
        for action_tensor in action_tensors:
            motor.maybe_print_env_action_debug(action_tensor)
            final_obs, reward, terminated, truncated, info = env.step(action_tensor)
            model.executed_actions += 1
            if bool(adapter._to_numpy(truncated).reshape(-1)[0]) or bool(
                adapter._to_numpy(terminated).reshape(-1)[0]
            ) or bool(adapter._to_numpy(env.get_wrapper_attr("is_task_success")()).reshape(-1)[0]):
                break
        return final_obs, info, truncated, times
    return model.adapter.eval(env, model.motor_model, obs)


def reset_model(model):
    model.executed_actions = 0
    model.last_plan_action = -model.planner_interval_actions
    model.command = None
    model.planner_calls = model.planner_valid_responses = model.refresh_count = 0
    if model.planner is not None:
        model.planner.reset()
    if model.motor_policy == "smolvla" and model.motor_random_seed is not None:
        import json
        seed = (model.motor_random_seed + model.motor_episode_index) % 2**32
        # Match native reset's debug bookkeeping, but seed only this episode RPC.
        model.motor_model._debug_chunks_printed = 0
        model.motor_model._rpc({"cmd": "reset", "random_seed": seed})
        print('[sprint motor seed] ' + json.dumps({
            'episode_index': model.motor_episode_index, 'random_seed': seed,
            'variant': 'controlled episode-seeded motor; queue-only resets preserve RNG',
        }), flush=True)
        model.motor_episode_index += 1
        return None
    return model.adapter.reset_model(model.motor_model)
