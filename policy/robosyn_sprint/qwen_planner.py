"""Local Qwen3-VL worker; camera/state input and internal language output only."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import sys
import tempfile
import time


def _enabled(value):
    return value if isinstance(value, bool) else str(value).lower() in {"true", "1", "yes"}


def validate_instruction(text: str) -> str:
    """Validate syntax/size; this does not establish physical correctness."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    payload = json.loads(text)
    instruction = payload.get("instruction") if isinstance(payload, dict) else None
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Qwen must return a JSON object with a nonempty instruction.")
    instruction = instruction.strip()
    if len(instruction) > 400 or any(ord(c) < 32 for c in instruction):
        raise ValueError("Instruction exceeds the bounded single-line command format.")
    return instruction


def validate_refresh(text):
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[len("```json\n"):-3].strip()
    elif text.startswith("```\n") and text.endswith("```"):
        text = text[4:-3].strip()
    payload = json.loads(text)
    if not isinstance(payload, dict) or set(payload) != {"refresh", "reason"} or type(payload.get("refresh")) is not bool:
        raise ValueError("Refresh must be a JSON boolean.")
    reason = payload.get("reason")
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 200 or any(ord(c) < 32 for c in reason):
        raise ValueError("Refresh reason must be a bounded nonempty single line.")
    return {"refresh": payload["refresh"], "reason": reason.strip()}


class LocalQwenPlanner:
    def __init__(self, args):
        model_path = Path(args.get("qwen_model_path") or "").expanduser()
        if not args.get("qwen_model_path") or not model_path.is_dir():
            raise ValueError("qwen_model_path must be a downloaded local Qwen3-VL directory.")
        self.timeout = float(args.get("qwen_timeout_seconds", 120))
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("qwen_timeout_seconds must be positive.")
        load_timeout = float(args.get("qwen_load_timeout_seconds", 600))
        if not math.isfinite(load_timeout) or load_timeout <= 0:
            raise ValueError("qwen_load_timeout_seconds must be finite and positive.")
        self.previous_instruction = None
        self.temporal_context = _enabled(args.get("qwen_temporal_context", False))
        self.previous_observation = None
        self._response_buffer = bytearray()
        self.executable = args.get("qwen_python") or sys.executable
        child_env = os.environ.copy()
        child_env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        child_env["CUDA_VISIBLE_DEVICES"] = str(args.get("qwen_cuda_visible_devices", args.get("gpu_id", 0)))
        self.proc = subprocess.Popen(
            [self.executable, str(Path(__file__).resolve()), "--worker",
             "--model-path", str(model_path.resolve()),
             "--max-new-tokens", str(int(args.get("qwen_max_new_tokens", 128))),
             "--max-image-edge", str(int(args.get("qwen_max_image_edge", 512)))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            text=True, bufsize=1, env=child_env,
        )
        try:
            self._read_response(timeout=load_timeout)
        except Exception:
            self.close()
            raise

    def _read_response(self, timeout):
        if self.proc.stdout is None:
            raise RuntimeError("Qwen worker has no output pipe.")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Worker response timeout must be finite and positive.")
        deadline = time.monotonic() + timeout
        with selectors.DefaultSelector() as selector:
            selector.register(self.proc.stdout, selectors.EVENT_READ)
            while b"\n" not in self._response_buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    self.close()
                    raise TimeoutError("Local Qwen worker exceeded its response deadline.")
                chunk = os.read(self.proc.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError("Local Qwen worker exited; inspect its stderr.")
                self._response_buffer.extend(chunk)
                if len(self._response_buffer) > 1048576:
                    self.close()
                    raise RuntimeError("Local Qwen exceeded the bounded response size.")
        line, _, remainder = self._response_buffer.partition(b"\n")
        self._response_buffer = bytearray(remainder)
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "Local Qwen failed."))
        return response

    def plan(self, obs, original_instruction):
        return self._request(obs, original_instruction, "subgoal")

    def monitor(self, obs, original_instruction):
        return self._request(obs, original_instruction, "refresh")

    def _request(self, obs, original_instruction, mode):
        import numpy as np

        arrays = {}
        for name in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            if name not in obs.get("sensor", {}):
                continue
            array = obs["sensor"][name]["color"]
            arrays[name] = array.detach().cpu().numpy() if hasattr(array, "detach") else np.asarray(array)
        if not arrays:
            raise ValueError("Qwen needs at least one permitted camera observation.")
        state = obs.get("robot", {}).get("qpos")
        if state is not None:
            arrays["qpos"] = state.detach().cpu().numpy() if hasattr(state, "detach") else np.asarray(state)
        if self.proc.poll() is not None or self.proc.stdin is None:
            raise RuntimeError("Local Qwen worker is unavailable.")
        with tempfile.TemporaryDirectory(prefix="robosyn-qwen-") as folder:
            path = Path(folder) / "observation.npz"
            temporal = getattr(self, "temporal_context", False)
            payload_arrays = dict(arrays)
            previous = getattr(self, "previous_observation", None)
            if temporal and previous is not None:
                payload_arrays.update({"previous_" + key: value for key, value in previous.items()})
            np.savez(path, **payload_arrays)
            self.proc.stdin.write(json.dumps({"path": str(path), "task": original_instruction,
                                            "previous_instruction": self.previous_instruction,
                                            "temporal_context": temporal, "mode": mode}) + "\n")
            self.proc.stdin.flush()
            result = self._read_response(self.timeout)
        if mode == "refresh":
            decision = validate_refresh(json.dumps({"refresh": result.get("refresh"), "reason": result.get("reason")}))
            self.previous_instruction = original_instruction
        else:
            instruction = validate_instruction(json.dumps({"instruction": result.get("instruction")}))
            print("[qwen planner] " + json.dumps({"original_task": original_instruction,
                                                    "internal_instruction": instruction}), flush=True)
            self.previous_instruction = instruction
            decision = instruction
        if temporal:
            self.previous_observation = {key: value.copy() for key, value in arrays.items()}
        return decision

    def reset(self):
        # Worker is stateless per request; clear parent-owned episode history.
        self.previous_instruction = None
        self.previous_observation = None

    def close(self):
        proc = getattr(self, "proc", None)
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        for pipe in (proc.stdin, proc.stdout):
            if pipe is not None:
                pipe.close()


def create_planner(usr_args):
    if not _enabled(usr_args.get("planner_enabled", True)):
        return None
    return LocalQwenPlanner(usr_args)


def build_request_content(request, data, max_image_edge):
    """Build stateless visual input; temporal mode adds one earlier observation."""
    import numpy as np
    from PIL import Image

    temporal = request.get("temporal_context") is True
    content = []
    states = {}
    prefixes = [("previous_", "Previous observation"), ("", "Current observation")] if temporal else [("", "")]
    for prefix, label in prefixes:
        present = any(prefix + name in data for name in ("cam_high", "cam_left_wrist", "cam_right_wrist"))
        if not present:
            continue
        if label:
            content.append({"type": "text", "text": label})
        for name in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            if prefix + name not in data:
                continue
            array = data[prefix + name]
            if array.ndim == 4:
                if array.shape[0] != 1:
                    raise ValueError("Qwen currently supports exactly one environment.")
                array = array[0]
            if array.ndim != 3:
                raise ValueError(f"Invalid {name} image shape: {array.shape}")
            if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
                array = np.moveaxis(array, 0, -1)
            if array.dtype.kind == "f" and float(np.nanmax(array)) <= 1.5:
                array = array * 255
            array = np.clip(array, 0, 255).astype(np.uint8)
            if array.shape[-1] == 1:
                array = np.repeat(array, 3, axis=-1)
            image = Image.fromarray(array[..., :3])
            image.thumbnail((max_image_edge, max_image_edge))
            content.extend([{"type": "text", "text": name}, {"type": "image", "image": image}])
        state = data[prefix + "qpos"].reshape(-1).tolist() if prefix + "qpos" in data else None
        states[prefix] = state
        if temporal:
            content.append({"type": "text", "text": f"{label} joint state: {state}"})
    prompt = (
        "Plan the next short robot instruction using only these current camera images and joint state. "
        "Complete the original task; do not introduce another task or unsupported objects. "
        "A previous instruction is not evidence of success. If uncertain, repeat the original task. "
        "Return only a JSON object with one string field instruction, at most 400 characters. "
        f"Original task: {request['task']}\n"
        f"Previous instruction: {request.get('previous_instruction')}\nJoint state: {states.get('')}"
    )
    if temporal:
        prompt += (
            "\nCompare the labeled previous and current observations. Advance a phase only when "
            "visible evidence supports completion; a command or elapsed time is not completion. "
            "Do not assume a grasp, transfer, placement or stable basket from an earlier command. "
            "If evidence is occluded or ambiguous, do not claim completion or advance the phase."
        )
    if request.get("mode") == "refresh":
        prompt = (
            "Monitor the current robot execution without changing its instruction. Return only JSON "
            "with fields refresh (boolean) and reason (nonempty single-line string, at most 200 characters). "
            "Request refresh only for visible missed grasp/target, persistent stall evidenced by previous "
            "and current images, or a visible contact change requiring a fresh prediction. "
            "Uncertainty, elapsed time, or an earlier command alone do not justify refresh. "
            "If no such evidence exists, return refresh false. Do not declare task success or termination. "
            f"Original instruction: {request['task']}\nPrevious instruction: {request.get('previous_instruction')}"
        )
    content.append({"type": "text", "text": prompt})
    return content


def load_local_model(model_path, torch, transformers):
    """Select only known local architectures; never download model or kernel code."""
    config = transformers.AutoConfig.from_pretrained(model_path, **{"local_files_only": True})
    if config.model_type == "qwen3_vl":
        cls = transformers.Qwen3VLForConditionalGeneration
    elif config.model_type == "qwen3_5":
        # Access only in the new branch so Transformers 4.57 keeps its old path.
        cls = transformers.Qwen3_5ForConditionalGeneration
    else:
        raise ValueError(f"Unsupported local planner architecture: {config.model_type}")
    processor = transformers.AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = cls.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, attn_implementation="sdpa",
    ).eval()
    return processor, model, config.model_type


def template_inputs(processor, content, architecture):
    kwargs = dict(tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    if architecture == "qwen3_5":
        kwargs["enable_thinking"] = False
    return processor.apply_chat_template([{"role": "user", "content": content}], **kwargs)


def _worker(args):
    import numpy as np
    from PIL import Image
    import torch
    import transformers

    def respond(payload):
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()

    try:
        with contextlib.redirect_stdout(sys.stderr):
            if not torch.cuda.is_available():
                raise RuntimeError("A CUDA GPU is required for this foundation-model sprint.")
            processor, model, architecture = load_local_model(args.model_path, torch, transformers)
            print("[qwen load] " + json.dumps({"architecture": architecture,
                "transformers_version": transformers.__version__, "dtype": "bfloat16",
                "attention": "sdpa", "local_files_only": True,
                "thinking_disabled": architecture == "qwen3_5",
                "kernel_policy": "installed optional kernels or PyTorch reference; no kernel downloads"}), file=sys.stderr, flush=True)
        respond({"ok": True, "ready": True, "architecture": architecture})
    except Exception as exc:
        respond({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1

    for line in sys.stdin:
        try:
            request = json.loads(line)
            with np.load(request["path"], allow_pickle=False) as data:
                content = build_request_content(request, data, args.max_image_edge)
            with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
                inputs = template_inputs(processor, content, architecture).to(model.device)
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                text = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:],
                                              skip_special_tokens=True)[0]
            if request.get("mode") == "refresh":
                respond({"ok": True, **validate_refresh(text)})
            else:
                respond({"ok": True, "instruction": validate_instruction(text)})
        except Exception as exc:
            respond({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-image-edge", type=int, default=512)
    arguments = parser.parse_args()
    if arguments.max_image_edge < 32 or arguments.max_new_tokens < 1:
        parser.error("max-image-edge must be >=32 and max-new-tokens positive")
    raise SystemExit(_worker(arguments))
