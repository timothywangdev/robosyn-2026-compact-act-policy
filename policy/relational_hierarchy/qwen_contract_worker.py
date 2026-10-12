"""Local-only Qwen contract annotations and online JSONL, using one loaded model.

Run with ``python -m policy.relational_hierarchy.qwen_contract_worker``. Both
modes use the same current-only request, prompt, schema, feature encoder and
generation path. This module does not launch a worker unless explicitly run.
"""

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import zipfile

from .contracts import CAMERAS, FEATURE_DIM, VERSION, WIRE_COORDINATE_SYSTEM, build_prompt, encode_contract, parse_contract

MODEL_ID = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
TRANSFORMERS_VERSION = "5.19.0"
MAX_LINE_BYTES = 65536


def validate_request(request):
    expected = {"request_id", "path", "task", "frame_ids"}
    if type(request) is not dict or set(request) != expected:
        raise ValueError(f"Request requires exactly {sorted(expected)}; future/label fields are forbidden")
    for key, limit in (("request_id", 160), ("path", 4096)):
        value = request[key]
        if type(value) is not str or not value.strip() or len(value) > limit or any(ord(c) < 32 for c in value):
            raise ValueError(f"Invalid bounded {key}")
    ids = request["frame_ids"]
    if type(ids) is not dict or set(ids) != set(CAMERAS):
        raise ValueError("frame_ids must identify exactly all three current cameras")
    if any(type(v) is not str or not v.strip() or len(v) > 256 or any(ord(c) < 32 for c in v) for v in ids.values()):
        raise ValueError("Frame IDs must be bounded nonempty strings")
    # Task validation shared with the actual prompt; no task IDs enter features.
    build_prompt(request["task"], [0.0] * 14)
    return request


def prepare_request(request, input_root, max_image_edge=512):
    import numpy as np
    from PIL import Image

    validate_request(request)
    if type(max_image_edge) is not int or not 32 <= max_image_edge <= 2048:
        raise ValueError("max_image_edge must be between32 and2048")
    root = Path(input_root).resolve(strict=True)
    given = Path(request["path"])
    path = (given if given.is_absolute() else root / given).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size > 64 * 1024**2:
        raise ValueError("Observation must be a file below input_root of at most64MiB")
    with zipfile.ZipFile(path) as archive:
        if len(archive.infolist()) != 4 or sum(x.file_size for x in archive.infolist()) > 128 * 1024**2:
            raise ValueError("Observation archive must contain only three images and qpos, within128MiB")
    content, frames = [], {}
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {*CAMERAS, "qpos"}:
            raise ValueError("Observation NPZ must contain exactly three current RGB cameras and qpos; futures forbidden")
        qpos_array = data["qpos"]
        if qpos_array.shape not in ((14,), (1, 14)) or qpos_array.dtype.kind not in "fiu" or not np.isfinite(qpos_array).all():
            raise ValueError("qpos must be finite14D")
        qpos = [float(x) for x in qpos_array.reshape(-1)]
        for name in CAMERAS:
            original = data[name]
            array = original
            if array.ndim == 4 and array.shape[0] == 1:
                array = array[0]
            if array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
                array = np.moveaxis(array, 0, -1)
            if array.ndim != 3 or array.shape[-1] != 3 or min(array.shape[:2]) < 2 or max(array.shape[:2]) > 4096:
                raise ValueError(f"{name} must be current RGB HWC/CHW with edges2..4096")
            if array.dtype == np.uint8:
                pass
            elif array.dtype.kind == "f" and np.isfinite(array).all() and array.min() >= 0 and array.max() <= 1:
                array = np.rint(array * 255).astype(np.uint8)
            else:
                raise ValueError(f"{name} requires uint8[0,255] or finite float[0,1], without hidden renormalization")
            image = Image.fromarray(np.ascontiguousarray(array), mode="RGB")
            image.thumbnail((max_image_edge, max_image_edge))
            frames[name] = {"frame_id": request["frame_ids"][name], "array_shape": list(original.shape),
                            "array_dtype": str(original.dtype), "array_sha256": hashlib.sha256(original.tobytes()).hexdigest(),
                            "presented_size": list(image.size), "presented_rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest()}
            content.extend([{"type": "text", "text": f"CURRENT camera: {name}"}, {"type": "image", "image": image}])
    prompt = build_prompt(request["task"], qpos)
    content.append({"type": "text", "text": prompt})
    audit = {"request_id": request["request_id"], "input_path": str(path), "task": request["task"],
             "frame_ids": request["frame_ids"], "frames": frames, "qpos": qpos, "prompt": prompt,
             "contract_version": VERSION, "feature_dim": FEATURE_DIM,
             "requested_coordinate_system": WIRE_COORDINATE_SYSTEM}
    return content, audit


class LocalContractModel:
    def __init__(self, model_path, max_new_tokens=1024):
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        import torch
        import transformers
        from policy.robosyn_sprint.qwen_planner import load_local_model

        if transformers.__version__ != TRANSFORMERS_VERSION:
            raise RuntimeError(f"Expected transformers{TRANSFORMERS_VERSION}, got{transformers.__version__}")
        if not torch.cuda.is_available():
            raise RuntimeError("Explicit GPU worker launch required; no CPU fallback/model download")
        model_path = Path(model_path).resolve(strict=True)
        config = json.loads((model_path / "config.json").read_text())
        if config.get("model_type") != "qwen3_5":
            raise ValueError("This contract worker requires the pinned Qwen3.5-9B architecture")
        self.processor, self.model, architecture = load_local_model(str(model_path), torch, transformers)
        self.processor.tokenizer.padding_side = "left"
        self.torch, self.max_new_tokens = torch, max_new_tokens
        self.metadata = {"model_id": MODEL_ID, "declared_revision": MODEL_REVISION,
                         "weight_identity": "launch coordinator must verify restored pinned model inventory",
                         "model_path": str(model_path), "config_sha256": hashlib.sha256((model_path / "config.json").read_bytes()).hexdigest(),
                         "architecture": architecture, "torch": torch.__version__, "transformers": transformers.__version__,
                         "dtype": "bfloat16", "attention": "sdpa", "thinking": False,
                         "do_sample": False, "local_files_only": True, "max_new_tokens": max_new_tokens}

    def generate(self, contents):
        messages = [[{"role": "user", "content": content}] for content in contents]
        with self.torch.inference_mode():
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                return_tensors="pt", padding=True, enable_thinking=False,
            ).to(self.model.device)
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            return self.processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def _write(sink, payload):
    sink.write(json.dumps(payload, ensure_ascii=True, allow_nan=False) + "\n")
    sink.flush()


def process_batch(requests, model, input_root, output, audit_sink, *, max_image_edge=512, online=False):
    prepared, audits = [], []
    for request in requests:
        started = time.monotonic()
        fallback = {"request_id": request.get("request_id") if isinstance(request, dict) else None,
                    "request": request, "contract_version": VERSION, "feature_dim": FEATURE_DIM}
        try:
            content, record = prepare_request(request, input_root, max_image_edge)
        except (TimeoutError, KeyboardInterrupt):
            raise
        except Exception as error:
            record = {**fallback, "ok": False, "stage": "input", "error": f"{type(error).__name__}: {error}"}
            _write(audit_sink, record); _write(output, record)
            if online:
                _write(sys.stdout, record)
            continue
        record["started_monotonic"] = started
        _write(audit_sink, {**record, "event": "generation_started"})
        prepared.append(content); audits.append(record)
    if not prepared:
        return
    try:
        with contextlib.redirect_stdout(sys.stderr):
            texts = model.generate(prepared)
        if len(texts) != len(audits):
            raise RuntimeError("Model response count differs from request count")
    except (TimeoutError, KeyboardInterrupt):
        raise
    except Exception as error:
        texts = [None] * len(audits)
        generation_error = f"{type(error).__name__}: {error}"
    for record, text in zip(audits, texts, strict=True):
        record["raw_response"] = text
        record["elapsed_seconds"] = time.monotonic() - record.pop("started_monotonic")
        try:
            if text is None:
                raise RuntimeError(generation_error)
            contract = parse_contract(text)
            features, valid = encode_contract(contract)
            record.update(ok=True, stage="complete", contract=contract, goal_features=features, goal_valid=valid,
                          grounding_review="unreviewed_model_annotation")
        except (TimeoutError, KeyboardInterrupt):
            raise
        except Exception as error:
            record.update(ok=False, stage="generation" if text is None else "parse", error=f"{type(error).__name__}: {error}")
        _write(audit_sink, record); _write(output, record)
        if online:
            _write(sys.stdout, record)


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--requests", help="Offline JSONL request file")
    mode.add_argument("--worker", action="store_true", help="Online request/response JSONL over stdin/stdout")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True, help="New annotations JSONL path; never overwritten")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-image-edge", type=int, default=512)
    parser.add_argument("--max-requests", type=int, default=20000)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    args = parser.parse_args(argv)
    if not 1 <= args.batch_size <= 8 or (args.worker and args.batch_size != 1):
        parser.error("batch-size must be1..8; online worker uses1")
    if not 32 <= args.max_image_edge <= 2048 or not 64 <= args.max_new_tokens <= 2048:
        parser.error("Image edge must be32..2048; new tokens64..2048")
    if not 1 <= args.max_requests <= 100000 or not 1 <= args.timeout_seconds <= 86400:
        parser.error("Request/lifetime bounds are required")
    def deadline(_signal, _frame):
        raise TimeoutError("Qwen contract worker lifetime cap reached")
    previous = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.timeout_seconds)
    count, seen, batch = 0, set(), []
    try:
        with contextlib.ExitStack() as stack:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output = stack.enter_context(output_path.open("x"))
            audit = stack.enter_context(Path(str(output_path) + ".audit.jsonl").open("x"))
            source = sys.stdin if args.worker else stack.enter_context(open(args.requests))
            with contextlib.redirect_stdout(sys.stderr):
                model = LocalContractModel(args.model_path, args.max_new_tokens)
            metadata = {"event": "ready", "ok": True, "contract_version": VERSION, "feature_dim": FEATURE_DIM,
                        "model": model.metadata, "batch_size": args.batch_size,
                        "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        "contract_source_sha256": hashlib.sha256(Path(__file__).with_name("contracts.py").read_bytes()).hexdigest(),
                        "loader_source_sha256": hashlib.sha256((Path(__file__).parents[1] / "robosyn_sprint/qwen_planner.py").read_bytes()).hexdigest()}
            _write(audit, metadata)
            if args.worker:
                _write(sys.stdout, metadata)
            while True:
                line = source.readline(MAX_LINE_BYTES + 1)
                if not line:
                    break
                if len(line.encode("utf-8")) > MAX_LINE_BYTES:
                    raise ValueError("Request line exceeds64KiB; refusing truncated input")
                count += 1
                if count > args.max_requests:
                    raise ValueError("Request count cap reached")
                request = json.loads(line)
                request_id = request.get("request_id") if isinstance(request, dict) else None
                if isinstance(request_id, str):
                    if request_id in seen:
                        raise ValueError(f"Duplicate request_id {request_id}; no automatic retry")
                    seen.add(request_id)
                batch.append(request)
                if len(batch) == args.batch_size:
                    process_batch(batch, model, args.input_root, output, audit,
                                  max_image_edge=args.max_image_edge, online=args.worker)
                    batch = []
            if batch:
                process_batch(batch, model, args.input_root, output, audit, max_image_edge=args.max_image_edge)
            _write(audit, {"event": "finished", "requests_seen": count})
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
