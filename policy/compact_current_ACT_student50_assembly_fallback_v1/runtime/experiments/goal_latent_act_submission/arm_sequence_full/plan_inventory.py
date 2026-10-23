"""CPU-only additive ARM12 payload plan; never materialize or infer a model.

Hash only new small sources/proofs/weights. Existing large assets are matched to
immutable published release inventories; this is not a new full-store hash proof.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import stat

DEPLOYMENT = "goal-latent-act-commanded50-full63000-temporal-arm12-deployment-v1"
CAMPAIGN_SHA = "7460621966d54bc9b1f0c70480183c9cfa76b768bf376853b5735483aba5a1e2"
PAIR_SHA = "2804140fac4487cd4ac7cbdd98b9e0b9eca432466ffff60c032587ee6c760d88"
MODES = ("parent", "goal", "goal_free")


def ref(path, expected=None):
    path = pathlib.Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("Regular exact source required: " + str(path))
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("Planning must not scan large base assets: " + str(path))
    raw = path.read_bytes()
    result = dict(path=str(path.resolve()), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                  mode=stat.S_IMODE(path.stat().st_mode))
    if expected and any(result[k] != expected[k] for k in ("bytes", "sha256")):
        raise ValueError("Bound source/proof bytes differ: " + str(path))
    return result


def read_bound(value):
    actual = ref(value["path"], value)
    return json.loads(pathlib.Path(actual["path"]).read_text()), actual


def plan(preparation, published_code, output):
    preparation = pathlib.Path(preparation).resolve(strict=True)
    published_code = pathlib.Path(published_code).resolve(strict=True)
    output = pathlib.Path(output).absolute()
    if output.exists():
        raise ValueError("Fresh additive plan directory required")
    campaign_path = preparation / "exploratory30-CPU-packet-v3/campaign.json"
    campaign_ref = ref(campaign_path)
    if campaign_ref["sha256"] != CAMPAIGN_SHA:
        raise ValueError("Exact root-admitted ARM campaign required")
    campaign = json.loads(campaign_path.read_text())
    entry = json.loads((published_code / "entry-pins.json").read_text())
    if entry.get("primary_variant") is not None:
        raise ValueError("Preserve unselected primary variant")
    prior_maps = []; old_digests = {}; old_paths = {}
    for path in sorted((published_code / "releases").glob("*.json")):
        release = json.loads(path.read_text()); prior_maps.append(ref(path))
        if not release.get("complete_offline_payload"):
            continue
        for name, row in release["files"].items():
            if row["sha256"] in old_digests and old_digests[row["sha256"]]["bytes"] != row["bytes"]:
                raise ValueError("Published digest inventory conflicts")
            old_digests[row["sha256"]] = dict(row, release=release["variant"])
            old_paths.setdefault(name, []).append(dict(row, release=release["variant"]))
    sources = {}; small = {}; shared = {}; configurations = []

    def add_small(value, role):
        actual = ref(value["path"], value)
        digest = actual["sha256"]
        if digest not in small:
            small[digest] = dict(actual, roles=[])
        if role not in small[digest]["roles"]:
            small[digest]["roles"].append(role)
        return actual

    def shared_base(value, role):
        row = old_digests.get(value["sha256"])
        if not row or row["bytes"] != value["bytes"]:
            raise ValueError("Base asset missing from existing immutable published payload: " + role)
        source = pathlib.Path(value["path"])
        if not source.is_file() or source.is_symlink() or source.stat().st_size != value["bytes"]:
            raise ValueError("Current admitted base asset type/size differs: " + role)
        shared.setdefault(value["sha256"], dict(bytes=value["bytes"], sha256=value["sha256"],
            published=row, current_local_path=str(source), validation="published digest/bytes plus local ordinary-file type/size; not a new whole-base hash scan"))

    pair_ref = None
    for record in campaign["plans"]:
        mode, task = record["mode"], record["task"]
        if mode not in MODES:
            raise ValueError("Only typed parent/role/control branches")
        cfg_path = campaign_path.parent / (mode + "-" + task + "-deployment.json")
        cfg_ref = ref(cfg_path); cfg = json.loads(cfg_path.read_text())
        if not (cfg["schema"] == DEPLOYMENT and cfg["task"] == task and cfg["mode"] == mode and
                cfg["native_chunk_generated"] == cfg["prefix_executed"] == 50 and cfg["motor_seed"] == 17 and
                cfg["act_core_mode"] == "eager" and cfg["act_preprocess_mode"] == "native_every_action" and
                cfg["native_grippers_bitexact"] is True and cfg["arm_correction_bound_rad"] == .15):
            raise ValueError("No older schema or protocol may be relabeled as ARM12")
        configurations.append(dict(task=task, mode=mode, original_deployment=cfg_ref))
        source_root = pathlib.Path(cfg["source_root"])
        for name, digest in cfg["source_sha256"].items():
            actual = ref(source_root / name)
            if actual["sha256"] != digest:
                raise ValueError("Frozen runtime/science source differs: " + name)
            if name in sources and sources[name]["sha256"] != digest:
                raise ValueError("Branch source versions conflict: " + name)
            sources[name] = dict(actual, shared_existing_digest=digest in old_digests)
        for group in ("act_files", "qwen_files"):
            for name, value in cfg[group].items():
                shared_base(value, task + "/" + group + "/" + name)
        for name, value in cfg["assets"].items():
            add_small(value, "runtime_asset:" + name)
        if mode != "parent":
            value = cfg["assets"]["training_pair"]
            if value["sha256"] != PAIR_SHA:
                raise ValueError("Exact actually trained paired qualification required")
            pair_ref = value
    if len(configurations) != 30 or len({(r["mode"], r["task"]) for r in configurations}) != 30:
        raise ValueError("Exactly all3/all10 explicit original deployments required")
    pair, _ = read_bound(pair_ref)
    for name in ("original_pair", "original_wait", "supervisor", "independent_terminal", "prepared", "head_contract"):
        add_small(pair[name], "arm_training:" + name)
    for mode, branch in pair["branches"].items():
        add_small(branch["contract"], mode + ":contract")
        add_small(branch["training_receipt"], mode + ":training_receipt")
    prepared, _ = read_bound(pair["prepared"])
    for name in ("original_training", "original_outer_wait"):
        add_small(prepared[name], "historical_full63000:" + name)
    original, _ = read_bound(prepared["original_training"])
    for run in original["runs"]:
        add_small(run["child_receipt"], "historical_full63000:" + run["name"])
    missing_source_pins = set(pair["source_pins"]) - set(sources)
    if missing_source_pins:
        raise ValueError("Scientific sources absent from frozen actor closure")
    new_weights = [r for r in small.values() if pathlib.Path(r["path"]).suffix == ".safetensors"
                   and r["sha256"] not in old_digests]
    new_small = [r for r in small.values() if r["sha256"] not in old_digests]
    new_sources = [r for r in sources.values() if not r["shared_existing_digest"]]
    report = dict(schema="goal-latent-act-full63000-arm12-additive-portable-plan-v1",
        status="exact_source_and_small_asset_inventory_prepared_not_a_complete_release",
        campaign=campaign_ref, published_entry_pins=ref(published_code / "entry-pins.json"),
        unchanged_old_release_maps=prior_maps, primary_variant=None,
        robot_promotion_criterion_passed=False, robot_performance_selected=False,
        proposed_variants={"full63000-arm-role-v1":"goal", "full63000-arm-control-v1":"goal_free", "full63000-arm-parent-v1":"parent"},
        original_configurations=configurations, source_inventory=sources, small_assets_and_scientific_proofs=small,
        reused_large_base_inventory=shared,
        counts=dict(configurations=len(configurations), source_members=len(sources), new_source_members=len(new_sources),
            small_unique_assets=len(small), new_small_unique_assets=len(new_small), new_weight_files=len(new_weights),
            new_weight_bytes=sum(r["bytes"] for r in new_weights), new_small_bytes=sum(r["bytes"] for r in new_small),
            new_source_bytes=sum(r["bytes"] for r in new_sources), large_shared_bytes=sum(r["bytes"] for r in shared.values())),
        remaining_steps=[
            "New typed portable schema, template/manifest relocation, exact qualify() regeneration and SHA-keyed historical JSON aliases; historical receipts and checkpoint.json bytes unchanged",
            "Sibling-preserving head50/highlevel.safetensors+checkpoint.json and role/control arm-sequence.safetensors+checkpoint.json layout; no FIT arrays, future labels or optimizer data at inference",
            "Reuse exact published offline SIM Torch2.7.1+cu128 and Qwen TF5.19 environments, complete immutable SDK/evaluator/assets and qualified cleanup bootstrap/hook",
            "New installer/runner and organizer visible get_model/eval/reset/close bridge with exact source/PYTHONPATH/cwd/command routing and physically short isolated writable simulator caches",
            "Fresh CPU relocated qualification, actual FILE/SDK prelude imports and all30 native argument parity; existing dependency-install proofs must be labeled as reused, not rerun",
            "Complete admitted native30 original waits/drain, accepted-seed pairing and role/control result assessment before any candidate selection or publication authorization",
        ],
        runtime_training_path_dependencies=dict(
            location="original pair, parent waits, independent terminal, prepared receipt, head contract, both arm contracts and original scientific child receipts",
            route="qualification.py explicitly accepts byte-preserving historical_reference_aliases; regenerate the typed qualified-training-pair using transported paths",
            source_loader="GoalArmSequence common.ROOT verifies source_pins relative to its reconstructed package root",
            current_admission="Literal exploratory-after-failed-promotion admission remains mandatory; do not relabel current actor as promoted runtime",
            excluded="No 63000 FIT tensor cache, monitor arrays, original GPU training models/optimizer state, hidden outcomes or future teacher inputs needed by native inference"),
        models_constructed=0, CUDA_initialized=False, GPU_queries=0, GPU_launches=0, external_uploads=0)
    output.mkdir(parents=True)
    (output / "plan.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "counts": report["counts"],
                      "plan": str(output / "plan.json")}, sort_keys=True))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ("preparation", "published-code", "output"):
        parser.add_argument("--" + name, required=True)
    plan(**vars(parser.parse_args()))
