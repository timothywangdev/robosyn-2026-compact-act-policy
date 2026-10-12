#!/usr/bin/env python3
"""One admitted GPU; explicit1/2 CUDA processes and fresh-lineage CPU helpers.

--validate-only checks metadata without querying or launching a GPU workload.
The root admission owner separately authorizes --execute. No job retry occurs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import threading
import time
import traceback

SCHEMA = "coupled_goal_ACT_all14_local32_supervisor_v1"
PARENT_SHA256 = "0281eacb5d8bd267e769395b20e71e7505bade69de2dff7a0da6e0091a61fb4a"
OWNER_KEY = "VPP2_GPU_SUPERVISOR_OWNER"
GIB = 1 << 30


def identity(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return dict(pid=pid, birth_ticks=int(fields[19]), parent_pid=int(fields[1]),
                group=int(fields[2]), session=int(fields[3]), state=fields[0])


def same_identity(a, b):
    return all(a[key] == b[key] for key in ("pid", "birth_ticks", "group", "session"))


def group_members(group):
    # Read only stat identities to locate this newly created group. Never read
    # other processes' arguments/environments or use historical ownership data.
    rows = []
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        try:
            row = identity(int(path.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
        if row["group"] == group and row["state"] not in ("Z", "X"):
            rows.append(row)
    return rows


class OwnershipLedger:
    """Fresh setsid lineage evidence; never infer ownership from a numeric PID."""
    def __init__(self, leader, token):
        self.leader = leader.copy()
        self.token = token
        self.events = []
        self.proven = {}
        self._event_keys = set()
        if leader["pid"] != leader["group"] or leader["pid"] != leader["session"]:
            raise RuntimeError("Leader must be the actual fresh setsid child")
        self.capture(leader, "fresh_Popen_setsid_leader", "not_inspected", [leader])

    def observe(self, row, environment="not_inspected"):
        key = (row["pid"], row["birth_ticks"], "observed", environment, row["parent_pid"])
        if key not in self._event_keys:
            self._event_keys.add(key)
            self.events.append(dict(identity=row.copy(), proof="observed_not_yet_verified", environment=environment, lineage=[], monotonic_seconds=time.monotonic()))

    def capture(self, row, method, environment, lineage):
        key = (row["pid"], row["birth_ticks"])
        self.proven[key] = dict(identity=row.copy(), lineage=[x.copy() for x in lineage], method=method)
        event_key = (*key, method, environment, row["parent_pid"])
        if event_key not in self._event_keys:
            self._event_keys.add(event_key)
            self.events.append(dict(identity=row.copy(), proof=method, environment=environment,
                                    lineage=[x.copy() for x in lineage], monotonic_seconds=time.monotonic()))

    def lineage(self, row):
        key = (row["pid"], row["birth_ticks"])
        known = self.proven.get(key)
        if known and same_identity(row, known["identity"]):
            return "previously_captured_exact_identity_lineage", known["lineage"]
        chain = [row.copy()]
        current = row
        seen = {key}
        for _ in range(64):
            if same_identity(current, self.leader):
                return "fresh_live_parent_chain_to_setsid_leader", chain
            parent = identity(current["parent_pid"])
            pkey = (parent["pid"], parent["birth_ticks"])
            if pkey in seen or parent["birth_ticks"] > current["birth_ticks"]:
                raise RuntimeError("Invalid or recycled parent lineage")
            if (parent["group"] != self.leader["group"] or parent["session"] != self.leader["session"]
                or parent["birth_ticks"] < self.leader["birth_ticks"]):
                raise RuntimeError("Parent chain leaves newly isolated session/group")
            parent_after = identity(parent["pid"])
            current_after = identity(current["pid"])
            if (not same_identity(parent, parent_after) or not same_identity(current, current_after)
                or current_after["parent_pid"] != current["parent_pid"]):
                raise RuntimeError("Parent lineage changed during inspection")
            chain.append(parent.copy()); seen.add(pkey); current = parent
            known = self.proven.get(pkey)
            if known and same_identity(parent, known["identity"]):
                return "fresh_live_chain_to_previously_proven_ancestor", chain + known["lineage"][1:]
        raise RuntimeError("Parent lineage exceeds bounded64-process inspection")


def verified_member(row, leader, token, *, ledger=None):
    if ledger is not None: ledger.observe(row)
    if row["group"] != leader["group"] or row["session"] != leader["session"] or row["birth_ticks"] < leader["birth_ticks"]:
        raise RuntimeError("Unverified identity in newly owned group")
    if row["pid"] == leader["pid"] and not same_identity(row, leader):
        raise RuntimeError("Owned leader PID was reused")
    for attempt in range(3):
        try:
            before = identity(row["pid"])
            if before["state"] in ("Z", "X"):
                return False
            if not same_identity(row, before):
                raise RuntimeError("Owned member identity changed")
            try:
                environment = Path(f"/proc/{row['pid']}/environ").read_bytes().split(b"\0")
                values = [x.split(b"=",1)[1] for x in environment if x.startswith((OWNER_KEY+"=").encode())]
                if values and (len(values) != 1 or values[0] != token.encode()):
                    if ledger is not None: ledger.observe(before, "readable_explicit_wrong_token")
                    raise RuntimeError("Accessible explicit ownership token differs")
                readability = "readable_fresh_token" if values else "readable_owner_key_absent"
            except PermissionError:
                readability = "PermissionError"
            except (FileNotFoundError, ProcessLookupError):
                return False
            if ledger is None:
                if readability != "readable_fresh_token":
                    raise RuntimeError("Missing token requires independently captured fresh lineage")
                method, lineage = "fresh_token_and_exact_group_session_birth", [before]
            else:
                method, lineage = ledger.lineage(before)
            after = identity(row["pid"])
            if after["state"] in ("Z", "X"):
                return False
            if not same_identity(before, after) or before["parent_pid"] != after["parent_pid"]:
                raise RuntimeError("Owned member changed during ownership inspection")
            if ledger is not None:
                ledger.capture(after, method, readability, lineage)
            return True
        except (FileNotFoundError, ProcessLookupError):
            try:
                current = identity(row["pid"])
            except (FileNotFoundError, ProcessLookupError):
                return False
            if not same_identity(row, current):
                raise RuntimeError("Process changed while rechecking vanished ancestor")
            if attempt == 2:
                raise RuntimeError("Live process lineage could not be completely inspected")
        if attempt < 2:
            time.sleep(.02)
    raise RuntimeError("Live owned member inspection failed")


def signal_owned(row, leader, token, sig, *, ledger=None):
    if not verified_member(row, leader, token, ledger=ledger):
        return None
    try:
        fd = os.pidfd_open(row["pid"], 0)
    except ProcessLookupError:
        return None
    try:
        current = identity(row["pid"])
        if not same_identity(row, current):
            raise RuntimeError("Identity changed before PIDFD signal")
        if not verified_member(current, leader, token, ledger=ledger):
            return None
        signal.pidfd_send_signal(fd, sig)
        return dict(identity=current, signal=int(sig), pidfd_identity_pinned=True)
    except (FileNotFoundError, ProcessLookupError):
        return None
    finally:
        os.close(fd)


def drain_group(leader, token, grace_seconds=2, *, ledger=None):
    started = time.monotonic()
    before = group_members(leader["group"])
    sent, refused = [], []
    for sig, duration in ((signal.SIGTERM, grace_seconds), (signal.SIGKILL, 2)):
        # Capture complete lineage before terminating a parent can orphan helpers.
        verified = []
        for row in group_members(leader["group"]):
            try:
                if verified_member(row, leader, token, ledger=ledger):
                    verified.append(row)
            except Exception as error:
                refused.append(dict(identity=row, stage="pre_signal", error=repr(error)))
        for row in verified:
            try:
                event = signal_owned(row, leader, token, sig, ledger=ledger)
                if event: sent.append(event)
            except Exception as error:
                refused.append(dict(identity=row, stage="pidfd_signal", error=repr(error)))
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            if not group_members(leader["group"]): break
            time.sleep(.05)
    after = group_members(leader["group"])
    return dict(before=before, signals=sent, signal_refusals=refused, after=after,
                drained=not after, seconds=time.monotonic()-started,
                natural=not before and not sent and not after)


class GPUProcessInspectionRace(RuntimeError):
    pass


def verify_gpu_processes(snapshot, leader, token, ledger, maximum):
    # NVIDIA reports CUDA context rows; Torch and DexSim may have multiple
    # contexts in ONE process. Count independently verified live identities,
    # retaining raw context rows in the job journal below.
    raw_pids = snapshot["compute_pids"]
    if any(type(pid) is not int or pid <= 0 for pid in raw_pids):
        raise RuntimeError("CUDA context rows require positive numeric PIDs")
    pids = list(dict.fromkeys(raw_pids))
    result = []
    for pid in pids:
        try:
            row = identity(pid)
        except (FileNotFoundError, ProcessLookupError):
            raise GPUProcessInspectionRace("GPU compute PID has no live independently verified identity")
        if not verified_member(row, leader, token, ledger=ledger):
            raise GPUProcessInspectionRace("GPU compute process vanished during verification")
        result.append(row)
        if len(result) > maximum:
            raise RuntimeError("Per-job explicitly admitted CUDA process count exceeded")
    return result


def completed_child_gpu_drain(child, leader, token, ledger, maximum, gpu_reader, uuid):
    """Resolve stale NVIDIA rows only after independent completed-child proof.

    No live PID is excused; no signals are sent. Ten refreshes and at most
    ten50ms delays are allowed, inside the original per-job wall timer.
    """
    code = child.poll()
    if code is None:
        raise GPUProcessInspectionRace("GPU inspection race while original child remains active")
    observations = []
    for attempt in range(10):
        if child.poll() != code:
            raise RuntimeError("Original child completion changed during GPU drain inspection")
        before = group_members(leader["group"])
        if before:
            raise RuntimeError("Live owned group survives completed child GPU inspection race")
        snapshot = gpu_reader(uuid)
        after = group_members(leader["group"])
        event = dict(attempt=attempt+1, original_child_exit_code=code,
                     owned_group_before=before, gpu_snapshot=snapshot, owned_group_after=after)
        observations.append(event)
        if after or child.poll() != code:
            raise RuntimeError("Owned group or original child changed during independent GPU refresh")
        if not snapshot["compute_pids"]:
            # Bracket a second independent empty GPU query; one empty query
            # alone must not admit a newly visible foreign/live GPU process.
            confirmation = gpu_reader(uuid)
            final_members = group_members(leader["group"])
            event.update(confirming_gpu_snapshot=confirmation, final_owned_group=final_members)
            if final_members or child.poll() != code:
                raise RuntimeError("Owned group or child changed during confirming GPU query")
            if not confirmation["compute_pids"]:
                return confirmation, dict(status="completed_child_independently_group_and_GPU_drained",
                    original_child_exit_code=code, observations=observations,
                    signals=[], child_completed=True, independent_GPU_empty=True,
                    independent_owned_group_empty=True)
            snapshot = confirmation
        # Retain the original strict positive-PID/count/ownership checks. A
        # live verified owner is also refused here: this route requires drain.
        try:
            live = verify_gpu_processes(snapshot, leader, token, ledger, maximum)
        except GPUProcessInspectionRace as error:
            event["stale_row_inspection_error"] = str(error)
        else:
            if live:
                raise RuntimeError("Live GPU process remains after completed child")
            raise RuntimeError("GPU refresh changed unexpectedly")
        time.sleep(.05)
    raise GPUProcessInspectionRace("Completed child NVIDIA rows did not clear in ten bounded refreshes")


def active_known_gpu_race(child, leader, token, ledger, maximum, gpu_reader,
                          memory_reader, uuid, snapshot, previous_gpu_identities):
    """At most1second: known departed rows plus freshly verified live owners.

    Two fresh consistent live-only snapshots resolve mixed ACT/Qwen cleanup.
    Every refresh retains memory/group checks; unknown/reused/foreign fails.
    """
    if child.poll() is not None:
        return completed_child_gpu_drain(child, leader, token, ledger, maximum, gpu_reader, uuid)
    known = {}
    for row in previous_gpu_identities:
        if row["pid"] in known and not same_identity(row, known[row["pid"]]):
            raise RuntimeError("Previously GPU-verified PID has conflicting birth identities")
        known[row["pid"]] = row
    observations = []
    started = time.monotonic()

    def inspect_rows(view):
        pids = view["compute_pids"]
        if any(type(pid) is not int or pid <= 0 for pid in pids):
            raise RuntimeError("Invalid CUDA rows during active-child race")
        live, departed = [], []
        for pid in dict.fromkeys(pids):
            prior = known.get(pid)
            captured = ledger.proven.get((pid, prior["birth_ticks"])) if prior else None
            if prior is None or captured is None or not same_identity(prior, captured["identity"]):
                raise GPUProcessInspectionRace("Active child has unknown or never GPU-verified CUDA PID")
            try:
                current = identity(pid)
            except (FileNotFoundError, ProcessLookupError):
                departed.append(prior.copy()); continue
            if not same_identity(prior, current):
                raise RuntimeError("Previously verified GPU PID birth/group/session changed")
            if current["state"] in ("Z", "X"):
                departed.append(current); continue
            if not verified_member(current, leader, token, ledger=ledger):
                try:
                    after = identity(pid)
                except (FileNotFoundError, ProcessLookupError):
                    departed.append(prior.copy()); continue
                if not same_identity(prior, after):
                    raise RuntimeError("Previously verified GPU PID birth/group/session changed")
                if after["state"] in ("Z", "X"):
                    departed.append(after); continue
                raise GPUProcessInspectionRace("Live known GPU owner cannot be completely verified")
            try:
                after = identity(pid)
            except (FileNotFoundError, ProcessLookupError):
                departed.append(prior.copy()); continue
            if not same_identity(prior, after):
                raise RuntimeError("Previously verified GPU PID birth/group/session changed")
            if after["state"] in ("Z", "X"):
                departed.append(after); continue
            live.append(after)
            if len(live) > maximum:
                raise RuntimeError("Per-job explicitly admitted live CUDA process count exceeded")
        return live, departed

    inspect_rows(snapshot)
    for attempt in range(20):
        if time.monotonic() - started > 1:
            break
        if child.poll() is not None:
            gpu, proof = completed_child_gpu_drain(child, leader, token, ledger, maximum, gpu_reader, uuid)
            proof["active_known_cleanup_observations"] = observations
            return gpu, proof
        memory = memory_reader()
        if memory["headroom_bytes"] < 8 * GIB:
            raise MemoryError("Live memory headroom fell below8GiB during GPU race")
        members = group_members(leader["group"])
        for member in members:
            verified_member(member, leader, token, ledger=ledger)
        view = gpu_reader(uuid)
        live, departed = inspect_rows(view)
        after_members = group_members(leader["group"])
        for member in after_members:
            verified_member(member, leader, token, ledger=ledger)
        event = dict(attempt=attempt+1, gpu_snapshot=view, memory=memory,
                     group_members=members, group_members_after=after_members,
                     verified_live_gpu_identities=live, known_departed_gpu_identities=departed,
                     original_child_poll=child.poll())
        observations.append(event)
        if not departed and len(observations) >= 2:
            previous = observations[-2]
            left = sorted(live, key=lambda row:row["pid"])
            right = sorted(previous["verified_live_gpu_identities"], key=lambda row:row["pid"])
            consistent = len(left)==len(right) and all(same_identity(a,b) for a,b in zip(left,right))
            if not previous["known_departed_gpu_identities"] and consistent:
                if time.monotonic() - started > 1:
                    break
                if child.poll() is not None:
                    gpu, proof = completed_child_gpu_drain(child, leader, token, ledger, maximum, gpu_reader, uuid)
                    proof["active_known_cleanup_observations"] = observations
                    return gpu, proof
                return view, dict(status="active_child_known_GPU_exit_race_two_consistent_live_only_queries",
                    observations=observations, previously_verified_gpu_identities=previous_gpu_identities,
                    verified_live_gpu_identities=live, signals=[], active_child_permitted=True,
                    independent_GPU_empty=not live, independent_departed_GPU_rows_empty=True,
                    original_child_exit_code=None)
        time.sleep(.05)
    raise GPUProcessInspectionRace("Known departed GPU rows did not clear within bounded1second")


def memory_snapshot():
    host = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(("MemAvailable:", "MemTotal:")):
            key, value, _ = line.split()
            host[key.rstrip(":")] = int(value) * 1024
    relative = next((line.split(":", 2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::")), None)
    if relative is None:
        raise RuntimeError("This supervisor requires cgroup-v2 memory accounting")
    mount = Path("/sys/fs/cgroup")
    leaf = mount / relative.lstrip("/")
    if not (leaf / "memory.current").is_file():
        leaf = mount  # Container cgroup namespaces can expose their root here.
    candidates, finite = [host["MemAvailable"]], []
    current = leaf
    while current.is_relative_to(mount):
        maxfile, usedfile = current / "memory.max", current / "memory.current"
        if maxfile.is_file() and usedfile.is_file():
            limit = maxfile.read_text().strip()
            used = int(usedfile.read_text().strip())
            if limit != "max":
                available = max(0, int(limit) - used)
                candidates.append(available)
                finite.append(dict(path=str(current), limit_bytes=int(limit), used_bytes=used, headroom_bytes=available))
        if current == mount:
            break
        current = current.parent
    return dict(headroom_bytes=min(candidates), host_available_bytes=host["MemAvailable"],
                finite_cgroup_limits=finite, cgroup_leaf=str(leaf))


def gpu_snapshot(uuid):
    command = ["nvidia-smi", "-i", uuid, "--query-gpu=uuid,memory.total,memory.free", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
    rows = list(csv.reader(io.StringIO(result.stdout)))
    if len(rows) != 1 or rows[0][0].strip().lower() != uuid.lower():
        raise RuntimeError("Physical GPU UUID query differs")
    compute = subprocess.run(["nvidia-smi", "-i", uuid, "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=10)
    processes = []
    for row in csv.reader(io.StringIO(compute.stdout)):
        if not row:
            continue
        if len(row) != 2 or row[0].strip().lower() != uuid.lower():
            raise RuntimeError("Unexpected compute-process GPU identity")
        processes.append(int(row[1].strip()))
    return dict(uuid=uuid, total_bytes=int(rows[0][1].strip()) << 20,
                free_bytes=int(rows[0][2].strip()) << 20, compute_pids=processes)


def validate_plan(plan):
    floor = plan.get("admission_memory_floor_gib", 32)
    if type(floor) is not int or floor != 32:
        raise ValueError("Temporal-arm local admission memory floor must be32GiB")
    if plan.get("schema") != SCHEMA or not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", str(plan.get("gpu_uuid", ""))):
        raise ValueError("Exact supervisor schema and physical GPU UUID required")
    output = Path(plan["output_root"])
    if not output.is_absolute() or output != output.resolve() or output.exists() or output.is_symlink():
        raise ValueError("Fresh absolute supervisor output required")
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 2 or [j.get("name") for j in jobs] != ["role_goal", "state_action_control"]:
        raise ValueError("Exactly two declared role/control sequential original children required")
    names, roots = set(), set()
    for job in jobs:
        name = job.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", name) or name in names:
            raise ValueError("Unique simple job name required")
        names.add(name)
        argv = job.get("argv")
        if not isinstance(argv, list) or not argv or any(not isinstance(x, str) or "\0" in x for x in argv):
            raise ValueError("Exact string argv array required; no shell command")
        if not Path(argv[0]).is_absolute() or not Path(argv[0]).is_file():
            raise ValueError("Existing absolute executable path required")
        cwd, target = Path(job["cwd"]), Path(job["output_root"])
        if not cwd.is_absolute() or not cwd.is_dir() or not target.is_absolute() or target != target.resolve() or target.exists() or target.is_symlink():
            raise ValueError("Existing absolute cwd and fresh absolute job output required")
        if target == output or not target.is_relative_to(output) or target in roots or any(target.is_relative_to(x) or x.is_relative_to(target) for x in roots):
            raise ValueError("Distinct non-overlapping job outputs under fresh supervisor root required")
        if target == output / "logs" or target.is_relative_to(output / "logs"):
            raise ValueError("Job output overlaps supervisor logs")
        roots.add(target)
        if argv.count("--output") != 1 or argv.index("--output") + 1 >= len(argv) or argv[argv.index("--output") + 1] != str(target):
            raise ValueError("Exact argv --output must match fresh job output")
        if (argv[1:4] != ["-B", "-m", "experiments.coupled_goal_act_all14.train"]
                or argv.count("--execute") != 1 or argv.count("--prepared") != 1
                or argv.count("--prepared-sha256") != 1
                or ("--state-action-only" in argv) != (name == "state_action_control")):
            raise ValueError("Exact new coupled all14 trainer and matching causal control required")
        maximum = job.get("max_owned_gpu_processes")
        if type(maximum) is not int or maximum != 1:
            raise ValueError("Exactly one CUDA process per temporal-arm branch required")
        seconds = job.get("wall_seconds")
        if type(seconds) not in (int, float) or not 0 < seconds <= 1250:
            raise ValueError("Each job requires a positive wall cap at most1700seconds")
    return plan


def run_plan(plan, *, memory_reader=memory_snapshot, gpu_reader=gpu_snapshot, cpu_fixture=False):
    validate_plan(plan)
    admission_bytes = plan.get("admission_memory_floor_gib", 32) * GIB
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Supervisor must own main-thread signal handling")
    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise RuntimeError("Supervisor requires exclusive wall timer")
    def interrupted(sig, frame):
        if sig == signal.SIGALRM:
            raise TimeoutError("Per-job wall cap exceeded")
        raise InterruptedError(f"Supervisor received signal {sig}")
    output = Path(plan["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    logs = output / "logs"; logs.mkdir()
    started = time.monotonic()
    report = dict(schema=SCHEMA, status="preparing", controller=identity(os.getpid()),
                  implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  derived_parent_sha256=PARENT_SHA256,
                  plan=plan, cpu_fixture=cpu_fixture, gpu_workload_qualification=False,
                  jobs=[], children_launched=0, exit_code=1)
    def write():
        report["elapsed_seconds"] = time.monotonic() - started
        temp = output / "receipt.json.tmp"
        temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temp.replace(output / "receipt.json")
    write()
    try:
        for sig in previous_handlers:
            signal.signal(sig, interrupted)
        initial_memory = memory_reader(); initial_gpu = gpu_reader(plan["gpu_uuid"])
        report.update(initial_memory=initial_memory, initial_gpu=initial_gpu)
        if initial_memory["headroom_bytes"] < admission_bytes or initial_gpu["compute_pids"]:
            raise RuntimeError("Admission requires the declared memory floor and no existing GPU compute processes")
        for job in plan["jobs"]:
            if Path(job["output_root"]).exists():
                raise RuntimeError("Job output appeared before launch")
            memory = memory_reader(); gpu = gpu_reader(plan["gpu_uuid"])
            if memory["headroom_bytes"] < admission_bytes or gpu["compute_pids"]:
                raise RuntimeError("Fresh per-job memory/GPU admission failed")
            token = secrets.token_hex(24)
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=plan["gpu_uuid"], PYTHONUNBUFFERED="1")
            env[OWNER_KEY] = token
            row = dict(name=job["name"], argv=job["argv"], cwd=job["cwd"], output_root=job["output_root"],
                       wall_seconds=job["wall_seconds"], status="launching", gpu_uuid=plan["gpu_uuid"],
                       owner_token_sha256=hashlib.sha256(token.encode()).hexdigest(), parent=identity(os.getpid()),
                       original_child_exit_code=None, signals=[], memory_min_headroom_bytes=memory["headroom_bytes"])
            report["jobs"].append(row); write()
            child, leader, ledger = None, None, None
            duplicate_contexts_seen = set()
            job_started = time.monotonic()
            try:
                signal.setitimer(signal.ITIMER_REAL, job["wall_seconds"])
                with (logs / (job["name"] + ".stdout.log")).open("xb") as stdout, (logs / (job["name"] + ".stderr.log")).open("xb") as stderr:
                    child = subprocess.Popen(job["argv"], cwd=job["cwd"], env=env, stdout=stdout, stderr=stderr, start_new_session=True)
                    report["children_launched"] += 1
                    leader = identity(child.pid)
                    ledger = OwnershipLedger(leader, token)
                    row.update(child=leader, status="running", max_owned_gpu_processes=job["max_owned_gpu_processes"], ownership_events=ledger.events)
                    if leader["session"] != child.pid or leader["group"] != child.pid:
                        raise RuntimeError("New child must lead its isolated session and group")
                    write()
                    while child.poll() is None:
                        if time.monotonic() - job_started >= job["wall_seconds"]:
                            raise TimeoutError("Per-job wall cap exceeded")
                        memory = memory_reader(); gpu = gpu_reader(plan["gpu_uuid"])
                        row["memory_min_headroom_bytes"] = min(row["memory_min_headroom_bytes"], memory["headroom_bytes"])
                        if memory["headroom_bytes"] < 8 * GIB:
                            raise MemoryError("Live memory headroom fell below8GiB")
                        members = group_members(leader["group"])
                        for member in members:
                            if member["pid"] == leader["pid"]:
                                continue
                            if verified_member(member, leader, token, ledger=ledger):
                                key = (member["pid"], member["birth_ticks"])
                                observed = row.setdefault("verified_descendants", [])
                                if not any((x["pid"], x["birth_ticks"]) == key for x in observed):
                                    observed.append(member)
                                    write()
                        for inspection in range(3):
                            row["last_gpu_snapshot"] = gpu
                            try:
                                gpu_identities = verify_gpu_processes(gpu, leader, token, ledger, job["max_owned_gpu_processes"])
                                break
                            except GPUProcessInspectionRace as error:
                                row.setdefault("gpu_process_reinspection_events", []).append(dict(snapshot=gpu, reason=str(error), attempt=inspection+1))
                                if inspection == 2:
                                    gpu, resolution = active_known_gpu_race(
                                        child, leader, token, ledger, job["max_owned_gpu_processes"],
                                        gpu_reader, memory_reader, plan["gpu_uuid"], gpu,
                                        row.get("previously_verified_gpu_identities", []))
                                    row["gpu_process_exit_race_resolution"] = resolution
                                    gpu_identities = resolution.get("verified_live_gpu_identities", [])
                                    break
                                gpu = gpu_reader(plan["gpu_uuid"])
                        row["last_verified_gpu_identities"] = gpu_identities
                        history = row.setdefault("previously_verified_gpu_identities", [])
                        for verified in gpu_identities:
                            if not any(same_identity(verified, prior) for prior in history):
                                history.append(verified.copy())
                        raw_context_rows = list(gpu["compute_pids"])
                        row["last_cuda_context_accounting"] = dict(
                            raw_compute_pid_rows=raw_context_rows,
                            raw_context_row_count=len(raw_context_rows),
                            independently_verified_process_count=len(gpu_identities),
                            verified_process_identities=gpu_identities,
                            NVIDIA_rows_include_birth_identity=False)
                        if len(raw_context_rows) != len(gpu_identities):
                            signature = (tuple(raw_context_rows), tuple((x["pid"],x["birth_ticks"]) for x in gpu_identities))
                            if signature not in duplicate_contexts_seen:
                                duplicate_contexts_seen.add(signature)
                                row.setdefault("duplicate_cuda_context_observations", []).append(
                                    dict(row["last_cuda_context_accounting"]))
                        write()
                        time.sleep(.1)
                    row["original_child_exit_code"] = child.wait()
                    if row["original_child_exit_code"] != 0:
                        raise RuntimeError("Original child exited nonzero; no sequential retry")
                    if group_members(leader["group"]):
                        raise RuntimeError("Unexpected owned group survives child exit")
                    row["status"] = "child_success"
            except BaseException as error:
                row.update(status="failed_preserved_no_retry", failure=dict(type=type(error).__name__, message=str(error)))
                raise
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                cleanup_errors = []
                if leader is not None:
                    try:
                        row["drain"] = drain_group(leader, token, ledger=ledger)
                        row["signals"] = row["drain"]["signals"]
                    except BaseException as error:
                        cleanup_errors.append(dict(stage="drain", error=repr(error)))
                        row["drain"] = dict(drained=False, natural=False, error=repr(error))
                if child is not None:
                    try:
                        row["original_child_exit_code"] = child.wait(timeout=5)
                        row["child_reaped"] = True
                    except BaseException as error:
                        cleanup_errors.append(dict(stage="original_child_reap", error=repr(error)))
                        row["child_reaped"] = False
                        row["observed_child_poll_after_reap_failure"] = child.poll()
                if leader is not None:
                    try:
                        row["drain"]["after_reap"] = group_members(leader["group"])
                        row["drain"]["drained"] = not row["drain"]["after_reap"]
                    except BaseException as error:
                        cleanup_errors.append(dict(stage="after_reap_scan", error=repr(error)))
                        row["drain"]["drained"] = False
                row["cleanup_errors"] = cleanup_errors
                row["elapsed_seconds"] = time.monotonic() - job_started
                write()
                if cleanup_errors or (leader is not None and not row["drain"]["drained"]):
                    raise RuntimeError("Owned cleanup failed; original status and reaping evidence preserved")

        final_gpu = gpu_reader(plan["gpu_uuid"])
        report["final_gpu"] = final_gpu
        if final_gpu["compute_pids"]:
            raise RuntimeError("GPU compute processes remain after final owned-group drain")
        report.update(status="cpu_fixture_commands_pass" if cpu_fixture else "admitted_sequential_commands_completed", exit_code=0)
    except BaseException as error:
        report.update(status="failed_preserved_no_retry", failure=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()), exit_code=1)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        report["all_owned_groups_drained"] = all(x.get("drain", {}).get("drained", False) for x in report["jobs"])
        write()
    return report


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--plan", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    plan = validate_plan(json.loads(args.plan.read_text()))
    if args.validate_only:
        print(json.dumps(dict(status="metadata_valid_no_launch", jobs=len(plan["jobs"]), gpu_uuid=plan["gpu_uuid"])))
        return 0
    result = run_plan(plan)
    print(json.dumps(dict(status=result["status"], exit_code=result["exit_code"], receipt=str(Path(plan["output_root"]) / "receipt.json"))))
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
