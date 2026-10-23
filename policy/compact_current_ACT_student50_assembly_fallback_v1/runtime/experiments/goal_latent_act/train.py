"""Bounded actual full-FIT high-level training; monitor is evaluation only."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from .model import VERSION, RolePhysicalGoal, physical_loss, goal_metrics
from .prepare import sha, write_json


def birth(pid):
    return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19])


def source_pins(root):
    names = ['experiments/goal_latent_act/model.py', 'experiments/goal_latent_act/prepare.py',
             'experiments/goal_latent_act/train.py', 'policy/object_relative_hierarchy/multitask_model.py',
             'policy/object_relative_hierarchy/observer.py']
    return {name: sha(root / name) for name in names}


def load_data(path):
    receipt = json.loads((path / 'receipt.json').read_text())
    if receipt['status'] != 'all70000_actual_roles_teacher_bound' or (receipt['fit'], receipt['monitor']) != (63000, 7000):
        raise ValueError('Actual full role/physics binding required')
    for name, spec in receipt['artifacts'].items():
        if sha(path / name) != spec['sha256'] or (path / name).stat().st_size != spec['bytes']:
            raise ValueError('Packed training artifact changed: ' + name)
    roles = np.load(path / 'roles.npy', mmap_mode='r', allow_pickle=False)
    with np.load(path / 'labels.npz', allow_pickle=False) as archive:
        labels = {k: archive[k].copy() for k in archive.files}
    stats = json.loads((path / 'statistics.json').read_text())
    fit = np.flatnonzero(labels['fit']); monitor = np.flatnonzero(~labels['fit'])
    if len(fit) != 63000 or len(monitor) != 7000 or set(fit) & set(monitor): raise ValueError('Exact independent split required')
    if roles.shape != (70000, 3, 4096) or roles.dtype != np.float16: raise ValueError('Actual roles shape/dtype')
    return receipt, roles, labels, stats, fit, monitor


def stats_tensors(stats):
    return {name: torch.tensor(stats[name], dtype=torch.float32) for name in ('qpos_mean', 'qpos_scale', 'eef_mean', 'eef_scale')}


def batch(roles, labels, ids):
    return (torch.from_numpy(np.asarray(roles[ids], dtype=np.float32)),
            {k: torch.from_numpy(np.ascontiguousarray(v[ids])) for k, v in labels.items() if k not in ('clip_id', 'fit', 'task_index')})


def forward(model, values, labels, stats, *, role_control='factual'):
    return model(roles=values, qpos=labels['qpos'], eef_now=labels['eef_now'], **stats, role_control=role_control)


@torch.no_grad()
def evaluate(model, roles, labels, ids, stats, tasks, batch_size, *, intervention=None):
    model.eval(); metric_sums = Counter(); losses = Counter(); task_sums = {t: Counter() for t in tasks}; counts = Counter()
    shuffled = None
    if intervention == 'within_task_role_shuffle':
        shuffled = ids.copy(); rng = np.random.default_rng(9917)
        for j in range(len(tasks)):
            selected = np.flatnonzero(labels['task_index'][ids] == j)
            shuffled[selected] = rng.permutation(ids[selected])
    for start in range(0, len(ids), batch_size):
        current = ids[start:start + batch_size]
        values, targets = batch(roles, labels, current)
        if shuffled is not None: values = torch.from_numpy(np.asarray(roles[shuffled[start:start + batch_size]], dtype=np.float32))
        pred = forward(model, values, targets, stats, role_control='zero' if intervention == 'role_zero' else 'factual')
        total, component = physical_loss(pred, targets)
        values_by_metric = goal_metrics(pred, targets)
        hold = dict(eef_goal_pose=targets['eef_now'])
        hold_metrics = goal_metrics(hold, targets)
        values_by_metric.update({'hold_' + k: v for k, v in hold_metrics.items()})
        for name, value in component.items(): losses[name] += float(value) * len(current)
        losses['total'] += float(total) * len(current)
        for name, value in values_by_metric.items():
            metric_sums[name] += float(value.sum())
            for j, task in enumerate(tasks):
                select = torch.from_numpy(labels['task_index'][current] == j)
                if bool(select.any()): task_sums[task][name] += float(value[select].sum())
        for j, task in enumerate(tasks): counts[task] += int((labels['task_index'][current] == j).sum())
    return dict(rows=len(ids), metrics={k: v / len(ids) for k, v in metric_sums.items()},
                loss={k: v / len(ids) for k, v in losses.items()},
                per_task={t: dict(rows=counts[t], metrics={k: v / counts[t] for k, v in task_sums[t].items()}) for t in tasks},
                intervention=intervention)


def checkpoint(output, model, contract):
    tmp = output / 'highlevel.safetensors.tmp'
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(tmp))
    with tmp.open('rb') as stream: os.fsync(stream.fileno())
    tmp.replace(output / 'highlevel.safetensors')
    contract['weights_sha256'] = sha(output / 'highlevel.safetensors')
    write_json(output / 'checkpoint.json', contract)


def load_checkpoint(output):
    contract = json.loads((output / 'checkpoint.json').read_text())
    if contract['schema'] != VERSION or sha(output / 'highlevel.safetensors') != contract['weights_sha256']:
        raise ValueError('Exact high-level checkpoint required')
    model = RolePhysicalGoal(**contract['model_config'])
    model.load_state_dict(load_file(str(output / 'highlevel.safetensors')), strict=True)
    return model.eval(), contract


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--data', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=8); p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--threads', type=int, default=8); p.add_argument('--wall-seconds', type=int, default=1800)
    p.add_argument('--seed', type=int, default=1701); p.add_argument('--role-zero-control', action='store_true')
    a = p.parse_args()
    if not 1 <= a.epochs <= 30 or not 1 <= a.wall_seconds <= 1800 or not 1 <= a.threads <= 8:
        raise ValueError('Bounded epochs/wall/CPU threads required')
    if not 32 <= a.batch_size <= 512: raise ValueError('Bounded batch size')
    a.output.mkdir(parents=True, exist_ok=False); torch.set_num_threads(a.threads); torch.set_num_interop_threads(1)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    report = dict(schema=VERSION, status='started', pid=os.getpid(), birth=birth(os.getpid()),
                  device='cpu', GPU_launched=False, optimizer_updates=0, examples_seen=0, epochs_completed=0,
                  role_zero_control=a.role_zero_control, seed=a.seed, arguments=vars(a) | {'data': str(a.data), 'output': str(a.output)})
    started = time.monotonic(); report['source_sha256'] = source_pins(Path(__file__).resolve().parents[2])
    def expired(*_): raise TimeoutError('High-level training wall budget')
    signal.signal(signal.SIGALRM, expired); signal.alarm(a.wall_seconds)
    try:
        preparation, roles, labels, stats, fit, monitor = load_data(a.data)
        report.update(data_receipt_sha256=sha(a.data / 'receipt.json'), preparation=preparation,
                      fit_rows=63000, monitor_rows=7000, fit_demonstrations=9000, monitor_demonstrations=1000,
                      parameter_training_scope='high-level goals only; no ACT correction, VLM, or motor weights trained')
        st = stats_tensors(stats); tasks = preparation['tasks']; model = RolePhysicalGoal()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.001)
        control = 'role_zero' if a.role_zero_control else None
        report['initial_monitor'] = evaluate(model, roles, labels, monitor, st, tasks, a.batch_size, intervention=control)
        write_json(a.output / 'receipt.json', report)
        history = []; rng = np.random.default_rng(a.seed); best = float('inf'); best_epoch = None
        contract = dict(schema=VERSION, model_config=model.config, feature_shape=[3,4096],
                        context_version='qwen35-canonical-current-role-hidden-v2', current_qpos_shape=[14],
                        current_eef_shape=[2,4,4], output_dual_eef_goal_shape=[2,4,4], statistics=stats,
                        goal_horizon_frames=10, state_units=stats['state_units'], eef_units=stats['eef_units'],
                        source_sha256=report['source_sha256'], data_receipt_sha256=report['data_receipt_sha256'],
                        role_zero_control=a.role_zero_control, lowlevel_ACT_correction_trained=False,
                        actor_inputs='current RGB/instruction Qwen roles, canonical qpos, measured-current FK only')
        for epoch in range(a.epochs):
            model.train(); order = rng.permutation(fit); sums = Counter()
            for start in range(0, len(order), a.batch_size):
                ids = order[start:start+a.batch_size]; values, target = batch(roles, labels, ids)
                optimizer.zero_grad(set_to_none=True)
                pred = forward(model, values, target, st, role_control='zero' if a.role_zero_control else 'factual')
                loss, components = physical_loss(pred, target)
                if not bool(torch.isfinite(loss)): raise ValueError('Nonfinite high-level loss')
                loss.backward()
                if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                    raise ValueError('Nonfinite high-level gradients')
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step(); report['optimizer_updates'] += 1; report['examples_seen'] += len(ids)
                sums['total'] += float(loss.detach()) * len(ids)
                if report['optimizer_updates'] % 100 == 0:
                    report['elapsed_seconds'] = time.monotonic()-started
                    write_json(a.output / 'receipt.json', report)
            monitor_result = evaluate(model, roles, labels, monitor, st, tasks, a.batch_size, intervention=control)
            row = dict(epoch=epoch+1, fit_rows=63000, fit_loss=sums['total']/63000,
                       monitor=monitor_result, elapsed_seconds=time.monotonic()-started,
                       fit_order_sha256=hashlib.sha256(order.tobytes()).hexdigest())
            history.append(row); report['history'] = history; report['epochs_completed'] = epoch+1
            score = monitor_result['metrics']['eef_goal_position_l2_m']
            if score < best:
                best = score; best_epoch = epoch+1
                checkpoint(a.output, model, contract | {'selected_epoch': best_epoch, 'optimizer_updates': report['optimizer_updates'],
                            'selection_metric':'independent_monitor_mean_dual_EEF_goal_position_L2_m'})
            write_json(a.output / 'receipt.json', report)
            print(json.dumps({'epoch':epoch+1,'updates':report['optimizer_updates'],'fit_loss':row['fit_loss'],
                              'monitor_goal_m':score,'elapsed_seconds':row['elapsed_seconds']}), flush=True)
        reloaded, saved = load_checkpoint(a.output)
        values, target = batch(roles, labels, monitor[:a.batch_size])
        with torch.no_grad():
            before = forward(reloaded, values, target, st, role_control='zero' if a.role_zero_control else 'factual')
            twice, _ = load_checkpoint(a.output)
            after = forward(twice, values, target, st, role_control='zero' if a.role_zero_control else 'factual')
        if any(not torch.equal(before[k], after[k]) for k in before): raise ValueError('Saved/reloaded physical prediction changed')
        report.update(status='completed_highlevel_training', selected_epoch=best_epoch, checkpoint=saved,
                      final_monitor=evaluate(reloaded, roles, labels, monitor, st, tasks, a.batch_size, intervention=control),
                      reload_prediction_bitexact=True, CUDA_initialized=torch.cuda.is_initialized(),
                      limitations=['Heldout teacher goal loss is not robot success', 'Monitor selects epoch; not a test score',
                                   'No low-level ACT correction trained', 'Bell historical unit lineage remains uncertified',
                                   'Object supervision absent for Bell/Basket/Table; allten dualEEF supervision used'])
        if not a.role_zero_control:
            report['role_interventions'] = {name:evaluate(reloaded, roles, labels, monitor, st, tasks, a.batch_size, intervention=name)
                                             for name in ('role_zero','within_task_role_shuffle')}
    except BaseException as error:
        report.update(status='failed_no_retry', error=repr(error)); raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        write_json(a.output / 'receipt.json', report); signal.alarm(0)


if __name__ == '__main__': main()
