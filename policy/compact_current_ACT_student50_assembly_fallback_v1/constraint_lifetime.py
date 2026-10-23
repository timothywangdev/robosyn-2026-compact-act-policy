"""Release borrowed Python constraint wrappers before their native arena owner.

DexSim returns borrowed FixedConstraint wrappers. Their pybind11 deallocator
still traverses virtual base classes, so the native object must remain alive
until the Python wrapper has been discarded. The affected SDK did the reverse.
This compatibility repair leaves constraint creation, physics, resets, episode
limits, policy actions and success computation with the official implementation.
"""
from hashlib import sha256
from functools import wraps
from importlib import import_module
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import sys

SDK_SOURCE_SHA256 = '49b704acfe77a4b5de9c50a1c1d7077f19f3a39fbf513b3eff970869b2c40e45'
DEXSIM_BINDING_SHA256 = '4d11a71bfe75b85f74aed55d29e9e7d11b45d952103ae40ad1e212a4fb74c2e4'
SDK_MANAGER_SHA256 = '195ab1a2bb0ef188ccd675ef56c2ff73c921520c1f41a00271c8fed8553ac395'
_MARKER = 'robosyn_borrowed_constraint_wrapper_before_native_owner_v1'


def _destroy_borrowed_first(self, env_ids=None, arena_resolver=None):
    for index in self._active_env_ids(env_ids):
        arena = arena_resolver(index) if arena_resolver is not None else None
        if arena_resolver is not None and (arena is None
                or not callable(getattr(arena, 'remove_constraint', None))):
            raise RuntimeError('Constraint arena resolver returned no removable native arena')
        name = self.get_name(index)
        # Do not introduce a local handle alias here: this must discard the last
        # borrowed wrapper while its native owner and virtual bases are alive.
        if arena is not None and sys.getrefcount(self.constraint_handles[index]) != 2:
            raise RuntimeError(
                'Cannot safely remove a native constraint with additional Python '
                'handle references; release those references before detaching')
        self.constraint_handles[index] = None
        if arena is None:
            continue
        try:
            arena.remove_constraint(name)
        except BaseException as error:
            # If removal failed, reacquire a fresh borrowed wrapper so the SDK
            # can still track a constraint which the arena continues to own.
            try:
                self.constraint_handles[index] = arena.get_constraint(name)
            except Exception as lookup_error:
                error.add_note('Constraint lookup after removal failure: '
                               + type(lookup_error).__name__)
            raise
        print('[RoboSyn constraint lifetime] ' + json.dumps(
            dict(event='removed', name=name, env_id=int(index),
                 Python_wrapper_released_before_native_owner=True)), flush=True)


def _constraints_before_shutdown(original):
    @wraps(original)
    def destroy(self, *args, **kwargs):
        for name in list(self._constraints):
            if not self.remove_rigid_constraint(name):
                raise RuntimeError('Registered constraint was not removed: ' + name)
        if self._constraints:
            raise RuntimeError('Native constraint handles remain before scene shutdown')
        return original(self, *args, **kwargs)
    destroy._robosyn_lifetime_repair = _MARKER
    return destroy


def install():
    from embodichain.lab.sim.objects.constraint import RigidConstraint
    from embodichain.lab.sim.sim_manager import SimulationManager
    import dexsim

    if getattr(RigidConstraint.destroy, '_robosyn_lifetime_repair', None) == _MARKER:
        if all(getattr(method, '_robosyn_lifetime_repair', None) == _MARKER
               for method in (SimulationManager.destroy, SimulationManager._deferred_destroy)):
            return
        raise RuntimeError('Incomplete simulator constraint lifetime repair')
    if sys.implementation.name != 'cpython':
        raise RuntimeError('Constraint lifetime repair requires CPython reference counting')
    source = Path(inspect.getfile(RigidConstraint)).resolve(strict=True)
    binding = Path(import_module('dexsim.cuda.pybind').__file__).resolve(strict=True)
    manager = Path(inspect.getfile(SimulationManager)).resolve(strict=True)
    actual = dict(SDK_version=version('embodichain'), DexSim_version=dexsim.__version__,
                  SDK_source_sha256=sha256(source.read_bytes()).hexdigest(),
                  SDK_manager_sha256=sha256(manager.read_bytes()).hexdigest(),
                  binding_sha256=sha256(binding.read_bytes()).hexdigest())
    if (actual['SDK_version'] != '0.2.4.post1'
            or actual['DexSim_version'] != '0.4.3'
            or actual['SDK_source_sha256'] != SDK_SOURCE_SHA256
            or actual['SDK_manager_sha256'] != SDK_MANAGER_SHA256
            or actual['binding_sha256'] != DEXSIM_BINDING_SHA256):
        raise RuntimeError('Unrecognized simulator constraint implementation; '
                           'lifetime repair requires its verified source and binding: '
                           + json.dumps(actual, sort_keys=True))
    if (RigidConstraint.destroy.__module__ != 'embodichain.lab.sim.objects.constraint'
            or Path(RigidConstraint.destroy.__code__.co_filename).resolve() != source):
        raise RuntimeError('Simulator constraint destruction was already replaced')
    if any(method.__module__ != 'embodichain.lab.sim.sim_manager'
           or Path(method.__code__.co_filename).resolve() != manager
           for method in (SimulationManager.destroy, SimulationManager._deferred_destroy)):
        raise RuntimeError('Simulator scene destruction was already replaced')
    _destroy_borrowed_first._robosyn_lifetime_repair = _MARKER
    RigidConstraint.destroy = _destroy_borrowed_first
    SimulationManager.destroy = _constraints_before_shutdown(SimulationManager.destroy)
    SimulationManager._deferred_destroy = _constraints_before_shutdown(
        SimulationManager._deferred_destroy)
    print('[RoboSyn constraint lifetime] ' + json.dumps(
        dict(event='installed', repair=_MARKER, **actual)), flush=True)
