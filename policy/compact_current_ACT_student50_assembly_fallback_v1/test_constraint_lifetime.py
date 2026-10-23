"""CPU lifetime-order regressions; no simulator or binding imports/execution.

The historical destroy method and its index/name helpers are extracted from the
actual fresh SDK using AST, preserving their behavior without importing the SDK.
The borrowed-wrapper fixture records native-owner liveness in __del__.
Set EMBODICHAIN_ROOT to an SDK checkout or ROBOSYN_TEST_SDK_CONSTRAINT_SOURCE to
its constraint.py. Tests skip with a reason when that source fixture is absent.
"""
import ast
import contextlib
import copy
import __future__
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import weakref


_source_override = os.environ.get("ROBOSYN_TEST_SDK_CONSTRAINT_SOURCE")
_sdk_root = os.environ.get("EMBODICHAIN_ROOT")
SDK_CONSTRAINT_SOURCE = (Path(_source_override) if _source_override else
                         Path(_sdk_root) / "embodichain/lab/sim/objects/constraint.py"
                         if _sdk_root else None)
_modules_before_load = set(sys.modules)
_spec = importlib.util.spec_from_file_location(
    "constraint_lifetime_under_test", Path(__file__).with_name("constraint_lifetime.py")
)
repair = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(repair)
_modules_loaded_by_repair = set(sys.modules) - _modules_before_load


def extract_original_constraint():
    if SDK_CONSTRAINT_SOURCE is None or not SDK_CONSTRAINT_SOURCE.is_file():
        return None
    source = SDK_CONSTRAINT_SOURCE.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != repair.SDK_SOURCE_SHA256:
        raise AssertionError("Test requires the exact recognized fresh SDK source: " + digest)
    tree = ast.parse(source, filename=str(SDK_CONSTRAINT_SOURCE))
    original_class = next(node for node in tree.body
                          if isinstance(node, ast.ClassDef) and node.name == "RigidConstraint")
    wanted = {"num_envs", "get_name", "_active_env_ids", "destroy"}
    methods = [copy.deepcopy(node) for node in original_class.body
               if isinstance(node, ast.FunctionDef) and node.name in wanted]
    if {node.name for node in methods} != wanted:
        raise AssertionError("Required original SDK methods were not found")
    extracted = ast.ClassDef(name="OriginalConstraint", bases=[], keywords=[],
                             body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[extracted], type_ignores=[]))
    namespace = {}
    exec(compile(module, str(SDK_CONSTRAINT_SOURCE), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace["OriginalConstraint"]


OriginalConstraint = extract_original_constraint()
SDK_FIXTURE_REASON = ("Actual SDK source fixture is absent; set EMBODICHAIN_ROOT "
                      "or ROBOSYN_TEST_SDK_CONSTRAINT_SOURCE")


class NativeOwner:
    def __init__(self, name, events):
        self.name = name
        self.events = events
        self.alive = True
        self.wrapper_count = 0


class BorrowedWrapper:
    def __init__(self, owner):
        self.owner = owner
        owner.wrapper_count += 1

    def __del__(self):
        self.owner.events.append(("wrapper_deleted", self.owner.name, self.owner.alive))
        self.owner.wrapper_count -= 1


class Arena:
    def __init__(self, owner):
        self.owner = owner
        self.failure = None

    def remove_constraint(self, name):
        if name != self.owner.name:
            raise AssertionError("Wrong native constraint name: " + name)
        self.owner.events.append(("native_remove", name, self.owner.wrapper_count))
        if self.failure is not None:
            raise self.failure
        self.owner.alive = False

    def get_constraint(self, name):
        if name != self.owner.name or not self.owner.alive:
            raise AssertionError("A live native constraint is required for a fresh wrapper")
        self.owner.events.append(("wrapper_reacquired", name))
        return BorrowedWrapper(self.owner)


def fixture(count=1, inactive=(), name="latch", events=None):
    events = [] if events is None else events
    constraint = OriginalConstraint()
    constraint.cfg = SimpleNamespace(name=name)
    constraint.constraint_handles = []
    arenas = {}
    for index in range(count):
        owner = NativeOwner(name if count == 1 else f"{name}_{index}", events)
        arenas[index] = Arena(owner)
        # No separate fixture-held Python alias to the borrowed wrapper.
        constraint.constraint_handles.append(None if index in inactive else BorrowedWrapper(owner))
    return constraint, arenas, events


@unittest.skipUnless(OriginalConstraint is not None, SDK_FIXTURE_REASON)
class ConstraintLifetimeTests(unittest.TestCase):
    def setUp(self):
        self.output = io.StringIO()
        redirect = contextlib.redirect_stdout(self.output)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def test_module_load_does_not_import_sdk_or_native_binding(self):
        self.assertFalse(any(name == "dexsim" or name.startswith("dexsim.")
                             or name == "embodichain" or name.startswith("embodichain.")
                             for name in _modules_loaded_by_repair))

    def test_new_destroy_releases_wrapper_while_native_owner_is_alive(self):
        constraint, arenas, events = fixture()
        released = weakref.ref(constraint.constraint_handles[0])
        repair._destroy_borrowed_first(constraint, arena_resolver=arenas.__getitem__)
        self.assertIsNone(released())
        self.assertEqual(events, [("wrapper_deleted", "latch", True),
                                  ("native_remove", "latch", 0)])
        self.assertEqual(constraint.constraint_handles, [None])
        self.assertFalse(arenas[0].owner.alive)

    def test_actual_old_sdk_destroy_violates_same_lifetime_invariant(self):
        constraint, arenas, events = fixture()
        constraint.destroy(arena_resolver=arenas.__getitem__)
        self.assertEqual(events, [("native_remove", "latch", 1),
                                  ("wrapper_deleted", "latch", False)])

    def test_subset_indices_names_and_inactive_handles_use_real_sdk_helpers(self):
        constraint, arenas, events = fixture(count=4, inactive=(1,))
        resolved = []

        def resolve(index):
            resolved.append(index)
            return arenas[index]

        repair._destroy_borrowed_first(constraint, env_ids=[3, 1, 0], arena_resolver=resolve)
        self.assertEqual(resolved, [3, 0])
        self.assertEqual(events, [("wrapper_deleted", "latch_3", True),
                                  ("native_remove", "latch_3", 0),
                                  ("wrapper_deleted", "latch_0", True),
                                  ("native_remove", "latch_0", 0)])
        self.assertIsNone(constraint.constraint_handles[0])
        self.assertIsNone(constraint.constraint_handles[1])
        self.assertIsNotNone(constraint.constraint_handles[2])
        self.assertIsNone(constraint.constraint_handles[3])
        self.assertTrue(arenas[2].owner.alive)
        repair._destroy_borrowed_first(constraint, arena_resolver=resolve)
        self.assertEqual(resolved, [3, 0, 2])
        self.assertEqual(events[-2:], [("wrapper_deleted", "latch_2", True),
                                       ("native_remove", "latch_2", 0)])

    def test_extra_python_alias_fails_before_mutation_then_passes_after_release(self):
        constraint, arenas, events = fixture()
        alias = constraint.constraint_handles[0]
        with self.assertRaisesRegex(RuntimeError, "additional Python handle references"):
            repair._destroy_borrowed_first(constraint, arena_resolver=arenas.__getitem__)
        self.assertIs(constraint.constraint_handles[0], alias)
        self.assertTrue(arenas[0].owner.alive)
        self.assertEqual(events, [])
        del alias
        repair._destroy_borrowed_first(constraint, arena_resolver=arenas.__getitem__)
        self.assertEqual(events, [("wrapper_deleted", "latch", True),
                                  ("native_remove", "latch", 0)])

    def test_omitted_resolver_preserves_clear_only_semantics_even_with_alias(self):
        constraint, arenas, events = fixture()
        alias = constraint.constraint_handles[0]
        repair._destroy_borrowed_first(constraint, arena_resolver=None)
        self.assertEqual(constraint.constraint_handles, [None])
        self.assertTrue(arenas[0].owner.alive)
        self.assertEqual(events, [])
        del alias
        self.assertEqual(events, [("wrapper_deleted", "latch", True)])

    def test_explicit_unusable_arena_resolver_fails_before_mutation(self):
        for returned_arena in (None, object(), SimpleNamespace(remove_constraint=3)):
            with self.subTest(returned_arena=type(returned_arena).__name__):
                constraint, arenas, events = fixture()
                alias = constraint.constraint_handles[0]
                with self.assertRaisesRegex(RuntimeError, "no removable native arena"):
                    repair._destroy_borrowed_first(
                        constraint, arena_resolver=lambda index: returned_arena)
                self.assertIs(constraint.constraint_handles[0], alias)
                self.assertTrue(arenas[0].owner.alive)
                self.assertEqual(events, [])
                del alias

    def test_native_remove_failure_restores_fresh_wrapper_and_original_exception(self):
        constraint, arenas, events = fixture()
        old_wrapper = weakref.ref(constraint.constraint_handles[0])
        error = ValueError("native removal refused")
        arenas[0].failure = error
        with self.assertRaises(ValueError) as raised:
            repair._destroy_borrowed_first(constraint, arena_resolver=arenas.__getitem__)
        self.assertIs(raised.exception, error)
        self.assertIsNone(old_wrapper())
        self.assertIsNotNone(constraint.constraint_handles[0])
        self.assertTrue(arenas[0].owner.alive)
        self.assertEqual(arenas[0].owner.wrapper_count, 1)
        self.assertEqual(events, [("wrapper_deleted", "latch", True),
                                  ("native_remove", "latch", 0),
                                  ("wrapper_reacquired", "latch")])
        arenas[0].failure = None
        repair._destroy_borrowed_first(constraint, arena_resolver=arenas.__getitem__)
        self.assertEqual(events[-2:], [("wrapper_deleted", "latch", True),
                                       ("native_remove", "latch", 0)])


def manager_fixture(events, remove_returns_false=False, native_error=None):
    returned = object()

    def original(manager, *args, **kwargs):
        events.append(("original_shutdown", args, kwargs, list(manager._constraints)))
        return returned

    class Manager:
        destroy = repair._constraints_before_shutdown(original)
        _deferred_destroy = repair._constraints_before_shutdown(original)

        def __init__(self):
            self._constraints = {}
            for name in ("first", "second"):
                constraint, arenas, _ = fixture(name=name, events=events)
                arenas[0].failure = native_error
                self._constraints[name] = (constraint, arenas)

        def remove_rigid_constraint(self, name):
            events.append(("manager_remove", name))
            if remove_returns_false:
                return False
            constraint, arenas = self._constraints[name]
            repair._destroy_borrowed_first(constraint, arena_resolver=arenas.__getitem__)
            del self._constraints[name]
            return True

    return Manager(), returned


@unittest.skipUnless(OriginalConstraint is not None, SDK_FIXTURE_REASON)
class ManagerShutdownTests(unittest.TestCase):
    def setUp(self):
        redirect = contextlib.redirect_stdout(io.StringIO())
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def test_destroy_and_deferred_drain_constraints_before_original_and_forward(self):
        for method in ("destroy", "_deferred_destroy"):
            with self.subTest(method=method):
                events = []
                manager, expected = manager_fixture(events)
                actual = getattr(manager, method)("sentinel", exit_process=False)
                self.assertIs(actual, expected)
                self.assertEqual(manager._constraints, {})
                self.assertEqual(events, [
                    ("manager_remove", "first"),
                    ("wrapper_deleted", "first", True),
                    ("native_remove", "first", 0),
                    ("manager_remove", "second"),
                    ("wrapper_deleted", "second", True),
                    ("native_remove", "second", 0),
                    ("original_shutdown", ("sentinel",), {"exit_process": False}, []),
                ])

    def test_original_shutdown_not_called_when_registered_removal_returns_false(self):
        for method in ("destroy", "_deferred_destroy"):
            with self.subTest(method=method):
                events = []
                manager, _ = manager_fixture(events, remove_returns_false=True)
                with self.assertRaisesRegex(RuntimeError, "Registered constraint was not removed"):
                    getattr(manager, method)()
                self.assertEqual(events, [("manager_remove", "first")])
                self.assertEqual(list(manager._constraints), ["first", "second"])

    def test_original_shutdown_not_called_when_native_removal_raises(self):
        for method in ("destroy", "_deferred_destroy"):
            with self.subTest(method=method):
                events = []
                error = ValueError("native failure")
                manager, _ = manager_fixture(events, native_error=error)
                with self.assertRaises(ValueError) as raised:
                    getattr(manager, method)()
                self.assertIs(raised.exception, error)
                self.assertFalse(any(row[0] == "original_shutdown" for row in events))
                self.assertEqual(list(manager._constraints), ["first", "second"])
                self.assertIsNotNone(manager._constraints["first"][0].constraint_handles[0])
                self.assertEqual(events, [("manager_remove", "first"),
                                          ("wrapper_deleted", "first", True),
                                          ("native_remove", "first", 0),
                                          ("wrapper_reacquired", "first")])


if __name__ == "__main__":
    unittest.main(verbosity=2)
