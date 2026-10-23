"""CPU-only cache-binding regressions; no model, Torch, or simulator imports.

Only the expected initializer digest and size are replaced for tiny temporary
fixtures. The actual public policy helper and pinned runtime modules are used.
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


POLICY_DIRECTORY = Path(__file__).resolve().parent
_modules_before_load = set(sys.modules)
_spec = importlib.util.spec_from_file_location(
    "deploy_policy_cache_under_test", POLICY_DIRECTORY / "deploy_policy.py"
)
policy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(policy)
_modules_loaded_by_policy = set(sys.modules) - _modules_before_load


class ModelCacheTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        temporary = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.temporary = Path(temporary).resolve()
        self.root = self.temporary / "package"
        self.root.mkdir()
        self.home = self.root / "cache/torch"
        self.initializer = self.home / "hub/checkpoints/resnet18-f37072fd.pth"
        self.initializer.parent.mkdir(parents=True)
        self.expected = b"tiny exact initializer fixture\n"
        self.initializer.write_bytes(self.expected)
        self.digest = hashlib.sha256(self.expected).hexdigest()
        self.stack.enter_context(patch.object(policy, "_INITIALIZER_SHA256", self.digest))
        self.stack.enter_context(patch.object(policy, "_INITIALIZER_BYTES", len(self.expected)))
        self.stale_home = str(self.temporary / "unrelated-global-cache")
        self.stack.enter_context(patch.dict(os.environ, {"TORCH_HOME": self.stale_home}))
        self.stack.enter_context(patch.dict(sys.modules))
        sys.modules.pop("torch.hub", None)
        self.output = io.StringIO()
        self.stack.enter_context(contextlib.redirect_stdout(self.output))

    def fake_preloaded_hub(self):
        hub = SimpleNamespace(directory=str(self.temporary / "custom-preloaded-hub"))
        hub.set_dir = Mock(side_effect=lambda directory: setattr(hub, "directory", directory))
        sys.modules["torch.hub"] = hub
        return hub

    def assert_rejected_before_binding(self):
        hub = self.fake_preloaded_hub()
        original_hub_directory = hub.directory
        with self.assertRaisesRegex(ValueError, "Pinned packaged ResNet initializer is missing or differs"):
            policy._bind_torch_cache(self.root)
        self.assertEqual(os.environ["TORCH_HOME"], self.stale_home)
        self.assertEqual(hub.directory, original_hub_directory)
        hub.set_dir.assert_not_called()
        self.assertEqual(self.output.getvalue(), "")

    def test_outer_policy_import_preserves_pure_runtime_closure_and_source_pins(self):
        forbidden = ("torch", "torchvision", "dexsim", "embodichain")
        self.assertFalse(any(name == item or name.startswith(item + ".")
                             for name in _modules_loaded_by_policy for item in forbidden))
        inventory = json.loads((POLICY_DIRECTORY / "runtime-files.json").read_text())
        runtime = POLICY_DIRECTORY / "runtime"
        files = dict(inventory["files"], **inventory.get("supplemental_files", {}))
        for name, expected in files.items():
            with self.subTest(runtime_file=name):
                target = runtime / name
                self.assertFalse(target.is_symlink())
                self.assertEqual(target.stat().st_size, expected["bytes"])
                self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), expected["sha256"])
        for name in policy._policy.PINS:
            with self.subTest(pinned_module=name):
                module = policy._policy.pinned_module(name)
                self.assertTrue(Path(module.__file__).resolve().is_relative_to(runtime.resolve()))
        self.assertNotIn("torch", sys.modules)

    def test_valid_initializer_overrides_stale_home_and_preloaded_hub_directory(self):
        hub = self.fake_preloaded_hub()
        policy._bind_torch_cache(self.root)
        self.assertEqual(os.environ["TORCH_HOME"], str(self.home))
        self.assertEqual(hub.directory, str(self.home / "hub"))
        hub.set_dir.assert_called_once_with(str(self.home / "hub"))
        message = self.output.getvalue().strip()
        self.assertTrue(message.startswith("[RoboSyn model cache] "))
        proof = json.loads(message.removeprefix("[RoboSyn model cache] "))
        self.assertEqual(proof, dict(TORCH_HOME=str(self.home), initializer=str(self.initializer),
                                    sha256=self.digest, bytes=len(self.expected)))

    def test_unloaded_hub_is_not_imported_to_bind_valid_cache(self):
        self.assertNotIn("torch.hub", sys.modules)
        self.assertNotIn("torch", sys.modules)
        policy._bind_torch_cache(self.root)
        self.assertEqual(os.environ["TORCH_HOME"], str(self.home))
        self.assertNotIn("torch.hub", sys.modules)
        self.assertNotIn("torch", sys.modules)

    def test_unset_home_is_bound_to_packaged_initializer(self):
        os.environ.pop("TORCH_HOME")
        policy._bind_torch_cache(self.root)
        self.assertEqual(os.environ["TORCH_HOME"], str(self.home))
        self.assertNotIn("torch.hub", sys.modules)

    def test_missing_initializer_is_rejected_before_environment_or_hub_binding(self):
        self.initializer.unlink()
        self.assert_rejected_before_binding()

    def test_same_size_corrupt_initializer_is_rejected_before_binding(self):
        self.initializer.write_bytes(b"x" * len(self.expected))
        self.assert_rejected_before_binding()

    def test_wrong_size_initializer_is_rejected_before_binding(self):
        self.initializer.write_bytes(self.expected + b"extra")
        self.assert_rejected_before_binding()

    def test_symlink_initializer_is_rejected_even_when_bytes_match(self):
        actual = self.root / "same-valid-initializer.pth"
        actual.write_bytes(self.expected)
        self.initializer.unlink()
        self.initializer.symlink_to(actual)
        self.assert_rejected_before_binding()

    def test_parent_directory_symlink_escape_is_rejected_before_binding(self):
        outside = self.temporary / "outside-cache"
        self.home.rename(outside)
        self.home.symlink_to(outside, target_is_directory=True)
        self.assertEqual(self.initializer.read_bytes(), self.expected)
        self.assertFalse(self.initializer.resolve().is_relative_to(self.root))
        self.assert_rejected_before_binding()


if __name__ == "__main__":
    unittest.main(verbosity=2)
