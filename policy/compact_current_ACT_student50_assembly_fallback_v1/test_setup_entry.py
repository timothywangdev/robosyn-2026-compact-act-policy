"""Behavioral tests for the outer installer retry boundary; no network calls."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import format_datetime
import importlib.util
import hashlib
import io
from pathlib import Path
import runpy
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch
import urllib.error
import urllib.request


spec = importlib.util.spec_from_file_location(
    "setup_entry", Path(__file__).with_name("setup_entry.py")
)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class FakeTime:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def failure(status, retry_after=None):
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    return urllib.error.HTTPError(
        "https://example.invalid/private?token=must-not-log", status,
        "private error detail must not be logged", headers, None
    )


class RetryTests(unittest.TestCase):
    def retrying(self, opener):
        clock = FakeTime()
        output = io.StringIO()
        wrapper = entry.RetryingUrlopen(
            opener, monotonic=lambda: clock.now, sleep=clock.sleep,
            wall_time=lambda: 1000.0, stderr=output,
        )
        return wrapper, clock, output

    def test_429_then_success_preserves_arguments_and_server_wait(self):
        response = object()
        opener = Mock(side_effect=[failure(429, "108"), response])
        wrapper, clock, output = self.retrying(opener)
        request = urllib.request.Request("https://example.invalid/artifact")
        self.assertIs(wrapper(request, timeout=180), response)
        self.assertEqual(clock.sleeps, [108.0])
        self.assertEqual(opener.call_count, 2)
        self.assertEqual(opener.call_args.args, (request,))
        self.assertEqual(opener.call_args.kwargs, {"timeout": 180})
        self.assertIn("HTTP 429; attempt 1/6", output.getvalue())
        self.assertNotIn("token", output.getvalue())
        self.assertNotIn("private", output.getvalue())
        self.assertNotIn("https", output.getvalue())

    def test_retry_after_http_date(self):
        deadline = format_datetime(datetime.fromtimestamp(1108, timezone.utc), usegmt=True)
        opener = Mock(side_effect=[failure(503, deadline), object()])
        wrapper, clock, _ = self.retrying(opener)
        wrapper("https://example.invalid/artifact")
        self.assertEqual(clock.sleeps, [108.0])

    def test_transient_error_has_bounded_exponential_retries(self):
        errors = [failure(502) for _ in range(entry.MAX_ATTEMPTS)]
        opener = Mock(side_effect=errors)
        wrapper, clock, _ = self.retrying(opener)
        with self.assertRaises(urllib.error.HTTPError) as raised:
            wrapper("https://example.invalid/artifact")
        self.assertIs(raised.exception, errors[-1])
        self.assertEqual(opener.call_count, entry.MAX_ATTEMPTS)
        self.assertEqual(clock.sleeps, [2.0, 4.0, 8.0, 16.0, 32.0])

    def test_403_and_404_are_original_errors_without_retry(self):
        for status in (403, 404):
            with self.subTest(status=status):
                error = failure(status, "108")
                opener = Mock(side_effect=error)
                wrapper, clock, output = self.retrying(opener)
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    wrapper("https://example.invalid/artifact")
                self.assertIs(raised.exception, error)
                self.assertEqual(opener.call_count, 1)
                self.assertEqual(clock.sleeps, [])
                self.assertEqual(output.getvalue(), "")

    def test_excessive_server_wait_fails_without_shortening(self):
        error = failure(429, "901")
        opener = Mock(side_effect=error)
        wrapper, clock, output = self.retrying(opener)
        with self.assertRaises(urllib.error.HTTPError) as raised:
            wrapper("https://example.invalid/artifact")
        self.assertIs(raised.exception, error)
        self.assertEqual(opener.call_count, 1)
        self.assertEqual(clock.sleeps, [])
        self.assertIn("stopping without retry", output.getvalue())

    def test_non_http_failure_is_not_retried(self):
        error = ValueError("original integrity or path failure")
        opener = Mock(side_effect=error)
        wrapper, clock, _ = self.retrying(opener)
        with self.assertRaises(ValueError) as raised:
            wrapper("https://example.invalid/artifact")
        self.assertIs(raised.exception, error)
        self.assertEqual(opener.call_count, 1)
        self.assertEqual(clock.sleeps, [])

    def test_original_download_hash_failure_retains_partial_without_retry(self):
        installer = (Path(__file__).resolve().parent / "runtime" / "experiments" /
                     "coupled_act_student50_release" / "setup_release_v3.py")
        self.assertEqual(hashlib.sha256(installer.read_bytes()).hexdigest(),
                         "31e1c4332df3c5e438e9aef7f3f44a0d179ba3982c5f09fcbb97ff4c7e171213")
        frozen = runpy.run_path(str(installer), run_name="test_frozen_installer")
        opener = Mock(return_value=io.BytesIO(b"wrong bytes"))
        wrapper, clock, _ = self.retrying(opener)
        expected = {"bytes": len(b"expected bytes"),
                    "sha256": hashlib.sha256(b"expected bytes").hexdigest()}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch.object(urllib.request, "urlopen", wrapper):
                with self.assertRaisesRegex(ValueError, "Public artifact byte verification failed"):
                    frozen["download_file"](root, "artifact.bin", expected, "a" * 40)
            self.assertEqual((root / "artifact.bin.download-part").read_bytes(), b"wrong bytes")
            self.assertFalse((root / "artifact.bin").exists())
            self.assertEqual(opener.call_count, 1)
            self.assertEqual(clock.sleeps, [])

    def test_shared_host_cooldown_serializes_queued_worker_probes(self):
        lock = threading.Lock()
        calls = []
        active = 0
        maximum_active = 0

        def opener(request, **kwargs):
            nonlocal active, maximum_active
            with lock:
                calls.append(time.monotonic())
                number = len(calls)
                active += 1
                maximum_active = max(maximum_active, active)
            try:
                if number == 1:
                    raise failure(429, "0")
                time.sleep(0.01)
                return object()
            finally:
                with lock:
                    active -= 1

        wrapper = entry.RetryingUrlopen(opener, stderr=io.StringIO())
        # Prime shared cooldown before additional workers are queued.
        wrapper("https://example.invalid/first")
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(wrapper, ["https://example.invalid/next"] * 4))
        self.assertEqual(len(calls), 6)
        self.assertEqual(maximum_active, 1)
        for previous, current in zip(calls[1:], calls[2:]):
            self.assertGreaterEqual(current - previous, entry.RECOVERY_REQUEST_GAP_SECONDS)

    def test_original_installer_is_executed_and_opener_restored_on_error(self):
        original = urllib.request.urlopen
        with patch.object(entry.runpy, "run_path", side_effect=SystemExit(17)) as run:
            with self.assertRaises(SystemExit) as raised:
                entry.main()
        self.assertEqual(raised.exception.code, 17)
        self.assertIs(urllib.request.urlopen, original)
        self.assertEqual(run.call_args.kwargs, {"run_name": "__main__"})
        self.assertTrue(run.call_args.args[0].endswith(
            "runtime/experiments/coupled_act_student50_release/setup_release_v3.py"
        ))


if __name__ == "__main__":
    unittest.main()
