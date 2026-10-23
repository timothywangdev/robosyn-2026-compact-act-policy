"""Run the unchanged frozen installer with bounded HTTP request retries.

Only opening an existing request is retried. Artifact reads, filesystem writes,
and every original manifest, byte, source, and package check remain unchanged.
"""
from datetime import timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
import runpy
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


MAX_ATTEMPTS = 6
MAX_SERVER_WAIT_SECONDS = 900
RECOVERY_REQUEST_GAP_SECONDS = 0.125
RETRY_STATUSES = frozenset((429, 500, 502, 503, 504))


def retry_after_seconds(headers, wall_time):
    """Return a valid Retry-After delay, or None for absent/invalid headers."""
    value = headers.get("Retry-After") if headers is not None else None
    if value is None:
        return None
    value = value.strip()
    if value.isdigit():
        return int(value)
    try:
        deadline = parsedate_to_datetime(value)
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
        return max(0.0, deadline.timestamp() - wall_time())
    except (TypeError, ValueError, OverflowError):
        return None


class _HostBackoff:
    """Share server cooldowns and serialize recovery across installer threads."""

    def __init__(self, monotonic, sleep):
        self._monotonic = monotonic
        self._sleep = sleep
        self._lock = threading.Lock()
        self._recovery_gate = threading.Lock()
        self._until = 0.0
        self._recovering = False

    def acquire(self):
        with self._lock:
            recovering = self._recovering
        if not recovering:
            return False
        self._recovery_gate.acquire()
        try:
            while True:
                with self._lock:
                    delay = self._until - self._monotonic()
                if delay <= 0:
                    return True
                self._sleep(delay)
        except BaseException:
            self._recovery_gate.release()
            raise

    def defer(self, delay):
        with self._lock:
            self._recovering = True
            self._until = max(self._until, self._monotonic() + delay)

    def release(self, acquired, succeeded):
        if acquired:
            if succeeded:
                with self._lock:
                    self._until = max(
                        self._until,
                        self._monotonic() + RECOVERY_REQUEST_GAP_SECONDS,
                    )
            self._recovery_gate.release()


class RetryingUrlopen:
    def __init__(self, opener, *, monotonic=time.monotonic, sleep=time.sleep,
                 wall_time=time.time, stderr=None):
        self._opener = opener
        self._monotonic = monotonic
        self._sleep = sleep
        self._wall_time = wall_time
        self._stderr = sys.stderr if stderr is None else stderr
        self._hosts = {}
        self._hosts_lock = threading.Lock()

    def _host(self, request):
        url = request.full_url if isinstance(request, urllib.request.Request) else request
        host = urllib.parse.urlsplit(url).hostname or ""
        with self._hosts_lock:
            return self._hosts.setdefault(
                host, _HostBackoff(self._monotonic, self._sleep)
            )

    def _log(self, message):
        # Do not log URLs, requests, headers, authentication, or error bodies.
        print("setup HTTP retry: " + message, file=self._stderr, flush=True)

    def __call__(self, request, *args, **kwargs):
        host = self._host(request)
        for attempt in range(1, MAX_ATTEMPTS + 1):
            acquired = host.acquire()
            succeeded = False
            try:
                result = self._opener(request, *args, **kwargs)
                succeeded = True
                return result
            except urllib.error.HTTPError as error:
                if error.code not in RETRY_STATUSES:
                    raise
                delay = retry_after_seconds(error.headers, self._wall_time)
                if delay is None:
                    delay = min(60, 2 ** attempt)
                if delay > MAX_SERVER_WAIT_SECONDS:
                    self._log(
                        f"HTTP {error.code}; attempt {attempt}/{MAX_ATTEMPTS}; "
                        f"server wait {delay:.1f}s exceeds "
                        f"{MAX_SERVER_WAIT_SECONDS}s maximum; stopping without retry"
                    )
                    raise
                host.defer(delay)
                if attempt == MAX_ATTEMPTS:
                    self._log(
                        f"HTTP {error.code}; attempt {attempt}/{MAX_ATTEMPTS}; "
                        "retry limit reached"
                    )
                    raise
                self._log(
                    f"HTTP {error.code}; attempt {attempt}/{MAX_ATTEMPTS}; "
                    f"next attempt after at least {delay:.1f}s; shared host cooldown"
                )
                # The failing HTTP response is not used; release its resources.
                error.close()
            finally:
                host.release(acquired, succeeded)


def main():
    installer = (Path(__file__).resolve().parent / "runtime" / "experiments" /
                 "coupled_act_student50_release" / "setup_release_v3.py")
    original = urllib.request.urlopen
    urllib.request.urlopen = RetryingUrlopen(original)
    try:
        # run_path preserves the frozen script's __file__, imports, main(), and
        # argument handling, including --help and both offline install modes.
        runpy.run_path(str(installer), run_name="__main__")
    finally:
        urllib.request.urlopen = original


if __name__ == "__main__":
    main()
