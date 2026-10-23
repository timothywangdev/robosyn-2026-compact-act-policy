# Setup HTTP retries

Run the policy's `setup.sh` with the same interpreter and arguments documented
in `README.md`. The outer `setup_entry.py` executes the original frozen installer
without modifying its source or its manifest, byte, package, or source checks.
`--help`, `--download-only`, full installation, and `--offline-check-only` retain
the original argument handling. The original installer remains SHA256
`31e1c4332df3c5e438e9aef7f3f44a0d179ba3982c5f09fcbb97ff4c7e171213`.

HTTP 429, 500, 502, 503, and 504 request-opening failures receive at most six
total attempts. A valid `Retry-After` header is honored in seconds or HTTP date
format. Without a valid header, waits are 2, 4, 8, 16, and 32 seconds. Server
waits of up to 900 seconds are supported; a larger wait stops with the original
HTTP error and a clear message, without retrying early. Retries are bounded by
attempt count, not by a shorter wall-clock deadline.

Threads in this installer process share cooldowns by requested hostname. After
the first transient error, queued requests to that host use one recovery gate
and leave at least 0.125 seconds after a successful request opens before the
next opens. Requests already in flight when the error arrives can finish.
Separate installer processes do not share this gate.

Logs report only HTTP status, attempt number, and delay; no URLs, tokens,
headers, or response bodies are logged. Authentication and missing-file errors
(403/404) are raised immediately. Response-body read failures and integrity
failures are not retried. Existing mismatched files and `.download-part` files
remain subject to the original installer's explicit inspection requirement;
the wrapper never deletes them or treats them as verified.

Run the network-free behavioral checks with:

```bash
python3.11 -B policy/compact_current_ACT_student50_assembly_fallback_v1/test_setup_entry.py
```
