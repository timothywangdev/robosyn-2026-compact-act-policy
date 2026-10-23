# compact_current_ACT_student50_assembly_fallback_v1

Retain this entire directory when preparing the evaluation branch. It includes `setup.sh`, `eval.sh`, `entry_cli.py`, `deploy_policy.py`, `deploy_policy.yml`, the private `runtime/` source tree, and its hash inventory. No participant source file outside this directory is required.

Run from the official evaluation checkout root, using the organizer's environment:

```bash
PYTHON311=python3.11 bash policy/compact_current_ACT_student50_assembly_fallback_v1/setup.sh --download-only --root /tmp/rscp-policy --revision a164024eacdfb66ffdc3a2e3bbaad2bd5d5f46f8
export ROBOSYN_STUDENT50_PACKAGE_ROOT=/tmp/rscp-policy
export PYTHON_BIN=/path/to/official/python
bash policy/compact_current_ACT_student50_assembly_fallback_v1/eval.sh --task click_bell --seed 123 --episodes 20 --output /tmp/robosyn-click-bell
```

Replace the interpreter, seed, episode count, task and fresh output as appropriate. The named CLI is intentional. `--evaluator-root` or `ROBOSYN_EVALUATOR_ROOT` selects a different official checkout; automatic detection uses this directory's `../..`. By default the launcher calls that checkout's `scripts/eval_policy.py` with the selected interpreter and existing dependencies. The exact pinned LeRobot ACT implementation is loaded from the downloaded artifact's `upstream/lerobot-sim/src` to preserve the submitted actor source; installed dependencies still come from the organizer environment. A preexisting incompatible imported ACT module is rejected explicitly. All ten task names are supported. Assembly retains the exact released ACT route (50 generated commands, 10-command callbacks); the other nine retain the original trained Student50 route (50-command callbacks).

The default artifact root is `runtime/artifact/`; an explicitly short path such as `/tmp/rscp-policy` is recommended for native simulator paths. Download-only setup requires CPython 3.11 and does not install packages or replace the organizer's environment. Without `--download-only`, setup creates the standalone frozen environment; `eval.sh --standalone` explicitly uses its archived evaluator.

To verify a policy-only prepared copy, run:

```bash
python3.11 policy/compact_current_ACT_student50_assembly_fallback_v1/validate_package.py --python /path/to/python3.11 --artifact-root /tmp/rscp-policy --output /tmp/robosyn-packaging-checks
```

This validates source hashes, isolated direct hook imports, setup/eval help, and all ten actual shell commands in check-only mode against the immutable artifact. It builds no model or simulator world. Omit `--artifact-root` for source/import checks before downloading. Its inert official-evaluator fixture verifies command construction only; it does not qualify the native organizer environment.
