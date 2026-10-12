# Compact current-feature ACT policy for RoboSyn 2026

This repository contains the selected `distill_visual` policy and its all-ten-task deployment interface. A learned mapper uses the current feature maps from the same ACT image-backbone pass, plus current joint positions, to condition a shared recurrent correction of both arms and grippers. Each refill generates and enqueues 50 commands. The policy uses three official cameras, preserves complete-hand ACT fallback, and makes no external inference calls. Qwen is used offline for training targets and is absent from the deployed model.

**Release status:** public code is prepared; the checkpoint repository is being staged privately. A protected confirmation evaluation and the installed-package native test remain pending. Clean CPython 3.11 offline installation and all-ten-task CPU loader/API checks passed. A ten-case development screen scored 6/10 against 3/10 for the plain official ACT50 control; that small screen does not establish a leaderboard improvement. The comparison includes the selected policy's Drawer conversion and terminal handling, as well as learned corrections.

## Required files and environment

The organizer-facing adapter is `policy/compact_current_ACT_student50_release_v1/`:

- `eval.sh`
- `deploy_policy.yml`
- `deploy_policy.py`, exporting `get_model`, `eval`, `reset_model`, and `close_model`

Use Linux x86_64, **CPython 3.11**, and an NVIDIA GPU/driver compatible with the included Torch CUDA wheels. The intended competition device is one RTX 5090 32GB; direct RTX 5090 qualification is pending. The full package pins the simulator, evaluator, LeRobot and all Python dependencies in `requirements/sim.lock`, and includes 282 offline distributions. Native Linux libraries include `libGLU.so.1`, `libGL.so.1`, and `libgomp.so.1` (Ubuntu packages `libglu1-mesa`, `libgl1`, and `libgomp1`).

The artifact is about 9.01GB, including about 2.08GB of model assets. Allow at least 32GiB free disk before download/install. No models, datasets, credentials, or generated environments are stored in this GitHub repository. The immutable Hugging Face package preserves the exact source, simulator assets, dependency wheels, checkpoint metadata and path-alias closure verified during preparation. Historical metadata inside that immutable package is validation provenance, not an additional training-data download.

## Setup and evaluate

After the checkpoint repository is made public, use its exact 40-character revision shown in the final release notes. `HF_REVISION` below is a pending publication placeholder; it is not a runnable revision.

```bash
bash setup.sh --revision HF_REVISION
bash policy/compact_current_ACT_student50_release_v1/eval.sh \
  --task click_bell --seed 123 --episodes 20 --output /tmp/robosyn-bell-run
```

Setup downloads only the SHA-bound immutable package and uses the pinned offline CPython 3.11 install recipe. It verifies every file, creates `artifact/.venvs/sim`, and validates the installed versions. `PYTHON311` may select the installed CPython 3.11 executable. A failed/incomplete existing installation must be inspected before another install; setup never substitutes a different dependency lock.

For each new evaluation, pass the organizer's generator seed, episode count and a fresh output directory. A local metadata binder derives the installed paths automatically after full byte verification. It requires no team token, root session, or manual approval. The team token is used only in the official submission portal. The actor and action code remain unchanged. Byte verification adds constructor startup time; it is outside steady inference.

The command preserves the official full task horizon. The organizer can also import `policy.compact_current_ACT_student50_release_v1.deploy_policy` and pass the official task arguments to the four hooks; `ROBOSYN_STUDENT50_PACKAGE_ROOT` selects the downloaded artifact directory. The actor maintains its own 50-command queue. Its SDK callback cadence is 10 commands; that does not shorten the executed 50-command refill.

## Checkpoints and all-ten-task commands

All tasks share `models/current-student/distill_visual.pt` and `models/frozen-coupled-role/coupled-all14.safetensors`. Each task additionally loads its own released ACT checkpoint. The full machine-readable mapping is in [task-checkpoints.json](task-checkpoints.json). The selected artifact manifest SHA256 is `191e3a69625175b96c685304e19d5362522e16338943cd6c3065448cabb89515`.

| Task | ACT checkpoint at the pinned HF revision | Horizon |
|---|---|---:|
| click_bell | [models/act/click_bell](https://huggingface.co/0x00ff00ff/robosyn-2026-compact-act-policy/tree/HF_REVISION/models/act/click_bell) | 361 |
| drawer_open_place | [models/act/drawer_open_place](https://huggingface.co/0x00ff00ff/robosyn-2026-compact-act-policy/tree/HF_REVISION/models/act/drawer_open_place) | 900 |
| handle_basket | [models/act/handle_basket](https://huggingface.co/0x00ff00ff/robosyn-2026-compact-act-policy/tree/HF_REVISION/models/act/handle_basket) | 500 |
| item_assembly | [models/act/item_assembly](https://huggingface.co/0x00ff00ff/robosyn-2026-compact-act-policy/tree/HF_REVISION/models/act/item_assembly) | 500 |
| items_handover | [models/act/items_handover](https://huggingface.co/0x00ff00ff/robosyn-2026-compact-act-policy/tree/HF_REVISION/models/act/items_handover) | 350 |
| manipulate_pipette | [models/act/manipulate_pipette](https://huggingface.co/0x00ff00ff/robosyn-2026-compact-act-policy/tree/HF_REVISION/models/act/manipulate_pipette) | 1000 |
| mixer_operating | [models/act/mixer_operating](https://huggingface.co/0x00ff00ff/robosyn-2026-compact-act-policy/tree/HF_REVISION/models/act/mixer_operating) | 500 |
| sample_loading | [models/act/sample_loading](https://huggingface.co/0x00ff00ff/robosyn-2026-compact-act-policy/tree/HF_REVISION/models/act/sample_loading) | 500 |
| table_rearrangement | [models/act/table_rearrangement](https://huggingface.co/0x00ff00ff/robosyn-2026-compact-act-policy/tree/HF_REVISION/models/act/table_rearrangement) | 361 |
| water_pouring | [models/act/water_pouring](https://huggingface.co/0x00ff00ff/robosyn-2026-compact-act-policy/tree/HF_REVISION/models/act/water_pouring) | 500 |

Use the same command above with `--task` set to each table row. Seeds and episode counts are supplied by the organizer, not selected by the policy.

## Training and validation scope

Training uses the competition's provided demonstrations. The coupled correction uses the existing FIT current-observation/ACT-action caches; the visual student is fitted to offline teacher targets using current images and joint positions. Future demonstration actions are training labels only. At inference, no future frames, demonstration identifiers, hidden simulator state, or outcome labels are provided. The ACT backbone and task checkpoints are retained; this release does not claim a newly pretrained foundation model or a high-level text-generating planner.

The immutable package includes upstream source components used by the deployment; their existing licenses and notices apply. This repository contains the corresponding unchanged source files plus the automatic release binder and installer wrapper. It does not claim that every historical experiment or archived model was successful.
