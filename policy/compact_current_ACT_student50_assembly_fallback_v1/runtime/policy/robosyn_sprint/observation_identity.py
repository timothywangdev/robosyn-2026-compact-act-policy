"""Fingerprint policy-visible initial inputs for paired evaluation diagnostics."""

import hashlib
import json

import numpy as np


def observation_identity(encoded_obs):
    fields = {}
    numeric = {}
    for name, value in sorted(encoded_obs.items()):
        # Only RGB and joint positions actually consumed by the motor policy.
        if name != "observation.state" and not name.startswith("observation.images."):
            continue
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise TypeError(f"Cannot fingerprint object-valued observation: {name}")
        fields[name] = {
            "shape": list(array.shape),
            "dtype": array.dtype.str,
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
        # Keep numeric summaries outside the hashed fields: existing digest
        # comparisons remain valid after adding these magnitude diagnostics.
        values = array.astype(np.float64)
        if name == "observation.state":
            numeric[name] = {"values": values.reshape(-1).tolist()}
        else:
            channel_axis = 0 if array.ndim == 3 and array.shape[0] in (1, 3, 4) else -1
            axes = tuple(i for i in range(values.ndim) if i != channel_axis % values.ndim)
            numeric[name] = {
                "channel_axis": channel_axis,
                "mean": values.mean(axis=axes).tolist(),
                "std": values.std(axis=axes).tolist(),
                "min": values.min(axis=axes).tolist(),
                "max": values.max(axis=axes).tolist(),
                "units": "input pixel values; no normalization added",
            }
    return {
        "scope": "policy-visible RGB and qpos; does not establish hidden simulator-state equality",
        "fields": fields,
        "numeric_summary": numeric,
        "sha256": hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest(),
    }
