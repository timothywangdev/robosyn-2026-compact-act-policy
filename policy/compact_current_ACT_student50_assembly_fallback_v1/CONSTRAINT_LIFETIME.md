# Native constraint lifetime compatibility

The official SDK v0.2.4.post1 removes an arena-owned fixed constraint before
discarding its borrowed Python wrapper. DexSim 0.4.3 binds `FixedConstraint`
through virtual base classes. The pybind11 wrapper must traverse those bases
during deregistration, including for a borrowed reference, so its native object
must still be alive at that point.

A full native Assembly episode with generator seed `2026102201` reproduced a
segmentation fault on the 500-action horizon reset. A second diagnostic run
captured the native fault during Python list-slot replacement. Read-only binding
inspection confirmed a borrowed return policy and virtual-base deregistration;
the error was not a Python ownership-policy double-free.

`constraint_lifetime.py` installs a disclosed compatibility repair when Assembly's
model is constructed:

1. Cache the arena and constraint name while both remain valid.
2. Require the SDK slot to hold the sole Python reference to the borrowed handle.
3. Discard that wrapper while its native owner is still alive.
4. Call the ordinary native `arena.remove_constraint(name)` exactly once.
5. Remove registered constraints by the same SDK API before ordinary scene
   destruction, including deferred destruction.

If native removal raises, tracking is restored from the arena where possible and
the original exception propagates. Additional external handle references, invalid
arena resolvers, and unrecognized SDK/native implementations fail explicitly.
No native handles are retained or quarantined, and no binary is edited.

The exact SDK source files and native binding are SHA-guarded. This repair is
specific to the verified SDK v0.2.4.post1 / DexSim 0.4.3 implementation. Other
SDK builds need their own validation; this adapter will not silently patch them.

The policy weights, action outputs, callback sizes, physics configuration,
constraint creation, automatic resets, official horizons, feasibility filtering,
and success computation are preserved. The compatibility repair affects object
destruction order. The organizer entry also sets
`EMBODICHAIN_SIM_EXIT_PROCESS=0` so the SDK returns to the official metric writer.

The standard-library regression tests in `test_constraint_lifetime.py` compare
the original SDK ordering with the repaired ordering and cover aliases, partial
removal, failure recovery, and shutdown. Native validation must additionally
exercise the original failing scene, repeated episodes, and final process/GPU
cleanup. The diagnostic inspector and native debugger are not submission
dependencies.
