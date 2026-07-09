# Archive — Obsolete Experiment Files

These files are **preserved for reference only**. They contain bugs that
produced invalid experimental results. Do not use them as starting points
for new work. See the corrected versions in `project/planners/` and
`project/experiments/` instead.

---

## Why Each File Is Archived

### `baseline_astar_BROKEN.py`
Initial A* implementation. All three coordinate bugs active:
- Navigates `MsPacmanMaze.MAZES[0]` (wrong grid)
- Pellet mapping uses pixel arithmetic on wrong scale
- Action mapping inverted (2=RIGHT should be 3=RIGHT)

### `mspacman_explore_INITIAL.py`
Initial exploration script. No bugs, but superseded — no planner logic.
Preserved as a reference for what `obs` fields are available.

### `phase1_mf_planner_WRONG_COORDS.py`
MF baseline with all three bugs active. All reported metrics invalid:
- "479 oscillations" is an artifact, not a real measurement
- Score ~20 is random-equivalent

### `phase1_ablations_WRONG_COORDS.py`
Goal commitment and reduced-replanning ablations built on broken MF baseline.
Both ablation conclusions are invalid.

### `phase2_mb_v0_WRONG_COORDS.py`
MB v0 (linear velocity extrapolation) with all three bugs active.
"MBv0 ≡ MF (0% divergence)" finding was real for this planner type, but
the metric was computed in the wrong coordinate space.

### `phase2_diagnostics_SUPERSEDED.py`
First divergence comparison attempt. Wrong coordinate space throughout.
Superseded by `project/diagnostics/deep_diagnostics.py`.

### `phase2_evaluate_all_WRONG_COORDS.py`
Prediction error + planning comparison with wrong coords and actions.
All MB v0/v1 error values and score/pellet/oscillation numbers are invalid.

### `phase2_deep_diagnostics_v1_SUPERSEDED.py`
First version of deep diagnostics. Used un-JIT-compiled ghost step inside
Python loop — caused indefinite hang (traced every JAX op on each call).
Fixed in v2 (`project/diagnostics/deep_diagnostics.py`) with `@jax.jit`.

### `debug_pellet_coords_UTILITY.py`
One-off debugging script that found the correct pellet→DOF mapping:
`dof_gx = pellx*2+1, dof_gy = pelly*3+2`. Superseded by `utils/coords.py`.

### `test_exact_pellets_UTILITY.py` / `test_pellet_mapping_UTILITY.py`
One-off test scripts from early coordinate investigation. Superseded.

---

## Lessons Learned

1. **Always verify the coordinate space before building on top of it.**
   The three-bug cascade happened because no sanity check was done on whether
   A* was actually finding meaningful paths (mean path length should have been
   checked immediately).

2. **Check action effects empirically before assuming.**
   A one-line `env.step(state, action)` loop would have caught Bug C immediately.

3. **JAX-native functions are strongly preferred over custom re-implementations.**
   MB v0 and MB v1 hand-crafted simulators were fragile and inaccurate.
   MB v2 using JAXAtari's own `pathfind`/`get_chase_target` achieves 0.013 cell
   error at K=1 — near-perfect — with no custom ghost logic at all.

4. **JIT-compile before looping.**
   Calling traced JAX functions in a Python loop without `@jax.jit` caused
   catastrophic slowdown (re-tracing on every iteration).
