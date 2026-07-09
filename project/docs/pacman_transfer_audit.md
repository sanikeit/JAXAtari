# Pacman Transfer Audit

This document assesses the compatibility and transfer effort required to run the current Model-Free and Model-Based planners (built for JAXAtari `MsPacman`) on the base `Pacman` environment.

---

## 1. Observation Compatibility

Comparison of `PacmanObservation` vs `MsPacmanObservation`.

| Field | Compatibility | Notes |
|-------|---------------|-------|
| `player_position` | Identical | `(2,)` pixel coordinates |
| `ghost_positions` | Identical | `(4, 2)` pixel coordinates |
| `power_pellets` | Identical | `(4,)` boolean array |
| `player_action` | Minor Difference | Range is `0-5` (no diagonals) instead of `0-8` |
| `ghost_actions` | Minor Difference | Range is `0-5` instead of `0-8` |
| `fruit_action` | Minor Difference | Shape is `(2,)` instead of scalar `()` in the observation space, which will cause unwrap issues if accessed directly. |
| `pellets` | **Major Difference** | Shape is `(18, 8)` instead of `(18, 14)`. Vertical pellet spacing is vastly different (`tile_y = y // 24` instead of `y // 8`). |

---

## 2. Maze Compatibility

| Component | Status | Notes |
|-----------|--------|-------|
| Tile Scale | Identical | 4 pixels per tile |
| Maze Grid Dimensions | **Major Difference** | MsPacman DOF grids are `(40, 44, 4)`. Pacman DOF grids are `(40, 48, 4)`. |
| `pixel_to_dof()` | Identical | Division by 4 remains valid, but bounding logic must update. |
| `pellet_to_dof()` | **Major Difference** | MsPacman maps `y` to `y * 2 + 3`. Pacman uses a completely different spatial distribution and cannot use this mapping. |

---

## 3. Ghost Compatibility

Inspection of `jaxatari.games.jax_pacman`:

| Component | Compatibility | Notes |
|-----------|---------------|-------|
| `pathfind` | Identical | Exact same signature. |
| Ghost Modes | Identical | Same enum values and states. |
| `get_chase_target` | **Major Difference** | In MsPacman, the signature is `(ghost_type, ghost_pos, blinky_pos, pacman_pos, pacman_action, actions, scatter_targets)`. In Pacman, the signature is simply `(player_pos)` because all ghosts directly target the player. |
| Routing Logic | **Major Difference** | `jax_pacman.py` removes all unique ghost targeting behaviors (Pinky looking ahead, Inky using vectors, etc.) and collapses them all to target Pacman directly. |

---

## 4. Current Project Component Audit

| Component | Classification | Explanation & Effort |
|-----------|----------------|----------------------|
| `utils/coords.py` | **C = Requires rewrite** | `is_walkable()` hardcodes `y < 44` which will crash on Pacman's 48-height maze. `extract_pellet_targets()` hardcodes `(18, 14)` loops and `y*2+3` math. Needs ~40 lines of adapter code to support both shapes. |
| `utils/danger_map.py` | **A = Works unchanged** | Operates strictly on pixel coordinates and general constants. |
| `utils/astar.py` | **A = Works unchanged** | Operates on arbitrary graphs. |
| `utils/ghost_sim.py` | **C = Requires rewrite** | `build_jit_ghost_step()` explicitly imports and passes 7 arguments to `get_chase_target()`. In Pacman, this function takes 1 argument. Will crash immediately. Needs an `if is_pacman` branch or an adapter interface (~30 lines). |
| `planners/mf_baseline.py` | **A = Works unchanged** | Relies entirely on `coords.py` for target abstraction. |
| `planners/mb_v2_planner.py` | **A = Works unchanged** | Relies entirely on `ghost_sim.py` for predictor abstraction. |

---

## 5. Transfer Difficulty Estimate

### A. Current MF baseline on Pacman: **Small (1–4 hours)**
To run the MF planners, we only need to update `utils/coords.py` to dynamically handle the `(18, 8)` pellet grid and the 48-height maze bounds. Once the coordinate math is fixed, all MF targeting and A* logic will transfer seamlessly.

### B. Current MBv2 planner on Pacman: **Moderate (1 day)**
Running MBv2 requires fixing the coordinate math *and* rewriting the JAX-native forward simulator in `utils/ghost_sim.py` to handle the different `get_chase_target` signatures and slightly different ghost step semantics between `jax_mspacman` and `jax_pacman`. Since JAX requires static shapes and logic, handling both environments cleanly in the same JIT-compiled loop will require careful adapter design.

---

## 6. Validation Plan

To systematically validate transfer without polluting the MsPacman experiments, execute the following checklist:

1. **MsPacman Maze 0 Sanity Check**
   - Run: `python project/experiments/run_mf_ablations.py --env MsPacman`
   - Verify baseline scores remain ~158.
2. **Pacman MF Transfer**
   - Run: `python project/experiments/run_mf_ablations.py --env Pacman`
   - Verify coordinate fixes work and agent correctly navigates the `(18, 8)` pellet grid.
3. **MsPacman MBv2 Sanity Check**
   - Run: `python project/experiments/evaluate_planners.py --env MsPacman`
   - Verify MBv2 prediction error remains near zero.
4. **Pacman MBv2 Transfer**
   - Run: `python project/experiments/evaluate_planners.py --env Pacman`
   - Verify `ghost_sim.py` adapter functions correctly and MBv2 successfully leverages the simplified Pacman routing logic.

### Recommendation
**Do not transfer to Pacman yet.** 
We just conclusively proved that *Future Prediction* is the absolute bottleneck for MsPacman, and we have a fully functioning, perfectly-simulated MBv2 predictor ready to go in `MsPacman` Maze 0. Switching to Pacman right now would require 1–2 days of adapter engineering to fix the pellet grids and ghost simulators. It is far more efficient to finish the MBv2 experiments in MsPacman, prove the solution works, and *then* build the adapters to transfer the finalized MB architecture to Pacman.
