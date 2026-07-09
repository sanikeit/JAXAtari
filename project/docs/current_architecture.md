# Current Architecture

Describes what comes from JAXAtari (unchanged, imported only) versus what our project implements.

---

## What Comes from JAXAtari (Read-Only, Never Modified)

### Environment

| Symbol | Module | Description |
|--------|--------|-------------|
| `JaxPacman` | `jaxatari.games.jax_mspacman` | Main environment class; `.reset()`, `.step()`, `.consts` |
| `PacmanState` | `jaxatari.games.jax_mspacman` | Full internal state (ghosts, pellets, level, score, …) |
| `GhostsState` | `jaxatari.games.jax_mspacman` | Ghost sub-state: positions, actions, modes, types, timers |
| `GhostMode` | `jaxatari.games.jax_mspacman` | Enum: RANDOM=0, CHASE=1, SCATTER=2, FRIGHTENED=3, BLINKING=4, RETURNING=5, ENJAILED=6 |
| `MsPacmanConstants` | `jaxatari.games.jax_mspacman` | All game constants (speeds, durations, scatter targets, …) |

### Observations

| Field | Source | Notes |
|-------|--------|-------|
| `obs.player_position` | `JaxPacman.reset/step` | (2,) pixel coords |
| `obs.player_action` | `JaxPacman.reset/step` | Current direction (0–5) |
| `obs.ghost_positions` | `JaxPacman.reset/step` | (4, 2) pixel coords |
| `obs.ghost_actions` | `JaxPacman.reset/step` | (4,) directions |
| `obs.pellets` | `JaxPacman.reset/step` | (18, 14) bool — live pellets |
| `obs.power_pellets` | `JaxPacman.reset/step` | (4,) bool |
| `state.ghosts.modes` | `PacmanState` | (4,) GhostMode uint8 |
| `state.ghosts.types` | `PacmanState` | (4,) GhostType uint8 |
| `state.score` | `PacmanState` | Scalar |
| `state.level.collected_pellets` | `PacmanState` | Scalar |

### Maze Definitions

| Symbol | Module | Description |
|--------|--------|-------------|
| `MsPacmanMaze` | `jaxatari.games.mspacman_mazes` | Static maze data class |
| `MsPacmanMaze.MAZES` | — | List of binary 2D arrays (display/rendering grid — **do not use for navigation**) |
| `MsPacmanMaze.TILE_SCALE` | — | 4 (pixels per tile) |
| `MsPacmanMaze.WIDTH / HEIGHT` | — | 160 / 176 pixels |

### DOF Grid (Navigation Graph)

| Symbol | Access | Description |
|--------|--------|-------------|
| `consts.DOF_MAZES` | `env.consts.DOF_MAZES` | All maze DOF grids; `DOF_MAZES[0]` = Maze 0 |
| `dof[gx, gy]` | — | bool[4] — allowed directions (UP, RIGHT, LEFT, DOWN) |
| `dof[gx, gy].any()` | — | True if cell is walkable |

**Shape:** `(40, 44, 4)`. Index as `dof[gx, gy]` where `gx = pixel_x // 4`, `gy = pixel_y // 4`.

### Ghost Routing Functions

These are the JAXAtari functions our MB v2 simulator wraps:

| Function | Module | Description |
|----------|--------|-------------|
| `get_allowed_directions(pos, action, dofmaze, directions, is_ghost)` | `jax_mspacman` | Returns array of valid turn directions from a pixel position |
| `get_chase_target(ghost_type, ghost_pos, blinky_pos, pacman_pos, pacman_action, actions, scatter_targets)` | `jax_mspacman` | Computes chase-mode target for each ghost (Blinky→Pacman, Pinky→4-ahead, Inky→Blinky-vector, Sue→distance-gated) |
| `pathfind(position, direction, target, allowed, key, actions, directions)` | `jax_mspacman` | Single-step direction choice: Manhattan-cost minimisation with axis priority tie-breaking |
| `get_new_position(position, action, consts)` | `jax_mspacman` | Apply one pixel-step of movement with tunnel wrap |

### Constants Used

| Constant | Value | Description |
|----------|-------|-------------|
| `consts.ACTIONS` | `[(0,0),(0,0),(0,-1),(1,0),(-1,0),(0,1)]` | Pixel delta per action index |
| `consts.DIRECTIONS` | `[UP, RIGHT, LEFT, DOWN]` | Turn direction set |
| `consts.SCATTER_TARGETS` | 4 corner positions | Ghost scatter targets by type |
| `consts.INITIAL_PACMAN_POSITION` | `[75, 102]` | Start in pixel coords |

---

## What Our Project Implements

### `project/utils/` — Shared Utilities

| File | Responsibility |
|------|----------------|
| `coords.py` | Coordinate conversions: `pixel_to_dof`, `pellet_to_dof`, `is_walkable`, `extract_pellet_targets`, `get_action_for_path` |
| `danger_map.py` | `build_danger_maps(predicted_positions, modes, radius, mult)` → list of cost dicts |
| `ghost_sim.py` | `build_jit_ghost_step(consts)` + `forward_roll(jit_step, ghost_state, ...)` — wraps JAXAtari routing in a JIT-compiled MB predictor |
| `astar.py` | `spatiotemporal_astar(dof, start, targets, danger_maps)` — temporally-indexed A* over the DOF grid |

### `project/planners/` — Planning Agents

| File | Responsibility |
|------|----------------|
| `mf_baseline.py` | Model-Free planner: freezes t=0 ghost positions as a static K-step danger map |
| `mf_ablation_planner.py` | Extends MF planner with Reversal Penalty and Goal Commitment tracking |
| `mf_global_planner.py` | Extends ablation planner with Global Target Selection strategies (Farthest, Cluster, Utility) |
| `mb_v2_planner.py` | Model-Based v2: rolls ghost state forward K steps with JAXAtari-native simulator, then runs spatiotemporal A* |

### `project/diagnostics/` — Validation Scripts

| File | Responsibility |
|------|----------------|
| `deep_diagnostics.py` | Full suite: prediction accuracy (MBv0 vs MBv2), danger-radius ablation, planner decision log, MF vs MB divergence |
| `mf_target_diagnostic.py` | Deep diagnostic of MF target selection distance distribution; generates matplotlib snapshot visualisations |

### `project/experiments/` — Runnable Comparisons

| File | Responsibility |
|------|----------------|
| `evaluate_planners.py` | 3-way MF / MBv0 / MBv2 comparison; danger-radius sweep |
| `run_mf_ablations.py` | Sweep over Reversal Penalty and Goal Commitment magnitudes across 5 seeds |
| `run_mf_global_ablations.py` | Sweep over global target heuristics; computes Maze Coverage and renders Trajectory Heatmaps |

---

## Data Flow Diagram

```
JAXAtari                          Our Project
─────────────────────────────     ────────────────────────────────
JaxPacman.reset() ──────────────→ obs, state
JaxPacman.step(action) ─────────→ obs', state', reward, done

obs.player_position ────────────→ utils/coords.pixel_to_dof()
obs.ghost_positions ────────────→ utils/ghost_sim.forward_roll()
  │  uses get_allowed_directions  │
  │       get_chase_target        │
  │       pathfind                │
  │       get_new_position        ↓
                             danger_maps (K danger cost dicts)
                                  │
obs.pellets ────────────────────→ utils/coords.extract_pellet_targets()
consts.DOF_MAZES[0] ────────────→ utils/astar.spatiotemporal_astar()
                                  │
                                  ↓
                             action (2/3/4/5)
                                  │
                                  └─→ JaxPacman.step(action)
```
