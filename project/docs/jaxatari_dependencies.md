# JAXAtari Dependency Map

Every JAXAtari symbol currently imported by our project, where it comes from, and which of our files uses it.

---

## `jaxatari.games.jax_mspacman`

| Symbol | Type | Used In |
|--------|------|---------|
| `JaxPacman` | Class | `experiments/evaluate_planners.py`, `diagnostics/deep_diagnostics.py` |
| `GhostMode` | IntEnum | `utils/ghost_sim.py`, `utils/danger_map.py`, `planners/mf_baseline.py`, `planners/mb_v2_planner.py` |
| `get_allowed_directions` | Function | `utils/ghost_sim.py` |
| `get_chase_target` | Function | `utils/ghost_sim.py` |
| `pathfind` | Function | `utils/ghost_sim.py` |
| `get_new_position` | Function | `utils/ghost_sim.py` |
| `MsPacmanConstants` | Class | Accessed via `env.consts` — not directly imported |

## `jaxatari.games.mspacman_mazes`

| Symbol | Type | Used In |
|--------|------|---------|
| `MsPacmanMaze` | Class | `diagnostics/deep_diagnostics.py` (for `MsPacmanMaze.MAZES` in prediction accuracy section only) |

---

## Accessed via `env.consts` (MsPacmanConstants fields)

These are not imported directly but accessed through `env.consts`:

| Field | Access Path | Used In |
|-------|-------------|---------|
| `DOF_MAZES` | `env.consts.DOF_MAZES[0]` | `utils/ghost_sim.py`, `utils/coords.py` (caller passes it), `planners/*` |
| `ACTIONS` | `env.consts.ACTIONS` | `utils/ghost_sim.py` (passed to `pathfind`, `get_chase_target`) |
| `DIRECTIONS` | `env.consts.DIRECTIONS` | `utils/ghost_sim.py` (passed to `get_allowed_directions`, `pathfind`) |
| `SCATTER_TARGETS` | `env.consts.SCATTER_TARGETS` | `utils/ghost_sim.py` (passed to `get_chase_target`) |

---

## JAX / Standard Library

| Symbol | Package | Used In |
|--------|---------|---------|
| `jax.jit` | `jax` | `utils/ghost_sim.py` |
| `jax.random.split` | `jax` | `utils/ghost_sim.py`, `planners/*`, `experiments/*` |
| `jax.lax.cond` | `jax` | `utils/ghost_sim.py` (inside JIT boundary) |
| `jnp.array`, `jnp.where`, `jnp.sum` | `jax.numpy` | `utils/ghost_sim.py`, `planners/*` |
| `numpy` | `numpy` | `utils/ghost_sim.py` (output conversion), `utils/danger_map.py` |
| `heapq` | stdlib | `utils/astar.py` |

---

## Functions We Do NOT Need to Reimplement

The following JAXAtari-internal functions handle logic we previously tried to reimplement (and failed at in MB v0/v1):

| JAXAtari Function | What It Does | Our Equivalent |
|---|---|---|
| `pathfind` | Ghost direction selection (Manhattan cost + axis priority) | **No equivalent needed** — we call JAXAtari's directly |
| `get_chase_target` | Blinky/Pinky/Inky/Sue target computation | **No equivalent needed** |
| `get_new_position` | Pixel movement with tunnel wrap | **No equivalent needed** |
| `get_allowed_directions` | Valid turns from pixel position | **No equivalent needed** |

---

## Import Template

Canonical imports for any new project file:

```python
import jax
import jax.numpy as jnp
import numpy as np

# JAXAtari — import only, never modify
from jaxatari.games.jax_mspacman import (
    JaxPacman,
    GhostMode,
    get_allowed_directions,
    get_chase_target,
    pathfind,
    get_new_position,
)
from jaxatari.games.mspacman_mazes import MsPacmanMaze   # only if needed

# Our project utilities
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from project.utils.coords import pixel_to_dof, pellet_to_dof, extract_pellet_targets, get_action_for_path
from project.utils.danger_map import build_danger_maps, static_danger_map
from project.utils.ghost_sim import build_jit_ghost_step, forward_roll
from project.utils.astar import spatiotemporal_astar
```
