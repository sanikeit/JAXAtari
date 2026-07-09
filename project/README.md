# JAXAtari MsPacman — Praktikum Project

**TU Darmstadt Praktikum** | JAXAtari Maze Solving  
Comparing Model-Free, Model-Based, and Neuro-Symbolic planners for MsPacman.

---

## Repository Layout

```
project/
├── docs/                          # All research documentation
│   ├── 2026-06-15_root_cause_analysis.md   # Bug findings & fix record
│   ├── current_architecture.md             # What comes from JAX vs us
│   ├── jaxatari_dependencies.md            # Full dependency map
│   └── project_status.md                   # Current state & open questions
│
├── planners/                      # Planner implementations
│   ├── mf_baseline.py             # Model-Free: static danger map
│   └── mb_v2_planner.py           # Model-Based v2: JAXAtari-native sim
│
├── diagnostics/                   # Diagnostic and validation scripts
│   └── deep_diagnostics.py        # Full root-cause diagnostic suite
│
├── experiments/                   # Runnable experiment scripts
│   └── evaluate_planners.py       # 3-way MF / MBv0 / MBv2 comparison
│
├── utils/                         # Shared project utilities
│   ├── coords.py                  # Coordinate system reference & converters
│   ├── danger_map.py              # Danger map builder
│   ├── ghost_sim.py               # JAX-native ghost forward simulator
│   └── astar.py                   # Spatiotemporal A* planner
│
├── results/                       # Experiment output (CSVs, logs, plots)
│
├── archive/                       # Obsolete scripts preserved for reference
│   └── README.md                  # Why each file was archived
│
└── README.md                      # This file
```

---

## Guiding Principle

**JAXAtari is the source of truth.** We do not copy or modify JAXAtari source code.  
We only *import* from it:

```python
from jaxatari.games.jax_mspacman import (
    JaxPacman, GhostMode,
    get_allowed_directions, get_chase_target,
    pathfind, get_new_position,
)
from jaxatari.games.mspacman_mazes import MsPacmanMaze
```

See `docs/jaxatari_dependencies.md` for the full dependency list.

---

## Quick Start

All scripts are run from the **repository root** (`JAXAtari/`):

```bash
# Run the 3-way planner comparison (MF vs MBv0 vs MBv2)
conda run -n jaxatari python project/experiments/evaluate_planners.py

# Run the full diagnostic suite
conda run -n jaxatari python project/diagnostics/deep_diagnostics.py
```

---

## Current Best Result (2026-06-15)

| Planner | Score | Pellets | Oscillations | Survival |
|---------|-------|---------|--------------|----------|
| MF (static danger map) | 30 | 3 | 169 | 300 |
| MBv0 (linear extrapolation) | 30 | 3 | 179 | 300 |
| **MBv2 (JAXAtari-native)** | **60** | **6** | **151** | 300 |

Settings: K=15, radius=3, mult=5, 300 steps, seed=42.

**Key finding:** MBv2 uses JAXAtari's own ghost routing logic as the predictor.  
At K=1 it achieves 0.013 cell prediction error — near-perfect.  
The remaining bottleneck is **planner integration at ghost-blocked corridors**, not prediction.

---

## Research Plan

- [x] Phase 1 — Model-Free baseline (A* + static danger map)
- [x] Phase 2 — Model-Based (MBv0 linear, MBv2 JAX-native)
- [x] Root-cause analysis & coordinate bug fixes
- [x] Phase 3 — Fix corridor-blocking oscillation (path commitment / reversal penalty)
- [x] MFv2 Global Targeting — Proved target selection is NOT the dominant bottleneck; future prediction is required.
- [ ] Phase 4 — MBv2 Formal Ablations and Optimization
- [ ] Phase 5 — Transfer across mazes
