# Project Status — 2026-06-15

---

## What Is Working

| Component | Status | Evidence |
|-----------|--------|---------|
| DOF grid coordinate system | ✅ Confirmed correct | `dof[18,25].any() = True` at Pacman start |
| Pellet → DOF mapping | ✅ Confirmed correct | 236/236 pellets walk-able at `(pellx*2+1, pelly*3+2)` |
| Action mapping | ✅ Confirmed correct | Empirical test: action=3 moves pixel_x +1 |
| JAX-native ghost simulator | ✅ Working | K=1 prediction error = 0.013 cells |
| Spatiotemporal A* | ✅ Working | Score=60, 6 pellets in 300 steps |
| MBv2 > MF | ✅ Validated | 2× score and pellets vs MF baseline |
| JIT compilation | ✅ Fast | Ghost step compiles once, runs quickly |

---

## What Is Not Working / Still Broken

| Issue | Status | Root Cause |
|-------|--------|------------|
| High oscillation (151/300 steps) | ❌ Unresolved | Danger map inflates cost of only viable corridor → reversal loop |
| Low plan divergence (0.7%) | ⚠️ Partially | Ghosts are usually far from chosen path; MB only activates at key junctions |
| Score still low (60 in 300 steps) | ⚠️ Partially | Directly follows from oscillation |
| No waiting/retreat strategy | ❌ Not implemented | Planner defaults to NOOP when all paths blocked |

---

## Current Best Baseline

**MBv2 with JAXAtari-native ghost simulator**

```
Planner : MB v2 (JAX-native)
K       : 15 steps ahead
Radius  : 3 DOF cells
Mult    : 5 cost per unit distance
Score   : 60 (vs 30 for MF)
Pellets : 6  (vs 3 for MF)
Osc     : 151/300 steps
Survival: 300/300 (never dies)
Diverge : 0.7% of steps vs MF
```

Files: `project/planners/mb_v2_planner.py`

---

## Open Research Questions

### Q1: Can reversal penalty fix the oscillation?
**Hypothesis:** Adding a cost `alpha` when the chosen step reverses the last action will prevent corridor-entrance bouncing.  
**Next step:** Implement in `spatiotemporal_astar` and run 300-step episode.

### Q2: Does MB actually help at ghost-blocked junctions?
**Hypothesis:** The 0.7% divergence steps are specifically when a ghost is predicted to enter a corridor ahead. MB reroutes; MF walks in.  
**Next step:** Log the exact game state at each divergent step. Visualise ghost position vs chosen path.

### Q3: How does performance scale across mazes?
**Status:** All experiments run on Maze 0 only. Mazes 1–4 have different topologies.  
**Next step:** Run `evaluate_planners.py` for maze_idx in [0, 1, 2, 3, 4].

### Q4: Is Neuro-Symbolic necessary?
**Current answer:** No, for prediction. MBv2 already achieves near-perfect ghost prediction using JAXAtari's own functions.  
**What a learned predictor would add:** Only if we need to predict in partially-observable or multi-maze settings where we cannot access internal ghost state.  
**Decision:** Defer until Phase 3 (oscillation fix) and Phase 4 (cross-maze transfer) are evaluated.

### Q5: What is the theoretical upper-bound score?
**Status:** Unknown. Need to compute the maximum pellets collectable in 300 steps given ghost movement constraints.

---

## Experimental Baseline Record (Valid Results Only)

| Date | Script | Planner | Score | Pellets | Osc | Notes |
|------|--------|---------|-------|---------|-----|-------|
| 2026-06-15 | evaluate_planners.py | MF | 30 | 3 | 169 | First valid run |
| 2026-06-15 | evaluate_planners.py | MBv0 | 30 | 3 | 179 | No better than MF |
| 2026-06-15 | evaluate_planners.py | MBv2 | 60 | 6 | 151 | **Best so far** |
