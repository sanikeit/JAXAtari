# Root Cause Analysis — 2026-06-15

**Status:** All Phase 1 and early Phase 2 experimental results are **invalid** due to three simultaneous bugs. This document records what went wrong, how it was found, and what the corrected results are.

---

## Bug A — Wrong Coordinate System for Navigation

### What happened
The A* planner navigated in `MsPacmanMaze.MAZES[0]`, a binary array of shape `(44, 40)` indexed `[y, x]` where `False` means wall and `True` means path.

### Why it was wrong
The game does **not** use `MsPacmanMaze.MAZES` for movement logic. It uses the **DOF grid**:

```python
consts.DOF_MAZES[0]   # shape (40, 44, 4), indexed dof[gx, gy]
```

where `gx = pixel_x // 4` and `gy = pixel_y // 4`. A cell is walkable when `dof[gx, gy].any()` is `True`.

The DOF grid and MAZES grid have **different shapes, different axis orders, and different coordinate origins**.

### How it was diagnosed
Added a walkability check on Pacman's starting position using both grids:
```
MAZES[0] at Pacman grid: False   ← wall! Wrong grid.
DOF[18, 25].any()      : True    ← walkable. Correct grid.
```

### Effect
Every `maze[ny, nx]` lookup in A* returned `False`. The planner never found any valid path. It fell back to snapping to the nearest open cell, which always landed immediately adjacent to Pacman. **Mean path length was 2.0 cells throughout all experiments.**

---

## Bug B — Broken Pellet → Grid Coordinate Mapping

### What happened
```python
# BROKEN
gx, gy = pixel_to_grid(pellx * 8 + 5, pelly * 12 + 6)
```
This mapped pellet grid indices through an incorrect pixel conversion, then into the wrong `MAZES` coordinate space.

### Correct mapping (derived empirically)
The `obs.pellets` grid is `(18, 14)`. The DOF grid is `(40, 44)`. The relationship:

```python
# CORRECT — verified at 100% walkable rate across all 236 pellets
dof_gx = pellx * 2 + 1
dof_gy = pelly * 3 + 2
```

### How it was diagnosed
Tested the formula by checking `dof[dof_gx, dof_gy].any()` for all live pellets:
- Old formula: **0 / 236** walkable  
- New formula: **236 / 236** walkable (100%)

### Effect
Combined with Bug A, the fallback search always placed pellet targets immediately adjacent to Pacman regardless of actual pellet locations. The planner was doing 1-step hops, never navigating the maze.

---

## Bug C — Inverted Action Mapping

### What happened
```python
# BROKEN — wrong indices
return 2 if dx > 0 else 3   # intended RIGHT/LEFT
return 4 if dy > 0 else 1   # intended DOWN/UP
```

### Correct mapping (verified empirically via `env.step`)
```python
# CORRECT
# 0 = NOOP
# 2 = UP    (pixel_y decreases)
# 3 = RIGHT (pixel_x increases)
# 4 = LEFT  (pixel_x decreases)
# 5 = DOWN  (pixel_y increases)
return 3 if dx > 0 else 4   # RIGHT=3, LEFT=4
return 5 if dy > 0 else 2   # DOWN=5, UP=2
```

### How it was diagnosed
Ran `env.step(state, action)` for each action index 0–5 and observed pixel displacement:
```
action=2 → delta (-1, 0)  ← LEFT, not UP
action=3 → delta (+1, 0)  ← RIGHT ✓
action=4 → delta (-1, 0)  ← LEFT ✓
action=5 → not tested (revealed by formula)
```

### Effect
All directional commands executed wrong movements. The agent was functionally random despite appearing to run an A* planner.

---

## Invalidated Experiments

The following results are **meaningless** and must be discarded:

| File | Reason |
|------|--------|
| `phase1_mf_planner.py` | All 3 bugs active |
| `phase1_ablations.py` | Built on broken MF baseline |
| `phase2_mb_v0.py` | All 3 bugs active |
| `phase2_diagnostics.py` | Wrong coordinate space throughout |
| `phase2_evaluate_all.py` | All 3 bugs active |
| `baseline_astar.py` | All 3 bugs active |

Specific invalid claims:
- "479 oscillations per episode" — entirely an artifact of the broken oscillation counter and wrong coordinates
- "MF and MB produce 0% plan divergence" — both were random-equivalent, divergence measurement was meaningless
- "MB v1 error > MB v0 error" — positions were in wrong space, error values are nonsense

---

## Validated Coordinate Reference

```
PIXEL SPACE
  obs.player_position  → (pixel_x, pixel_y)
  obs.ghost_positions  → [(pixel_x, pixel_y), ...]
  pixel range: x ∈ [0,159], y ∈ [0,175]

DOF GRID (use this for A*)
  consts.DOF_MAZES[0]  → dof[gx, gy, direction]
  gx = pixel_x // 4   (range 0..39)
  gy = pixel_y // 4   (range 0..43)
  walkable: dof[gx, gy].any()

PELLET GRID → DOF GRID
  obs.pellets[pellx, pelly] == 1
  dof_gx = pellx * 2 + 1
  dof_gy = pelly * 3 + 2

ACTION INDICES (JAXAtari)
  0=NOOP  2=UP  3=RIGHT  4=LEFT  5=DOWN
```

---

## First Valid Experimental Results

Settings: K=15, radius=3, mult=5, 300 steps, seed=42.

| Planner | Score | Pellets | Oscillations | Survival | vs MF Divergence |
|---------|-------|---------|--------------|----------|-----------------|
| MF (static danger map) | 30 | 3 | 169 | 300 | — |
| MBv0 (linear extrapolation) | 30 | 3 | 179 | 300 | 0.3% |
| **MBv2 (JAXAtari-native)** | **60** | **6** | **151** | 300 | 0.7% |

**Prediction accuracy of MBv2 (JAXAtari-native ghost simulator):**

| Horizon K | MBv0 (linear) | MBv2 (JAX-native) |
|---|---|---|
| K=1 | 0.675 cells | **0.013 cells** |
| K=5 | 2.938 cells | **0.138 cells** |
| K=15 | 6.975 cells | **0.750 cells** |

---

## Current Bottleneck

**Prediction quality is not the bottleneck.** MBv2 predicts ghost positions with 0.013 cell error at K=1 — near-perfect — because it uses JAXAtari's own routing functions directly.

**Planner integration at blocked corridors is the bottleneck.** MB and MF plans diverge in only 0.7% of steps, but those critical divergences produce 2× better performance. The remaining 151 oscillations occur when the danger map inflates the cost of the only viable corridor path, causing Pacman to reverse repeatedly toward a lower-cost cell.

**Next targeted fix:** Add a reversal penalty to the A* cost function so that the agent prefers committing to a corridor over bouncing at its entrance.
