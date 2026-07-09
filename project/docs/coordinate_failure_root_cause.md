# Coordinate Failure Root Cause Analysis

## 1. Coordinate Consistency Audit

| Component | Legacy Coords (`x//4`) | JAX Coords (`(x+5)//4`) | Consistent? | Notes |
|---|---|---|---|---|
| Pacman Position | gx = x // 4 | gx = (x+5) // 4 | YES | Both transform Pacman correctly within their math. |
| Ghost Positions | gx = x // 4 | gx = (x+5) // 4 | YES | Both transform Ghosts correctly within their math. |
| Danger Map | gx = x // 4 | gx = (x+5) // 4 | YES | Both match the ghost coordinate system. |
| Pellet Positions | gx = pellx * 2 + 1, gy = pelly * 3 + 2 | py = pelly * 8 + 24... | **NO! (CRITICAL BUG)** | The JAX pellet mapping formula is completely wrong. |

## 2. The First-50-Step Trace Analysis
The trace logs revealed exactly why JAX-native coords failed with a score of 12. 
At Step 0, Pacman is at `y=102`, which is `pelly=8` in the JAXAtari engine (since `y = pelly * 12 + 6`, so `8 * 12 + 6 = 102`).
However, our `coords_jax.py` incorrectly calculated `py = pelly * 8 + 24`. 
Because of this bug, `pelly=10` evaluated to `py = 10 * 8 + 24 = 104`, which maps to `gy = 26`.
Pacman's starting position `y=102` also maps to `gy=26`.
Thus, the A* planner thought that the pellet `pelly=10` was right next to Pacman at `gy=26`.

When Pacman moved and successfully collected a pellet, he actually collected `pelly=8`. But because `pelly=10` was mapped to `gy=26`, A* kept seeing `pelly=10` as active in the environment, and it remained as a target at `gy=26`. 

Pacman became trapped in an infinite 3-cell loop trying to collect a "phantom" pellet that A* placed at `gy=26` (which actually lived at `y=126`), completely ignoring the rest of the maze.

## 3. Visual Failure Analysis & Reachability
The reachability check proves that 236/236 pellets are fully walkable in the JAX-native A* graph. The Danger Maps are also perfectly aligned. Pacman wasn't trapped by Danger Maps; he was trapped by a broken target list that mapped distant pellets directly into his current cell.

## 4. Final Verdict

### A. Is the JAX-native coordinate system correct?
Yes, `gx = (x+5)//4, gy = (y+3)//4` is correct.

### B. Is there still a mapping inconsistency?
Yes, the pellet mapping in `coords_jax.py` is entirely broken and does not match JAXAtari's actual `eat_pellet` math.
The correct mapping for pellets to pixels should be:
```python
px = pellx * 8 + 5  (for x < 75)
py = pelly * 12 + 6
```

### C. Did we introduce a new bug?
Yes. The `coords_jax.py` script hallucinated an incorrect pellet-to-pixel math which destroyed target selection. 

### D. Is the MF planner genuinely limited by static danger maps?
This remains unproven. Because the JAX-native MF planner never actually got to play the game (it got stuck hunting phantom pellets on step 1), we do not yet know how it performs.

### E. Is MB actually the next justified step?
No. We must fix the `coords_jax.py` pellet mapping and re-run the `MF_jax_coords` ablation to see the TRUE impact of accurate coordinates on the Model-Free planner. Only then can we conclude whether static danger maps are fundamentally flawed.
