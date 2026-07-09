# Repository Reuse Audit — Group 17 (MsPacman Planning)

This document catalogues the existing tools, frameworks, and utilities present in the core `JAXAtari` repository that we can leverage to accelerate our Model-Free (MF) vs Model-Based (MB) vs Neuro-Symbolic comparisons. 

It specifically highlights where we successfully avoided reimplementing complex arcade logic, and identifies scripts we can co-opt for Phase 3 and Phase 4.

---

## 1. Low-Level Ghost Simulation & Pathfinding
**Files:** `src/jaxatari/games/jax_mspacman.py` / `src/jaxatari/games/jax_pacman.py`

| Function | Purpose | Can we reuse it? | How it helps Group 17 | Priority |
| :--- | :--- | :--- | :--- | :--- |
| `pathfind()` | Computes the single next action for a ghost given its current position, allowed directions, and a target tile, using Manhattan distance tie-breaking (UP > LEFT > DOWN > RIGHT). | **Yes.** (Currently used in `MBv2Planner`) | Saves us from reverse-engineering the exact arcade movement logic. It perfectly mimics JAXAtari's ghost decisions. | **High** |
| `get_chase_target()` | Computes the target tile for Blinky, Pinky, Inky, and Sue based on their specific personality rules (e.g. Pinky aims 4 tiles ahead of Pacman). | **Yes.** (Currently used in `MBv2Planner`) | Crucial for accurate MB prediction. Without this, we fall back to inaccurate linear velocity extrapolation (MBv0). | **High** |
| `get_new_position()` | Computes pixel movements and handles complex edge cases like screen-wrapping through tunnels. | **Yes.** (Currently used in `MBv2Planner`) | Prevents prediction errors when ghosts travel through the side tunnels. | **High** |

*Did we reimplement this?* In our early (now archived) `MBv0` and `MBv1` experiments, we foolishly tried to hand-craft linear velocity simulators. We have since corrected this in `MBv2` by securely wrapping JAXAtari's native functions.

---

## 2. Maze Representations & Navigation Graphs
**File:** `src/jaxatari/games/mspacman_mazes.py`

| Component | Purpose | Can we reuse it? | How it helps Group 17 | Priority |
| :--- | :--- | :--- | :--- | :--- |
| `DOF_MAZES` | Precomputed, 4-channel (Up, Right, Down, Left) binary masks representing walkable paths for all 4 maze configurations. | **Yes.** (Used in `utils/astar.py`) | Acts as the immediate graph for our A* spatiotemporal planner. Completely removes the need for us to parse wall collisions from pixel observations. | **High** |
| `MAZES` | 2D tile arrays defining the visual layout of walls, pellets, and power-pellets. | **Yes.** | Useful for rendering custom visualizations of our danger-maps over the maze topology. | **Medium** |

*Did we reimplement this?* We originally tried to plan over `MsPacmanMaze.MAZES[0]` (the rendering grid) rather than `DOF_MAZES` (the pathfinding graph). We fixed this in the Phase 2 cleanup.

---

## 3. Evaluators & Agent Interfaces
**Files:** `scripts/benchmarks/ppo_jaxatari_vmap_eval.py`, `pqn_agent.py`

| Component | Purpose | Can we reuse it? | How it helps Group 17 | Priority |
| :--- | :--- | :--- | :--- | :--- |
| `ppo_jaxatari_vmap_eval.py` | Runs fully JIT-compiled evaluation loops using `jax.lax.scan` across multiple environments concurrently. | **No.** | Our A* planner relies on Python data structures (`heapq`, dynamic lists) which cannot be compiled via `jax.vmap` or `jax.lax.scan`. We *must* use our custom Python-side `evaluate_planners.py` loop. | **N/A** |
| `pqn_agent.py` | Proximal Q-Network training loop utilizing PPO-style data collection. | **Maybe.** (Phase 3) | If we eventually decide to train a Neuro-Symbolic CNN predictor, we can lift their rollout buffer / PPO loss framework rather than writing our own RL loop from scratch. | **Low** |

---

## 4. Visualizers & Renderers
**Files:** `scripts/benchmarks/pqn_agent_visualisation.py`, `scripts/play.py`

| Component | Purpose | Can we reuse it? | How it helps Group 17 | Priority |
| :--- | :--- | :--- | :--- | :--- |
| `pqn_agent_visualisation.py` | Connects a PyGame window to an agent's decision loop, allowing real-time viewing of actions. | **Yes.** | While built for `.safetensors` NN agents, we can easily adapt the PyGame rendering loop to accept actions from our `MBv2Planner` instead. This would allow us to visually see the planner getting stuck in corridor oscillations. | **High** |
| `play.py` | Human keyboard-control interface for JAXAtari. | **Yes.** | We could modify it to draw our A* paths or Danger Maps directly onto the screen as an overlay while the game runs. | **Medium** |

---

## 5. Existing Planners (Search / Heuristic)
**Files:** Various

| Component | Purpose | Can we reuse it? | How it helps Group 17 | Priority |
| :--- | :--- | :--- | :--- | :--- |
| None Found | - | - | - | - |

*Analysis:* A repository-wide grep for `A*`, `MCTS`, `BFS`, and `heuristic` reveals that JAXAtari is strictly a Model-Free Deep RL codebase. There are absolutely no existing search planners or classical pathfinding utilities available. **Our `spatiotemporal_astar` is completely novel to this repository and strictly necessary.**

---

## Final Conclusion for Current Bottleneck (Phase 3: Oscillations)
We are extremely efficient right now. Our `project/` directory contains zero redundant code—we are importing `pathfind` and `DOF_MAZES` natively. 

**Next Immediate Step:** We should build a customized version of `pqn_agent_visualisation.py` specifically for our `MBv2Planner` so we can physically watch the "corridor-blocking oscillation" happen in PyGame. This will make it much easier to tune the A* reversal penalties.
