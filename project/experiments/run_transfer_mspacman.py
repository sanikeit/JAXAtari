"""
experiments/run_transfer_mspacman.py

TIER 1 TRANSFER EXPERIMENT
==========================
Question: does our planner generalize across Ms. Pac-Man mazes (0, 1, 2, 3)?

Runs the MF and MBv2 planners on all four mazes and tabulates
score / pellets / oscillations / survival per maze.

HOW MAZE SELECTION WORKS
------------------------
The env chooses the maze from the level via get_level_maze():
    level 1-2 -> maze 0,   3-4 -> maze 1,   5-6 -> maze 2,   7-8 -> maze 3
So we set RESET_LEVEL to 1 / 3 / 5 / 7 to start directly on maze 0 / 1 / 2 / 3.
(Confirmed: RESET_LEVEL=3 loads maze 1, DOF grid differs from maze 0, and the
pellet mapping produces 0 targets inside walls -> the representation transfers.)

THE ONE FIX vs THE MAZE-0 PIPELINE
----------------------------------
project/utils/ghost_sim.build_jit_ghost_step hardcodes DOF_MAZES[0]. On maze 1
that means MBv2 would predict ghosts on the WRONG walls. Below,
build_jit_ghost_step_maze() is the same function parameterized by maze_id.
Once this is validated, fold the maze_id parameter back into ghost_sim.py.

MF needs NO change -- it only uses current ghost positions, no maze walls.

Run from the repository root:
    uv run python project/experiments/run_transfer_mspacman.py
"""
import sys
import os
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from jaxatari.games.jax_mspacman import (
    JaxPacman, MsPacmanConstants, GhostMode,
    get_allowed_directions, get_chase_target, pathfind, get_new_position,
)
from project.utils.coords import (
    pixel_to_dof, extract_pellet_targets, get_action_for_path,
)
from project.utils.danger_map import build_danger_maps, static_danger_map
from project.utils.ghost_sim import forward_roll
from project.utils.astar import spatiotemporal_astar


# ────────────────────────────────────────────────────────────────────
# Maze-parameterized ghost step  (== ghost_sim.build_jit_ghost_step but
# indexes DOF_MAZES[maze_id] instead of [0])
# ────────────────────────────────────────────────────────────────────
def build_jit_ghost_step_maze(consts, maze_id):
    dofmaze = consts.DOF_MAZES[maze_id]

    @jax.jit
    def jit_ghost_step(positions, actions, modes, types,
                       pacman_pos, pacman_action, key):
        blinky_pos = positions[0]

        def step_one(ghost_type, mode, action, pos, subkey):
            skip = (mode == GhostMode.ENJAILED) | (mode == GhostMode.RETURNING)

            def move(_):
                allowed = get_allowed_directions(
                    pos, action, dofmaze, consts.DIRECTIONS, is_ghost=True)
                n_allowed = jnp.sum(allowed != 0)
                is_random_mode = ((mode == GhostMode.FRIGHTENED) |
                                  (mode == GhostMode.BLINKING) |
                                  (mode == GhostMode.RANDOM))

                def _target(_):
                    return jax.lax.cond(
                        mode == GhostMode.CHASE,
                        lambda: get_chase_target(
                            ghost_type, pos, blinky_pos,
                            pacman_pos, pacman_action,
                            consts.ACTIONS, consts.SCATTER_TARGETS),
                        lambda: consts.SCATTER_TARGETS[ghost_type]
                    )

                new_act = jax.lax.cond(
                    n_allowed == 0, lambda _: action,
                    lambda _: jax.lax.cond(
                        n_allowed == 1, lambda _: allowed[0],
                        lambda _: jax.lax.cond(
                            is_random_mode,
                            lambda _: allowed[
                                jax.random.randint(subkey, (), 0, n_allowed)],
                            lambda _: pathfind(
                                pos, action, _target(None), allowed,
                                subkey, consts.ACTIONS, consts.DIRECTIONS),
                            None),
                        None),
                    None)
                return get_new_position(pos, new_act, consts), new_act

            return jax.lax.cond(skip, lambda _: (pos, action), move, None)

        keys = jax.random.split(key, 4)
        results = [step_one(types[i], modes[i], actions[i], positions[i], keys[i])
                   for i in range(4)]
        return (jnp.stack([r[0] for r in results]),
                jnp.stack([r[1] for r in results]))

    return jit_ghost_step


# ────────────────────────────────────────────────────────────────────
# Oscillation counter (pixel-level; same as evaluate_planners.py)
# ────────────────────────────────────────────────────────────────────
class OscillationCounter:
    def __init__(self):
        self.history = []

    def update(self, px, py):
        pos = (int(px), int(py))
        self.history.append(pos)
        if len(self.history) > 8:
            self.history.pop(0)
        if len(self.history) >= 6:
            recent = self.history[-3:]
            older = self.history[:-3]
            return sum(p in older for p in recent) >= 2
        return False


# ────────────────────────────────────────────────────────────────────
# One episode on one maze with one planner
# ────────────────────────────────────────────────────────────────────
MAZE_TO_LEVEL = {0: 1, 1: 3, 2: 5, 3: 7}


def run_episode(maze_id, mode, seed, jit_step, K, radius, mult, n_steps):
    """mode: 'MF' or 'MBv2'.  jit_step: maze-specific (only used by MBv2)."""
    level = MAZE_TO_LEVEL[maze_id]
    consts = MsPacmanConstants(RESET_LEVEL=level)
    env = JaxPacman(consts)
    step_fn = jax.jit(env.step)
    dof = env.consts.DOF_MAZES[maze_id]

    key = jax.random.PRNGKey(seed)
    obs, state = env.reset(key)
    rkey = jax.random.PRNGKey(seed + 1)

    osc_counter = OscillationCounter()
    score = pellets = osc = survival = 0

    for _ in range(n_steps):
        survival += 1
        px, py = obs.player_position[0].item(), obs.player_position[1].item()
        start = pixel_to_dof(px, py)
        targets = extract_pellet_targets(dof, obs.pellets)
        if not targets:
            break

        rkey, sk = jax.random.split(rkey)
        if mode == 'MF':
            dmaps = static_danger_map(obs.ghost_positions, state.ghosts.modes,
                                      K=K, radius=radius, mult=mult)
        else:  # MBv2
            preds = forward_roll(jit_step, state.ghosts,
                                 obs.player_position, obs.player_action, sk, K=K)
            dmaps = build_danger_maps(preds, state.ghosts.modes, radius, mult)

        path = spatiotemporal_astar(dof, start, targets, dmaps)
        action = get_action_for_path(start, path)

        if osc_counter.update(px, py):
            osc += 1

        obs, state, _, done, _ = step_fn(state, jnp.array(action, dtype=jnp.int32))
        score = state.score.item()
        pellets = state.level.collected_pellets.item()
        if done:
            break

    return dict(score=score, pellets=pellets, osc=osc, survival=survival)


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # --- config (start small; add seeds once you see it works) ---
    SEEDS = [42, 123, 456, 789, 999]              # e.g. [42, 123, 456] for final numbers
    N_STEPS = 300
    K, RADIUS, MULT = 15, 3, 5
    MAZES = [0, 1, 2, 3]

    # Build one ghost simulator per maze (compiled once, reused across seeds).
    # RESET_LEVEL is irrelevant to ghost stepping, so a base consts is fine.
    base_consts = MsPacmanConstants()
    print("Compiling ghost simulators per maze (first run only)...")
    jit_steps = {m: build_jit_ghost_step_maze(base_consts, m) for m in MAZES}
    print("Done.\n")

    header = (f"TIER-1 TRANSFER  |  seeds={SEEDS}  steps={N_STEPS}  "
              f"K={K} radius={RADIUS} mult={MULT}")
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    print(f"{'maze':>4} | {'planner':<6} | {'score':>6} | {'pellets':>7} | "
          f"{'osc':>4} | {'survival':>8}")
    print("-" * 52)

    for maze_id in MAZES:
        for mode in ['MF', 'MBv2']:
            runs = [run_episode(maze_id, mode, s, jit_steps[maze_id],
                                K, RADIUS, MULT, N_STEPS) for s in SEEDS]
            avg = {k: np.mean([r[k] for r in runs]) for k in runs[0]}
            print(f"{maze_id:>4} | {mode:<6} | {avg['score']:>6.0f} | "
                  f"{avg['pellets']:>7.1f} | {avg['osc']:>4.0f} | "
                  f"{avg['survival']:>8.0f}")
        print("-" * 52)

    print("\nRead across rows: does each planner still collect pellets / avoid")
    print("ghosts on mazes 1-3, or does performance collapse off maze 0?")
    print("That collapse (or lack of it) IS your tier-1 transfer result.")