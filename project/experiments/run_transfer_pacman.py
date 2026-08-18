"""
experiments/run_transfer_pacman.py

TIER 2 TRANSFER EXPERIMENT — Pac-Man (not Ms. Pac-Man)
======================================================
Question: does our planner, built for Ms. Pac-Man, run on Pac-Man unchanged?

Pac-Man is a near-clone of Ms. Pac-Man but with three structural differences,
all handled below:
  1. DOF grid lives on state.level.dofmaze  (Ms. Pac-Man: consts.DOF_MAZES[i])
  2. Only 2 mazes: RESET_LEVEL 1 -> MAZE0, RESET_LEVEL 2 -> MAZE2  (indices 0,1)
  3. Different maze size (40x48 vs 40x44) and pellet grid (18x8 vs 18x14)

KEY TIER-2 FINDING (validated by try_pacman.py):
  Our pixel->DOF and pellet->DOF mappings transfer to Pac-Man with ZERO changes
  (0 pellet targets land inside walls) despite the different maze/pellet sizes.
  So the *representation* is reused as-is; only the ghost simulator is rebound
  to Pac-Man's own routing functions (same names, different module).

NOTE: Pac-Man's env class is ALSO named JaxPacman, so we import it aliased.

Run from the repository root:
    uv run python project/experiments/run_transfer_pacman.py
"""
import sys
import os
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Pac-Man env + its OWN ghost-routing functions (same names as Ms. Pac-Man,
# but imported from jax_pacman so the maze logic matches Pac-Man).
from jaxatari.games.jax_pacman import (
    JaxPacman as PacmanEnv, PacmanConstants, GhostMode,
    get_allowed_directions, get_chase_target, pathfind, get_new_position,
)
# Reused UNCHANGED from the Ms. Pac-Man pipeline:
from project.utils.coords import (
    pixel_to_dof, extract_pellet_targets, get_action_for_path,
)
from project.utils.danger_map import build_danger_maps, static_danger_map
from project.utils.astar import spatiotemporal_astar


# ────────────────────────────────────────────────────────────────────
# Pac-Man ghost step (mirrors utils/ghost_sim but binds Pac-Man's funcs
# and takes the dof grid explicitly, since Pac-Man has no consts.DOF_MAZES)
# ────────────────────────────────────────────────────────────────────
def build_jit_ghost_step_pacman(consts, dofmaze):
    @jax.jit
    def jit_ghost_step(positions, actions, modes,
                       pacman_pos, pacman_action, key):
        def step_one(mode, action, pos, subkey):
            skip = (mode == GhostMode.ENJAILED) | (mode == GhostMode.RETURNING)

            def move(_):
                allowed = get_allowed_directions(
                    pos, action, dofmaze, consts.DIRECTIONS, is_ghost=True)
                n_allowed = jnp.sum(allowed != 0)
                is_random_mode = ((mode == GhostMode.FRIGHTENED) |
                                  (mode == GhostMode.BLINKING) |
                                  (mode == GhostMode.RANDOM))

                # Pac-Man: ALL ghosts chase the player directly (no per-ghost type)
                def _target(_):
                    return jax.lax.cond(
                        mode == GhostMode.CHASE,
                        lambda: get_chase_target(pacman_pos).astype(jnp.int32),
                        lambda: pacman_pos.astype(jnp.int32)
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
        results = [step_one(modes[i], actions[i], positions[i], keys[i])
                   for i in range(4)]
        return (jnp.stack([r[0] for r in results]),
                jnp.stack([r[1] for r in results]))

    return jit_ghost_step


def forward_roll_pacman(jit_step, ghost_state, pacman_pos, pacman_action, key, K):
    predicted = []
    cur_pos = jnp.array(ghost_state.positions, dtype=jnp.int32)
    cur_act = jnp.array(ghost_state.actions,   dtype=jnp.uint8)
    modes   = jnp.array(ghost_state.modes,     dtype=jnp.uint8)
    # NOTE: Pac-Man has no ghost 'types' — removed.
    for t in range(K):
        key, sk = jax.random.split(key)
        new_pos, new_act = jit_step(cur_pos, cur_act, modes,
                                    pacman_pos, pacman_action, sk)   # no types arg
        slow = ((modes == GhostMode.FRIGHTENED) |
                (modes == GhostMode.BLINKING) |
                (modes == GhostMode.RETURNING))
        if t % 2 == 0:
            new_pos = jnp.where(slow[:, None], cur_pos, new_pos)
            new_act = jnp.where(slow, cur_act, new_act)
        cur_pos, cur_act = new_pos, new_act
        predicted.append(np.array(cur_pos))
    return predicted


# ────────────────────────────────────────────────────────────────────
# Oscillation counter (same as Ms. Pac-Man)
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
# One episode.  maze_id: 0 -> RESET_LEVEL 1 (MAZE0), 1 -> RESET_LEVEL 2 (MAZE2)
# ────────────────────────────────────────────────────────────────────
MAZE_TO_LEVEL = {0: 1, 1: 2}


def run_episode(maze_id, mode, seed, K, radius, mult, n_steps):
    level = MAZE_TO_LEVEL[maze_id]
    consts = PacmanConstants(RESET_LEVEL=level)
    env = PacmanEnv(consts)
    step_fn = jax.jit(env.step)

    key = jax.random.PRNGKey(seed)
    obs, state = env.reset(key)
    dof = state.level.dofmaze     
    rkey = jax.random.PRNGKey(seed + 1)

    jit_step = None
    if mode == 'MBv2':
        jit_step = build_jit_ghost_step_pacman(env.consts, dof)

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
        else:
            preds = forward_roll_pacman(jit_step, state.ghosts,
                                        obs.player_position, obs.player_action,
                                        sk, K=K)
            dmaps = build_danger_maps(preds, state.ghosts.modes, radius, mult)

        path = spatiotemporal_astar(dof, start, targets, dmaps)
        action = get_action_for_path(start, path)

        if osc_counter.update(px, py):
            osc += 1
        obs, state, _, done, _ = step_fn(state, jnp.array(action, dtype=jnp.int32))
        score = int(state.score)
        # Pac-Man tracks collected pellets as level.eaten_pellets
        pellets = int(state.level.eaten_pellets)
        if done:
            break

    return dict(score=score, pellets=pellets, osc=osc, survival=survival)


# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    SEEDS = [42]                 # add [42,123,456] for final numbers
    N_STEPS = 300
    K, RADIUS, MULT = 15, 3, 5
    MAZES = [0, 1]               # Pac-Man has only 2 mazes (MAZE0, MAZE2)

    header = (f"TIER-2 TRANSFER (Pac-Man)  |  seeds={SEEDS}  steps={N_STEPS}  "
              f"K={K} radius={RADIUS} mult={MULT}")
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    print(f"{'maze':>4} | {'planner':<6} | {'score':>6} | {'pellets':>7} | "
          f"{'osc':>4} | {'survival':>8}")
    print("-" * 52)

    for maze_id in MAZES:
        for mode in ['MF', 'MBv2']:
            runs = [run_episode(maze_id, mode, s, K, RADIUS, MULT, N_STEPS)
                    for s in SEEDS]
            avg = {k: np.mean([r[k] for r in runs]) for k in runs[0]}
            print(f"{maze_id:>4} | {mode:<6} | {avg['score']:>6.0f} | "
                  f"{avg['pellets']:>7.1f} | {avg['osc']:>4.0f} | "
                  f"{avg['survival']:>8.0f}")
        print("-" * 52)

    print("\nTier-2 result: does the SAME planner (reused from Ms. Pac-Man,")
    print("representation unchanged) navigate and collect pellets on Pac-Man?")