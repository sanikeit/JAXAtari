"""
utils/ghost_sim.py

JAXAtari-native ghost forward simulator (MB v2).

Uses JAXAtari's own pathfinding functions directly:
  - get_allowed_directions
  - get_chase_target
  - pathfind
  - get_new_position

Prediction accuracy (avg L1 cell error over 20 samples, 4 ghosts):
  K=1  : 0.013 cells  (50x better than linear extrapolation)
  K=5  : 0.138 cells
  K=15 : 0.750 cells

Usage:
    from jaxatari.games.jax_mspacman import JaxPacman
    from project.utils.ghost_sim import build_jit_ghost_step, forward_roll

    env = JaxPacman()
    jit_step = build_jit_ghost_step(env.consts)
    preds = forward_roll(jit_step, state.ghosts, obs.player_position,
                          obs.player_action, key, K=15)
    # preds: list[K] of (4, 2) numpy arrays in pixel coords
"""
import jax
import jax.numpy as jnp
import numpy as np

from jaxatari.games.jax_mspacman import (
    GhostMode,
    get_allowed_directions,
    get_chase_target,
    pathfind,
    get_new_position,
)


def build_jit_ghost_step(consts):
    """
    Build and return a JIT-compiled single-step ghost update function.
    
    Must be called once at startup; the result can be reused indefinitely.
    First call incurs JIT compilation overhead (~1-2 seconds).
    
    Args:
        consts: MsPacmanConstants from env.consts
    
    Returns:
        jit_ghost_step: callable(positions, actions, modes, types,
                                  pacman_pos, pacman_action, key)
                         -> (new_positions, new_actions)
    """
    dofmaze = consts.DOF_MAZES[0]

    @jax.jit
    def jit_ghost_step(positions, actions, modes, types,
                        pacman_pos, pacman_action, key):
        """
        Step all 4 ghosts one frame forward using JAXAtari's own logic.
        
        Args:
            positions: (4, 2) int32 — ghost pixel positions
            actions:   (4,)   uint8 — ghost current directions
            modes:     (4,)   uint8 — ghost behavioral modes
            types:     (4,)   uint8 — ghost identities (0=Blinky, 1=Pinky, ...)
            pacman_pos:    (2,) — Pacman pixel position (for chase targeting)
            pacman_action: ()   — Pacman current direction (for Pinky/Inky targeting)
            key: JAX PRNGKey
        
        Returns:
            new_positions: (4, 2) int32
            new_actions:   (4,)   uint8
        """
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


def forward_roll(jit_step, ghost_state, pacman_pos, pacman_action, key, K=15):
    """
    Roll ghost positions forward K steps using the JAX-native simulator.
    
    Args:
        jit_step:      compiled function from build_jit_ghost_step()
        ghost_state:   state.ghosts (GhostsState NamedTuple)
        pacman_pos:    obs.player_position
        pacman_action: obs.player_action
        key:           JAX PRNGKey
        K:             prediction horizon (frames)
    
    Returns:
        list[K] of numpy arrays of shape (4, 2) — ghost PIXEL positions at
        steps t=1, 2, ..., K (does NOT include t=0 current positions)
    
    Notes:
        - Frightened/Blinking/Returning ghosts move at half speed
          (every other frame), implemented via the slow_mask rule.
        - The returned positions are in PIXEL space; convert with
          pixel_to_dof() from utils.coords before building danger maps.
    """
    predicted = []
    cur_pos = jnp.array(ghost_state.positions, dtype=jnp.int32)
    cur_act = jnp.array(ghost_state.actions,   dtype=jnp.uint8)
    modes   = jnp.array(ghost_state.modes,     dtype=jnp.uint8)
    types   = jnp.array(ghost_state.types,     dtype=jnp.uint8)

    for t in range(K):
        key, sk = jax.random.split(key)
        new_pos, new_act = jit_step(
            cur_pos, cur_act, modes, types, pacman_pos, pacman_action, sk)

        # Half-speed rule for frightened/blinking/returning ghosts
        slow = ((modes == GhostMode.FRIGHTENED) |
                (modes == GhostMode.BLINKING)   |
                (modes == GhostMode.RETURNING))
        if t % 2 == 0:
            new_pos = jnp.where(slow[:, None], cur_pos, new_pos)
            new_act = jnp.where(slow, cur_act, new_act)

        cur_pos, cur_act = new_pos, new_act
        predicted.append(np.array(cur_pos))

    return predicted
