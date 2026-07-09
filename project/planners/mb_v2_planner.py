"""
planners/mb_v2_planner.py

Model-Based v2 planner for MsPacman — uses JAXAtari's own ghost routing.

Strategy:
  - Roll ghost state forward K steps using JAXAtari's own pathfind /
    get_chase_target / get_new_position (via a JIT-compiled wrapper).
  - Build a temporally-indexed danger map from predicted positions.
  - Run spatiotemporal A* that consults danger[t][cell] at expansion depth t.

Why "v2" beats "v0" (linear extrapolation) and "v1" (hand-crafted simulator):
  - v0: 0.675 cell error at K=1. No awareness of maze topology or ghost modes.
  - v1: Hand-written targeting rules desynced from real emulator logic.
  - v2: 0.013 cell error at K=1. Uses the exact same logic JAXAtari uses.

Validated result (2026-06-15):
  Score=60, Pellets=6, Oscillations=151/300, Survival=300
  (K=15, radius=3, mult=5, seed=42)
  2× score and pellets vs MF baseline.

Usage:
    from jaxatari.games.jax_mspacman import JaxPacman
    from project.planners.mb_v2_planner import MBv2Planner

    env = JaxPacman()
    planner = MBv2Planner(env.consts)
    # …
    action = planner.plan(obs, state, key)
"""
import sys
import os
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from project.utils.coords import pixel_to_dof, extract_pellet_targets, get_action_for_path
from project.utils.danger_map import build_danger_maps
from project.utils.ghost_sim import build_jit_ghost_step, forward_roll
from project.utils.astar import spatiotemporal_astar


class MBv2Planner:
    """
    Model-Based v2 planner using JAXAtari-native ghost forward simulation.

    Ghost prediction uses JAXAtari's own routing functions, JIT-compiled
    for efficiency. Near-perfect prediction at short horizons (K=1 error
    = 0.013 DOF cells).
    """

    def __init__(self, consts, K=15, radius=3, mult=5):
        """
        Args:
            consts: MsPacmanConstants from env.consts — passed through to
                    the ghost simulator (provides ACTIONS, DIRECTIONS,
                    SCATTER_TARGETS, DOF_MAZES)
            K:      prediction horizon in frames
            radius: danger blob radius in DOF cells (best: 3)
            mult:   danger cost multiplier (best: 5)
        """
        self.dof = consts.DOF_MAZES[0]
        self.K = K
        self.radius = radius
        self.mult = mult
        self._jit_step = build_jit_ghost_step(consts)

    def plan(self, obs, state, key):
        """
        Compute next action using MB v2 spatiotemporal planning.

        Args:
            obs:   JAXAtari observation
            state: JAXAtari state (state.ghosts for ghost sim input)
            key:   JAX PRNGKey (consumed; split internally)

        Returns:
            (action, key_out):
                action  — int: JAXAtari action (2=UP, 3=RIGHT, 4=LEFT, 5=DOWN)
                key_out — updated PRNGKey for next call
        """
        key, sk = jax.random.split(key)
        px, py = obs.player_position[0].item(), obs.player_position[1].item()
        start = pixel_to_dof(px, py)
        targets = extract_pellet_targets(self.dof, obs.pellets)
        if not targets:
            return 0, key

        preds = forward_roll(
            self._jit_step, state.ghosts,
            obs.player_position, obs.player_action,
            sk, K=self.K
        )
        dmaps = build_danger_maps(preds, state.ghosts.modes,
                                   radius=self.radius, mult=self.mult)
        path = spatiotemporal_astar(self.dof, start, targets, dmaps)
        return get_action_for_path(start, path), key
