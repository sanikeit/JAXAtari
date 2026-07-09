"""
planners/mf_baseline.py

Model-Free (MF) baseline planner for MsPacman.

Strategy:
  - Snapshot current ghost pixel positions at t=0.
  - Replicate that snapshot as a static danger map across K future steps.
  - Run spatiotemporal A* to find a path to the nearest safe pellet.

This is the simplest valid baseline. It does not predict ghost movement.
It avoids ghosts only based on their current position, not where they will be.

Validated result (2026-06-15):
  Score=30, Pellets=3, Oscillations=169/300, Survival=300
  (K=15, radius=3, mult=5, seed=42)

Usage (from repository root):
    conda run -n jaxatari python project/experiments/evaluate_planners.py
"""
import sys
import os
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from project.utils.coords import pixel_to_dof, extract_pellet_targets, get_action_for_path
from project.utils.danger_map import static_danger_map
from project.utils.astar import spatiotemporal_astar


class MFPlanner:
    """
    Model-Free A* planner.
    
    Uses current ghost positions as a static K-step danger map.
    No prediction of future ghost movement.
    """

    def __init__(self, dof, K=15, radius=3, mult=5):
        """
        Args:
            dof:    consts.DOF_MAZES[0] — shape (40, 44, 4)
            K:      number of danger-map timesteps (reused statically)
            radius: danger blob radius in DOF cells
            mult:   cost multiplier per unit distance from ghost center
        """
        self.dof = dof
        self.K = K
        self.radius = radius
        self.mult = mult

    def plan(self, obs, state):
        """
        Compute next action.

        Args:
            obs:   JAXAtari observation (player_position, ghost_positions, pellets, …)
            state: JAXAtari state (state.ghosts.modes for mode filtering)

        Returns:
            int: JAXAtari action (0=NOOP, 2=UP, 3=RIGHT, 4=LEFT, 5=DOWN)
        """
        px, py = obs.player_position[0].item(), obs.player_position[1].item()
        start = pixel_to_dof(px, py)
        targets = extract_pellet_targets(self.dof, obs.pellets)
        if not targets:
            return 0

        dmaps = static_danger_map(
            obs.ghost_positions, state.ghosts.modes,
            K=self.K, radius=self.radius, mult=self.mult
        )
        path = spatiotemporal_astar(self.dof, start, targets, dmaps)
        return get_action_for_path(start, path)
