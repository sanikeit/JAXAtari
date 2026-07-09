"""
planners/mf_jax_coords_planner.py

Ablation variants for the Model-Free baseline planner using JAX coordinate system.
"""
import sys
import os
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from project.utils.coords_jax import pixel_to_dof, extract_pellet_targets, get_action_for_path
from project.utils.danger_map_jax import static_danger_map
from project.utils.astar import spatiotemporal_astar

class MFJaxPlanner:
    def __init__(self, dof, K=15, radius=3, mult=5, reversal_penalty=0, commit_duration=0):
        self.dof = dof
        self.K = K
        self.radius = radius
        self.mult = mult
        self.reversal_penalty = reversal_penalty
        self.commit_duration = commit_duration
        
        self.prev_pos = None
        self.committed_target = None
        self.commitment_counter = 0
        
        self.target_switches = 0
        self.immediate_reversals = 0
        self.last_path_length = 0

    def reset(self):
        self.prev_pos = None
        self.committed_target = None
        self.commitment_counter = 0
        self.target_switches = 0
        self.immediate_reversals = 0
        self.last_path_length = 0

    def plan(self, obs, state):
        px, py = obs.player_position[0].item(), obs.player_position[1].item()
        start = pixel_to_dof(px, py)
        targets = extract_pellet_targets(self.dof, obs.pellets)
        
        if not targets:
            return 0

        dmaps = static_danger_map(
            obs.ghost_positions, state.ghosts.modes,
            K=self.K, radius=self.radius, mult=self.mult
        )
        
        danger_at_pacman = dmaps[0].get((start[0], start[1]), 0)
        
        if self.commit_duration > 0:
            if self.committed_target and self.committed_target in targets and danger_at_pacman < 30:
                self.commitment_counter += 1
                search_targets = {self.committed_target}
                if self.commitment_counter > self.commit_duration:
                    self.committed_target = None
            else:
                self.committed_target = None
                search_targets = targets
        else:
            search_targets = targets

        path = spatiotemporal_astar(self.dof, start, search_targets, dmaps,
                                    reversal_pos=self.prev_pos, reversal_penalty=self.reversal_penalty)
                                    
        self.last_path_length = len(path) - 1 if len(path) > 1 else 0

        if path and len(path) > 1:
            chosen_target = path[-1]
            if chosen_target != self.committed_target:
                self.target_switches += 1
                self.committed_target = chosen_target
                self.commitment_counter = 0

            next_pos = path[1]
            if self.prev_pos and next_pos == self.prev_pos:
                self.immediate_reversals += 1
            
            self.prev_pos = start
            return get_action_for_path(start, path)

        self.prev_pos = start
        return 0
