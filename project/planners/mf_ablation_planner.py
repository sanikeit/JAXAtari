"""
planners/mf_ablation_planner.py

Ablation variants for the Model-Free baseline planner.
Extends MFPlanner to support:
1. Reversal Penalty (cost added if the first step reverses to the previous tile)
2. Goal Commitment (locking onto a target pellet for N steps)
"""
import sys
import os
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from project.utils.coords import pixel_to_dof, extract_pellet_targets, get_action_for_path
from project.utils.danger_map import static_danger_map
from project.utils.astar import spatiotemporal_astar
from project.planners.mf_baseline import MFPlanner

class MFAblationPlanner(MFPlanner):
    def __init__(self, dof, K=15, radius=3, mult=5, reversal_penalty=0, commit_duration=0):
        super().__init__(dof, K, radius, mult)
        self.reversal_penalty = reversal_penalty
        self.commit_duration = commit_duration
        
        # Stateful tracking
        self.prev_pos = None
        self.committed_target = None
        self.commitment_counter = 0
        
        # Analytics
        self.target_switches = 0
        self.immediate_reversals = 0
        self.last_path_length = 0

    def reset(self):
        """Reset internal state across episodes."""
        self.prev_pos = None
        self.committed_target = None
        self.commitment_counter = 0
        self.target_switches = 0
        self.immediate_reversals = 0
        self.last_path_length = 0

    def plan(self, obs, state):
        px, py = obs.player_position[0].item(), obs.player_position[1].item()
        start = pixel_to_dof(px, py)
        all_targets = extract_pellet_targets(self.dof, obs.pellets)
        
        if not all_targets:
            return 0

        dmaps = static_danger_map(
            obs.ghost_positions, state.ghosts.modes,
            K=self.K, radius=self.radius, mult=self.mult
        )

        # Handle Goal Commitment
        targets_to_search = all_targets
        if self.commit_duration > 0:
            if self.commitment_counter > 0 and self.committed_target in all_targets:
                # Maintain commitment
                targets_to_search = {self.committed_target}
            else:
                # Need to pick a new target, but we wait until A* finds one
                pass

        # Run A*
        path = spatiotemporal_astar(
            self.dof, start, targets_to_search, dmaps,
            reversal_pos=self.prev_pos, 
            reversal_penalty=self.reversal_penalty
        )
        
        if not path:
            return 0
            
        self.last_path_length = len(path) - 1  # -1 because path includes start
        
        # Extract chosen target (the end of the path)
        chosen_target = path[-1]
        
        # Track target switches
        if self.committed_target is not None and chosen_target != self.committed_target:
            # Note: We only count it as a switch if we didn't just collect the target.
            # If the target is no longer in all_targets, we successfully collected it, so it's not an "abandonment" switch.
            if self.committed_target in all_targets:
                self.target_switches += 1
                
        # Update commitment tracking
        if chosen_target != self.committed_target:
            self.committed_target = chosen_target
            self.commitment_counter = self.commit_duration
        else:
            self.commitment_counter -= 1

        # Track immediate reversals
        next_pos = path[1] if len(path) > 1 else start
        if self.prev_pos is not None and next_pos == self.prev_pos:
            self.immediate_reversals += 1
            
        # Update previous position tracking
        self.prev_pos = start
        
        return get_action_for_path(start, path)
