"""
planners/mf_midrange_planner.py

Implements Mid-Range Utility targeting to test if a middle ground between local myopia and extreme global targeting is viable.
"""
import sys
import os
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from project.planners.mf_ablation_planner import MFAblationPlanner
from project.utils.coords import pixel_to_dof, extract_pellet_targets, get_action_for_path
from project.utils.danger_map import static_danger_map
from project.utils.astar import spatiotemporal_astar

def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

class MFMidRangePlanner(MFAblationPlanner):
    def __init__(self, dof, strategy='midrange', K=15, radius=3, mult=5, reversal_penalty=5, commit_duration=20):
        super().__init__(dof, K, radius, mult, reversal_penalty, commit_duration)
        self.strategy = strategy

    def _select_targets(self, start, all_targets, dmaps, ghosts):
        """Applies Mid-Range Utility filtering and scoring."""
        if self.strategy == 'nearest':
            return all_targets
            
        safe_targets = [t for t in all_targets if dmaps[0].get(t, 0) < 30]
        if not safe_targets:
            safe_targets = list(all_targets) # fallback
            
        if self.strategy == 'midrange':
            # 1. Filter candidate pellets: 5 <= distance <= 15
            candidates = [t for t in safe_targets if 5 <= manhattan(start, t) <= 15]
            
            if not candidates:
                return all_targets  # fallback to nearest if no midrange targets exist
                
            # 2. Score remaining candidates
            best_target = None
            best_util = -float('inf')
            
            for t in candidates:
                density = sum(1 for other in all_targets if manhattan(t, other) <= 5)
                
                ghost_dists = [manhattan(t, g) for g in ghosts]
                min_g_dist = min(ghost_dists) if ghost_dists else float('inf')
                # Avoid division by zero
                risk = max(0.1, 10 - min_g_dist)
                
                util = density / risk
                
                if util > best_util:
                    best_util = util
                    best_target = t
                    
            return {best_target} if best_target else all_targets

        return all_targets

    def plan(self, obs, state):
        px, py = obs.player_position[0].item(), obs.player_position[1].item()
        start = pixel_to_dof(px, py)
        all_targets = extract_pellet_targets(self.dof, obs.pellets)
        
        if not all_targets:
            return 0

        ghosts = [pixel_to_dof(obs.ghost_positions[i][0].item(), obs.ghost_positions[i][1].item()) for i in range(4)]
        dmaps = static_danger_map(
            obs.ghost_positions, state.ghosts.modes,
            K=self.K, radius=self.radius, mult=self.mult
        )

        # Handle Goal Commitment
        if self.commit_duration > 0 and self.commitment_counter > 0 and self.committed_target in all_targets:
            targets_to_search = {self.committed_target}
        else:
            targets_to_search = self._select_targets(start, all_targets, dmaps, ghosts)

        # Run A*
        path = spatiotemporal_astar(
            self.dof, start, targets_to_search, dmaps,
            reversal_pos=self.prev_pos, 
            reversal_penalty=self.reversal_penalty
        )
        
        if not path:
            return 0
            
        self.last_path_length = len(path) - 1
        chosen_target = path[-1]
        
        if self.committed_target is not None and chosen_target != self.committed_target:
            if self.committed_target in all_targets:
                self.target_switches += 1
                
        if chosen_target != self.committed_target:
            self.committed_target = chosen_target
            self.commitment_counter = self.commit_duration
        else:
            self.commitment_counter -= 1

        next_pos = path[1] if len(path) > 1 else start
        if self.prev_pos is not None and next_pos == self.prev_pos:
            self.immediate_reversals += 1
            
        self.prev_pos = start
        
        return get_action_for_path(start, path)
