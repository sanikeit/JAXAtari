"""
validate_coords.py
"""
import sys
import os
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from jaxatari.games.jax_mspacman import JaxPacman
from project.planners.mf_baseline import MFPlanner
from project.utils.coords import pellet_to_dof

def run_validation():
    env = JaxPacman()
    step_fn = jax.jit(env.step)
    dof_maze = env.consts.DOF_MAZES[0]
    
    seeds = [42, 123, 456, 789, 999]
    n_steps = 300
    
    samples = []
    
    # Let's collect 100 random states
    print("Collecting states...")
    count = 0
    
    for seed in seeds:
        key = jax.random.PRNGKey(seed)
        obs, state = env.reset(key)
        planner = MFPlanner(dof_maze)
        
        for _ in range(n_steps):
            action = planner.plan(obs, state)
            obs, state, _, done, _ = step_fn(state, jnp.array(action, dtype=jnp.int32))
            
            samples.append({
                'pacman': obs.player_position,
                'ghosts': obs.ghost_positions,
                'pellets': obs.pellets
            })
            count += 1
            if count >= 100:
                break
        if count >= 100:
            break
            
    print(f"Collected {len(samples)} states.")
    
    def test_pacman_ghosts(samples):
        # A: Current mapping
        # gx = px // 4
        # gy = py // 4
        
        # B: JAX mapping
        # gx = (px + 5) // 4
        # gy = (py + 3) // 4
        
        A_walkable = 0
        B_walkable = 0
        total = len(samples) * 5 # pacman + 4 ghosts
        
        for s in samples:
            positions = [s['pacman']] + [s['ghosts'][i] for i in range(4)]
            for pos in positions:
                px, py = pos[0].item(), pos[1].item()
                
                # A
                gx_a = px // 4
                gy_a = py // 4
                if 0 <= gx_a < 40 and 0 <= gy_a < 44 and dof_maze[gx_a, gy_a].any():
                    A_walkable += 1
                    
                # B
                gx_b = (px + 5) // 4
                gy_b = (py + 3) // 4
                if 0 <= gx_b < 40 and 0 <= gy_b < 44 and dof_maze[gx_b, gy_b].any():
                    B_walkable += 1
                    
        return A_walkable / total, B_walkable / total

    pac_ghost_A, pac_ghost_B = test_pacman_ghosts(samples)
    print(f"Pacman/Ghosts Walkable: A (Current)={pac_ghost_A*100:.1f}% | B (JAX)={pac_ghost_B*100:.1f}%")

    def test_pellets(samples):
        # A: Current mapping
        # gx = pellx * 2 + 1
        # gy = pelly * 3 + 2
        
        # B: JAX mapping derived from pixels
        # px = pellx * 8 + 8, py = pelly * 8 + 24
        # gx = (px + 5) // 4 = pellx * 2 + 3
        # gy = (py + 3) // 4 = pelly * 2 + 6
        
        A_walkable = 0
        B_walkable = 0
        total = 0
        
        for s in samples:
            pellets = s['pellets']
            for pelly in range(pellets.shape[1]):
                for pellx in range(pellets.shape[0]):
                    if pellets[pellx, pelly]:
                        total += 1
                        
                        # A
                        gx_a = pellx * 2 + 1
                        gy_a = pelly * 3 + 2
                        if 0 <= gx_a < 40 and 0 <= gy_a < 44 and dof_maze[gx_a, gy_a].any():
                            A_walkable += 1
                            
                        # B
                        px = pellx * 8 + 8
                        if px > 74: px += 4
                        py = pelly * 8 + 24
                        gx_b = (px + 5) // 4
                        gy_b = (py + 3) // 4
                        if 0 <= gx_b < 40 and 0 <= gy_b < 44 and dof_maze[gx_b, gy_b].any():
                            B_walkable += 1
                            
        return A_walkable / total, B_walkable / total
        
    pellets_A, pellets_B = test_pellets(samples)
    print(f"Pellets Walkable: A (Current)={pellets_A*100:.1f}% | B (JAX)={pellets_B*100:.1f}%")

    # Path reconstruction consistency
    # Let's see if JAX mapping consistently places Pacman on a path segment.
    def test_consistency(samples):
        A_consistent = 0
        B_consistent = 0
        
        # We define consistency as being able to move smoothly through the grid.
        # Actually, let's just check if Pacman's pixel location is EXACTLY on the grid center.
        # JAXAtari checks: on_vertical_grid = x % 4 == 1, on_horizontal_grid = y % 12 == 6
        
        # Let's trace Pacman's movement over the 100 steps.
        A_path_length = 0
        B_path_length = 0
        
        for i in range(1, len(samples)):
            pos_prev = samples[i-1]['pacman']
            pos_curr = samples[i]['pacman']
            
            gx_a_p, gy_a_p = pos_prev[0].item() // 4, pos_prev[1].item() // 4
            gx_a_c, gy_a_c = pos_curr[0].item() // 4, pos_curr[1].item() // 4
            
            if gx_a_p != gx_a_c or gy_a_p != gy_a_c:
                A_path_length += 1
                
            gx_b_p, gy_b_p = (pos_prev[0].item() + 5) // 4, (pos_prev[1].item() + 3) // 4
            gx_b_c, gy_b_c = (pos_curr[0].item() + 5) // 4, (pos_curr[1].item() + 3) // 4
            
            if gx_b_p != gx_b_c or gy_b_p != gy_b_c:
                B_path_length += 1
                
        return A_path_length, B_path_length

    A_len, B_len = test_consistency(samples)
    print(f"Path Lengths (Transitions): A={A_len}, B={B_len}")

if __name__ == "__main__":
    run_validation()
