"""
diagnostics/coordinate_failure_trace.py

Traces the first 50 steps of MF_legacy vs MF_jax on Seed 42 to determine why JAX-native coordinates fail.
"""
import sys
import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from jaxatari.games.jax_mspacman import JaxPacman
from project.planners.mf_ablation_planner import MFAblationPlanner
from project.planners.mf_jax_coords_planner import MFJaxPlanner
from project.utils.coords import extract_pellet_targets as legacy_extract
from project.utils.coords_jax import extract_pellet_targets as jax_extract
from project.utils.astar import spatiotemporal_astar
from project.utils.danger_map import static_danger_map as legacy_danger
from project.utils.danger_map_jax import static_danger_map as jax_danger
from project.utils.coords import pixel_to_dof as legacy_px_to_dof
from project.utils.coords_jax import pixel_to_dof as jax_px_to_dof

def get_reachable_targets(dof, start, targets, dmaps):
    reachable = 0
    unreachable = 0
    # simplified reachability using A*
    for t in targets:
        path = spatiotemporal_astar(dof, start, {t}, dmaps)
        if path:
            reachable += 1
        else:
            unreachable += 1
    return reachable, unreachable

def visualize_state(obs, dof, start_pos, target, path, dmap, step, suffix, title):
    plt.figure(figsize=(8, 8))
    
    # Draw walls (from DOF)
    maze_x, maze_y = np.where(~np.array(dof)[..., 0])
    plt.scatter(maze_x, maze_y, c='black', s=10, marker='s')
    
    # Draw danger map
    for (gx, gy), cost in dmap.items():
        if cost > 0:
            plt.scatter(gx, gy, c='red', s=40, alpha=min(cost/30.0, 1.0))
            
    # Draw path
    if path:
        px = [p[0] for p in path]
        py = [p[1] for p in path]
        plt.plot(px, py, c='white', alpha=0.5, linewidth=2)
        
    # Draw ghosts
    for i in range(4):
        gpx, gpy = obs.ghost_positions[i][0].item(), obs.ghost_positions[i][1].item()
        if suffix == 'legacy':
            gx, gy = legacy_px_to_dof(gpx, gpy)
        else:
            gx, gy = jax_px_to_dof(gpx, gpy)
        plt.scatter(gx, gy, c=['red', 'pink', 'cyan', 'orange'][i], s=100)
        
    # Draw Pacman
    plt.scatter(start_pos[0], start_pos[1], c='yellow', s=150, zorder=5)
    
    # Draw Target
    if target:
        plt.scatter(target[0], target[1], c='green', s=100, marker='*')
        
    plt.title(f"{title} - Step {step}")
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.axis('off')
    
    os.makedirs('project/docs/snapshots', exist_ok=True)
    plt.savefig(f'project/docs/snapshots/{suffix}_step_{step:02d}.png', facecolor='gray')
    plt.close()

def run_trace():
    env = JaxPacman()
    step_fn = jax.jit(env.step)
    dof = env.consts.DOF_MAZES[0]
    
    planners = {
        'legacy': MFAblationPlanner(dof, reversal_penalty=5, commit_duration=20),
        'jax': MFJaxPlanner(dof, reversal_penalty=5, commit_duration=20)
    }
    
    traces = {'legacy': [], 'jax': []}
    
    for name, planner in planners.items():
        print(f"\nRunning {name}...")
        key = jax.random.PRNGKey(42)
        obs, state = env.reset(key)
        planner.reset()
        
        for step in range(51):
            px, py = obs.player_position[0].item(), obs.player_position[1].item()
            
            if name == 'legacy':
                start = legacy_px_to_dof(px, py)
                targets = legacy_extract(dof, obs.pellets)
                dmaps = legacy_danger(obs.ghost_positions, state.ghosts.modes, 1)
            else:
                start = jax_px_to_dof(px, py)
                targets = jax_extract(dof, obs.pellets)
                dmaps = jax_danger(obs.ghost_positions, state.ghosts.modes, 1)
                
            danger_cost = dmaps[0].get(start, 0)
            
            # Step 0 reachability check
            if step == 0:
                reachable, unreachable = get_reachable_targets(dof, start, targets, dmaps)
                print(f"[{name}] Step 0: Total Pellets = {len(targets)}, Reachable = {reachable}, Unreachable = {unreachable}")
            
            action = planner.plan(obs, state)
            path_len = planner.last_path_length
            chosen_target = planner.committed_target
            
            if step in [0, 10, 20, 30, 40, 50]:
                visualize_state(obs, dof, start, chosen_target, [], dmaps[0], step, name, f"MF {name.capitalize()} Planner")
                
            traces[name].append({
                'step': step,
                'pos': start,
                'target': chosen_target,
                'path_len': path_len,
                'danger': danger_cost,
                'action': action,
                'targets': len(targets)
            })
            
            obs, state, _, _, _ = step_fn(state, jnp.array(action, dtype=jnp.int32))
            
    with open('project/results/trace_comparison.md', 'w') as f:
        f.write("# First 50 Steps Trace Comparison\n\n")
        f.write("| Step | Legacy Pos | Legacy Target | L-Path | L-Danger | L-Act | L-Pellets | | JAX Pos | JAX Target | J-Path | J-Danger | J-Act | J-Pellets |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        
        for i in range(50):
            l = traces['legacy'][i]
            j = traces['jax'][i]
            f.write(f"| {i:02d} | {l['pos']} | {l['target']} | {l['path_len']} | {l['danger']} | {l['action']} | {l['targets']} | | "
                    f"{j['pos']} | {j['target']} | {j['path_len']} | {j['danger']} | {j['action']} | {j['targets']} |\n")
                    
    print("\nTrace written to project/results/trace_comparison.md")

if __name__ == "__main__":
    run_trace()
