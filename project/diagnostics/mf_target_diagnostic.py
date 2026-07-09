"""
diagnostics/mf_target_diagnostic.py

Deep diagnostic of target selection and danger map integration for the Best MF Variant (H).
"""
import sys
import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from jaxatari.games.jax_mspacman import JaxPacman
from project.planners.mf_ablation_planner import MFAblationPlanner
from project.utils.coords import pixel_to_dof, extract_pellet_targets, is_walkable
from project.utils.danger_map import static_danger_map
from project.utils.astar import spatiotemporal_astar
from project.experiments.run_mf_ablations import OscillationCounter

def render_snapshot(dof, start, ghosts, all_targets, chosen_target, path, dmap, filepath, title):
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='black')
    ax.set_facecolor('black')
    ax.invert_yaxis()
    ax.set_xlim(-1, 41)
    ax.set_ylim(45, -1)
    
    # Draw walls / walkable
    walk_x, walk_y = [], []
    for x in range(40):
        for y in range(44):
            if is_walkable(dof, x, y):
                walk_x.append(x)
                walk_y.append(y)
    ax.scatter(walk_x, walk_y, color='#222222', marker='s', s=20, label='Walkable')
    
    # Draw danger map
    if dmap:
        dx, dy, dc = [], [], []
        for (x, y), cost in dmap.items():
            if cost > 0:
                dx.append(x)
                dy.append(y)
                dc.append(min(cost, 100) / 100.0) # normalize 0-1
        ax.scatter(dx, dy, c=dc, cmap='Reds', marker='s', s=40, alpha=0.5, label='Danger')

    # Draw pellets
    px, py = zip(*all_targets) if all_targets else ([], [])
    ax.scatter(px, py, color='white', s=5, label='Pellets')
    
    # Draw path
    if path:
        path_x, path_y = zip(*path)
        ax.plot(path_x, path_y, color='lime', linewidth=2, label='A* Path')
        
    # Draw chosen target
    if chosen_target:
        ax.scatter([chosen_target[0]], [chosen_target[1]], color='lime', s=80, facecolors='none', edgecolors='lime', linewidth=2, label='Target')
        
    # Draw ghosts
    gx, gy = zip(*ghosts)
    ax.scatter(gx, gy, color='red', s=60, zorder=5, label='Ghosts')
    
    # Draw pacman
    ax.scatter([start[0]], [start[1]], color='yellow', s=60, zorder=5, label='Pacman')
    
    ax.set_title(title, color='white')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(filepath, facecolor=fig.get_facecolor())
    plt.close()


def run_diagnostic(seed, n_steps=300):
    env = JaxPacman()
    step_fn = jax.jit(env.step)
    dof = env.consts.DOF_MAZES[0]
    
    planner = MFAblationPlanner(dof, reversal_penalty=5, commit_duration=20)
    key = jax.random.PRNGKey(seed)
    obs, state = env.reset(key)
    
    osc_counter = OscillationCounter()
    
    metrics = {
        "distances": [],
        "path_lengths": [],
        "remaining_pellets": [],
        "safe_pellets": [],
        "target_switches": 0,
        "immediate_reversals": 0,
        "snapshots_taken": []
    }
    
    last_safe_count = None
    last_path_len = None
    
    print(f"Running Diagnostic for Seed {seed}")
    for step in range(n_steps):
        px, py = obs.player_position[0].item(), obs.player_position[1].item()
        start = pixel_to_dof(px, py)
        all_targets = extract_pellet_targets(dof, obs.pellets)
        
        if not all_targets:
            break
            
        ghosts = [pixel_to_dof(obs.ghost_positions[i][0].item(), obs.ghost_positions[i][1].item()) for i in range(4)]
        dmaps = static_danger_map(obs.ghost_positions, state.ghosts.modes, K=15, radius=3, mult=5)
        
        safe_pellets = [t for t in all_targets if dmaps[0].get(t, 0) < 30]
        num_safe = len(safe_pellets)
        
        # Determine target switch before calling plan()
        prev_target = planner.committed_target
        
        # Plan action
        action = planner.plan(obs, state)
        path_len = planner.last_path_length
        chosen_target = planner.committed_target
        
        is_oscillation = osc_counter.update(px, py)
        
        # Distances
        if chosen_target:
            man_dist = abs(start[0] - chosen_target[0]) + abs(start[1] - chosen_target[1])
            metrics["distances"].append(man_dist)
            
        metrics["path_lengths"].append(path_len)
        metrics["remaining_pellets"].append(len(all_targets))
        metrics["safe_pellets"].append(num_safe)
        
        if prev_target is not None and chosen_target != prev_target and prev_target in all_targets:
            metrics["target_switches"] += 1
            trigger = "Target Switch"
        elif is_oscillation:
            trigger = "Oscillation"
        elif path_len <= 1 and len(all_targets) > 0:
            trigger = "Stuck (Path <= 1)"
        elif last_safe_count is not None and (last_safe_count - num_safe) > 5 and path_len < (last_path_len or 0) / 2:
            trigger = "Blocked Corridor (Path Collapsed)"
        else:
            trigger = None
            
        # Snapshot logic (limit to ~5 per seed to avoid clutter)
        if trigger and len(metrics["snapshots_taken"]) < 5:
            # Recompute path just for drawing
            targets_to_search = {chosen_target} if chosen_target else all_targets
            path = spatiotemporal_astar(dof, start, targets_to_search, dmaps, reversal_pos=planner.prev_pos, reversal_penalty=planner.reversal_penalty)
            
            fname = f"snapshot_s{seed}_step{step}_{trigger.split()[0]}.png"
            fpath = os.path.join(os.path.dirname(__file__), '..', 'results', 'snapshots', fname)
            render_snapshot(dof, start, ghosts, all_targets, chosen_target, path, dmaps[0], fpath, f"{trigger} at Step {step}")
            metrics["snapshots_taken"].append(fname)
            
        last_safe_count = num_safe
        last_path_len = path_len
        
        obs, state, _, done, _ = step_fn(state, jnp.array(action, dtype=jnp.int32))
        if done:
            break
            
    metrics["immediate_reversals"] = planner.immediate_reversals
    return metrics


if __name__ == "__main__":
    seeds = [42, 789]
    all_metrics = {}
    
    for s in seeds:
        all_metrics[s] = run_diagnostic(s)
        
    # Aggregate arrays
    dists = []
    paths = []
    safes = []
    totals = []
    for s, m in all_metrics.items():
        dists.extend(m["distances"])
        paths.extend(m["path_lengths"])
        safes.extend(m["safe_pellets"])
        totals.extend(m["remaining_pellets"])
        
    d1_3 = sum(1 for d in dists if 1 <= d <= 3)
    d4_10 = sum(1 for d in dists if 4 <= d <= 10)
    d11_plus = sum(1 for d in dists if d > 10)
    total_d = len(dists) or 1
    
    out_lines = [
        "# MF Target Selection Diagnostic",
        "",
        "## Overall Distributions (Seeds 42 & 789)",
        "",
        f"- **Path Lengths**: Mean={np.mean(paths):.2f}, Median={np.median(paths):.2f}, Min={np.min(paths)}, Max={np.max(paths)}",
        f"- **Safe Pellets vs Total**: Average {np.mean(safes):.1f} safe out of {np.mean(totals):.1f} remaining ({(np.mean(safes)/np.mean(totals))*100:.1f}%)",
        "",
        "### Target Distance Categories",
        f"- **1-3 cells**:  {d1_3} targets ({(d1_3/total_d)*100:.1f}%)",
        f"- **4-10 cells**: {d4_10} targets ({(d4_10/total_d)*100:.1f}%)",
        f"- **>10 cells**:  {d11_plus} targets ({(d11_plus/total_d)*100:.1f}%)",
        "",
        "## Snapshots",
        "Visual evidence of planner failures:"
    ]
    
    for s, m in all_metrics.items():
        for snap in m["snapshots_taken"]:
            out_lines.append(f"![{snap}](./snapshots/{snap})")
            
    with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'mf_target_analysis.md'), 'w') as f:
        f.write("\n".join(out_lines))
        
    print("Diagnostics complete. Written to project/results/mf_target_analysis.md")
