"""
experiments/run_mf_global_ablations.py

Runs the MFv2 Global Targeting Ablation.
"""
import sys
import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from jaxatari.games.jax_mspacman import JaxPacman
from project.planners.mf_global_planner import MFGlobalPlanner, manhattan
from project.utils.coords import pixel_to_dof, extract_pellet_targets, is_walkable
from project.experiments.run_mf_ablations import OscillationCounter
from project.diagnostics.mf_target_diagnostic import render_snapshot
from project.utils.danger_map import static_danger_map

def plot_heatmap(dof, visited_cells, start, end, filepath, title):
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='black')
    ax.set_facecolor('black')
    ax.invert_yaxis()
    ax.set_xlim(-1, 41)
    ax.set_ylim(45, -1)
    
    walk_x, walk_y = [], []
    for x in range(40):
        for y in range(44):
            if is_walkable(dof, x, y):
                walk_x.append(x)
                walk_y.append(y)
    ax.scatter(walk_x, walk_y, color='#222222', marker='s', s=20)
    
    vx, vy = zip(*visited_cells) if visited_cells else ([], [])
    # Heatmap color based on visitation frequency
    from collections import Counter
    counts = Counter(visited_cells)
    
    hx, hy, hc = [], [], []
    for (x, y), c in counts.items():
        hx.append(x)
        hy.append(y)
        hc.append(c)
        
    scatter = ax.scatter(hx, hy, c=hc, cmap='plasma', marker='s', s=40, alpha=0.8)
    plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    
    ax.scatter([start[0]], [start[1]], color='white', s=100, marker='*')
    ax.scatter([end[0]], [end[1]], color='lime', s=100, marker='X')
    
    ax.set_title(title, color='white')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(filepath, facecolor=fig.get_facecolor())
    plt.close()

def run_variant(label, strategy, seeds, n_steps=300):
    env = JaxPacman()
    step_fn = jax.jit(env.step)
    dof = env.consts.DOF_MAZES[0]
    
    total_walkable = sum(1 for x in range(40) for y in range(44) if is_walkable(dof, x, y))
    
    results = []
    
    print(f"Running Variant: {label}")
    for seed in seeds:
        planner = MFGlobalPlanner(dof, strategy=strategy)
        key = jax.random.PRNGKey(seed)
        obs, state = env.reset(key)
        
        planner.reset()
        osc_counter = OscillationCounter()
        
        score, pellets, osc, survival = 0, 0, 0, 0
        path_lengths = []
        distances = []
        visited = []
        
        start_pos = None
        last_safe_count = None
        last_path_len = None
        snapshots_taken = 0
        
        for step in range(n_steps):
            survival += 1
            px, py = obs.player_position[0].item(), obs.player_position[1].item()
            pos = pixel_to_dof(px, py)
            visited.append(pos)
            
            if start_pos is None: start_pos = pos
            
            all_targets = extract_pellet_targets(dof, obs.pellets)
            if not all_targets:
                break
                
            prev_target = planner.committed_target
            action = planner.plan(obs, state)
            path_len = planner.last_path_length
            chosen_target = planner.committed_target
            
            if path_len > 0:
                path_lengths.append(path_len)
            if chosen_target:
                distances.append(manhattan(pos, chosen_target))
                
            is_osc = osc_counter.update(px, py)
            if is_osc: osc += 1
            
            # Snapshots for seed 789
            if seed == 789 and snapshots_taken < 3:
                ghosts = [pixel_to_dof(obs.ghost_positions[i][0].item(), obs.ghost_positions[i][1].item()) for i in range(4)]
                dmaps = static_danger_map(obs.ghost_positions, state.ghosts.modes, K=15, radius=3, mult=5)
                safe_pellets = [t for t in all_targets if dmaps[0].get(t, 0) < 30]
                num_safe = len(safe_pellets)
                
                trigger = None
                if prev_target is not None and chosen_target != prev_target and prev_target in all_targets:
                    trigger = "Target Switch"
                elif is_osc:
                    trigger = "Oscillation"
                elif path_len <= 1 and len(all_targets) > 0:
                    trigger = "Stuck"
                elif last_safe_count is not None and (last_safe_count - num_safe) > 5 and path_len < (last_path_len or 0)/2:
                    trigger = "Blocked Corridor"
                    
                if trigger:
                    from project.utils.astar import spatiotemporal_astar
                    targets_to_search = {chosen_target} if chosen_target else all_targets
                    path = spatiotemporal_astar(dof, pos, targets_to_search, dmaps, reversal_pos=planner.prev_pos, reversal_penalty=planner.reversal_penalty)
                    
                    fname = f"snap_{strategy}_s{seed}_step{step}_{trigger.replace(' ', '')}.png"
                    fpath = os.path.join(os.path.dirname(__file__), '..', 'results', 'global_targeting_visuals', fname)
                    render_snapshot(dof, pos, ghosts, all_targets, chosen_target, path, dmaps[0], fpath, f"[{strategy}] {trigger} Step {step}")
                    snapshots_taken += 1
                
                last_safe_count = num_safe
                last_path_len = path_len
                
            obs, state, _, done, _ = step_fn(state, jnp.array(action, dtype=jnp.int32))
            score = state.score.item()
            pellets = state.level.collected_pellets.item()
            if done: break
            
        coverage = len(set(visited)) / total_walkable
        
        # Heatmap for seed 789
        if seed == 789:
            hname = f"heatmap_{strategy}_s{seed}.png"
            hpath = os.path.join(os.path.dirname(__file__), '..', 'results', 'global_targeting_visuals', hname)
            plot_heatmap(dof, visited, start_pos, visited[-1], hpath, f"[{strategy}] Trajectory Heatmap")
            
        metrics = {
            "seed": seed,
            "score": score,
            "pellets": pellets,
            "survival": survival,
            "osc": osc,
            "coverage": coverage,
            "distances": distances,
            "mean_path": np.mean(path_lengths) if path_lengths else 0,
            "min_path": np.min(path_lengths) if path_lengths else 0,
            "max_path": np.max(path_lengths) if path_lengths else 0,
        }
        results.append(metrics)
        print(f"  Seed {seed:3d}: Score={score:5d}  Cov={coverage*100:4.1f}%  PathLen={metrics['mean_path']:.1f}")
              
    return results

def format_markdown_table(label, results):
    lines = [f"### {label}"]
    lines.append("| Seed | Score | Pellets | Coverage | Mean Path | Min Path | Max Path | Dist 1-3 | Dist 4-10 | Dist >10 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    
    all_dists = []
    
    for r in results:
        dists = r["distances"]
        all_dists.extend(dists)
        tot = len(dists) or 1
        d1_3 = sum(1 for d in dists if 1 <= d <= 3) / tot * 100
        d4_10 = sum(1 for d in dists if 4 <= d <= 10) / tot * 100
        d11_plus = sum(1 for d in dists if d > 10) / tot * 100
        
        lines.append(f"| {r['seed']} | {r['score']} | {r['pellets']} | {r['coverage']*100:.1f}% | "
                     f"{r['mean_path']:.1f} | {r['min_path']} | {r['max_path']} | {d1_3:.1f}% | {d4_10:.1f}% | {d11_plus:.1f}% |")
                     
    mean_score = np.mean([r['score'] for r in results])
    mean_cov = np.mean([r['coverage'] for r in results]) * 100
    mean_path = np.mean([r['mean_path'] for r in results])
    
    tot_all = len(all_dists) or 1
    ad1_3 = sum(1 for d in all_dists if 1 <= d <= 3) / tot_all * 100
    ad4_10 = sum(1 for d in all_dists if 4 <= d <= 10) / tot_all * 100
    ad11_plus = sum(1 for d in all_dists if d > 10) / tot_all * 100
    
    lines.append(f"| **AVG** | **{mean_score:.1f}** | **{np.mean([r['pellets'] for r in results]):.1f}** | **{mean_cov:.1f}%** | "
                 f"**{mean_path:.1f}** | - | - | **{ad1_3:.1f}%** | **{ad4_10:.1f}%** | **{ad11_plus:.1f}%** |")
    lines.append("")
    return "\n".join(lines), mean_path, mean_cov, mean_score

if __name__ == "__main__":
    seeds = [42, 123, 456, 789, 999]
    n_steps = 300
    
    variants = {
        "A: Nearest Pellet (Baseline)": "nearest",
        "B: Farthest Safe Pellet": "farthest",
        "C: Cluster Targeting": "cluster",
        "D: Utility Targeting": "utility",
        "E: Random-Far Baseline": "random_far"
    }
    
    all_results = {}
    stats = {}
    
    for label, strat in variants.items():
        res = run_variant(label, strat, seeds, n_steps)
        all_results[label] = res
        
    with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'mf_global_targeting_ablation.md'), 'w') as f:
        f.write("# MFv2 Global Targeting Ablation\n\n")
        
        for label, results in all_results.items():
            tbl, path, cov, score = format_markdown_table(label, results)
            f.write(tbl)
            stats[label] = {"path": path, "cov": cov, "score": score}
            
        f.write("## Analysis\n\n")
        
        # Conclusion logic
        base_path = stats["A: Nearest Pellet (Baseline)"]["path"]
        base_cov = stats["A: Nearest Pellet (Baseline)"]["cov"]
        base_score = stats["A: Nearest Pellet (Baseline)"]["score"]
        
        best_label = max(stats.keys(), key=lambda k: stats[k]["score"])
        best = stats[best_label]
        
        q1 = "Yes" if best["path"] > base_path + 0.5 else "No"
        q2 = "Yes" # we can see this from the tables visually
        q3 = "Yes" if best["cov"] > base_cov + 2.0 else "No"
        q4 = "Yes" if best["score"] > base_score + 20 else "No"
        
        if q4 == "Yes" and q1 == "Yes" and q3 == "Yes":
            q5 = "Target selection WAS the dominant bottleneck."
        elif q1 == "Yes" and q3 == "Yes" and q4 == "No":
            q5 = "Future prediction / planner integration is the next bottleneck. Agent travels further but still fails."
        else:
            q5 = "Inconclusive / No significant improvement."
            
        f.write(f"1. Does average path length increase significantly? **{q1}** (Baseline: {base_path:.1f} -> Best: {best['path']:.1f})\n")
        f.write(f"3. Does maze coverage improve? **{q3}** (Baseline: {base_cov:.1f}% -> Best: {best['cov']:.1f}%)\n")
        f.write(f"4. Does score improve? **{q4}** (Baseline: {base_score:.1f} -> Best: {best['score']:.1f})\n")
        f.write(f"\n**Final Conclusion:** {q5}\n")
        
    print("\nExperiments complete. Results written to project/results/mf_global_targeting_ablation.md")
