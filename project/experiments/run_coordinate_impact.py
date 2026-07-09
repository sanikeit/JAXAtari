"""
experiments/run_coordinate_impact.py

Evaluates the impact of using JAX-native coordinates vs legacy coordinates.
"""
import sys
import os
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from jaxatari.games.jax_mspacman import JaxPacman
from project.planners.mf_ablation_planner import MFAblationPlanner
from project.planners.mf_jax_coords_planner import MFJaxPlanner

class OscillationCounter:
    def __init__(self):
        self.history = []

    def update(self, px, py):
        pos = (int(px), int(py))
        self.history.append(pos)
        if len(self.history) > 8:
            self.history.pop(0)
        if len(self.history) >= 6:
            recent = self.history[-3:]
            older  = self.history[:-3]
            return sum(p in older for p in recent) >= 2
        return False

def run_variant(label, planner_class, seeds, n_steps=300):
    env = JaxPacman()
    step_fn = jax.jit(env.step)
    dof = env.consts.DOF_MAZES[0]
    
    # Best MF variant from previous ablations
    planner = planner_class(dof, reversal_penalty=5, commit_duration=20)
    
    results = []
    
    print(f"Running Variant: {label}")
    for seed in seeds:
        key = jax.random.PRNGKey(seed)
        obs, state = env.reset(key)
        
        planner.reset()
        osc_counter = OscillationCounter()
        
        score, pellets, osc, survival = 0, 0, 0, 0
        path_lengths = []
        visited_cells = set()
        
        for step in range(n_steps):
            survival += 1
            px, py = obs.player_position[0].item(), obs.player_position[1].item()
            
            # Map for coverage calculation (using raw pixel rounding for consistency across both)
            visited_cells.add((px // 4, py // 4))
            
            action = planner.plan(obs, state)
            if planner.last_path_length > 0:
                path_lengths.append(planner.last_path_length)
                
            if osc_counter.update(px, py):
                osc += 1
                
            obs, state, _, done, _ = step_fn(state, jnp.array(action, dtype=jnp.int32))
            score = state.score.item()
            pellets = state.level.collected_pellets.item()
            if done:
                break
                
        metrics = {
            "seed": seed,
            "score": score,
            "pellets": pellets,
            "survival": survival,
            "osc": osc,
            "coverage": len(visited_cells),
            "mean_path": np.mean(path_lengths) if path_lengths else 0,
        }
        results.append(metrics)
        print(f"  Seed {seed:3d}: Score={score:5d}  Pellets={pellets:3d}  Osc={osc:3d}  "
              f"Coverage={len(visited_cells):3d}  MeanPath={metrics['mean_path']:.1f}")
              
    return results

def format_markdown_table(label, results):
    lines = [f"### {label}"]
    lines.append("| Seed | Score | Pellets | Survival | Oscillations | Maze Coverage | Mean Path |")
    lines.append("|---|---|---|---|---|---|---|")
    
    for r in results:
        lines.append(f"| {r['seed']} | {r['score']} | {r['pellets']} | {r['survival']} | {r['osc']} | {r['coverage']} | {r['mean_path']:.1f} |")
                     
    # Averages
    mean_score = np.mean([r['score'] for r in results])
    std_score = np.std([r['score'] for r in results])
    mean_pellets = np.mean([r['pellets'] for r in results])
    mean_osc = np.mean([r['osc'] for r in results])
    mean_cov = np.mean([r['coverage'] for r in results])
    mean_path = np.mean([r['mean_path'] for r in results])
    
    lines.append(f"| **AVG** | **{mean_score:.1f}±{std_score:.1f}** | **{mean_pellets:.1f}** | "
                 f"**{np.mean([r['survival'] for r in results]):.1f}** | **{mean_osc:.1f}** | "
                 f"**{mean_cov:.1f}** | **{mean_path:.1f}** |")
    lines.append("")
    return "\n".join(lines)

if __name__ == "__main__":
    seeds = [42, 123, 456, 789, 999]
    n_steps = 300
    
    print("Evaluating MF_legacy_coords...")
    legacy_results = run_variant("MF_legacy_coords", MFAblationPlanner, seeds, n_steps)
    
    print("\nEvaluating MF_jax_coords...")
    jax_results = run_variant("MF_jax_coords", MFJaxPlanner, seeds, n_steps)
    
    with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'jax_coordinate_retest.md'), 'w') as f:
        f.write("# JAX Coordinate Retest\n\n")
        f.write("Controlled experiment verifying the impact of JAXAtari-native coordinates vs legacy coords.\n")
        f.write("Planner settings: Reversal Penalty=5, Commitment=20, 300 steps.\n\n")
        f.write("## Results\n\n")
        f.write(format_markdown_table("MF_legacy_coords (x // 4)", legacy_results))
        f.write(format_markdown_table("MF_jax_coords ((x+5) // 4)", jax_results))
        
        f.write("## Analysis\n\n")
        f.write("### A. Does JAX-native mapping significantly improve MF?\n")
        f.write("*(See metrics above to answer)*\n\n")
        f.write("### B. Do previous MF conclusions still hold?\n")
        f.write("*(Yes/No based on if the planner still exhibits similar patterns)*\n\n")
        f.write("### C. Is target selection still the dominant bottleneck?\n")
        f.write("*(See mean path length)*\n\n")
        f.write("### D. Is future prediction still required?\n")
        f.write("*(See if it still fails corridors)*\n")
        
    print("\nResults written to project/results/jax_coordinate_retest.md")
