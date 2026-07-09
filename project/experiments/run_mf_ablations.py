"""
experiments/run_mf_ablations.py

Runs the MF baseline validation and ablation study.
Variants:
A: MF Baseline
B: MF + Reversal(5)
C: MF + Reversal(10)
D: MF + Reversal(20)
E: MF + Commit(5)
F: MF + Commit(10)
G: MF + Commit(20)

Metrics: Score, Pellets, Survival, Oscillations, Path Lengths (Mean/Min/Max).
Additional logs for seed 42: Target Switches, Immediate Reversals.
"""
import sys
import os
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from jaxatari.games.jax_mspacman import JaxPacman
from project.planners.mf_ablation_planner import MFAblationPlanner

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

def run_variant(label, planner_kwargs, seeds, n_steps=300):
    env = JaxPacman()
    step_fn = jax.jit(env.step)
    dof = env.consts.DOF_MAZES[0]
    
    # Initialize planner
    planner = MFAblationPlanner(dof, **planner_kwargs)
    
    results = []
    
    print(f"Running Variant: {label}")
    for seed in seeds:
        key = jax.random.PRNGKey(seed)
        obs, state = env.reset(key)
        
        planner.reset()
        osc_counter = OscillationCounter()
        
        score, pellets, osc, survival = 0, 0, 0, 0
        path_lengths = []
        
        for step in range(n_steps):
            survival += 1
            px, py = obs.player_position[0].item(), obs.player_position[1].item()
            
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
            "target_switches": planner.target_switches,
            "immediate_reversals": planner.immediate_reversals,
            "mean_path": np.mean(path_lengths) if path_lengths else 0,
            "min_path": np.min(path_lengths) if path_lengths else 0,
            "max_path": np.max(path_lengths) if path_lengths else 0,
        }
        results.append(metrics)
        print(f"  Seed {seed:3d}: Score={score:5d}  Pellets={pellets:3d}  Osc={osc:3d}  "
              f"Switches={planner.target_switches:3d}  Revs={planner.immediate_reversals:3d}")
              
    return results

def format_markdown_table(label, results):
    lines = [f"### {label}"]
    lines.append("| Seed | Score | Pellets | Survival | Oscillations | Mean Path | Min Path | Max Path | Switches | Reversals |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    
    for r in results:
        lines.append(f"| {r['seed']} | {r['score']} | {r['pellets']} | {r['survival']} | {r['osc']} | "
                     f"{r['mean_path']:.1f} | {r['min_path']} | {r['max_path']} | {r['target_switches']} | {r['immediate_reversals']} |")
                     
    # Averages
    mean_score = np.mean([r['score'] for r in results])
    std_score = np.std([r['score'] for r in results])
    mean_pellets = np.mean([r['pellets'] for r in results])
    std_pellets = np.std([r['pellets'] for r in results])
    mean_osc = np.mean([r['osc'] for r in results])
    
    lines.append(f"| **AVG** | **{mean_score:.1f}±{std_score:.1f}** | **{mean_pellets:.1f}±{std_pellets:.1f}** | "
                 f"**{np.mean([r['survival'] for r in results]):.1f}** | **{mean_osc:.1f}** | "
                 f"**{np.mean([r['mean_path'] for r in results]):.1f}** | - | - | "
                 f"**{np.mean([r['target_switches'] for r in results]):.1f}** | **{np.mean([r['immediate_reversals'] for r in results]):.1f}** |")
    lines.append("")
    return "\n".join(lines)

if __name__ == "__main__":
    seeds = [42, 123, 456, 789, 999]
    n_steps = 300
    
    variants = {
        "A: MF Baseline": {"reversal_penalty": 0, "commit_duration": 0},
        "B: MF + Reversal(5)": {"reversal_penalty": 5, "commit_duration": 0},
        "C: MF + Reversal(10)": {"reversal_penalty": 10, "commit_duration": 0},
        "D: MF + Reversal(20)": {"reversal_penalty": 20, "commit_duration": 0},
        "E: MF + Commit(5)": {"reversal_penalty": 0, "commit_duration": 5},
        "F: MF + Commit(10)": {"reversal_penalty": 0, "commit_duration": 10},
        "G: MF + Commit(20)": {"reversal_penalty": 0, "commit_duration": 20},
    }
    
    all_results = {}
    for label, kwargs in variants.items():
        all_results[label] = run_variant(label, kwargs, seeds, n_steps)
        
    # Pick best (simple heuristic: highest avg score)
    best_rev_label = max(["B: MF + Reversal(5)", "C: MF + Reversal(10)", "D: MF + Reversal(20)"], 
                         key=lambda l: np.mean([r['score'] for r in all_results[l]]))
    best_com_label = max(["E: MF + Commit(5)", "F: MF + Commit(10)", "G: MF + Commit(20)"], 
                         key=lambda l: np.mean([r['score'] for r in all_results[l]]))
                         
    best_rev_val = variants[best_rev_label]["reversal_penalty"]
    best_com_val = variants[best_com_label]["commit_duration"]
    
    h_label = f"H: MF + Reversal({best_rev_val}) + Commit({best_com_val})"
    print(f"\nDetermined Best Combined Variant: {h_label}")
    
    all_results[h_label] = run_variant(h_label, {"reversal_penalty": best_rev_val, "commit_duration": best_com_val}, seeds, n_steps)
    
    # Write Baseline Validation Report
    with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'mf_baseline_validation.md'), 'w') as f:
        f.write("# MF Baseline Validation\n\n")
        f.write("This report captures the clean Model-Free baseline performance across 5 seeds.\n\n")
        f.write(format_markdown_table("A: MF Baseline", all_results["A: MF Baseline"]))
        
    # Write Ablation Study Report
    with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'mf_ablation_study.md'), 'w') as f:
        f.write("# MF Ablation Study\n\n")
        for label, results in all_results.items():
            f.write(format_markdown_table(label, results))
            
    print("\nExperiments complete. Results written to project/results/")
