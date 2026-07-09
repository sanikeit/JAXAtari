"""
experiments/evaluate_planners.py

3-way MF vs MBv0 vs MBv2 comparison + danger-radius sweep.

Run from repository root:
    conda run -n jaxatari python project/experiments/evaluate_planners.py

All results use:
  - Correct DOF coordinate system (gx=pixel_x//4, gy=pixel_y//4)
  - Correct pellet mapping (pellx*2+1, pelly*3+2)
  - Correct action mapping (2=UP, 3=RIGHT, 4=LEFT, 5=DOWN)
  - Pixel-level oscillation counter (true reversals, not DOF-cell repetition)
"""
import sys
import os
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from jaxatari.games.jax_mspacman import JaxPacman
from project.planners.mf_baseline import MFPlanner
from project.planners.mb_v2_planner import MBv2Planner
from project.utils.coords import pixel_to_dof, extract_pellet_targets, get_action_for_path
from project.utils.danger_map import build_danger_maps, static_danger_map
from project.utils.ghost_sim import build_jit_ghost_step, forward_roll
from project.utils.astar import spatiotemporal_astar


# ─────────────────────────────────────────────
# Oscillation counter (pixel-level)
# ─────────────────────────────────────────────

class OscillationCounter:
    """
    Detects true reversals at pixel resolution.
    Fires when ≥2 of the last 3 positions appeared in the 3 positions before that.
    """
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


# ─────────────────────────────────────────────
# Linear velocity predictor (MBv0)
# ─────────────────────────────────────────────

def mbv0_predict(obs, last_ghost_pix, K):
    velocities = []
    for i in range(4):
        dx = obs.ghost_positions[i][0].item() - last_ghost_pix[i][0]
        dy = obs.ghost_positions[i][1].item() - last_ghost_pix[i][1]
        vx = 1 if dx > 0 else (-1 if dx < 0 else 0)
        vy = 1 if dy > 0 else (-1 if dy < 0 else 0)
        if abs(dx) > abs(dy): vy = 0
        else: vx = 0
        velocities.append((vx, vy))

    cur = [[obs.ghost_positions[i][0].item(), obs.ghost_positions[i][1].item()]
           for i in range(4)]
    preds = []
    for _ in range(K):
        for i in range(4):
            cur[i][0] += velocities[i][0] * 4   # v0 works in pixel space
            cur[i][1] += velocities[i][1] * 4
        preds.append(np.array([[c[0], c[1]] for c in cur]))
    return preds


# ─────────────────────────────────────────────
# Generic run loop
# ─────────────────────────────────────────────

def run_episode(label, mode, env, step_fn, jit_step, consts,
                K=15, radius=3, mult=5, n_steps=300, seed=42):
    """
    Run one episode and report metrics.

    mode: 'MF' | 'MBv0' | 'MBv2'
    """
    dof = consts.DOF_MAZES[0]
    key = jax.random.PRNGKey(seed)
    obs, state = env.reset(key)
    rkey = jax.random.PRNGKey(seed + 1)

    osc_counter = OscillationCounter()
    last_ghost_pix = [[obs.ghost_positions[i][0].item(),
                        obs.ghost_positions[i][1].item()] for i in range(4)]

    score, pellets, osc, survival, diverge = 0, 0, 0, 0, 0

    for step in range(n_steps):
        survival += 1
        px, py = obs.player_position[0].item(), obs.player_position[1].item()
        start = pixel_to_dof(px, py)
        targets = extract_pellet_targets(dof, obs.pellets)
        if not targets:
            break

        rkey, sk = jax.random.split(rkey)

        # Build predictions
        if mode == 'MF':
            cur = np.array([[obs.ghost_positions[i][0].item(),
                              obs.ghost_positions[i][1].item()] for i in range(4)])
            preds = [cur] * K
        elif mode == 'MBv0':
            preds = mbv0_predict(obs, last_ghost_pix, K)
        else:  # MBv2
            preds = forward_roll(jit_step, state.ghosts,
                                  obs.player_position, obs.player_action, sk, K=K)

        dmaps = build_danger_maps(preds, state.ghosts.modes, radius, mult)
        path  = spatiotemporal_astar(dof, start, targets, dmaps)
        action = get_action_for_path(start, path)

        # MF divergence comparison (for non-MF modes)
        if mode != 'MF':
            cur = np.array([[obs.ghost_positions[i][0].item(),
                              obs.ghost_positions[i][1].item()] for i in range(4)])
            mf_dmaps = build_danger_maps([cur] * K, state.ghosts.modes, radius, mult)
            mf_path  = spatiotemporal_astar(dof, start, targets, mf_dmaps)
            if mf_path != path:
                diverge += 1

        if osc_counter.update(px, py):
            osc += 1

        last_ghost_pix = [[obs.ghost_positions[i][0].item(),
                            obs.ghost_positions[i][1].item()] for i in range(4)]
        obs, state, _, done, _ = step_fn(state, jnp.array(action, dtype=jnp.int32))
        score   = state.score.item()
        pellets = state.level.collected_pellets.item()
        if done:
            break

    div_str = (f"  divergence={diverge}/{n_steps} ({100*diverge/n_steps:.1f}%)"
               if mode != 'MF' else "")
    print(f"  {label:38s} | score={score:5d}  pellets={pellets:3d}  "
          f"osc={osc:3d}  survival={survival:3d}{div_str}")
    return dict(label=label, score=score, pellets=pellets, osc=osc,
                survival=survival, diverge=diverge)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    env    = JaxPacman()
    consts = env.consts
    step_fn  = jax.jit(env.step)
    jit_step = build_jit_ghost_step(consts)

    print("Compiling…")
    obs, state = env.reset(jax.random.PRNGKey(0))
    step_fn(state, jnp.array(0, dtype=jnp.int32))
    forward_roll(jit_step, state.ghosts, obs.player_position,
                  obs.player_action, jax.random.PRNGKey(0), K=15)
    print("Done.\n")

    K, radius, mult, n_steps, seed = 15, 3, 5, 300, 42

    print("=" * 78)
    print(f"3-WAY COMPARISON  K={K}  radius={radius}  mult={mult}  "
          f"steps={n_steps}  seed={seed}")
    print("=" * 78)
    run_episode("MF  (static danger map)",     'MF',   env, step_fn, jit_step, consts, K, radius, mult, n_steps, seed)
    run_episode("MBv0 (linear extrapolation)", 'MBv0', env, step_fn, jit_step, consts, K, radius, mult, n_steps, seed)
    run_episode("MBv2 (JAXAtari-native sim)",  'MBv2', env, step_fn, jit_step, consts, K, radius, mult, n_steps, seed)

    print()
    print("=" * 78)
    print("DANGER RADIUS SWEEP  (MBv2 only)")
    print("=" * 78)
    for r, m in [(6, 20), (3, 5), (2, 2), (1, 1)]:
        run_episode(f"MBv2  radius={r}  mult={m}", 'MBv2',
                    env, step_fn, jit_step, consts, K, r, m, n_steps, seed)
