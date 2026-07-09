"""
diagnostics/deep_diagnostics.py

Full diagnostic suite — used to investigate root causes of planner failure.

Sections:
  1. Ghost prediction accuracy  (MBv0 linear vs MBv2 JAX-native)
  2. Danger radius ablation
  3. Planner decision log       (target switches, danger dominance, path length)
  4. MF vs MBv2 plan divergence

Run from repository root:
    conda run -n jaxatari python project/diagnostics/deep_diagnostics.py

This script does NOT run any training. It only steps the environment and
calls the planner, so results are fast and deterministic.
"""
import sys
import os
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from jaxatari.games.jax_mspacman import JaxPacman
from jaxatari.games.mspacman_mazes import MsPacmanMaze
from project.utils.coords import pixel_to_dof, pellet_to_dof, extract_pellet_targets, get_action_for_path
from project.utils.danger_map import build_danger_maps, static_danger_map
from project.utils.ghost_sim import build_jit_ghost_step, forward_roll
from project.utils.astar import spatiotemporal_astar


# ─────────────────────────────────────────────
# Section 1: Ghost prediction accuracy
# ─────────────────────────────────────────────

def run_prediction_accuracy(env, step_fn, jit_step, n_samples=20):
    print("=" * 60)
    print("SECTION 1: Ghost Prediction Accuracy")
    print("MBv0 (linear extrapolation) vs MBv2 (JAXAtari-native)")
    print("=" * 60)

    key = jax.random.PRNGKey(77)
    obs, state = env.reset(key)
    for _ in range(50):
        obs, state, _, _, _ = step_fn(state, jnp.array(0, dtype=jnp.int32))

    horizons = [1, 3, 5, 10, 15]
    v0_err = {h: [] for h in horizons}
    v2_err = {h: [] for h in horizons}

    last_pix = [[obs.ghost_positions[i][0].item(),
                  obs.ghost_positions[i][1].item()] for i in range(4)]

    for sample in range(n_samples):
        obs, state, _, _, _ = step_fn(state, jnp.array(0, dtype=jnp.int32))
        cur_pix = [[obs.ghost_positions[i][0].item(),
                    obs.ghost_positions[i][1].item()] for i in range(4)]

        # MBv0 velocity
        velocities = []
        for i in range(4):
            dx, dy = cur_pix[i][0]-last_pix[i][0], cur_pix[i][1]-last_pix[i][1]
            vx = 1 if dx>0 else (-1 if dx<0 else 0)
            vy = 1 if dy>0 else (-1 if dy<0 else 0)
            if abs(dx)>abs(dy): vy = 0
            else: vx = 0
            velocities.append((vx, vy))
        last_pix = [[p[0],p[1]] for p in cur_pix]

        # MBv0 predictions (DOF grid space)
        v0_cur = [list(pixel_to_dof(cur_pix[i][0], cur_pix[i][1])) for i in range(4)]
        v0_preds = [list(v0_cur)]
        for _ in range(max(horizons)):
            for i in range(4):
                vx, vy = velocities[i]
                v0_cur[i] = [v0_cur[i][0]+vx, v0_cur[i][1]+vy]
            v0_preds.append([list(c) for c in v0_cur])

        # MBv2 predictions
        rkey = jax.random.PRNGKey(sample * 7 + 13)
        v2_raw = forward_roll(jit_step, state.ghosts,
                               obs.player_position, obs.player_action,
                               rkey, K=max(horizons))
        v2_preds = [None] + [[list(pixel_to_dof(int(v2_raw[t][i][0]),
                                                  int(v2_raw[t][i][1])))
                               for i in range(4)] for t in range(max(horizons))]

        # Roll env forward to get ground truth
        saved_state, saved_obs = state, obs
        actuals = {}
        for f in range(1, max(horizons)+1):
            obs, state, _, _, _ = step_fn(state, jnp.array(0, dtype=jnp.int32))
            actuals[f] = [list(pixel_to_dof(obs.ghost_positions[i][0].item(),
                                             obs.ghost_positions[i][1].item()))
                          for i in range(4)]

        for h in horizons:
            act = actuals[h]
            p0  = v0_preds[h]
            p2  = v2_preds[h]
            for i in range(4):
                v0_err[h].append(abs(act[i][0]-p0[i][0]) + abs(act[i][1]-p0[i][1]))
                v2_err[h].append(abs(act[i][0]-p2[i][0]) + abs(act[i][1]-p2[i][1]))

        state, obs = saved_state, saved_obs
        for _ in range(10):
            obs, state, _, _, _ = step_fn(state, jnp.array(0, dtype=jnp.int32))

    print(f"{'K':<6} | {'MBv0 (linear)':<18} | {'MBv2 (JAX-native)':<20}")
    print("-" * 52)
    for h in horizons:
        e0 = np.mean(v0_err[h])
        e2 = np.mean(v2_err[h])
        tag = " ← BETTER" if e2 < e0 else (" ← WORSE" if e2 > e0 else " (tie)")
        print(f"K={h:<4} | {e0:<18.3f} | {e2:<20.3f}{tag}")


# ─────────────────────────────────────────────
# Section 2: Danger radius ablation
# ─────────────────────────────────────────────

def run_danger_radius_ablation(env, step_fn, jit_step, n_steps=300):
    print("\n" + "=" * 60)
    print("SECTION 2: Danger Radius Ablation (MBv2)")
    print("=" * 60)
    dof = env.consts.DOF_MAZES[0]
    configs = [
        ("r=6, m=20 (original baseline)", 6, 20),
        ("r=3, m=5  (current best)",      3,  5),
        ("r=2, m=2  (weak signal)",        2,  2),
        ("r=1, m=1  (minimal)",            1,  1),
    ]
    for label, radius, mult in configs:
        key = jax.random.PRNGKey(42)
        obs, state = env.reset(key)
        rkey = jax.random.PRNGKey(0)
        score, pellets, osc, survival = 0, 0, 0, 0
        last_10 = []

        for _ in range(n_steps):
            survival += 1
            px, py = obs.player_position[0].item(), obs.player_position[1].item()
            start   = pixel_to_dof(px, py)
            targets = extract_pellet_targets(dof, obs.pellets)
            if not targets: break
            rkey, sk = jax.random.split(rkey)
            preds = forward_roll(jit_step, state.ghosts,
                                  obs.player_position, obs.player_action, sk, K=15)
            dmaps  = build_danger_maps(preds, state.ghosts.modes, radius, mult)
            path   = spatiotemporal_astar(dof, start, targets, dmaps)
            action = get_action_for_path(start, path)

            last_10.append(start)
            if len(last_10) > 10: last_10.pop(0)
            if last_10.count(start) > 3: osc += 1

            obs, state, _, done, _ = step_fn(state, jnp.array(action, dtype=jnp.int32))
            score, pellets = state.score.item(), state.level.collected_pellets.item()
            if done: break

        print(f"  {label:35s} | score={score:5d}  pellets={pellets:3d}  "
              f"osc={osc:3d}  survival={survival}")


# ─────────────────────────────────────────────
# Section 3: Planner decision log
# ─────────────────────────────────────────────

def run_decision_log(env, step_fn, jit_step, radius=3, mult=5, n_steps=200):
    print("\n" + "=" * 60)
    print(f"SECTION 3: Planner Decision Log  (radius={radius}, mult={mult})")
    print("=" * 60)
    dof = env.consts.DOF_MAZES[0]
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    rkey = jax.random.PRNGKey(1)

    prev_target   = None
    target_sw     = 0
    danger_dom    = 0
    path_lengths  = []

    for step in range(n_steps):
        px, py = obs.player_position[0].item(), obs.player_position[1].item()
        start  = pixel_to_dof(px, py)
        targets = extract_pellet_targets(dof, obs.pellets)
        if not targets: break
        rkey, sk = jax.random.split(rkey)
        preds  = forward_roll(jit_step, state.ghosts,
                               obs.player_position, obs.player_action, sk, K=15)
        dmaps  = build_danger_maps(preds, state.ghosts.modes, radius, mult)

        if dmaps[0].get(start, 0) > 100:
            danger_dom += 1

        path   = spatiotemporal_astar(dof, start, targets, dmaps)
        cur_tgt = path[-1] if path else None
        if cur_tgt != prev_target:
            target_sw += 1
        prev_target = cur_tgt
        path_lengths.append(len(path))

        action = get_action_for_path(start, path)
        obs, state, _, done, _ = step_fn(state, jnp.array(action, dtype=jnp.int32))
        if done: break

    print(f"  Target switches           : {target_sw}/{n_steps} ({100*target_sw/n_steps:.1f}%)")
    print(f"  Danger-dominated starts   : {danger_dom}/{n_steps} ({100*danger_dom/n_steps:.1f}%)")
    print(f"  Mean path length          : {np.mean(path_lengths):.1f} cells")
    print(f"  Max  path length          : {max(path_lengths)} cells")


# ─────────────────────────────────────────────
# Section 4: MF vs MBv2 divergence
# ─────────────────────────────────────────────

def run_plan_divergence(env, step_fn, jit_step, radius=3, mult=5, n_steps=200):
    print("\n" + "=" * 60)
    print(f"SECTION 4: MF vs MBv2 Plan Divergence  (radius={radius}, mult={mult})")
    print("=" * 60)
    dof = env.consts.DOF_MAZES[0]
    key = jax.random.PRNGKey(99)
    obs, state = env.reset(key)
    rkey = jax.random.PRNGKey(2)
    differ = 0

    for step in range(n_steps):
        px, py = obs.player_position[0].item(), obs.player_position[1].item()
        start  = pixel_to_dof(px, py)
        targets = extract_pellet_targets(dof, obs.pellets)
        if not targets: break
        rkey, sk = jax.random.split(rkey)

        preds   = forward_roll(jit_step, state.ghosts,
                                obs.player_position, obs.player_action, sk, K=15)
        mb_dmaps = build_danger_maps(preds, state.ghosts.modes, radius, mult)
        mb_path  = spatiotemporal_astar(dof, start, targets, mb_dmaps)

        cur     = np.array([[obs.ghost_positions[i][0].item(),
                              obs.ghost_positions[i][1].item()] for i in range(4)])
        mf_dmaps = build_danger_maps([cur]*15, state.ghosts.modes, radius, mult)
        mf_path  = spatiotemporal_astar(dof, start, targets, mf_dmaps)

        if mf_path != mb_path:
            differ += 1
            if differ <= 3:
                print(f"  [step {step:3d}] "
                      f"MF→{mf_path[-1] if mf_path else '?'}  "
                      f"MB→{mb_path[-1] if mb_path else '?'}  "
                      f"MF_action={get_action_for_path(start,mf_path)}  "
                      f"MB_action={get_action_for_path(start,mb_path)}")

        action = get_action_for_path(start, mb_path)
        obs, state, _, done, _ = step_fn(state, jnp.array(action, dtype=jnp.int32))
        if done: break

    print(f"\n  Plans differed: {differ}/{n_steps} ({100*differ/n_steps:.1f}%)")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    env      = JaxPacman()
    consts   = env.consts
    step_fn  = jax.jit(env.step)
    jit_step = build_jit_ghost_step(consts)

    print("Compiling…")
    obs, state = env.reset(jax.random.PRNGKey(0))
    step_fn(state, jnp.array(0, dtype=jnp.int32))
    forward_roll(jit_step, state.ghosts, obs.player_position,
                  obs.player_action, jax.random.PRNGKey(0), K=15)
    print("Done.\n")

    run_prediction_accuracy(env, step_fn, jit_step, n_samples=20)
    run_danger_radius_ablation(env, step_fn, jit_step)
    run_decision_log(env, step_fn, jit_step, radius=3, mult=5)
    run_plan_divergence(env, step_fn, jit_step, radius=3, mult=5)
