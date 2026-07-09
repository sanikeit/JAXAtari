"""
phase2_deep_diagnostics.py  (v2)

Root-cause analysis of the MF vs MB planner failure.
Fixes JAX-native ghost simulator by JIT-compiling everything upfront.
"""
import jax
import jax.numpy as jnp
import heapq
import numpy as np

from jaxatari.games.jax_mspacman import (
    JaxPacman,
    GhostMode,
    get_allowed_directions,
    get_chase_target,
    pathfind,
    get_new_position,
)
from jaxatari.games.mspacman_mazes import MsPacmanMaze

# =====================================================================
# Coordinate utils
# =====================================================================

def pixel_to_grid(pixel_x, pixel_y):
    return int(round(pixel_x / 4.0)), int(round(pixel_y / 4.0))

def extract_targets(maze, pellets):
    targets = set()
    for py in range(pellets.shape[1]):
        for px in range(pellets.shape[0]):
            if pellets[px, py] == 1:
                gx, gy = pixel_to_grid(px * 8 + 5, py * 12 + 6)
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        ny, nx = gy + dy, gx + dx
                        if 0 <= ny < maze.shape[0] and 0 <= nx < maze.shape[1] and maze[ny, nx] == 0:
                            targets.add((nx, ny))
                            break
                    else:
                        continue
                    break
    return targets

# =====================================================================
# JIT-compiled JAXAtari-native ghost step
# =====================================================================

def build_jit_ghost_step(consts):
    """Build a JIT-compiled single-step ghost function using JAXAtari internals."""
    dofmaze = consts.DOF_MAZES[0]

    @jax.jit
    def jit_ghost_step(positions, actions, modes, types, pacman_pos, pacman_action, key):
        """
        Single forward step for all 4 ghosts.
        Returns new_positions (4,2), new_actions (4,).
        """
        blinky_pos = positions[0]

        def step_one(ghost_type, mode, action, pos, subkey):
            # Skip jailed/returning ghosts
            skip = (mode == GhostMode.ENJAILED) | (mode == GhostMode.RETURNING)

            def move(_):
                allowed = get_allowed_directions(pos, action, dofmaze,
                                                 consts.DIRECTIONS, is_ghost=True)
                n_allowed = jnp.sum(allowed != 0)

                def random_dir(_):
                    return allowed[jax.random.randint(subkey, (), 0, n_allowed)]

                def pathfind_dir(_):
                    chase_target = jax.lax.cond(
                        mode == GhostMode.CHASE,
                        lambda: get_chase_target(ghost_type, pos, blinky_pos,
                                                  pacman_pos, pacman_action,
                                                  consts.ACTIONS, consts.SCATTER_TARGETS),
                        lambda: consts.SCATTER_TARGETS[ghost_type]
                    )
                    return pathfind(pos, action, chase_target, allowed,
                                   subkey, consts.ACTIONS, consts.DIRECTIONS)

                is_random_mode = ((mode == GhostMode.FRIGHTENED) |
                                  (mode == GhostMode.BLINKING) |
                                  (mode == GhostMode.RANDOM))

                new_act = jax.lax.cond(
                    n_allowed == 0,
                    lambda _: action,
                    lambda _: jax.lax.cond(
                        n_allowed == 1,
                        lambda _: allowed[0],
                        lambda _: jax.lax.cond(
                            is_random_mode,
                            random_dir,
                            pathfind_dir,
                            None
                        ),
                        None
                    ),
                    None
                )
                new_pos = get_new_position(pos, new_act, consts)
                return new_pos, new_act

            def stay(_):
                return pos, action

            return jax.lax.cond(skip, stay, move, None)

        keys = jax.random.split(key, 4)
        new_positions = []
        new_actions = []
        for i in range(4):
            np_, na_ = step_one(types[i], modes[i], actions[i], positions[i], keys[i])
            new_positions.append(np_)
            new_actions.append(na_)

        return (jnp.stack(new_positions),
                jnp.stack(new_actions))

    return jit_ghost_step


def forward_roll_jax(jit_step, ghost_state, pacman_pos, pacman_action, key, K):
    """
    Roll ghost state forward K steps using the JAX-native JIT step.
    Returns positions list: length K+1, each element shape (4,2) numpy.
    """
    positions = [np.array(ghost_state.positions)]
    cur_pos = jnp.array(ghost_state.positions, dtype=jnp.int32)
    cur_act = jnp.array(ghost_state.actions, dtype=jnp.uint8)
    modes   = jnp.array(ghost_state.modes, dtype=jnp.uint8)
    types   = jnp.array(ghost_state.types, dtype=jnp.uint8)

    for t in range(K):
        key, sk = jax.random.split(key)
        new_pos, new_act = jit_step(cur_pos, cur_act, modes, types,
                                     pacman_pos, pacman_action, sk)
        # Slow down rule: frightened/blinking/returning move every 2 frames
        slow_mask = ((modes == GhostMode.FRIGHTENED) |
                     (modes == GhostMode.BLINKING) |
                     (modes == GhostMode.RETURNING))
        if t % 2 == 0:
            new_pos = jnp.where(slow_mask[:, None], cur_pos, new_pos)
            new_act = jnp.where(slow_mask, cur_act, new_act)

        cur_pos, cur_act = new_pos, new_act
        positions.append(np.array(cur_pos))

    return positions  # list[K+1] of (4,2) arrays


# =====================================================================
# Danger map from predicted positions
# =====================================================================

def danger_map_from_positions(positions_K, modes, radius, mult):
    """positions_K: list[K+1] where [0] = current. Skip t=0."""
    dmaps = []
    for t, pos4 in enumerate(positions_K[1:]):
        dmap = {}
        for i in range(4):
            if int(modes[i]) in (5, 6):
                continue
            gx, gy = pixel_to_grid(int(pos4[i][0]), int(pos4[i][1]))
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    d = abs(dx) + abs(dy)
                    if d <= radius:
                        cost = (radius + 1 - d) * mult
                        key = (gx + dx, gy + dy)
                        dmap[key] = dmap.get(key, 0) + cost
        dmaps.append(dmap)
    return dmaps


# =====================================================================
# A* planner
# =====================================================================

def spatiotemporal_astar(maze, start, targets, danger_maps):
    K = len(danger_maps)
    d0 = danger_maps[0]
    safe_targets = {t for t in targets if d0.get(t, 0) < 30}
    if not safe_targets:
        safe_targets = targets
    if not safe_targets:
        return []

    def h(pos):
        return min(abs(pos[0]-tx) + abs(pos[1]-ty) for tx, ty in safe_targets)

    queue = [(h(start), 0, 0, start, [start])]
    visited = {(0, start): 0}
    for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
        pass  # just to ensure no shadowing

    while queue:
        _, cost, t, cur, path = heapq.heappop(queue)
        if cur in safe_targets:
            return path
        nt = min(t + 1, K - 1)
        dn = danger_maps[nt]
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx, ny = cur[0]+dx, cur[1]+dy
            if 0 <= ny < maze.shape[0] and 0 <= nx < maze.shape[1] and maze[ny,nx] == 0:
                pos = (nx, ny)
                nc = cost + 1 + dn.get(pos, 0)
                sk = (nt, pos)
                if sk not in visited or nc < visited[sk]:
                    visited[sk] = nc
                    heapq.heappush(queue, (nc + h(pos), nc, nt, pos, path+[pos]))
    return []


def get_action(px, py, path):
    if len(path) < 2:
        return 0
    tx, ty = path[1][0]*4, path[1][1]*4
    dx, dy = tx-px, ty-py
    if abs(dx) > abs(dy): return 2 if dx > 0 else 3
    elif abs(dy) > 0: return 4 if dy > 0 else 1
    return 0


# =====================================================================
# Section 1: Ghost prediction accuracy
# =====================================================================

def run_prediction_accuracy(env, step_fn, jit_step, n_samples=20):
    print("=" * 60)
    print("SECTION 1: Ghost Prediction Accuracy (MBv0 vs MBv2-JAX)")
    print("=" * 60)
    consts = env.consts
    key = jax.random.PRNGKey(77)
    obs, state = env.reset(key)
    maze = MsPacmanMaze.MAZES[0]

    # Warm-up 50 frames
    for _ in range(50):
        obs, state, _, _, _ = step_fn(state, jnp.array(0, dtype=jnp.int32))

    horizons = [1, 3, 5, 10, 15]
    v0_err = {h: [] for h in horizons}
    v2_err = {h: [] for h in horizons}

    last_pix = [[obs.ghost_positions[i][0].item(), obs.ghost_positions[i][1].item()]
                for i in range(4)]

    for sample in range(n_samples):
        obs, state, _, _, _ = step_fn(state, jnp.array(0, dtype=jnp.int32))
        cur_pix = [[obs.ghost_positions[i][0].item(), obs.ghost_positions[i][1].item()]
                   for i in range(4)]

        # Velocity for v0
        velocities = []
        for i in range(4):
            dx = cur_pix[i][0] - last_pix[i][0]
            dy = cur_pix[i][1] - last_pix[i][1]
            vx = 1 if dx>0 else (-1 if dx<0 else 0)
            vy = 1 if dy>0 else (-1 if dy<0 else 0)
            if abs(dx)>abs(dy): vy = 0
            else: vx = 0
            velocities.append((vx, vy))
        last_pix = [[p[0],p[1]] for p in cur_pix]

        # v0 linear prediction
        v0_cur = [list(pixel_to_grid(cur_pix[i][0], cur_pix[i][1])) for i in range(4)]
        v0_preds = [list(v0_cur)]
        for _ in range(max(horizons)):
            for i in range(4):
                vx, vy = velocities[i]
                nx, ny = v0_cur[i][0]+vx, v0_cur[i][1]+vy
                if 0<=ny<maze.shape[0] and 0<=nx<maze.shape[1] and maze[ny,nx]==0:
                    v0_cur[i] = [nx, ny]
            v0_preds.append([list(p) for p in v0_cur])

        # v2 JAX-native prediction
        rkey = jax.random.PRNGKey(sample * 7 + 13)
        v2_preds = forward_roll_jax(jit_step, state.ghosts,
                                    obs.player_position, obs.player_action,
                                    rkey, K=max(horizons))

        # Roll env forward to get ground truth
        saved_state, saved_obs = state, obs
        actuals = {}
        for f in range(1, max(horizons)+1):
            obs, state, _, _, _ = step_fn(state, jnp.array(0, dtype=jnp.int32))
            actuals[f] = [list(pixel_to_grid(obs.ghost_positions[i][0].item(),
                                              obs.ghost_positions[i][1].item()))
                          for i in range(4)]

        for h in horizons:
            act = actuals[h]
            p0 = v0_preds[h]
            p2 = [list(pixel_to_grid(int(v2_preds[h][i][0]),
                                      int(v2_preds[h][i][1]))) for i in range(4)]
            for i in range(4):
                v0_err[h].append(abs(act[i][0]-p0[i][0]) + abs(act[i][1]-p0[i][1]))
                v2_err[h].append(abs(act[i][0]-p2[i][0]) + abs(act[i][1]-p2[i][1]))

        # Restore and advance 10 frames for diversity
        state, obs = saved_state, saved_obs
        for _ in range(10):
            obs, state, _, _, _ = step_fn(state, jnp.array(0, dtype=jnp.int32))

    print(f"{'K':<6} | {'MBv0 (linear)':<18} | {'MBv2 (JAX-native)':<20}")
    print("-" * 50)
    for h in horizons:
        e0 = np.mean(v0_err[h])
        e2 = np.mean(v2_err[h])
        win = " <-- BETTER" if e2 < e0 else (" <-- WORSE" if e2 > e0 else " (tie)")
        print(f"K={h:<4} | {e0:<18.3f} | {e2:<20.3f}{win}")


# =====================================================================
# Section 2: Danger-radius ablation
# =====================================================================

def run_danger_radius_ablation(env, step_fn, jit_step):
    print("\n" + "=" * 60)
    print("SECTION 2: Danger Radius Ablation")
    print("=" * 60)
    maze = MsPacmanMaze.MAZES[0]
    configs = [
        ("old baseline (r=6, m=20)", 6, 20),
        ("tight       (r=3, m=5)",   3,  5),
        ("minimal     (r=2, m=2)",   2,  2),
    ]
    for label, radius, mult in configs:
        key = jax.random.PRNGKey(42)
        obs, state = env.reset(key)
        rkey = jax.random.PRNGKey(0)
        score, pellets, osc, survival = 0, 0, 0, 0
        last_10 = []

        for _ in range(300):
            survival += 1
            px, py = obs.player_position[0].item(), obs.player_position[1].item()
            start = pixel_to_grid(px, py)
            targets = extract_targets(maze, obs.pellets)
            if not targets: break

            rkey, sk = jax.random.split(rkey)
            preds = forward_roll_jax(jit_step, state.ghosts,
                                     obs.player_position, obs.player_action, sk, K=15)
            dmaps = danger_map_from_positions(preds, state.ghosts.modes, radius, mult)
            path = spatiotemporal_astar(maze, start, targets, dmaps)
            action = get_action(px, py, path)

            last_10.append(start)
            if len(last_10) > 10: last_10.pop(0)
            if last_10.count(start) > 3: osc += 1

            obs, state, _, done, _ = step_fn(state, jnp.array(action, dtype=jnp.int32))
            score = state.score.item()
            pellets = state.level.collected_pellets.item()
            if done: break

        print(f"  {label:35s} | score={score:5d}  pellets={pellets:3d}  osc={osc:3d}  survival={survival}")


# =====================================================================
# Section 3: Planner decision log
# =====================================================================

def run_decision_log(env, step_fn, jit_step, radius=3, mult=5, steps=200):
    print("\n" + "=" * 60)
    print(f"SECTION 3: Planner Decision Log (r={radius}, m={mult})")
    print("=" * 60)
    maze = MsPacmanMaze.MAZES[0]
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    rkey = jax.random.PRNGKey(1)

    prev_target = None
    target_switches = 0
    danger_dominated = 0
    path_lengths = []

    for step in range(steps):
        px, py = obs.player_position[0].item(), obs.player_position[1].item()
        start = pixel_to_grid(px, py)
        targets = extract_targets(maze, obs.pellets)
        if not targets: break

        rkey, sk = jax.random.split(rkey)
        preds = forward_roll_jax(jit_step, state.ghosts,
                                 obs.player_position, obs.player_action, sk, K=15)
        dmaps = danger_map_from_positions(preds, state.ghosts.modes, radius, mult)

        if dmaps[0].get(start, 0) > 100:
            danger_dominated += 1

        path = spatiotemporal_astar(maze, start, targets, dmaps)
        cur_target = path[-1] if path else None
        if cur_target != prev_target:
            target_switches += 1
        prev_target = cur_target
        path_lengths.append(len(path))

        action = get_action(px, py, path)
        obs, state, _, done, _ = step_fn(state, jnp.array(action, dtype=jnp.int32))
        if done: break

    print(f"  Target switches          : {target_switches}/{steps} ({100*target_switches/steps:.1f}%)")
    print(f"  Danger-dominated starts  : {danger_dominated}/{steps} ({100*danger_dominated/steps:.1f}%)")
    print(f"  Mean path length         : {np.mean(path_lengths):.1f}")


# =====================================================================
# Section 4: MF vs MBv2 plan divergence
# =====================================================================

def run_plan_divergence(env, step_fn, jit_step, radius=3, mult=5, steps=200):
    print("\n" + "=" * 60)
    print(f"SECTION 4: MF vs MBv2 Plan Divergence (r={radius}, m={mult})")
    print("=" * 60)
    maze = MsPacmanMaze.MAZES[0]
    key = jax.random.PRNGKey(99)
    obs, state = env.reset(key)
    rkey = jax.random.PRNGKey(2)
    differ = 0

    for step in range(steps):
        px, py = obs.player_position[0].item(), obs.player_position[1].item()
        start = pixel_to_grid(px, py)
        targets = extract_targets(maze, obs.pellets)
        if not targets: break

        rkey, sk = jax.random.split(rkey)
        preds = forward_roll_jax(jit_step, state.ghosts,
                                 obs.player_position, obs.player_action, sk, K=15)
        mb_dmaps = danger_map_from_positions(preds, state.ghosts.modes, radius, mult)
        mb_path = spatiotemporal_astar(maze, start, targets, mb_dmaps)

        # MF: freeze t=0 danger map across all steps
        static = [mb_dmaps[0]] * 15
        mf_path = spatiotemporal_astar(maze, start, targets, static)

        if mf_path != mb_path:
            differ += 1
            if differ <= 3:
                print(f"  [step {step}] MF target={mf_path[-1] if mf_path else None}, "
                      f"MB target={mb_path[-1] if mb_path else None}, "
                      f"MF action={get_action(px,py,mf_path)}, "
                      f"MB action={get_action(px,py,mb_path)}")

        action = get_action(px, py, mb_path)
        obs, state, _, done, _ = step_fn(state, jnp.array(action, dtype=jnp.int32))
        if done: break

    print(f"\n  Plans differed: {differ}/{steps} ({100*differ/steps:.1f}%)")


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    env = JaxPacman()
    consts = env.consts
    step_fn = jax.jit(env.step)

    print("Building JIT-compiled ghost step...")
    jit_step = build_jit_ghost_step(consts)

    # Warm-up compile both
    print("Compiling env and ghost step...")
    obs, state = env.reset(jax.random.PRNGKey(0))
    step_fn(state, jnp.array(0, dtype=jnp.int32))
    dummy_key = jax.random.PRNGKey(0)
    _ = forward_roll_jax(jit_step, state.ghosts, obs.player_position,
                          obs.player_action, dummy_key, K=15)
    print("Done. Starting experiments.\n")

    run_prediction_accuracy(env, step_fn, jit_step, n_samples=20)
    run_danger_radius_ablation(env, step_fn, jit_step)
    run_decision_log(env, step_fn, jit_step, radius=3, mult=5)
    run_plan_divergence(env, step_fn, jit_step, radius=3, mult=5)
