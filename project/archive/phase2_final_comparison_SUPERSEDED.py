"""
phase2_final_comparison.py

3-way comparison of MF vs MBv0 vs MBv2, ALL with:
  - Correct DOF coordinate system (gx=pixel_x//4, gy=pixel_y//4)
  - Correct pellet mapping (pellx*2+1, pelly*3+2)
  - Correct action mapping (2=UP,3=RIGHT,4=LEFT,5=DOWN)
  - Fixed oscillation counter (pixel-level, not DOF-level)
"""
import jax
import jax.numpy as jnp
import heapq
import numpy as np
from jaxatari.games.jax_mspacman import (
    JaxPacman, GhostMode,
    get_allowed_directions, get_chase_target, pathfind, get_new_position,
)
from jaxatari.games.mspacman_mazes import MsPacmanMaze

# =====================================================================
# Correct coordinate utils
# =====================================================================

def pixel_to_dof(px, py):
    return int(px) // 4, int(py) // 4

def pellet_to_dof(pellx, pelly):
    return pellx * 2 + 1, pelly * 3 + 2

def is_walkable(dof, gx, gy):
    return (0 <= gx < dof.shape[0] and 0 <= gy < dof.shape[1]
            and bool(dof[gx, gy].any()))

def extract_targets(dof, pellets):
    targets = set()
    for pelly in range(pellets.shape[1]):
        for pellx in range(pellets.shape[0]):
            if pellets[pellx, pelly]:
                gx, gy = pellet_to_dof(pellx, pelly)
                if is_walkable(dof, gx, gy):
                    targets.add((gx, gy))
    return targets

def get_action(cur_dof, path):
    """Correct JAXAtari action: 2=UP, 3=RIGHT, 4=LEFT, 5=DOWN."""
    if len(path) < 2:
        return 0
    dx = path[1][0] - cur_dof[0]
    dy = path[1][1] - cur_dof[1]
    if abs(dx) > abs(dy):
        return 3 if dx > 0 else 4
    elif abs(dy) > 0:
        return 5 if dy > 0 else 2
    return 0

# =====================================================================
# Ghost simulator (JAX-native, JIT-compiled)
# =====================================================================

def build_jit_ghost_step(consts):
    dofmaze = consts.DOF_MAZES[0]

    @jax.jit
    def jit_ghost_step(positions, actions, modes, types, pacman_pos, pacman_action, key):
        blinky_pos = positions[0]

        def step_one(ghost_type, mode, action, pos, subkey):
            skip = (mode == GhostMode.ENJAILED) | (mode == GhostMode.RETURNING)

            def move(_):
                allowed = get_allowed_directions(pos, action, dofmaze,
                                                  consts.DIRECTIONS, is_ghost=True)
                n_allowed = jnp.sum(allowed != 0)
                is_random_mode = ((mode == GhostMode.FRIGHTENED) |
                                  (mode == GhostMode.BLINKING) |
                                  (mode == GhostMode.RANDOM))
                new_act = jax.lax.cond(
                    n_allowed == 0, lambda _: action,
                    lambda _: jax.lax.cond(
                        n_allowed == 1, lambda _: allowed[0],
                        lambda _: jax.lax.cond(
                            is_random_mode,
                            lambda _: allowed[jax.random.randint(subkey, (), 0, n_allowed)],
                            lambda _: pathfind(
                                pos, action,
                                jax.lax.cond(
                                    mode == GhostMode.CHASE,
                                    lambda: get_chase_target(ghost_type, pos, blinky_pos,
                                                              pacman_pos, pacman_action,
                                                              consts.ACTIONS, consts.SCATTER_TARGETS),
                                    lambda: consts.SCATTER_TARGETS[ghost_type]
                                ),
                                allowed, subkey, consts.ACTIONS, consts.DIRECTIONS),
                            None), None), None)
                return get_new_position(pos, new_act, consts), new_act

            return jax.lax.cond(skip, lambda _: (pos, action), move, None)

        keys = jax.random.split(key, 4)
        results = [step_one(types[i], modes[i], actions[i], positions[i], keys[i])
                   for i in range(4)]
        return jnp.stack([r[0] for r in results]), jnp.stack([r[1] for r in results])

    return jit_ghost_step


def forward_roll_v0(ghost_pos, velocities, K):
    """Linear velocity extrapolation (MBv0)."""
    positions = []
    cur = [list(pixel_to_dof(ghost_pos[i][0].item(), ghost_pos[i][1].item()))
           for i in range(4)]
    for t in range(K):
        for i in range(4):
            vx, vy = velocities[i]
            nx, ny = cur[i][0] + vx, cur[i][1] + vy
            cur[i] = [nx, ny]
        positions.append([list(c) for c in cur])
    return positions  # list[K] of (4,2)


def forward_roll_v2(jit_step, ghost_state, pacman_pos, pacman_action, key, K):
    """JAX-native predictor (MBv2)."""
    positions = []
    cur_pos = jnp.array(ghost_state.positions, dtype=jnp.int32)
    cur_act = jnp.array(ghost_state.actions,   dtype=jnp.uint8)
    modes   = jnp.array(ghost_state.modes,     dtype=jnp.uint8)
    types   = jnp.array(ghost_state.types,     dtype=jnp.uint8)

    for t in range(K):
        key, sk = jax.random.split(key)
        new_pos, new_act = jit_step(cur_pos, cur_act, modes, types,
                                     pacman_pos, pacman_action, sk)
        slow = ((modes == GhostMode.FRIGHTENED) |
                (modes == GhostMode.BLINKING) |
                (modes == GhostMode.RETURNING))
        if t % 2 == 0:
            new_pos = jnp.where(slow[:, None], cur_pos, new_pos)
            new_act = jnp.where(slow, cur_act, new_act)
        cur_pos, cur_act = new_pos, new_act
        positions.append(np.array(cur_pos))
    return positions  # list[K] of (4,2) arrays

# =====================================================================
# Danger map builder
# =====================================================================

def build_danger_maps(pred_positions, modes, radius=3, mult=5):
    """pred_positions: list[K] of (4,2) arrays (pixel coords)."""
    dmaps = []
    for pos4 in pred_positions:
        dmap = {}
        for i in range(4):
            if int(modes[i]) in (5, 6):
                continue
            gx, gy = pixel_to_dof(int(pos4[i][0]), int(pos4[i][1]))
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    d = abs(dx) + abs(dy)
                    if d <= radius:
                        k = (gx+dx, gy+dy)
                        dmap[k] = dmap.get(k, 0) + (radius+1-d)*mult
        dmaps.append(dmap)
    return dmaps

# =====================================================================
# Spatiotemporal A*
# =====================================================================

def astar(dof, start, targets, danger_maps):
    K = len(danger_maps)
    safe = {t for t in targets if danger_maps[0].get(t, 0) < 30}
    if not safe: safe = targets
    if not safe: return []

    def h(p): return min(abs(p[0]-t[0])+abs(p[1]-t[1]) for t in safe)

    queue = [(h(start), 0, 0, start, [start])]
    vis = {(0, start): 0}
    dirs = [(0,1),(0,-1),(1,0),(-1,0)]

    while queue:
        _, cost, t, cur, path = heapq.heappop(queue)
        if cur in safe: return path
        nt = min(t+1, K-1)
        dn = danger_maps[nt]
        for dx, dy in dirs:
            nx, ny = cur[0]+dx, cur[1]+dy
            if is_walkable(dof, nx, ny):
                pos = (nx, ny)
                nc = cost + 1 + dn.get(pos, 0)
                sk = (nt, pos)
                if sk not in vis or nc < vis[sk]:
                    vis[sk] = nc
                    heapq.heappush(queue, (nc+h(pos), nc, nt, pos, path+[pos]))
    return []

# =====================================================================
# Oscillation counter (pixel-level, not DOF-level)
# =====================================================================

class OscillationCounter:
    """
    Detects true oscillation: Pacman reversing direction within a short window.
    Counts events where pixel position repeats within last 8 frames.
    """
    def __init__(self):
        self.history = []
    def update(self, px, py):
        pos = (px, py)
        self.history.append(pos)
        if len(self.history) > 8: self.history.pop(0)
        # Oscillation: current position appeared in earlier half of window
        if len(self.history) >= 6:
            recent_half = self.history[-3:]
            older_half  = self.history[:-3]
            return sum(p in older_half for p in recent_half) >= 2
        return False

# =====================================================================
# Main run loop
# =====================================================================

def run_planner(label, env, step_fn, jit_step, consts, mode='MF', K=15, radius=3, mult=5, n_steps=300, seed=42):
    dof = consts.DOF_MAZES[0]
    key = jax.random.PRNGKey(seed)
    obs, state = env.reset(key)
    rkey = jax.random.PRNGKey(seed + 1)

    osc_counter = OscillationCounter()
    score = 0
    pellets_collected = 0
    survival = 0
    osc = 0
    diverge = 0

    # For MBv0 velocity tracking
    last_pix = [[obs.ghost_positions[i][0].item(), obs.ghost_positions[i][1].item()] for i in range(4)]

    for step in range(n_steps):
        survival += 1
        px, py = obs.player_position[0].item(), obs.player_position[1].item()
        start = pixel_to_dof(px, py)
        targets = extract_targets(dof, obs.pellets)
        if not targets: break

        rkey, sk = jax.random.split(rkey)

        # Compute velocity for v0
        cur_pix = [[obs.ghost_positions[i][0].item(), obs.ghost_positions[i][1].item()] for i in range(4)]
        velocities = []
        for i in range(4):
            dx = cur_pix[i][0] - last_pix[i][0]
            dy = cur_pix[i][1] - last_pix[i][1]
            vx = 1 if dx>0 else (-1 if dx<0 else 0)
            vy = 1 if dy>0 else (-1 if dy<0 else 0)
            if abs(dx) > abs(dy): vy = 0
            else: vx = 0
            velocities.append((vx, vy))
        last_pix = [[p[0],p[1]] for p in cur_pix]

        # Build predictions
        if mode == 'MF':
            # Static: replicate current ghost positions K times
            cur_pos_arr = np.array([[obs.ghost_positions[i][0].item(),
                                      obs.ghost_positions[i][1].item()] for i in range(4)])
            preds = [cur_pos_arr] * K
        elif mode == 'MBv0':
            preds = forward_roll_v0(obs.ghost_positions, velocities, K)
            # Convert DOF coords back to pixel for danger map builder
            preds_px = []
            for t_preds in preds:
                preds_px.append(np.array([[t_preds[i][0]*4, t_preds[i][1]*4] for i in range(4)]))
            preds = preds_px
        else:  # MBv2
            preds = forward_roll_v2(jit_step, state.ghosts,
                                     obs.player_position, obs.player_action, sk, K=K)

        dmaps = build_danger_maps(preds, state.ghosts.modes, radius, mult)
        path = astar(dof, start, targets, dmaps)
        action = get_action(start, path)

        # MF divergence comparison (only for non-MF modes)
        if mode != 'MF':
            cur_pos_arr = np.array([[obs.ghost_positions[i][0].item(),
                                      obs.ghost_positions[i][1].item()] for i in range(4)])
            static_preds = [cur_pos_arr] * K
            mf_dmaps = build_danger_maps(static_preds, state.ghosts.modes, radius, mult)
            mf_path = astar(dof, start, targets, mf_dmaps)
            if mf_path != path:
                diverge += 1

        if osc_counter.update(px, py):
            osc += 1

        obs, state, _, done, _ = step_fn(state, jnp.array(action, dtype=jnp.int32))
        score = state.score.item()
        pellets_collected = state.level.collected_pellets.item()
        if done: break

    div_str = f"  divergence={diverge}/{n_steps} ({100*diverge/n_steps:.1f}%)" if mode != 'MF' else ""
    print(f"  {label:35s} | score={score:5d}  pellets={pellets_collected:3d}  "
          f"osc={osc:3d}  survival={survival:3d}{div_str}")
    return score, pellets_collected, osc, survival


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    env = JaxPacman()
    consts = env.consts
    step_fn = jax.jit(env.step)
    jit_step = build_jit_ghost_step(consts)

    print("Compiling...")
    obs, state = env.reset(jax.random.PRNGKey(0))
    step_fn(state, jnp.array(0, dtype=jnp.int32))
    forward_roll_v2(jit_step, state.ghosts, obs.player_position,
                     obs.player_action, jax.random.PRNGKey(0), K=15)
    print("Done.\n")

    print("=" * 80)
    print("FINAL COMPARISON: MF vs MBv0 vs MBv2 (all bugs fixed)")
    print("K=15, radius=3, mult=5, 300 steps, seed=42")
    print("=" * 80)

    run_planner("MF  (static danger map)",  env, step_fn, jit_step, consts, mode='MF',   n_steps=300)
    run_planner("MBv0 (linear extrapolation)", env, step_fn, jit_step, consts, mode='MBv0', n_steps=300)
    run_planner("MBv2 (JAX-native simulator)", env, step_fn, jit_step, consts, mode='MBv2', n_steps=300)

    print()
    print("=" * 80)
    print("Varying danger radius (MBv2 only)")
    print("=" * 80)
    for radius, mult in [(6, 20), (3, 5), (2, 2), (1, 1)]:
        run_planner(f"MBv2 r={radius} m={mult}", env, step_fn, jit_step, consts,
                    mode='MBv2', K=15, radius=radius, mult=mult, n_steps=300)
