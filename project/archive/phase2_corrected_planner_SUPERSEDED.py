"""
phase2_corrected_planner.py

Fixes ALL coordinate bugs identified by deep diagnostics.

Root cause confirmed:
  - Planner operated in the wrong coordinate space
  - pixel_to_grid(px/4) ≠ DOF grid
  - Correct: DOF grid = (pixel_x//4, pixel_y//4)  [same formula but validated]
  - Pellet DOF grid: (pellx*2+1, pelly*3+2)  [NOT pixel arithmetic]
  - POWER_PELLET_TILES are already in DOF grid coords
  - A* must navigate in DOF grid space, NOT in MsPacmanMaze.MAZES space

The DOF maze (40×44) is the correct navigation grid, not MAZES[0] (44×40).
DOF indexing: dof[gx, gy, direction] -- so dof[gx, gy].any() = walkable
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
# CORRECTED coordinate system
# =====================================================================

def pixel_to_dof(px, py):
    """Convert pixel coordinates to DOF grid coordinates."""
    return int(px) // 4, int(py) // 4

def pellet_to_dof(pellx, pelly):
    """Convert pellet grid index to DOF grid coordinates."""
    return pellx * 2 + 1, pelly * 3 + 2

def is_walkable(dof, gx, gy):
    if 0 <= gx < dof.shape[0] and 0 <= gy < dof.shape[1]:
        return bool(dof[gx, gy].any())
    return False

def extract_targets_correct(dof, pellets):
    """Return set of (gx, gy) DOF grid positions for all live pellets."""
    targets = set()
    for pelly in range(pellets.shape[1]):
        for pellx in range(pellets.shape[0]):
            if pellets[pellx, pelly]:
                gx, gy = pellet_to_dof(pellx, pelly)
                if is_walkable(dof, gx, gy):
                    targets.add((gx, gy))
    return targets

# =====================================================================
# JIT-compiled JAXAtari-native ghost simulator
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
                                allowed, subkey, consts.ACTIONS, consts.DIRECTIONS
                            ),
                            None
                        ),
                        None
                    ),
                    None
                )
                new_pos = get_new_position(pos, new_act, consts)
                return new_pos, new_act

            return jax.lax.cond(skip, lambda _: (pos, action), move, None)

        keys = jax.random.split(key, 4)
        results = [step_one(types[i], modes[i], actions[i], positions[i], keys[i])
                   for i in range(4)]
        return jnp.stack([r[0] for r in results]), jnp.stack([r[1] for r in results])

    return jit_ghost_step


def forward_roll(jit_step, ghost_state, pacman_pos, pacman_action, key, K):
    positions = [np.array(ghost_state.positions)]
    cur_pos = jnp.array(ghost_state.positions, dtype=jnp.int32)
    cur_act = jnp.array(ghost_state.actions,   dtype=jnp.uint8)
    modes   = jnp.array(ghost_state.modes,     dtype=jnp.uint8)
    types   = jnp.array(ghost_state.types,     dtype=jnp.uint8)

    for t in range(K):
        key, sk = jax.random.split(key)
        new_pos, new_act = jit_step(cur_pos, cur_act, modes, types,
                                     pacman_pos, pacman_action, sk)
        slow = (modes == GhostMode.FRIGHTENED) | (modes == GhostMode.BLINKING) | (modes == GhostMode.RETURNING)
        if t % 2 == 0:
            new_pos = jnp.where(slow[:, None], cur_pos, new_pos)
            new_act = jnp.where(slow, cur_act, new_act)
        cur_pos, cur_act = new_pos, new_act
        positions.append(np.array(cur_pos))
    return positions

# =====================================================================
# Danger map (in DOF grid coordinates)
# =====================================================================

def danger_map_from_preds(positions_K, modes, radius=3, mult=5):
    dmaps = []
    for pos4 in positions_K[1:]:
        dmap = {}
        for i in range(4):
            if int(modes[i]) in (5, 6):
                continue
            gx, gy = pixel_to_dof(int(pos4[i][0]), int(pos4[i][1]))
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    d = abs(dx) + abs(dy)
                    if d <= radius:
                        dmap[(gx+dx, gy+dy)] = dmap.get((gx+dx, gy+dy), 0) + (radius+1-d)*mult
        dmaps.append(dmap)
    return dmaps

# =====================================================================
# A* in DOF grid coordinates
# =====================================================================

def astar_dof(dof, start, targets, danger_maps):
    """Navigate DOF grid. start=(gx,gy), targets=set of (gx,gy)."""
    K = len(danger_maps)
    d0 = danger_maps[0]
    safe_targets = {t for t in targets if d0.get(t, 0) < 30}
    if not safe_targets:
        safe_targets = targets
    if not safe_targets:
        return []

    def h(p):
        return min(abs(p[0]-t[0]) + abs(p[1]-t[1]) for t in safe_targets)

    # DOF directions: up=(-y), down=(+y), left=(-x), right=(+x) in pixel space
    # But DOF is indexed [gx, gy], so neighbors are (gx±1, gy) or (gx, gy±1)
    # Check walkability using dof[gx, gy].any()
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    queue = [(h(start), 0, 0, start, [start])]
    visited = {(0, start): 0}

    while queue:
        _, cost, t, cur, path = heapq.heappop(queue)
        if cur in safe_targets:
            return path
        nt = min(t+1, K-1)
        dn = danger_maps[nt]
        for dx, dy in neighbors:
            nx, ny = cur[0]+dx, cur[1]+dy
            if is_walkable(dof, nx, ny):
                pos = (nx, ny)
                nc = cost + 1 + dn.get(pos, 0)
                sk = (nt, pos)
                if sk not in visited or nc < visited[sk]:
                    visited[sk] = nc
                    heapq.heappush(queue, (nc + h(pos), nc, nt, pos, path+[pos]))
    return []


def get_action(cur_dof, path):
    """Convert next DOF grid step to JAXAtari action.
    
    JAXAtari action indices (confirmed empirically):
      2 = UP    (pixel_y decreases, dy = -1)
      3 = RIGHT (pixel_x increases, dx = +1)
      4 = LEFT  (pixel_x decreases, dx = -1)
      5 = DOWN  (pixel_y increases, dy = +1)
      0 = NOOP
    DOF grid: gx=pixel_x//4 (increases right), gy=pixel_y//4 (increases down)
    """
    if len(path) < 2:
        return 0
    gx0, gy0 = cur_dof
    gx1, gy1 = path[1]
    dx, dy = gx1 - gx0, gy1 - gy0
    if abs(dx) > abs(dy):
        return 3 if dx > 0 else 4   # RIGHT=3, LEFT=4
    elif abs(dy) > 0:
        return 5 if dy > 0 else 2   # DOWN=5, UP=2
    return 0

# =====================================================================
# Comparison: MF vs MBv2 with CORRECTED coordinates
# =====================================================================

def run_comparison(env, step_fn, jit_step, K=15, radius=3, mult=5, n_steps=300, label=""):
    consts = env.consts
    dof = consts.DOF_MAZES[0]
    key = jax.random.PRNGKey(42)
    obs, state = env.reset(key)
    rkey = jax.random.PRNGKey(0)

    score, pellets_collected, osc, survival = 0, 0, 0, 0
    last_10 = []
    diverge_count = 0

    for step in range(n_steps):
        survival += 1
        px, py = obs.player_position[0].item(), obs.player_position[1].item()
        start = pixel_to_dof(px, py)
        targets = extract_targets_correct(dof, obs.pellets)
        if not targets:
            break

        rkey, sk = jax.random.split(rkey)
        preds = forward_roll(jit_step, state.ghosts, obs.player_position,
                             obs.player_action, sk, K=K)
        dmaps = danger_map_from_preds(preds, state.ghosts.modes, radius, mult)

        path = astar_dof(dof, start, targets, dmaps)
        action = get_action(start, path)

        # MF comparison (freeze t=0 danger map)
        static = [dmaps[0]] * K
        mf_path = astar_dof(dof, start, targets, static)
        if mf_path != path:
            diverge_count += 1

        last_10.append(start)
        if len(last_10) > 10: last_10.pop(0)
        if last_10.count(start) > 3: osc += 1

        obs, state, _, done, _ = step_fn(state, jnp.array(action, dtype=jnp.int32))
        score = state.score.item()
        pellets_collected = state.level.collected_pellets.item()
        if done:
            break

    print(f"  {label:30s} | score={score:5d}  pellets={pellets_collected:3d}  "
          f"osc={osc:3d}  survival={survival:3d}  "
          f"divergence={diverge_count}/{n_steps} ({100*diverge_count/n_steps:.1f}%)")
    return score, pellets_collected, osc, survival


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    env = JaxPacman()
    consts = env.consts
    dof = consts.DOF_MAZES[0]
    step_fn = jax.jit(env.step)

    print("Compiling...")
    jit_step = build_jit_ghost_step(consts)
    obs, state = env.reset(jax.random.PRNGKey(0))
    step_fn(state, jnp.array(0, dtype=jnp.int32))
    forward_roll(jit_step, state.ghosts, obs.player_position,
                  obs.player_action, jax.random.PRNGKey(0), K=15)
    print("Done.\n")

    # Quick sanity check
    px, py = obs.player_position[0].item(), obs.player_position[1].item()
    start = pixel_to_dof(px, py)
    targets = extract_targets_correct(dof, obs.pellets)
    print(f"Sanity: Pacman DOF={start}, walkable={is_walkable(dof, *start)}, "
          f"n_targets={len(targets)}")
    if targets:
        dists = sorted(abs(t[0]-start[0])+abs(t[1]-start[1]) for t in targets)
        print(f"Nearest 5 target distances: {dists[:5]}")

    print("\n" + "=" * 80)
    print("CORRECTED PLANNER: MF vs MBv2 Comparison")
    print("=" * 80)
    run_comparison(env, step_fn, jit_step, K=15, radius=3, mult=5,
                   n_steps=300, label="MBv2 (JAX-native, correct coords)")
