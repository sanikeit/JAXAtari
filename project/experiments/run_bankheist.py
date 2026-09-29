"""
experiments/run_bankheist.py  (v4 - A* waypoint-following controller)

TIER 3 - Bank Heist cross-game transfer: A* path + waypoint-following controller.

Greedy pixel-chasing got trapped at walls (it only knows "closer/farther", not the
route). This version FOLLOWS THE A* PATH cell-by-cell: A* traces the exact corridor
route around walls; the low-level controller steers the car toward the NEXT waypoint
in that path, advancing as each is reached. A stuck-escape handles pixel-box vs grid
mismatch at cell boundaries.

Action mapping (confirmed): UP=2, RIGHT=3, LEFT=4, DOWN=5, NOOP=0.

Run:
    uv run python project/experiments/run_bankheist.py
"""
import sys, os, heapq
import jax, jax.numpy as jnp, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import jaxatari
from importlib import import_module
adapter = import_module("project.experiments.bankheist_adapter")
get_bankheist_state = adapter.get_bankheist_state
TILE = adapter.TILE

A_NOOP, A_UP, A_RIGHT, A_LEFT, A_DOWN = 0, 2, 3, 4, 5


def astar_grid(grid, start, goal):
    GH, GW = grid.shape
    def h(c): return abs(c[0]-goal[0]) + abs(c[1]-goal[1])
    openq = [(h(start), 0, start)]
    came, g = {start: None}, {start: 0}
    while openq:
        _, gc, cur = heapq.heappop(openq)
        if cur == goal:
            path = [cur]
            while came[cur] is not None:
                cur = came[cur]; path.append(cur)
            return path[::-1]
        cx, cy = cur
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = cx+dx, cy+dy
            if 0 <= ny < GH and 0 <= nx < GW and grid[ny, nx]:
                ng = gc + 1
                if (nx,ny) not in g or ng < g[(nx,ny)]:
                    g[(nx,ny)] = ng; came[(nx,ny)] = cur
                    heapq.heappush(openq, (ng + h((nx,ny)), ng, (nx,ny)))
    return None


def cell_center(cell, tile=TILE):
    return cell[0]*tile + tile//2, cell[1]*tile + tile//2


def dir_toward(px, py, cx, cy):
    """Single action moving the car pixel toward (cx,cy); dominant axis first."""
    dx, dy = cx - px, cy - py
    if abs(dx) >= abs(dy):
        if dx >= 1:  return A_RIGHT
        if dx <= -1: return A_LEFT
        if dy >= 1:  return A_DOWN
        if dy <= -1: return A_UP
    else:
        if dy >= 1:  return A_DOWN
        if dy <= -1: return A_UP
        if dx >= 1:  return A_RIGHT
        if dx <= -1: return A_LEFT
    return A_NOOP


def nearest_bank_cell(pcell, banks):
    return min(banks, key=lambda b: abs(b[0]-pcell[0]) + abs(b[1]-pcell[1]))


def main():
    env = jaxatari.make("bankheist")
    obs, state = env.reset(jax.random.PRNGKey(0))
    step_fn = jax.jit(env.step)
    obs, state, _, _, _ = step_fn(state, jnp.array(0, dtype=jnp.int32))

    N_STEPS = 1000
    last_score = 0
    path, wp = None, 1
    stuck, escape_dir = 0, 0
    ESCAPE_SEQ = [A_DOWN, A_UP, A_RIGHT, A_LEFT]

    for step in range(N_STEPS):
        grid, pcell, banks, enemies = get_bankheist_state(env, obs, state)
        px, py = int(obs.player.x), int(obs.player.y)
        if not banks:
            print(f"step {step}: no active banks; stopping."); break

        # (re)plan when needed
        if path is None or wp >= len(path):
            goal = nearest_bank_cell(pcell, banks)
            path = astar_grid(grid, pcell, goal)
            wp = 1
            stuck = 0
            if step == 0 and path:
                print(f"  initial path head: {path[:6]} (car at {pcell})")

        if not path or len(path) < 2:
            action = A_NOOP
        else:
            # advance waypoint(s) we've already reached
            while wp < len(path) - 1 and pcell == path[wp]:
                wp += 1
            target = path[min(wp, len(path) - 1)]
            cx, cy = cell_center(target)
            action = dir_toward(px, py, cx, cy)

        prev = (px, py)
        obs, state, _, done, _ = step_fn(state, jnp.array(int(action), dtype=jnp.int32))
        px2, py2 = int(obs.player.x), int(obs.player.y)

        # stuck escape: pixel-box vs grid mismatch can wedge us at a boundary
        if (px2, py2) == prev:
            stuck += 1
            if stuck >= 2:
                action = ESCAPE_SEQ[escape_dir % len(ESCAPE_SEQ)]
                escape_dir += 1
                obs, state, _, done, _ = step_fn(state, jnp.array(int(action), dtype=jnp.int32))
                stuck = 0
        else:
            stuck = 0; escape_dir = 0

        sc = int(np.array(state.score)) if hasattr(state, "score") else 0
        if sc != last_score:
            print(f"step {step}: *** SCORE {last_score} -> {sc} *** robbed a bank!")
            last_score = sc
            path = None  # replan
        if step % 50 == 0:
            tw = path[wp] if path and wp < len(path) else '-'
            print(f"step {step}: pos=({px},{py}) cell={pcell} wp={wp} target={tw} "
                  f"path_len={len(path) if path else 0} score={sc}")
        if done:
            print(f"episode ended at step {step}, score={sc}"); break

    print(f"\nFinal score: {last_score}")
    if last_score > 0:
        print("A* + waypoint-following controller DROVE Bank Heist and robbed banks")
        print("-> cross-GAME transfer fully demonstrated.")
    else:
        print("Car navigating the path - if not robbing, check bank-contact tolerance.")


if __name__ == "__main__":
    main()