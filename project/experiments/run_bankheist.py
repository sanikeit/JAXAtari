"""
experiments/run_bankheist.py

TIER 3 - Bank Heist cross-game transfer: run the A* planner.

Uses the adapter (get_bankheist_state) to get a walkable grid + player/bank cells,
runs grid A* from the car to the nearest bank, and drives the car there. This is
the cross-GAME transfer demonstration: the SAME A* idea, on a game with a totally
different representation (pixel collision map, no DOF grid, banks not pellets).

The A* here reads the plain walkable grid directly (4-connected), rather than the
DOF format spatiotemporal_astar expects - that's the point: the ALGORITHM transfers,
only the representation adapter is new.

Run:
    uv run python project/experiments/run_bankheist.py
"""
# Tier 3: Bank Heist waypoint controller + collision-accurate grid (partial actuation)
import sys, os, heapq
import jax, jax.numpy as jnp, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import jaxatari
# Bank Heist ACTION_SET order: [NOOP, FIRE, UP, RIGHT, LEFT, DOWN, UPRIGHT, ...]
# step() does jnp.take(ACTION_SET, action) -> we pass the INDEX, not the enum value.
A_NOOP, A_UP, A_RIGHT, A_LEFT, A_DOWN = 0, 2, 3, 4, 5
# reuse the validated adapter
from importlib import import_module
adapter = import_module("project.experiments.bankheist_adapter")
get_bankheist_state = adapter.get_bankheist_state
TILE = adapter.TILE


def astar_grid(grid, start, goal):
    """Plain 4-connected A* on a boolean walkable grid. grid[y,x]."""
    GH, GW = grid.shape
    def h(c): return abs(c[0]-goal[0]) + abs(c[1]-goal[1])
    openq = [(h(start), 0, start)]
    came, gscore = {start: None}, {start: 0}
    while openq:
        _, g, cur = heapq.heappop(openq)
        if cur == goal:
            path = [cur]
            while came[cur] is not None:
                cur = came[cur]; path.append(cur)
            return path[::-1]
        cx, cy = cur
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = cx+dx, cy+dy
            if 0 <= ny < GH and 0 <= nx < GW and grid[ny, nx]:
                ng = g + 1
                if (nx,ny) not in gscore or ng < gscore[(nx,ny)]:
                    gscore[(nx,ny)] = ng
                    came[(nx,ny)] = cur
                    heapq.heappush(openq, (ng + h((nx,ny)), ng, (nx,ny)))
    return None


def dir_to_action(cur, nxt):
    dx, dy = nxt[0]-cur[0], nxt[1]-cur[1]
    if dx > 0: return A_RIGHT
    if dx < 0: return A_LEFT
    if dy > 0: return A_DOWN
    if dy < 0: return A_UP
    return A_NOOP


def nearest_bank(pcell, banks):
    return min(banks, key=lambda b: abs(b[0]-pcell[0]) + abs(b[1]-pcell[1]))


def main():
    env = jaxatari.make("bankheist")
    obs, state = env.reset(jax.random.PRNGKey(0))
    step_fn = jax.jit(env.step)
    obs, state, _, _, _ = step_fn(state, jnp.array(0, dtype=jnp.int32))  # warmup

    N_STEPS = 400
    banks_robbed_start = int(np.array(obs.banks.active).sum())  # not a robbery metric; info only
    print("Driving the A* planner in Bank Heist (city 0)...")
    last_score = 0
    for step in range(N_STEPS):
        grid, pcell, banks, enemies = get_bankheist_state(env, obs, state)
        if not banks:
            print(f"step {step}: no active banks; stopping.")
            break

        goal = nearest_bank(pcell, banks)
        path = astar_grid(grid, pcell, goal)
        if not path or len(path) < 2:
            action = A_NOOP
        else:
            action = dir_to_action(path[0], path[1])

        prev_xy = (int(obs.player.x), int(obs.player.y))
        obs, state, _, done, _ = step_fn(state, jnp.array(int(action), dtype=jnp.int32))
        new_xy = (int(obs.player.x), int(obs.player.y))
        if step < 8:
            print(f"  step {step}: action={action} player {prev_xy} -> {new_xy}")
        sc = int(np.array(state.score)) if hasattr(state, "score") else 0
        if sc != last_score:
            print(f"step {step}: score {last_score} -> {sc}  (drove toward bank at {goal})")
            last_score = sc
        if step % 100 == 0:
            print(f"step {step}: player_cell={pcell} nearest_bank={goal} "
                  f"path_len={len(path) if path else 'none'} score={sc}")
        if done:
            print(f"episode ended at step {step}, score={sc}")
            break

    print(f"\nFinal score: {last_score}")
    print("If the car reached banks / score increased, the A* planner drives Bank Heist")
    print("-> cross-GAME transfer: same algorithm, new representation adapter.")


if __name__ == "__main__":
    main()