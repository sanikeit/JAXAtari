"""
experiments/bankheist_adapter.py  (v3 - final)

TIER 3 - Bank Heist representation adapter.

Findings from diagnosis:
  - collision map is (W,H), indexed [x,y]; 0=wall, nonzero=walkable
  - object (x,y) is a sprite CORNER, which can land 1-2 px inside a wall edge
    even though the 8x8 sprite occupies a drivable corridor
  => build a center-sampled walkable grid, then SNAP each object (player/banks)
     to the nearest walkable grid cell. That cell is where the car actually
     navigates. This is the correct model, not a hack.

Provides build_bankheist_grid() + object cell mapping for the A* planner.

Run:
    uv run python project/experiments/bankheist_adapter.py
"""
import sys, os
import jax, jax.numpy as jnp, numpy as np
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import jaxatari

TILE = 8


def build_walkable_grid(city_map, tile=TILE, thresh=0.25):
    """(W,H) collision map, 0=wall -> (GH,GW) bool grid.
    A cell is walkable if >= `thresh` fraction of its pixels are open. Bank Heist's
    corridors are narrow relative to the 8px tile, so a permissive threshold (0.25)
    is needed to keep the maze CONNECTED - stricter rules (center-sample, >=0.5)
    disconnect drivable corridors and A* finds no path. Verified: 0.25 connects the
    player to all banks; 0.5 connects only 1/3; center-sample connects 0/3."""
    walk_px = (np.array(city_map) != 0).T          # (H,W) True=walkable
    H, W = walk_px.shape
    GH, GW = H // tile, W // tile
    grid = np.zeros((GH, GW), dtype=bool)
    for gy in range(GH):
        for gx in range(GW):
            grid[gy, gx] = walk_px[gy*tile:(gy+1)*tile, gx*tile:(gx+1)*tile].mean() >= thresh
    return grid


def snap_to_walkable(cell, grid):
    """BFS from cell to the nearest walkable grid cell (handles corner-on-wall)."""
    GH, GW = grid.shape
    gx, gy = cell
    if 0 <= gy < GH and 0 <= gx < GW and grid[gy, gx]:
        return (gx, gy)
    seen = {(gx, gy)}
    q = deque([(gx, gy)])
    while q:
        cx, cy = q.popleft()
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = cx+dx, cy+dy
            if (nx, ny) in seen:
                continue
            if 0 <= ny < GH and 0 <= nx < GW:
                if grid[ny, nx]:
                    return (nx, ny)
                seen.add((nx, ny)); q.append((nx, ny))
    return cell  # nothing walkable (shouldn't happen)


def get_bankheist_state(env, obs, state, tile=TILE):
    """Returns (grid, player_cell, [bank_cells], [enemy_cells]) ready for A*."""
    city_map = np.array(env.city_collision_maps)[int(np.array(state.map_id))]
    grid = build_walkable_grid(city_map, tile)
    pcell = snap_to_walkable((int(obs.player.x)//tile, int(obs.player.y)//tile), grid)
    banks = [snap_to_walkable((int(bx)//tile, int(by)//tile), grid)
             for bx, by, a in zip(np.array(obs.banks.x), np.array(obs.banks.y),
                                  np.array(obs.banks.active)) if a]
    enemies = [snap_to_walkable((int(ex)//tile, int(ey)//tile), grid)
               for ex, ey, a in zip(np.array(obs.enemies.x), np.array(obs.enemies.y),
                                    np.array(obs.enemies.active)) if a]
    return grid, pcell, banks, enemies


def main():
    env = jaxatari.make("bankheist")
    obs, state = env.reset(jax.random.PRNGKey(0))
    step_fn = jax.jit(env.step)
    obs, state, _, _, _ = step_fn(state, jnp.array(0, dtype=jnp.int32))

    grid, pcell, banks, enemies = get_bankheist_state(env, obs, state)
    GH, GW = grid.shape
    print(f"grid {grid.shape}, walkable {100*grid.mean():.1f}%")
    print(f"player -> {pcell}  walkable? {grid[pcell[1], pcell[0]]}")
    for i, b in enumerate(banks):
        print(f"bank {i} -> {b}  walkable? {grid[b[1], b[0]]}")
    print(f"enemies: {enemies}")
    bad = sum(1 for b in banks if not grid[b[1], b[0]])
    print(f"\nBanks (snapped) inside walls (should be 0): {bad}")

    print("\n=== CITY MAP  #=wall .=open P=player B=bank E=enemy ===")
    for gy in range(GH):
        row = ""
        for gx in range(GW):
            ch = "." if grid[gy, gx] else "#"
            if (gx, gy) == pcell: ch = "P"
            if (gx, gy) in banks: ch = "B"
            if (gx, gy) in enemies: ch = "E"
            row += ch
        print(row)


if __name__ == "__main__":
    main()