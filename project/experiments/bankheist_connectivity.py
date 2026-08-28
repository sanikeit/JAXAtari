"""
experiments/bankheist_connectivity.py

Diagnose why A* finds no path: is the player in the same connected region as the
banks? Tests several grid-building rules and reports, for each, how many banks are
reachable from the player via flood fill. Pick the rule that connects them.

Run:
    uv run python project/experiments/bankheist_connectivity.py
"""
import sys, os
import jax, jax.numpy as jnp, numpy as np
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import jaxatari

TILE = 8


def grid_center(cm, tile=TILE):
    wp = (np.array(cm) != 0).T
    H, W = wp.shape; GH, GW = H//tile, W//tile
    g = np.zeros((GH, GW), bool)
    for gy in range(GH):
        for gx in range(GW):
            g[gy, gx] = wp[gy*tile+tile//2, gx*tile+tile//2]
    return g

def grid_frac(cm, thresh, tile=TILE):
    wp = (np.array(cm) != 0).T
    H, W = wp.shape; GH, GW = H//tile, W//tile
    g = np.zeros((GH, GW), bool)
    for gy in range(GH):
        for gx in range(GW):
            g[gy, gx] = wp[gy*tile:(gy+1)*tile, gx*tile:(gx+1)*tile].mean() >= thresh
    return g

def grid_any(cm, tile=TILE):
    wp = (np.array(cm) != 0).T
    H, W = wp.shape; GH, GW = H//tile, W//tile
    g = np.zeros((GH, GW), bool)
    for gy in range(GH):
        for gx in range(GW):
            g[gy, gx] = wp[gy*tile:(gy+1)*tile, gx*tile:(gx+1)*tile].any()
    return g

def snap(cell, grid):
    GH, GW = grid.shape; gx, gy = cell
    if 0<=gy<GH and 0<=gx<GW and grid[gy,gx]: return (gx,gy)
    seen={(gx,gy)}; q=deque([(gx,gy)])
    while q:
        cx,cy=q.popleft()
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx,ny=cx+dx,cy+dy
            if (nx,ny) in seen: continue
            if 0<=ny<GH and 0<=nx<GW:
                if grid[ny,nx]: return (nx,ny)
                seen.add((nx,ny)); q.append((nx,ny))
    return cell

def reachable(grid, start):
    GH, GW = grid.shape
    seen={start}; q=deque([start])
    while q:
        cx,cy=q.popleft()
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx,ny=cx+dx,cy+dy
            if 0<=ny<GH and 0<=nx<GW and grid[ny,nx] and (nx,ny) not in seen:
                seen.add((nx,ny)); q.append((nx,ny))
    return seen


env = jaxatari.make("bankheist")
obs, state = env.reset(jax.random.PRNGKey(0))
step_fn = jax.jit(env.step)
obs, state, _, _, _ = step_fn(state, jnp.array(0, dtype=jnp.int32))
cm = np.array(env.city_collision_maps)[int(np.array(state.map_id))]

builders = {
    "center":     lambda: grid_center(cm),
    "any":        lambda: grid_any(cm),
    "frac>=0.25": lambda: grid_frac(cm, 0.25),
    "frac>=0.5":  lambda: grid_frac(cm, 0.5),
    "frac>=0.75": lambda: grid_frac(cm, 0.75),
}

px, py = int(obs.player.x)//TILE, int(obs.player.y)//TILE
banks_px = [(int(bx)//TILE, int(by)//TILE)
            for bx,by,a in zip(np.array(obs.banks.x), np.array(obs.banks.y),
                               np.array(obs.banks.active)) if a]

print(f"{'builder':<12} | {'walk%':>6} | reachable banks from player")
print("-"*55)
for name, build in builders.items():
    g = build()
    p = snap((px,py), g)
    reach = reachable(g, p)
    banks = [snap(b, g) for b in banks_px]
    hit = sum(1 for b in banks if b in reach)
    print(f"{name:<12} | {100*g.mean():5.1f}% | {hit}/{len(banks)} reachable  "
          f"(player region size {len(reach)})")