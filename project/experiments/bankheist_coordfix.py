"""
experiments/bankheist_coordfix.py

Diagnose the coordinate mismatch: object pixels report as WALL in the collision
map, so object-space and collision-space don't align. This probe tests:
  1. both axis orders (city_map[x,y] vs city_map[y,x])
  2. the nearest OPEN pixel to each object (reveals any constant offset)
  3. prints a zoomed ASCII around one bank so we can SEE the misalignment

Run:
    uv run python project/experiments/bankheist_coordfix.py
"""
import sys, os
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import jaxatari

env = jaxatari.make("bankheist")
obs, state = env.reset(jax.random.PRNGKey(0))
step_fn = jax.jit(env.step)
obs, state, _, _, _ = step_fn(state, jnp.array(0, dtype=jnp.int32))

cm = np.array(env.city_collision_maps)[int(np.array(state.map_id))]  # (160, 210) = (W,H)?
print("collision map shape:", cm.shape)

objs = [("player", int(obs.player.x), int(obs.player.y))]
for i, (bx, by, a) in enumerate(zip(np.array(obs.banks.x), np.array(obs.banks.y), np.array(obs.banks.active))):
    if a:
        objs.append((f"bank{i}", int(bx), int(by)))

def val(arr, a, b):
    if 0 <= a < arr.shape[0] and 0 <= b < arr.shape[1]:
        return int(arr[a, b])
    return -1

print("\n--- axis order test (value 0=wall, nonzero=open) ---")
for name, ox, oy in objs:
    print(f"{name:8s} obj=({ox},{oy}):  cm[x,y]={val(cm, ox, oy):3d}   cm[y,x]={val(cm, oy, ox):3d}")

print("\n--- nearest OPEN pixel (search radius 12) to each object, both orders ---")
def nearest_open(arr, ox, oy, order):
    best = None
    for r in range(0, 13):
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                if max(abs(dx), abs(dy)) != r:
                    continue
                a, b = (ox+dx, oy+dy) if order == "xy" else (oy+dy, ox+dx)
                if 0 <= a < arr.shape[0] and 0 <= b < arr.shape[1] and arr[a, b] != 0:
                    return (dx, dy, r)
    return None

for name, ox, oy in objs:
    for order in ["xy", "yx"]:
        res = nearest_open(cm, ox, oy, order)
        print(f"{name:8s} order={order}: nearest open offset (dx,dy,dist)={res}")

# zoomed view around bank0 in BOTH orders
print("\n--- zoom around bank0 (21x21), '#'=wall '.'=open 'B'=object cell ---")
bx, by = objs[1][1], objs[1][2]
for order, arr_get in [("cm[x,y]", lambda a,b: val(cm,a,b)), ("cm[y,x]", lambda a,b: val(cm,b,a))]:
    print(f"\norder {order}:")
    for dy in range(-10, 11):
        row = ""
        for dx in range(-10, 11):
            x, y = bx+dx, by+dy
            v = arr_get(x, y)
            ch = "B" if (dx==0 and dy==0) else ("." if v != 0 else "#")
            row += ch
        print("  " + row)