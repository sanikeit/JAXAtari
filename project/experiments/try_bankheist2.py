"""
experiments/try_bankheist2.py

Follow-up inspector. The first pass showed two problems:
  1. state.map_collision was all 255 (useless) -> the real maze is in
     env.city_collision_maps[city_id]. Inspect THAT.
  2. banks/enemies were inactive at reset -> step forward to find when they spawn.

Run:
    uv run python project/experiments/try_bankheist2.py
"""
import sys
import os
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import jaxatari

env = jaxatari.make("bankheist")
obs, state = env.reset(jax.random.PRNGKey(0))

# ------------------------------------------------------------------
# 1. Look at the ACTUAL city collision map (not the live all-255 field)
# ------------------------------------------------------------------
print("=" * 64)
print("CITY COLLISION MAP 0  (the real maze walls)")
print("=" * 64)
ccm = np.array(env.city_collision_maps)          # (8, 160, 210)
city0 = ccm[0]
print("city0 shape:", city0.shape, "dtype:", city0.dtype)
vals, counts = np.unique(city0, return_counts=True)
print("unique values + counts:", dict(zip(vals.tolist(), counts.tolist())))

# The map is stored (W, H) = (160, 210). Transpose to (H, W) for a natural preview.
grid = city0.T   # now (210, 160): rows = y (top-down), cols = x
print("previewing as (H,W) =", grid.shape)

# Try BOTH interpretations so we can see which one looks like a maze
for wall_val_label, mask in [("nonzero = wall", grid != 0), ("zero = wall", grid == 0)]:
    print(f"\n--- interpretation: {wall_val_label} ---")
    step_y = max(1, grid.shape[0] // 50)
    step_x = max(1, grid.shape[1] // 80)
    for y in range(0, grid.shape[0], step_y):
        row = "".join("#" if mask[y, x] else " " for x in range(0, grid.shape[1], step_x))
        print(row)

# ------------------------------------------------------------------
# 2. Step forward until banks / enemies become active
# ------------------------------------------------------------------
print("\n" + "=" * 64)
print("STEPPING FORWARD to find when banks/police spawn")
print("=" * 64)
step_fn = jax.jit(env.step)
NOOP = 0
for step in range(1, 401):
    obs, state, _, done, _ = step_fn(state, jnp.array(NOOP, dtype=jnp.int32))
    banks_active = np.array(obs.banks.active)
    enemies_active = np.array(obs.enemies.active)
    if step % 50 == 0 or banks_active.any() or enemies_active.any():
        print(f"step {step:3d}: banks.active={banks_active.tolist()} "
              f"banks.xy={list(zip(np.array(obs.banks.x).tolist(), np.array(obs.banks.y).tolist()))} | "
              f"enemies.active={enemies_active.tolist()} | "
              f"player=({int(obs.player.x)},{int(obs.player.y)}) fuel={float(obs.fuel):.0f}")
        if banks_active.any() and enemies_active.any():
            print("  -> both banks and enemies now active; stopping.")
            break
    if done:
        print(f"  episode ended at step {step}")
        break