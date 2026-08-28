"""
experiments/try_bankheist.py

TIER 3 — Bank Heist inspector.
Dumps everything we need to design the representation adapter:
  - observation structure (player, enemies, banks, dynamite, fuel, ...)
  - the collision map: shape + what "wall" looks like in the pixel bitmap
  - object coordinate ranges (to derive a pixel->grid mapping)

Bank Heist is a DIFFERENT game from Ms. Pac-Man:
  - no DOF grid — just a raw pixel collision map (state.map_collision)
  - banks (targets) instead of pellets, police (enemies) instead of ghosts
  - a fuel constraint with no Pac-Man equivalent
  - 8 cities instead of 4/2 mazes

Run from the repository root:
    uv run python project/experiments/try_bankheist.py
"""
import sys
import os
import jax
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import jaxatari

env = jaxatari.make("bankheist")
obs, state = env.reset(jax.random.PRNGKey(0))

print("=" * 60)
print("OBSERVATION STRUCTURE")
print("=" * 60)
print("obs type:", type(obs).__name__)
# Print each top-level field and, for object fields, their attributes
for field in ["player", "enemies", "banks", "dynamite", "fuel", "fuel_refill", "lives", "score"]:
    try:
        val = getattr(obs, field)
    except AttributeError:
        print(f"  {field}: <not present>")
        continue
    if hasattr(val, "shape"):
        print(f"  {field}: array shape={val.shape} dtype={val.dtype}")
        print(f"       value={np.array(val).tolist()}")
    else:
        # ObjectObservation-like: dump its public attributes
        attrs = [a for a in dir(val) if not a.startswith("_")]
        print(f"  {field}: {type(val).__name__}  attrs={attrs}")
        for a in attrs:
            try:
                av = getattr(val, a)
                if hasattr(av, "shape"):
                    print(f"       .{a}: shape={av.shape}  value={np.array(av).tolist()}")
            except Exception:
                pass

print()
print("=" * 60)
print("COLLISION MAP  (this becomes the maze walls)")
print("=" * 60)
cm = np.array(state.map_collision)
print("state.map_collision shape:", cm.shape, "dtype:", cm.dtype)
print("unique values:", np.unique(cm)[:20], "..." if np.unique(cm).size > 20 else "")
print("nonzero fraction:", float((cm != 0).mean()))
# If it's HxW (or HxWxC), show a tiny downsampled ASCII preview
cm2 = cm if cm.ndim == 2 else cm[..., 0]
print("2D shape used for preview:", cm2.shape)
step_y = max(1, cm2.shape[0] // 40)
step_x = max(1, cm2.shape[1] // 60)
print(f"ASCII preview (downsampled every {step_y}x{step_x}px, '#'=nonzero/wall):")
for y in range(0, cm2.shape[0], step_y):
    row = "".join("#" if cm2[y, x] != 0 else " " for x in range(0, cm2.shape[1], step_x))
    print("   " + row)

print()
print("=" * 60)
print("HOW MANY CITIES?")
print("=" * 60)
if hasattr(env, "city_collision_maps"):
    ccm = np.array(env.city_collision_maps)
    print("env.city_collision_maps shape:", ccm.shape, "(first dim = number of cities)")
print("current map_id:", int(np.array(state.map_id)) if hasattr(state, "map_id") else "?")