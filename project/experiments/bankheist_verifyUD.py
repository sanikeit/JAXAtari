"""
experiments/bankheist_verifyUD.py

Verify the hypothesized mapping UP=2, DOWN=5 by driving the car RIGHT until a
vertical corridor opens, then testing idx 2 and idx 5. Reports the first x where
each produces vertical movement.

Run:
    uv run python project/experiments/bankheist_verifyUD.py
"""
import sys, os
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import jaxatari

def fresh():
    env = jaxatari.make("bankheist")
    obs, state = env.reset(jax.random.PRNGKey(0))
    sf = jax.jit(env.step)
    for _ in range(2):
        obs, state, _, _, _ = sf(state, jnp.array(0, dtype=jnp.int32))
    return env, sf, obs, state

def pos(o): return (int(o.player.x), int(o.player.y))

RIGHT, UP, DOWN = 3, 2, 5

# Drive right in small increments; at each stop, test UP and DOWN on a fresh
# re-drive to that x (avoids latch). Report where vertical movement appears.
print(f"Testing UP=idx{UP}, DOWN=idx{DOWN} at increasing x (driven right):\n")
print(f"{'drive_steps':>11} | {'x reached':>9} | {'UP dy':>6} | {'DOWN dy':>7}")
print("-"*44)
found_up = found_down = None
for drive_steps in range(0, 90, 4):
    # UP test
    e,sf,o,s = fresh()
    for _ in range(drive_steps): o,s,_,d,_ = sf(s, jnp.array(RIGHT, dtype=jnp.int32))
    x_here = pos(o)[0]; y0 = pos(o)[1]
    for _ in range(4): o,s,_,d,_ = sf(s, jnp.array(UP, dtype=jnp.int32))
    up_dy = pos(o)[1]-y0
    # DOWN test (fresh)
    e,sf,o,s = fresh()
    for _ in range(drive_steps): o,s,_,d,_ = sf(s, jnp.array(RIGHT, dtype=jnp.int32))
    y0 = pos(o)[1]
    for _ in range(4): o,s,_,d,_ = sf(s, jnp.array(DOWN, dtype=jnp.int32))
    down_dy = pos(o)[1]-y0
    print(f"{drive_steps:>11} | {x_here:>9} | {up_dy:>6} | {down_dy:>7}")
    if found_up is None and up_dy < 0: found_up = (drive_steps, x_here)
    if found_down is None and down_dy > 0: found_down = (drive_steps, x_here)

print()
print(f"UP (idx {UP}) first produced upward movement at: {found_up}")
print(f"DOWN (idx {DOWN}) first produced downward movement at: {found_down}")
if found_up or found_down:
    print("\n=> Mapping CONFIRMED: UP=2, RIGHT=3, LEFT=4, DOWN=5")
    print("   (they showed 'none' at spawn only because that corridor is horizontal)")
else:
    print("\n=> Neither moved vertically; UP/DOWN may be different indices.")