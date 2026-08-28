"""
experiments/bankheist_actiontest2.py

Map ALL four directions to action indices by testing from an OPEN position
(drive into the maze first, then probe each action). Also measure pixels/step
so we can build a waypoint controller.

Run:
    uv run python project/experiments/bankheist_actiontest2.py
"""
import sys, os
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import jaxatari

def fresh():
    env = jaxatari.make("bankheist")
    obs, state = env.reset(jax.random.PRNGKey(0))
    step_fn = jax.jit(env.step)
    for _ in range(2):
        obs, state, _, _, _ = step_fn(state, jnp.array(0, dtype=jnp.int32))
    return env, step_fn, obs, state

def drive(step_fn, obs, state, action_idx, n):
    for _ in range(n):
        obs, state, _, done, _ = step_fn(state, jnp.array(action_idx, dtype=jnp.int32))
        if done: break
    return obs, state

# 1. Drive right ~10 steps with idx 3 to get into a more open area
env, step_fn, obs, state = fresh()
obs, state = drive(step_fn, obs, state, 3, 12)
base = (int(obs.player.x), int(obs.player.y))
print(f"After driving right, car at {base}. Now probing each action from here:\n")

print(f"{'idx':>3} | {'end':>10} | {'dx':>4} {'dy':>4} | pixels/step | direction")
print("-"*62)
mapping = {}
for idx in range(10):
    env2, sf2, o2, s2 = fresh()
    o2, s2 = drive(sf2, o2, s2, 3, 12)          # same starting point
    start = (int(o2.player.x), int(o2.player.y))
    o2, s2 = drive(sf2, o2, s2, idx, 6)         # test this action 6 steps
    end = (int(o2.player.x), int(o2.player.y))
    dx, dy = end[0]-start[0], end[1]-start[1]
    pps = (abs(dx)+abs(dy)) / 6.0
    d = "none"
    if dx==0 and dy==0: d = "NO MOVE"
    elif abs(dx) > abs(dy): d = "RIGHT" if dx>0 else "LEFT"
    else: d = "DOWN" if dy>0 else "UP"
    if d not in ("none","NO MOVE") and d not in mapping:
        mapping[d] = idx
    print(f"{idx:>3} | {str(end):>10} | {dx:>4} {dy:>4} | {pps:>10.1f} | {d}")

print(f"\nDirection -> action index mapping found: {mapping}")
print("(UP/DOWN/LEFT/RIGHT indices to use in the waypoint controller)")