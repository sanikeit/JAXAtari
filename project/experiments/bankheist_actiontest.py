"""
experiments/bankheist_actiontest.py

Empirically find which action INDEX moves the car in which direction, and whether
the car can move at all from its spawn (wall-wedging check). We send each action
index for several steps and record the net pixel displacement.

Run:
    uv run python project/experiments/bankheist_actiontest.py
"""
import sys, os
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import jaxatari

def run_with_action(action_idx, n=10, warmup=2):
    env = jaxatari.make("bankheist")
    obs, state = env.reset(jax.random.PRNGKey(0))
    step_fn = jax.jit(env.step)
    for _ in range(warmup):
        obs, state, _, _, _ = step_fn(state, jnp.array(0, dtype=jnp.int32))
    start = (int(obs.player.x), int(obs.player.y))
    traj = [start]
    for _ in range(n):
        obs, state, _, done, _ = step_fn(state, jnp.array(action_idx, dtype=jnp.int32))
        traj.append((int(obs.player.x), int(obs.player.y)))
        if done: break
    end = traj[-1]
    dx, dy = end[0]-start[0], end[1]-start[1]
    return start, end, dx, dy, traj

print("Testing action indices 0-9 (10 steps each) from spawn:\n")
print(f"{'idx':>3} | {'start':>10} | {'end':>10} | {'dx':>4} {'dy':>4} | interpretation")
print("-"*66)
labels = {}
for idx in range(10):
    s, e, dx, dy, traj = run_with_action(idx)
    interp = ""
    if dx==0 and dy==0: interp = "NO MOVEMENT"
    elif abs(dx) > abs(dy): interp = "RIGHT (+x)" if dx>0 else "LEFT (-x)"
    else: interp = "DOWN (+y)" if dy>0 else "UP (-y)"
    labels[idx] = (dx, dy, interp)
    print(f"{idx:>3} | {str(s):>10} | {str(e):>10} | {dx:>4} {dy:>4} | {interp}")

print("\n--- Can the car move AT ALL from spawn? ---")
moved = [i for i,(dx,dy,_) in labels.items() if dx!=0 or dy!=0]
if moved:
    print(f"YES - indices {moved} produce movement. Spawn is NOT wall-wedged.")
    print("Use these to map A* directions (up/down/left/right) to action indices.")
else:
    print("NO index moves the car from spawn -> wall-wedged at spawn pixel.")
    print("Next: nudge the car to a nearby open pixel before planning, or the")
    print("collision model needs the grid aligned to it.")

# If movement works, try to reach the interior: send the index that moves toward
# larger x (into the maze) and see how far it gets before hitting a wall.
if moved:
    print("\n--- Trying a longer drive with the first RIGHT-ish action ---")
    right_idx = next((i for i,(dx,dy,_) in labels.items() if dx>abs(dy)), moved[0])
    s,e,dx,dy,traj = run_with_action(right_idx, n=40)
    print(f"action {right_idx}: {s} -> {e} over 40 steps  (net dx={dx}, dy={dy})")
    print(f"trajectory sample: {traj[::5]}")