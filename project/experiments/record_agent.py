"""
experiments/record_agent.py

Watch an agent play. Produces TWO things:
  1. agent_play.mp4          -- full episode video (for your poster / playback)
  2. snap_00.png .. snap_07.png -- 8 evenly-spaced still frames

Upload the 8 snapshot PNGs to Claude -- Claude can read stills (not video),
so the snapshots are how it can actually see what the agent does.

Config below: choose PLANNER ('MF' or 'MBv2') and MAZE_ID (0/1/2/3).

Run from the repository root:
    uv run python project/experiments/record_agent.py

First install the video writer (bundles ffmpeg, no system install needed):
    uv pip install imageio imageio-ffmpeg
"""
import sys
import os
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import imageio.v2 as imageio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from jaxatari.games.jax_mspacman import JaxPacman, MsPacmanConstants
from project.planners.mf_baseline import MFPlanner
from project.planners.mb_v2_planner import MBv2Planner

# ─────────────── CONFIG ───────────────
PLANNER = 'MFGlobal'        # 'MF' or 'MBv2'
MAZE_ID = 0           # 0, 1, 2, 3
N_STEPS = 300
SEED = 42
FPS = 25
N_SNAPSHOTS = 8
# ──────────────────────────────────────

LEVEL = {0: 1, 1: 3, 2: 5, 3: 7}[MAZE_ID]
consts = MsPacmanConstants(RESET_LEVEL=LEVEL)
env = JaxPacman(consts)
step_fn = jax.jit(env.step)
dof = env.consts.DOF_MAZES[MAZE_ID]

if PLANNER == 'MF':
    planner = MFPlanner(dof)
elif PLANNER == 'MFGlobal':
    from project.planners.mf_global_planner import MFGlobalPlanner
    planner = MFGlobalPlanner(dof, strategy='farthest')
    planner.reset()
else:
    planner = MBv2Planner(env.consts)

key = jax.random.PRNGKey(SEED)
obs, state = env.reset(key)
pkey = jax.random.PRNGKey(SEED + 1)     # only used by MBv2

print(f"Recording {PLANNER} on maze {MAZE_ID} ...")
frames = []
for _ in range(N_STEPS):
    frames.append(np.array(env.render(state)).astype(np.uint8))

    if PLANNER == 'MBv2':
        action, pkey = planner.plan(obs, state, pkey)
    else:
        action = planner.plan(obs, state)

    obs, state, _, done, _ = step_fn(state, jnp.array(action, dtype=jnp.int32))
    if done:
        break

# --- write the mp4 ---
out_mp4 = f"agent_play_{PLANNER}_maze{MAZE_ID}.mp4"
# mp4 encoders need even width/height; pad by 1px if odd
h, w = frames[0].shape[:2]
if h % 2 or w % 2:
    frames = [np.pad(f, ((0, h % 2), (0, w % 2), (0, 0))) for f in frames]
imageio.mimsave(out_mp4, frames, fps=FPS)
print(f"Saved {out_mp4}  ({len(frames)} frames)")

# --- dump 8 evenly-spaced snapshot PNGs (upload THESE to Claude) ---
idxs = np.linspace(0, len(frames) - 1, N_SNAPSHOTS).astype(int)
for i, fi in enumerate(idxs):
    Image.fromarray(frames[fi]).save(f"snap_{i:02d}.png")
print(f"Saved {N_SNAPSHOTS} snapshots: snap_00.png .. snap_{N_SNAPSHOTS-1:02d}.png")
print("Upload the snap_*.png files to Claude to have it read what the agent did.")