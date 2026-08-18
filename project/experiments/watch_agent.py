import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import jax, jax.numpy as jnp, numpy as np
from PIL import Image
from jaxatari.games.jax_mspacman import JaxPacman, MsPacmanConstants
from project.planners.mf_baseline import MFPlanner

MAZE_ID = 1                       # 0/1/2/3
LEVEL = {0:1, 1:3, 2:5, 3:7}[MAZE_ID]
N_STEPS = 300

consts = MsPacmanConstants(RESET_LEVEL=LEVEL)
env = JaxPacman(consts)
step_fn = jax.jit(env.step)
planner = MFPlanner(env.consts.DOF_MAZES[MAZE_ID])

obs, state = env.reset(jax.random.PRNGKey(42))
frames = []
for _ in range(N_STEPS):
    frames.append(Image.fromarray(np.array(env.render(state)).astype(np.uint8)))
    action = planner.plan(obs, state)
    obs, state, _, done, _ = step_fn(state, jnp.array(action, dtype=jnp.int32))
    if done:
        break

frames[0].save("agent_play.gif", save_all=True, append_images=frames[1:],
               duration=40, loop=0)
print(f"Saved agent_play.gif ({len(frames)} frames)")