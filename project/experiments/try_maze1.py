import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import jax
from jaxatari.games.jax_mspacman import JaxPacman, MsPacmanConstants
from project.utils.coords import extract_pellet_targets, is_walkable

consts = MsPacmanConstants(RESET_LEVEL=3)   # level 3 = maze 1
env = JaxPacman(consts)
obs, state = env.reset(jax.random.PRNGKey(42))

dof = env.consts.DOF_MAZES[1]
print("Level id:", state.level.id)
print("Pacman pos:", obs.player_position)
print("Num pellets in obs:", int(obs.pellets.sum()))

targets = extract_pellet_targets(dof, obs.pellets)
print("Targets extracted:", len(targets))

bad = [t for t in targets if not is_walkable(dof, t[0], t[1])]
print("Targets INSIDE walls (should be 0):", len(bad))

# --- NEW: render the frame and save it so we can see which maze it is ---
import numpy as np
from PIL import Image
img = env.render(state)
Image.fromarray(np.array(img).astype(np.uint8)).save("maze_check.png")
print("Saved maze_check.png - open it and check which maze it is")


from jaxatari.games.jax_mspacman import get_level_maze
import numpy as np

active_idx = int(get_level_maze(state.level.id))
print("Active maze index from level:", active_idx)   # want 1

# Are DOF_MAZES[0] and DOF_MAZES[1] actually different?
d0 = np.array(env.consts.DOF_MAZES[0])
d1 = np.array(env.consts.DOF_MAZES[1])
print("DOF maze 0 vs 1 identical?", np.array_equal(d0, d1))
print("Cells differing between maze 0 and 1:", int((d0 != d1).sum()))