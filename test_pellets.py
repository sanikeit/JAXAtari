import jax
import jax.numpy as jnp
from jaxatari.games.jax_mspacman import JaxPacman
from project.utils.coords import pellet_to_dof, is_walkable

env = JaxPacman()
dof_maze = env.consts.DOF_MAZES[0]
key = jax.random.PRNGKey(42)
obs, state = env.reset(key)
pellets = state.level.pellets

valid_our = 0
total = 0
for pelly in range(pellets.shape[1]):
    for pellx in range(pellets.shape[0]):
        if pellets[pellx, pelly]:
            total += 1
            gx, gy = pellet_to_dof(pellx, pelly)
            if is_walkable(dof_maze, gx, gy):
                valid_our += 1

print(f"Our pellet mapping: {valid_our}/{total} walkable")

valid_jax = 0
for pelly in range(pellets.shape[1]):
    for pellx in range(pellets.shape[0]):
        if pellets[pellx, pelly]:
            px = pellx * 8 + 8
            if px > 74: px += 4
            py = pelly * 8 + 24
            
            gx = (px + 5) // 4
            gy = (py + 3) // 4
            if 0 <= gx < 40 and 0 <= gy < 44 and dof_maze[gx, gy].any():
                valid_jax += 1
                
print(f"JAX pixel-based mapping: {valid_jax}/{total} walkable")
