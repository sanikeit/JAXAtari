import jax.numpy as jnp
from jaxatari.games.jax_mspacman import JaxPacman, dof
import sys

env = JaxPacman()
dof_maze = env.consts.DOF_MAZES[0]
pos = jnp.array([75, 102]) # Initial pacman position
print("Initial Pos:", pos)
print("jaxatari dof calculation:")
gx_jax = (pos[0] + 5) // 4
gy_jax = (pos[1] + 3) // 4
print("gx, gy =", gx_jax, gy_jax)
print("walkable?", dof_maze[gx_jax, gy_jax].any())

print("our coords.py pixel_to_dof:")
gx_our = pos[0] // 4
gy_our = pos[1] // 4
print("gx, gy =", gx_our, gy_our)
print("walkable?", dof_maze[gx_our, gy_our].any())
