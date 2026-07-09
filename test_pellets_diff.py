import jax
import jax.numpy as jnp
from jaxatari.games.jax_mspacman import JaxPacman
from project.utils.coords import pellet_to_dof

env = JaxPacman()
dof_maze = env.consts.DOF_MAZES[0]
key = jax.random.PRNGKey(42)
obs, state = env.reset(key)
pellets = state.level.pellets

diff_count = 0
for pelly in range(pellets.shape[1]):
    for pellx in range(pellets.shape[0]):
        if pellets[pellx, pelly]:
            gx_our, gy_our = pellet_to_dof(pellx, pelly)
            
            px = pellx * 8 + 8
            if px > 74: px += 4
            py = pelly * 8 + 24
            
            gx_jax = (px + 5) // 4
            gy_jax = (py + 3) // 4
            
            if (gx_our, gy_our) != (gx_jax, gy_jax):
                diff_count += 1
                if diff_count <= 5:
                    print(f"[{pellx},{pelly}] Our: {gx_our},{gy_our} | JAX: {gx_jax},{gy_jax}")

print(f"Total diffs: {diff_count}")
