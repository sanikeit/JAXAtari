import jax
import jax.numpy as jnp
from jaxatari.games.jax_mspacman import JaxPacman
from jaxatari.games.mspacman_mazes import MsPacmanMaze

env = JaxPacman()
key = jax.random.PRNGKey(42)
obs, state = env.reset(key)

maze = MsPacmanMaze.MAZES[0]
pellets = obs.pellets

pellet_to_grid = {}

for gy in range(maze.shape[0]):
    for gx in range(maze.shape[1]):
        if maze[gy, gx] == 0: # path
            pixel_x = gx * 4
            pixel_y = gy * 4
            px = (pixel_x - 2) // 8
            py = (pixel_y + 4) // 12
            
            if 0 <= px < 18 and 0 <= py < 14:
                if pellets[px, py] == 1:
                    if (px, py) not in pellet_to_grid:
                        pellet_to_grid[(px, py)] = []
                    pellet_to_grid[(px, py)].append((gx, gy))

print(f"Total pellets in obs: {jnp.sum(pellets)}")
print(f"Total pellets mapped to grid: {len(pellet_to_grid)}")

# Check if any mapped pellets have NO grid cell
print("All mapped pellets have at least one grid cell:", all(len(v) > 0 for v in pellet_to_grid.values()))
