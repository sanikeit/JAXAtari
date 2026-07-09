import jax
import jax.numpy as jnp
from jaxatari.games.jax_mspacman import JaxPacman
from jaxatari.games.mspacman_mazes import MsPacmanMaze

env = JaxPacman()
key = jax.random.PRNGKey(42)
obs, state = env.reset(key)

maze = MsPacmanMaze.MAZES[0]
pellets = obs.pellets

def check_pellet(x, y):
    x_offset = 5 if x < 75 else 1
    return (x % 8 == x_offset) and (y % 12 == 6)

eatable_pellets_pixels = []
eatable_pellets_grid = set()

# The pixel space is 40*4 x 71*4
for y in range(71 * 4):
    for x in range(40 * 4):
        gy = y // 4
        gx = x // 4
        if maze[gy, gx] == 0:
            if check_pellet(x, y):
                px = (x - 2) // 8
                py = (y + 4) // 12
                if 0 <= px < 18 and 0 <= py < 14:
                    if pellets[px, py] == 1:
                        eatable_pellets_pixels.append((x, y))
                        eatable_pellets_grid.add((gx, gy))

print(f"Total eatable pellets found: {len(eatable_pellets_pixels)}")
print(f"Unique grid cells for eatable pellets: {len(eatable_pellets_grid)}")
