"""
Investigate the root cause: mean path length = 2.0
This means targets are always 1 step from Pacman.
"""
import jax
import jax.numpy as jnp
from jaxatari.games.jax_mspacman import JaxPacman
from jaxatari.games.mspacman_mazes import MsPacmanMaze

def pixel_to_grid(pixel_x, pixel_y):
    return int(round(pixel_x / 4.0)), int(round(pixel_y / 4.0))

env = JaxPacman()
step_fn = jax.jit(env.step)
key = jax.random.PRNGKey(42)
obs, state = env.reset(key)
step_fn(state, jnp.array(0, dtype=jnp.int32))  # compile

maze = MsPacmanMaze.MAZES[0]

print("=== Pellet array info ===")
print(f"obs.pellets.shape: {obs.pellets.shape}")
print(f"obs.pellets.dtype: {obs.pellets.dtype}")
total = int(obs.pellets.sum())
print(f"Total pellets: {total}")

print("\n=== Pacman position ===")
px, py = obs.player_position[0].item(), obs.player_position[1].item()
print(f"Pixel: ({px}, {py})")
gx, gy = pixel_to_grid(px, py)
print(f"Grid: ({gx}, {gy})")
print(f"Maze cell at grid: {maze[gy, gx]}")

print("\n=== First 10 pellet positions ===")
count = 0
for pelly in range(obs.pellets.shape[1]):
    for pellx in range(obs.pellets.shape[0]):
        if obs.pellets[pellx, pelly] == 1:
            # Current (broken?) conversion
            pg_x = pixel_to_grid(pellx * 8 + 5, pelly * 12 + 6)
            # Direct grid coords
            print(f"  pellet[{pellx},{pelly}] -> pixel ({pellx*8+5},{pelly*12+6}) -> grid {pg_x}  maze_cell={maze[pg_x[1], pg_x[0]] if 0<=pg_x[1]<maze.shape[0] and 0<=pg_x[0]<maze.shape[1] else 'OOB'}")
            count += 1
            if count >= 10:
                break
    if count >= 10:
        break

print("\n=== Pacman neighbors ===")
for dx, dy, name in [(0,-1,'UP'),(1,0,'RIGHT'),(-1,0,'LEFT'),(0,1,'DOWN')]:
    nx, ny = gx+dx, gy+dy
    if 0<=ny<maze.shape[0] and 0<=nx<maze.shape[1]:
        print(f"  {name}: grid ({nx},{ny}) maze={maze[ny,nx]}")

print("\n=== Checking if any pellet target is adjacent to Pacman ===")
targets = set()
for pelly in range(obs.pellets.shape[1]):
    for pellx in range(obs.pellets.shape[0]):
        if obs.pellets[pellx, pelly] == 1:
            pg_x = pixel_to_grid(pellx * 8 + 5, pelly * 12 + 6)
            best = None
            for dy in [-1,0,1]:
                for ddx in [-1,0,1]:
                    ny, nx = pg_x[1]+dy, pg_x[0]+ddx
                    if 0<=ny<maze.shape[0] and 0<=nx<maze.shape[1] and maze[ny,nx]==0:
                        best = (nx, ny)
                        break
                if best: break
            if best: targets.add(best)

adj_count = 0
for t in targets:
    if abs(t[0]-gx) + abs(t[1]-gy) <= 1:
        adj_count += 1
        print(f"  Adjacent target: {t}")
print(f"Total adjacent targets: {adj_count} / {len(targets)}")

print("\n=== Checking MsPacmanMaze for correct pellet grid scale ===")
print(f"Maze shape: {maze.shape}")
print(f"MsPacmanMaze.TILE_SCALE: {MsPacmanMaze.TILE_SCALE}")
print(f"MsPacmanMaze.WIDTH: {MsPacmanMaze.WIDTH}")
print(f"MsPacmanMaze.HEIGHT: {MsPacmanMaze.HEIGHT}")

# Try alternative conversions
print("\n=== Alternative: direct pellet grid as maze grid ===")
count = 0
for pelly in range(obs.pellets.shape[1]):
    for pellx in range(obs.pellets.shape[0]):
        if obs.pellets[pellx, pelly] == 1:
            # Try: pellet coords ARE maze grid coords
            if 0<=pelly<maze.shape[0] and 0<=pellx<maze.shape[1]:
                print(f"  pellet[{pellx},{pelly}] maze_cell={maze[pelly, pellx]}")
            count += 1
            if count >= 10:
                break
    if count >= 10:
        break
