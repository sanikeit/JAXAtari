import jax
import jax.numpy as jnp
import heapq
from jaxatari.games.jax_mspacman import JaxPacman
from jaxatari.games.mspacman_mazes import MsPacmanMaze

def pixel_to_grid(pixel_x, pixel_y):
    """
    Converts a pixel coordinate to maze grid coordinates.
    The tile scale is 4 pixels per grid cell.
    """
    return int(round(pixel_x / 4.0)), int(round(pixel_y / 4.0))

def extract_targets(maze, pellets):
    """
    Converts the 18x14 pellet grid into valid (x, y) maze grid coordinates.
    """
    targets = set()
    for py in range(pellets.shape[1]):
        for px in range(pellets.shape[0]):
            if pellets[px, py] == 1:
                # Reconstruct the center pixel of the pellet
                # based on the jax_mspacman check_pellet logic
                pixel_x = px * 8 + 5
                pixel_y = py * 12 + 6
                gx, gy = pixel_to_grid(pixel_x, pixel_y)
                
                # We need to map it to the closest path cell, since 
                # pellets might be slightly off the strict 4x4 grid center.
                best_gx, best_gy = None, None
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = gy + dy, gx + dx
                        if 0 <= ny < maze.shape[0] and 0 <= nx < maze.shape[1]:
                            if maze[ny, nx] == 0:
                                best_gx, best_gy = nx, ny
                                break
                    if best_gx is not None:
                        break
                
                if best_gx is not None:
                    targets.add((best_gx, best_gy))
    return targets

def astar_search(maze, start, targets):
    """
    A* search algorithm to find the shortest path from start to the nearest target.
    """
    if not targets:
        return []
    
    if start in targets:
        return [start]
    
    # Heuristic: Manhattan distance to the nearest target
    def h(pos):
        return min(abs(pos[0] - tx) + abs(pos[1] - ty) for tx, ty in targets)
    
    # Priority Queue items: (f_score, g_score, (x, y), path_so_far)
    queue = [(h(start), 0, start, [start])]
    visited = set([start])
    
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)] # Down, Up, Right, Left
    
    while queue:
        _, cost, current, path = heapq.heappop(queue)
        
        if current in targets:
            return path
        
        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy
            
            # Check bounds and if it is a valid path
            if 0 <= ny < maze.shape[0] and 0 <= nx < maze.shape[1]:
                # Wrap-around tunnels in Pacman could be added here, 
                # but standard grid bounds are fine for this initial baseline
                if maze[ny, nx] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    new_path = path + [(nx, ny)]
                    g_score = cost + 1
                    f_score = g_score + h((nx, ny))
                    heapq.heappush(queue, (f_score, g_score, (nx, ny), new_path))
                    
    return [] # No path found

def visualize(maze, path, targets, start):
    """
    ASCII visualization of the maze with the path.
    """
    out = []
    for y in range(maze.shape[0]):
        row = []
        for x in range(maze.shape[1]):
            if (x, y) == start:
                row.append('P') # Pacman start
            elif path and (x, y) in path:
                row.append('*') # Path
            elif (x, y) in targets:
                row.append('.') # Uneaten pellet
            elif maze[y, x] == 1:
                row.append('#') # Wall
            else:
                row.append(' ') # Empty path
        out.append("".join(row))
    return "\n".join(out)

def main():
    # 1. Init environment
    env = JaxPacman()
    key = jax.random.PRNGKey(42)
    obs, state = env.reset(key)
    
    maze = MsPacmanMaze.MAZES[0] # Boolean array (71, 40)
    
    # 2. Coordinate Conversion
    pixel_x, pixel_y = obs.player_position[0].item(), obs.player_position[1].item()
    grid_start = pixel_to_grid(pixel_x, pixel_y)
    print(f"Pacman Pixel Position: ({pixel_x}, {pixel_y}) -> Grid Position: {grid_start}")
    
    # 3. Extract Pellet Targets
    targets = extract_targets(maze, obs.pellets)
    print(f"Total valid pellet targets found in maze graph: {len(targets)}")
    
    # 4. Run A*
    print("Running A* search to nearest pellet...")
    path = astar_search(maze, grid_start, targets)
    
    print(f"Path length: {len(path)} cells")
    if path:
        print(f"Nearest pellet found at: {path[-1]}")
    
    # 5. Visualize
    print("\n--- ASCII Visualization ---")
    print("Legend: P=Pacman, *=Path, .=Pellet, #=Wall\n")
    print(visualize(maze, path, targets, grid_start))

if __name__ == "__main__":
    main()
