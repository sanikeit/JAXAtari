import jax
import jax.numpy as jnp
import heapq
from jaxatari.games.jax_mspacman import JaxPacman
from jaxatari.games.mspacman_mazes import MsPacmanMaze

def pixel_to_grid(pixel_x, pixel_y):
    return int(round(pixel_x / 4.0)), int(round(pixel_y / 4.0))

def extract_targets(maze, pellets):
    targets = set()
    for py in range(pellets.shape[1]):
        for px in range(pellets.shape[0]):
            if pellets[px, py] == 1:
                pixel_x = px * 8 + 5
                pixel_y = py * 12 + 6
                gx, gy = pixel_to_grid(pixel_x, pixel_y)
                best_gx, best_gy = None, None
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = gy + dy, gx + dx
                        if 0 <= ny < maze.shape[0] and 0 <= nx < maze.shape[1] and maze[ny, nx] == 0:
                            best_gx, best_gy = nx, ny
                            break
                    if best_gx is not None: break
                if best_gx is not None: targets.add((best_gx, best_gy))
    return targets

def compute_danger_map(maze, ghosts_pos, ghosts_modes):
    danger_map = {}
    for i in range(4):
        mode = ghosts_modes[i].item()
        if mode in [5, 6]: # RETURNING or ENJAILED
            continue
            
        px, py = ghosts_pos[i][0].item(), ghosts_pos[i][1].item()
        gx, gy = pixel_to_grid(px, py)
        
        if mode in [3, 4]: # FRIGHTENED or BLINKING (Safe)
            pass
        else: # Dangerous
            # Spread danger in a Manhattan distance of 6
            for dx in range(-6, 7):
                for dy in range(-6, 7):
                    dist = abs(dx) + abs(dy)
                    if dist <= 6:
                        cost = (7 - dist) * 20
                        pos = (gx + dx, gy + dy)
                        danger_map[pos] = danger_map.get(pos, 0) + cost
    return danger_map

def astar_search(maze, start, targets, danger_map):
    # Safe pellet selection: filter out targets that are too dangerous
    safe_targets = {t for t in targets if danger_map.get(t, 0) < 40}
    if not safe_targets:
        safe_targets = targets # Fallback if all targets are dangerous
        if not safe_targets: return []

    def h(pos):
        return min(abs(pos[0] - tx) + abs(pos[1] - ty) for tx, ty in safe_targets)

    queue = [(h(start), 0, start, [start])]
    visited = {start: 0}
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    while queue:
        _, cost, current, path = heapq.heappop(queue)
        
        if current in safe_targets:
            return path
            
        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= ny < maze.shape[0] and 0 <= nx < maze.shape[1] and maze[ny, nx] == 0:
                pos = (nx, ny)
                new_cost = cost + 1 + danger_map.get(pos, 0)
                if pos not in visited or new_cost < visited[pos]:
                    visited[pos] = new_cost
                    f_score = new_cost + h(pos)
                    heapq.heappush(queue, (f_score, new_cost, pos, path + [pos]))
    return []

def get_action_for_path(current_pixel, target_grid):
    cx, cy = current_pixel
    tx, ty = target_grid[0] * 4, target_grid[1] * 4
    
    dx = tx - cx
    dy = ty - cy
    
    # Standard Enum Action mapping in JaxAtari:
    # 0: NOOP, 1: UP, 2: RIGHT, 3: LEFT, 4: DOWN
    if abs(dx) > abs(dy):
        return 2 if dx > 0 else 3 # RIGHT / LEFT
    elif abs(dy) > 0:
        return 4 if dy > 0 else 1 # DOWN / UP
    return 0

def main():
    env = JaxPacman()
    step_fn = jax.jit(env.step)
    
    key = jax.random.PRNGKey(42)
    obs, state = env.reset(key)
    maze = MsPacmanMaze.MAZES[0]
    
    print("--- Starting Phase 1 Model-Free Loop ---")
    
    total_reward = 0
    steps = 0
    done = False
    
    # We will simulate 300 steps (about 15 seconds of gameplay)
    while not done and steps < 300:
        px, py = obs.player_position[0].item(), obs.player_position[1].item()
        start_grid = pixel_to_grid(px, py)
        
        targets = extract_targets(maze, obs.pellets)
        if not targets:
            print("No more targets!")
            break
            
        danger_map = compute_danger_map(maze, obs.ghost_positions, state.ghosts.modes)
        path = astar_search(maze, start_grid, targets, danger_map)
        
        if path and len(path) > 1:
            action = get_action_for_path((px, py), path[1])
        else:
            action = 0
            
        obs, state, reward, done, info = step_fn(state, jnp.array(action, dtype=jnp.int32))
        total_reward += reward.item()
        steps += 1
        
        if steps % 50 == 0:
            print(f"Step {steps}: Pos={start_grid}, Action={action}, Reward={total_reward}, Done={done}")

    print(f"Finished after {steps} steps. Total Reward: {total_reward}")

if __name__ == '__main__':
    main()
