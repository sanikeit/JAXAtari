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

class GhostTracker:
    def __init__(self):
        self.last_pos = None
        
    def update(self, current_pos):
        if self.last_pos is None:
            velocities = [(0, 0) for _ in range(4)]
        else:
            velocities = []
            for i in range(4):
                px, py = current_pos[i][0].item(), current_pos[i][1].item()
                lx, ly = self.last_pos[i][0].item(), self.last_pos[i][1].item()
                dx = px - lx
                dy = py - ly
                
                # Assume 1 grid step per frame for extrapolation direction
                vx = 1 if dx > 0 else (-1 if dx < 0 else 0)
                vy = 1 if dy > 0 else (-1 if dy < 0 else 0)
                
                # Resolve diagonals
                if abs(dx) > abs(dy):
                    vy = 0
                else:
                    vx = 0
                velocities.append((vx, vy))
                
        self.last_pos = current_pos
        return velocities

def predict_ghost_occupancy(maze, ghost_pos, ghost_modes, velocities, K=5):
    danger_maps = []
    
    current_grid_pos = []
    for i in range(4):
        px, py = ghost_pos[i][0].item(), ghost_pos[i][1].item()
        current_grid_pos.append(list(pixel_to_grid(px, py)))
        
    for t in range(K):
        dmap = {}
        for i in range(4):
            mode = ghost_modes[i].item()
            if mode in [5, 6]:
                continue
                
            gx, gy = current_grid_pos[i]
            
            if t > 0:
                vx, vy = velocities[i]
                if vx != 0 or vy != 0:
                    nx, ny = gx + vx, gy + vy
                    if 0 <= ny < maze.shape[0] and 0 <= nx < maze.shape[1] and maze[ny, nx] == 0:
                        gx, gy = nx, ny
                        current_grid_pos[i] = [gx, gy]
                    
            if mode not in [3, 4]:
                for dx in range(-6, 7):
                    for dy in range(-6, 7):
                        dist = abs(dx) + abs(dy)
                        if dist <= 6:
                            cost = (7 - dist) * 20
                            pos = (gx + dx, gy + dy)
                            dmap[pos] = dmap.get(pos, 0) + cost
        danger_maps.append(dmap)
    return danger_maps

def spatiotemporal_astar(maze, start, targets, danger_maps):
    K = len(danger_maps)
    dmap0 = danger_maps[0]
    
    safe_targets = {t for t in targets if dmap0.get(t, 0) < 40}
    if not safe_targets:
        safe_targets = targets
        if not safe_targets: return []

    def h(pos):
        return min(abs(pos[0] - tx) + abs(pos[1] - ty) for tx, ty in safe_targets)

    # Priority queue: (f_score, cost, time, pos, path)
    queue = [(h(start), 0, 0, start, [start])]
    visited = {(0, start): 0}
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    while queue:
        _, cost, t, current, path = heapq.heappop(queue)
        
        if current in safe_targets:
            return path
            
        next_t = min(t + 1, K - 1)
        dmap_next = danger_maps[next_t]
        
        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= ny < maze.shape[0] and 0 <= nx < maze.shape[1] and maze[ny, nx] == 0:
                pos = (nx, ny)
                new_cost = cost + 1 + dmap_next.get(pos, 0)
                state_key = (next_t, pos)
                if state_key not in visited or new_cost < visited[state_key]:
                    visited[state_key] = new_cost
                    f_score = new_cost + h(pos)
                    heapq.heappush(queue, (f_score, new_cost, next_t, pos, path + [pos]))
    return []

def get_action_for_path(current_pixel, target_grid):
    cx, cy = current_pixel
    tx, ty = target_grid[0] * 4, target_grid[1] * 4
    dx = tx - cx
    dy = ty - cy
    if abs(dx) > abs(dy):
        return 2 if dx > 0 else 3
    elif abs(dy) > 0:
        return 4 if dy > 0 else 1
    return 0

def main():
    env = JaxPacman()
    step_fn = jax.jit(env.step)
    
    key = jax.random.PRNGKey(42)
    obs, state = env.reset(key)
    maze = MsPacmanMaze.MAZES[0]
    
    print("--- Starting Phase 2 Model-Based v0 Loop ---")
    
    total_reward = 0
    steps = 0
    done = False
    
    tracker = GhostTracker()
    
    pos_history = []
    oscillation_count = 0
    pellets_collected = 0
    
    # We will simulate 500 steps for direct comparison with ablations
    while not done and steps < 500:
        px, py = obs.player_position[0].item(), obs.player_position[1].item()
        start_grid = pixel_to_grid(px, py)
        
        # Oscillation check
        pos_history.append(start_grid)
        if len(pos_history) > 10:
            pos_history.pop(0)
        
        if pos_history.count(start_grid) >= 4:
            oscillation_count += 1
        
        targets = extract_targets(maze, obs.pellets)
        if not targets:
            print("No more targets!")
            break
            
        velocities = tracker.update(obs.ghost_positions)
        
        # Predict K=5 steps into the future
        danger_maps = predict_ghost_occupancy(maze, obs.ghost_positions, state.ghosts.modes, velocities, K=5)
        path = spatiotemporal_astar(maze, start_grid, targets, danger_maps)
        
        if path and len(path) > 1:
            action = get_action_for_path((px, py), path[1])
        else:
            action = 0
            
        obs, state, reward, done, info = step_fn(state, jnp.array(action, dtype=jnp.int32))
        r = reward.item()
        total_reward += r
        if r > 0 and r <= 50:
            pellets_collected += 1
        steps += 1
        
    print("\nRunning: Model-Based Planner (MB v0)")
    print(f"  Score: {total_reward}")
    print(f"  Pellets: {pellets_collected}")
    print(f"  Oscillations: {oscillation_count}")
    print(f"  Survival: {steps}")

if __name__ == '__main__':
    main()
