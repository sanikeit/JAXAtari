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

def compute_mf_danger_map(maze, ghosts_pos, ghosts_modes):
    danger_map = {}
    for i in range(4):
        mode = ghosts_modes[i].item()
        if mode in [5, 6]: continue
        px, py = ghosts_pos[i][0].item(), ghosts_pos[i][1].item()
        gx, gy = pixel_to_grid(px, py)
        if mode not in [3, 4]:
            for dx in range(-6, 7):
                for dy in range(-6, 7):
                    dist = abs(dx) + abs(dy)
                    if dist <= 6:
                        cost = (7 - dist) * 20
                        pos = (gx + dx, gy + dy)
                        danger_map[pos] = danger_map.get(pos, 0) + cost
    return danger_map

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
                vx = 1 if dx > 0 else (-1 if dx < 0 else 0)
                vy = 1 if dy > 0 else (-1 if dy < 0 else 0)
                if abs(dx) > abs(dy): vy = 0
                else: vx = 0
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
            if mode in [5, 6]: continue
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

def astar_search(maze, start, targets, danger_map):
    safe_targets = {t for t in targets if danger_map.get(t, 0) < 40}
    if not safe_targets: safe_targets = targets
    if not safe_targets: return []
    def h(pos): return min(abs(pos[0] - tx) + abs(pos[1] - ty) for tx, ty in safe_targets)
    queue = [(h(start), 0, start, [start])]
    visited = {start: 0}
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    while queue:
        _, cost, current, path = heapq.heappop(queue)
        if current in safe_targets: return path
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

def spatiotemporal_astar(maze, start, targets, danger_maps):
    K = len(danger_maps)
    dmap0 = danger_maps[0]
    safe_targets = {t for t in targets if dmap0.get(t, 0) < 40}
    if not safe_targets: safe_targets = targets
    if not safe_targets: return []
    def h(pos): return min(abs(pos[0] - tx) + abs(pos[1] - ty) for tx, ty in safe_targets)
    queue = [(h(start), 0, 0, start, [start])]
    visited = {(0, start): 0}
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    while queue:
        _, cost, t, current, path = heapq.heappop(queue)
        if current in safe_targets: return path
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
    if abs(dx) > abs(dy): return 2 if dx > 0 else 3
    elif abs(dy) > 0: return 4 if dy > 0 else 1
    return 0

def visualize_comparison(maze, start, mf_path, mb_path):
    out = []
    # Only print rows 10 to 35 to save terminal space
    for y in range(10, 35):
        row = []
        for x in range(maze.shape[1]):
            if (x, y) == start: row.append('P')
            elif mf_path and (x, y) in mf_path and mb_path and (x, y) in mb_path: row.append('X') # Shared
            elif mf_path and (x, y) in mf_path: row.append('F') # MF
            elif mb_path and (x, y) in mb_path: row.append('B') # MB
            elif maze[y, x] == 1: row.append('#')
            else: row.append(' ')
        out.append("".join(row))
    return "\n".join(out)

def main():
    env = JaxPacman()
    step_fn = jax.jit(env.step)
    key = jax.random.PRNGKey(42)
    obs, state = env.reset(key)
    maze = MsPacmanMaze.MAZES[0]
    
    # Pre-compile
    print("Compiling step_fn...")
    step_fn(state, jnp.array(0, dtype=jnp.int32))
    
    tracker = GhostTracker()
    
    differ_count = 0
    total_steps = 200
    
    print("--- Running Diagnostics ---")
    
    for steps in range(total_steps):
        px, py = obs.player_position[0].item(), obs.player_position[1].item()
        start_grid = pixel_to_grid(px, py)
        targets = extract_targets(maze, obs.pellets)
        
        if not targets: break
            
        velocities = tracker.update(obs.ghost_positions)
        
        # MF
        mf_danger = compute_mf_danger_map(maze, obs.ghost_positions, state.ghosts.modes)
        mf_path = astar_search(maze, start_grid, targets, mf_danger)
        
        # MB with K=15 prediction horizon
        mb_danger_maps = predict_ghost_occupancy(maze, obs.ghost_positions, state.ghosts.modes, velocities, K=15)
        mb_path = spatiotemporal_astar(maze, start_grid, targets, mb_danger_maps)
        
        mf_target = mf_path[-1] if mf_path else None
        mb_target = mb_path[-1] if mb_path else None
        
        mf_action = get_action_for_path((px, py), mf_path[1]) if (mf_path and len(mf_path) > 1) else 0
        mb_action = get_action_for_path((px, py), mb_path[1]) if (mb_path and len(mb_path) > 1) else 0
        
        # We define a differing plan if the path itself is different
        if mf_path != mb_path:
            differ_count += 1
            if differ_count <= 2:
                print(f"\n--- Difference at Step {steps} ---")
                print(f"MF Target: {mf_target}, MF Action: {mf_action}")
                print(f"MB Target: {mb_target}, MB Action: {mb_action}")
                print("Visualization (F=MF, B=MB, X=Shared Path):")
                print(visualize_comparison(maze, start_grid, mf_path, mb_path))
                
        # Drive env with MB action
        obs, state, reward, done, info = step_fn(state, jnp.array(mb_action, dtype=jnp.int32))
        if done: break
        
    print(f"\nDiagnostics Complete.")
    print(f"Plans differed in {differ_count} out of {total_steps} steps ({(differ_count/total_steps)*100:.1f}%).")
    
if __name__ == '__main__':
    main()
