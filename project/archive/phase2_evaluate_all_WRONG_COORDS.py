import jax
import jax.numpy as jnp
import heapq
import random
from jaxatari.games.jax_mspacman import JaxPacman
from jaxatari.games.mspacman_mazes import MsPacmanMaze

def pixel_to_grid(pixel_x, pixel_y):
    return int(round(pixel_x / 4.0)), int(round(pixel_y / 4.0))

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

def get_pacman_offset(px, py, pdir, tiles):
    if pdir == 1: return px - tiles, py - tiles # UP (arcade bug)
    if pdir == 4: return px, py + tiles # DOWN
    if pdir == 3: return px - tiles, py # LEFT
    if pdir == 2: return px + tiles, py # RIGHT
    return px, py

def predict_mb_v0_centers(maze, ghost_pos, velocities, K=16):
    grids_over_time = []
    current_grid_pos = [list(pixel_to_grid(ghost_pos[i][0].item(), ghost_pos[i][1].item())) for i in range(4)]
    
    for t in range(K):
        grids_over_time.append([list(p) for p in current_grid_pos])
        for i in range(4):
            vx, vy = velocities[i]
            if vx != 0 or vy != 0:
                gx, gy = current_grid_pos[i]
                nx, ny = gx + vx, gy + vy
                if 0 <= ny < maze.shape[0] and 0 <= nx < maze.shape[1] and maze[ny, nx] == 0:
                    current_grid_pos[i] = [nx, ny]
    return grids_over_time

def predict_mb_v1_centers(maze, ghost_pos, ghost_modes, ghost_types, ghost_actions, pacman_pos, pacman_action, K=16):
    grids_over_time = []
    px, py = pixel_to_grid(pacman_pos[0].item(), pacman_pos[1].item())
    pdir = pacman_action.item()
    
    current_grid_pos = [list(pixel_to_grid(ghost_pos[i][0].item(), ghost_pos[i][1].item())) for i in range(4)]
    current_dirs = [ghost_actions[i].item() for i in range(4)]
    
    scatter_targets = {0: (39, 0), 1: (0, 0), 2: (39, 44), 3: (0, 44)}
    
    for t in range(K):
        grids_over_time.append([list(p) for p in current_grid_pos])
        bx, by = current_grid_pos[0]
        
        for i in range(4):
            mode = ghost_modes[i].item()
            gtype = ghost_types[i].item()
            if mode in [5, 6]: continue
            
            gx, gy = current_grid_pos[i]
            gdir = current_dirs[i]
            
            tx, ty = None, None
            if mode == 1:
                if gtype == 0: tx, ty = px, py
                elif gtype == 1: tx, ty = get_pacman_offset(px, py, pdir, 4)
                elif gtype == 2:
                    ox, oy = get_pacman_offset(px, py, pdir, 2)
                    tx, ty = ox + (ox - bx), oy + (oy - by)
                else:
                    if abs(gx - px) + abs(gy - py) > 8: tx, ty = px, py
                    else: tx, ty = scatter_targets[gtype]
            elif mode == 2:
                tx, ty = scatter_targets[gtype]
                
            directions = [(0, -1, 1), (1, 0, 2), (-1, 0, 3), (0, 1, 4)]
            reverse_dirs = {1:4, 4:1, 2:3, 3:2, 0:0}
            
            valid_moves = []
            for dx, dy, adir in directions:
                if adir == reverse_dirs.get(gdir, 0): continue
                nx, ny = gx + dx, gy + dy
                if 0 <= ny < maze.shape[0] and 0 <= nx < maze.shape[1] and maze[ny, nx] == 0:
                    valid_moves.append((nx, ny, adir))
                    
            if not valid_moves:
                for dx, dy, adir in directions:
                    nx, ny = gx + dx, gy + dy
                    if 0 <= ny < maze.shape[0] and 0 <= nx < maze.shape[1] and maze[ny, nx] == 0:
                        valid_moves.append((nx, ny, adir))
                        
            if valid_moves:
                if tx is None:
                    nx, ny, adir = random.choice(valid_moves)
                    current_grid_pos[i] = [nx, ny]
                    current_dirs[i] = adir
                else:
                    best_dist = 999999
                    best_pos = None
                    best_dir = gdir
                    for nx, ny, adir in valid_moves:
                        dist = (nx - tx)**2 + (ny - ty)**2
                        if dist < best_dist:
                            best_dist = dist
                            best_pos = [nx, ny]
                            best_dir = adir
                    if best_pos:
                        current_grid_pos[i] = best_pos
                        current_dirs[i] = best_dir
                        
    return grids_over_time

def evaluate_predictions():
    env = JaxPacman()
    step_fn = jax.jit(env.step)
    key = jax.random.PRNGKey(42)
    obs, state = env.reset(key)
    maze = MsPacmanMaze.MAZES[0]
    
    # Warmup
    print("Warming up prediction evaluation...")
    step_fn(state, jnp.array(0, jnp.int32))
    for _ in range(50):
        obs, state, _, _, _ = step_fn(state, jnp.array(0, jnp.int32))
        
    tracker = GhostTracker()
    tracker.update(obs.ghost_positions)
    obs, state, _, _, _ = step_fn(state, jnp.array(0, jnp.int32))
    
    eval_horizons = [1, 3, 5, 10, 15]
    env_frames = [h * 4 for h in eval_horizons]
    
    v0_errors = {h: [] for h in eval_horizons}
    v1_errors = {h: [] for h in eval_horizons}
    
    print("\n--- Ghost Prediction Evaluation ---")
    
    for sample in range(20):
        velocities = tracker.update(obs.ghost_positions)
        
        mb0_grids = predict_mb_v0_centers(maze, obs.ghost_positions, velocities, K=16)
        mb1_grids = predict_mb_v1_centers(maze, obs.ghost_positions, state.ghosts.modes, state.ghosts.types, state.ghosts.actions, obs.player_position, obs.player_action, K=16)
        
        saved_state = state
        saved_obs = obs
        
        actual_positions = {}
        for f in range(1, 61):
            obs, state, _, _, _ = step_fn(state, jnp.array(0, jnp.int32))
            if f in env_frames:
                actual_positions[f] = [list(pixel_to_grid(obs.ghost_positions[i][0].item(), obs.ghost_positions[i][1].item())) for i in range(4)]
                
        # Compare
        for h, f in zip(eval_horizons, env_frames):
            actual = actual_positions[f]
            p0 = mb0_grids[h]
            p1 = mb1_grids[h]
            
            for i in range(4):
                e0 = abs(actual[i][0] - p0[i][0]) + abs(actual[i][1] - p0[i][1])
                e1 = abs(actual[i][0] - p1[i][0]) + abs(actual[i][1] - p1[i][1])
                v0_errors[h].append(e0)
                v1_errors[h].append(e1)
                
        state = saved_state
        obs = saved_obs
        for _ in range(20):
            obs, state, _, _, _ = step_fn(state, jnp.array(0, jnp.int32))
            tracker.update(obs.ghost_positions)
            
    print(f"{'Horizon (K)':<15} | {'MB v0 Error (L1 cells)':<25} | {'MB v1 Error (L1 cells)':<25}")
    print("-" * 70)
    for h in eval_horizons:
        e0_avg = sum(v0_errors[h]) / len(v0_errors[h])
        e1_avg = sum(v1_errors[h]) / len(v1_errors[h])
        print(f"K={h:<13} | {e0_avg:<25.2f} | {e1_avg:<25.2f}")


# ---------------- PLANNING EVALUATION ----------------
def extract_targets(maze, pellets):
    targets = set()
    for py in range(pellets.shape[1]):
        for px in range(pellets.shape[0]):
            if pellets[px, py] == 1:
                gx, gy = pixel_to_grid(px * 8 + 5, py * 12 + 6)
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

def create_danger_maps_from_centers(maze, centers, K=16):
    maps = []
    for t in range(K):
        dmap = {}
        for i in range(4):
            gx, gy = centers[t][i]
            # Use tighter radius for MB to prevent washing out signal
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    dist = abs(dx) + abs(dy)
                    if dist <= 4:
                        cost = (5 - dist) * 20
                        dmap[(gx + dx, gy + dy)] = dmap.get((gx + dx, gy + dy), 0) + cost
        maps.append(dmap)
    return maps

def spatiotemporal_astar(maze, start, targets, danger_maps):
    K = len(danger_maps)
    dmap0 = danger_maps[0]
    safe_targets = {t for t in targets if dmap0.get(t, 0) < 30}
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
    dx, dy = tx - cx, ty - cy
    if abs(dx) > abs(dy): return 2 if dx > 0 else 3
    elif abs(dy) > 0: return 4 if dy > 0 else 1
    return 0

def run_planning_comparison():
    print("\n--- Planning Comparison (MF vs MBv0 vs MBv1) ---")
    env = JaxPacman()
    step_fn = jax.jit(env.step)
    maze = MsPacmanMaze.MAZES[0]
    
    planners = ['MF', 'MBv0', 'MBv1']
    results = {}
    
    for planner in planners:
        key = jax.random.PRNGKey(101)
        obs, state = env.reset(key)
        tracker = GhostTracker()
        
        score, pellets, osc = 0, 0, 0
        last_positions = []
        survival = 0
        
        for steps in range(300):
            survival += 1
            px, py = obs.player_position[0].item(), obs.player_position[1].item()
            start_grid = pixel_to_grid(px, py)
            targets = extract_targets(maze, obs.pellets)
            if not targets: break
            
            velocities = tracker.update(obs.ghost_positions)
            
            mb0_centers = predict_mb_v0_centers(maze, obs.ghost_positions, velocities, K=16)
            
            if planner == 'MF':
                # MF is just K=1 prediction (static) extended across time
                mf_centers = [mb0_centers[0] for _ in range(16)]
                dmaps = create_danger_maps_from_centers(maze, mf_centers, K=16)
            elif planner == 'MBv0':
                dmaps = create_danger_maps_from_centers(maze, mb0_centers, K=16)
            else:
                mb1_centers = predict_mb_v1_centers(maze, obs.ghost_positions, state.ghosts.modes, state.ghosts.types, state.ghosts.actions, obs.player_position, obs.player_action, K=16)
                dmaps = create_danger_maps_from_centers(maze, mb1_centers, K=16)
                
            path = spatiotemporal_astar(maze, start_grid, targets, dmaps)
            action = get_action_for_path((px, py), path[1]) if path and len(path) > 1 else 0
            
            last_positions.append(start_grid)
            if len(last_positions) > 10: last_positions.pop(0)
            if last_positions.count(start_grid) > 3: osc += 1
                
            obs, state, reward, done, info = step_fn(state, jnp.array(action, dtype=jnp.int32))
            score = state.score.item()
            pellets = state.level.collected_pellets.item()
            if done: break
            
        results[planner] = {'score': score, 'pellets': pellets, 'osc': osc, 'survival': survival}
        print(f"Finished {planner}: Score={score}, Pellets={pellets}, Osc={osc}, Survival={survival}")

if __name__ == '__main__':
    evaluate_predictions()
    run_planning_comparison()
