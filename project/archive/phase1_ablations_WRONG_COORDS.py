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
        if mode in [5, 6]:
            continue
        px, py = ghosts_pos[i][0].item(), ghosts_pos[i][1].item()
        gx, gy = pixel_to_grid(px, py)
        if mode in [3, 4]:
            pass
        else:
            for dx in range(-6, 7):
                for dy in range(-6, 7):
                    dist = abs(dx) + abs(dy)
                    if dist <= 6:
                        cost = (7 - dist) * 20
                        pos = (gx + dx, gy + dy)
                        danger_map[pos] = danger_map.get(pos, 0) + cost
    return danger_map

def astar_search(maze, start, targets, danger_map):
    safe_targets = {t for t in targets if danger_map.get(t, 0) < 40}
    if not safe_targets:
        safe_targets = targets
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
    if abs(dx) > abs(dy):
        return 2 if dx > 0 else 3
    elif abs(dy) > 0:
        return 4 if dy > 0 else 1
    return 0

def run_simulation(env, step_fn, maze, replan_rate=1, commit_steps=1, max_steps=500):
    key = jax.random.PRNGKey(42)
    obs, state = env.reset(key)
    
    total_reward = 0
    steps = 0
    done = False
    
    current_target = None
    target_commitment = 0
    path = []
    
    pos_history = []
    oscillation_count = 0
    pellets_collected = 0
    
    while not done and steps < max_steps:
        px, py = obs.player_position[0].item(), obs.player_position[1].item()
        start_grid = pixel_to_grid(px, py)
        
        # Oscillation check
        pos_history.append(start_grid)
        if len(pos_history) > 10:
            pos_history.pop(0)
        
        if pos_history.count(start_grid) >= 4:
            oscillation_count += 1
            
        danger_map = compute_danger_map(maze, obs.ghost_positions, state.ghosts.modes)
        targets = extract_targets(maze, obs.pellets)
        if not targets:
            break
            
        # 1. Goal Commitment Check
        if current_target is not None:
            if current_target not in targets or danger_map.get(current_target, 0) > 40:
                current_target = None
                target_commitment = 0
            else:
                target_commitment -= 1
                if target_commitment <= 0:
                    current_target = None
                    
        # 2. Replan Check
        if steps % replan_rate == 0 or not path:
            if current_target is None:
                path = astar_search(maze, start_grid, targets, danger_map)
                if path and len(path) > 1:
                    current_target = path[-1]
                    target_commitment = commit_steps
            else:
                path = astar_search(maze, start_grid, {current_target}, danger_map)
                if not path:
                    current_target = None
                    path = astar_search(maze, start_grid, targets, danger_map)
                    if path and len(path) > 1:
                        current_target = path[-1]
                        target_commitment = commit_steps
                    
        if path and len(path) > 1:
            if start_grid in path:
                idx = path.index(start_grid)
                if idx + 1 < len(path):
                    next_grid = path[idx + 1]
                else:
                    next_grid = start_grid
            else:
                next_grid = path[1]
            action = get_action_for_path((px, py), next_grid)
        else:
            action = 0
            
        obs, state, reward, done, info = step_fn(state, jnp.array(action, dtype=jnp.int32))
        r = reward.item()
        total_reward += r
        if r > 0 and r <= 50:
            pellets_collected += 1
        steps += 1
        
    return {
        "score": total_reward,
        "pellets": pellets_collected,
        "oscillations": oscillation_count,
        "survival_time": steps
    }

def main():
    env = JaxPacman()
    step_fn = jax.jit(env.step)
    maze = MsPacmanMaze.MAZES[0]
    
    # Pre-compile
    print("Compiling step_fn...")
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    step_fn(state, jnp.array(0, dtype=jnp.int32))
    
    configs = [
        {"name": "Pure MF Baseline", "replan_rate": 1, "commit_steps": 1},
        {"name": "Goal Commitment (15 steps)", "replan_rate": 1, "commit_steps": 15},
        {"name": "Reduced Replanning (Every 5 steps)", "replan_rate": 5, "commit_steps": 1},
        {"name": "Combined Ablation", "replan_rate": 5, "commit_steps": 15},
    ]
    
    for cfg in configs:
        print(f"\nRunning: {cfg['name']}")
        res = run_simulation(env, step_fn, maze, 
                             replan_rate=cfg["replan_rate"], 
                             commit_steps=cfg["commit_steps"], 
                             max_steps=500)
        print(f"  Score: {res['score']}")
        print(f"  Pellets: {res['pellets']}")
        print(f"  Oscillations: {res['oscillations']}")
        print(f"  Survival: {res['survival_time']}")

if __name__ == '__main__':
    main()
