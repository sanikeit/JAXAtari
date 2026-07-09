"""
utils/astar.py

Spatiotemporal A* planner for the MsPacman project.
Operates in DOF grid coordinates.
"""
import heapq
from .coords import is_walkable


def spatiotemporal_astar(dof, start, targets, danger_maps, reversal_pos=None, reversal_penalty=0):
    """
    Find the lowest-cost path from start to the nearest safe target,
    using a temporally-indexed danger map at each expansion depth.
    
    This is the core innovation over plain A*: at expansion depth t,
    the cost of entering cell (gx, gy) uses danger_maps[t][(gx, gy)],
    so the planner naturally avoids predicted ghost positions in the
    future frame where it would actually be at that cell.
    
    Args:
        dof:         consts.DOF_MAZES[0] — shape (40, 44, 4)
        start:       (gx, gy) in DOF grid
        targets:     set of (gx, gy) DOF grid positions (pellet locations)
        danger_maps: list[K] of dicts {(gx, gy): cost}
                     — built by utils.danger_map.build_danger_maps()
    
    Returns:
        list of (gx, gy) tuples from start to chosen target (inclusive),
        or [] if no path found.
    
    Notes on safe_target selection:
        - A target is "safe" if its danger cost at t=0 < 30.
        - If no safe target exists, falls back to all targets.
        - This prevents the planner from routing directly into a ghost.
    
    Notes on complexity:
        - State space: K × |walkable cells| ≈ 15 × 1730 = 25,950 states
        - Typically terminates in < 1 ms on CPU for K=15.
    """
    K = len(danger_maps)
    safe = {t for t in targets if danger_maps[0].get(t, 0) < 30}
    if not safe:
        safe = targets
    if not safe:
        return []

    def h(pos):
        return min(abs(pos[0] - t[0]) + abs(pos[1] - t[1]) for t in safe)

    queue = [(h(start), 0, 0, start, [start])]
    visited = {(0, start): 0}
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    while queue:
        _, cost, t, cur, path = heapq.heappop(queue)
        if cur in safe:
            return path
        nt = min(t + 1, K - 1)
        dn = danger_maps[nt]
        for dx, dy in neighbors:
            nx, ny = cur[0] + dx, cur[1] + dy
            if is_walkable(dof, nx, ny):
                pos = (nx, ny)
                nc = cost + 1 + dn.get(pos, 0)
                if len(path) == 1 and pos == reversal_pos:
                    nc += reversal_penalty
                sk = (nt, pos)
                if sk not in visited or nc < visited[sk]:
                    visited[sk] = nc
                    heapq.heappush(queue,
                                   (nc + h(pos), nc, nt, pos, path + [pos]))
    return []
