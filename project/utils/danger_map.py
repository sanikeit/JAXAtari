"""
utils/danger_map.py

Danger map construction for the MsPacman A* planner.
Operates entirely in DOF grid coordinates.
"""
from .coords import pixel_to_dof


def build_danger_maps(predicted_pixel_positions, ghost_modes, radius=3, mult=5):
    """
    Build K danger maps from a K-step ghost position forecast.
    
    Args:
        predicted_pixel_positions: list[K] of (4, 2) numpy arrays in PIXEL coords
        ghost_modes: array-like of 4 ghost mode integers
        radius: Manhattan-distance radius around each ghost center
        mult: cost multiplier (cost = (radius+1 - dist) * mult at each cell)
    
    Returns:
        list[K] of dicts mapping (gx, gy) -> danger_cost in DOF grid space
    
    Notes:
        - Ghost modes 5 (RETURNING) and 6 (ENJAILED) are harmless — skipped.
        - At radius=3, mult=5: max cost per ghost = 5+4+3+2+1 = 15 per cell.
        - A* treats any cell with total danger >= 30 as "unsafe target".
        - Empirically best: radius=3, mult=5. Below radius=2 the signal
          is too weak to avoid ghosts. Above radius=6 it blocks corridors.
    """
    dmaps = []
    for pos4 in predicted_pixel_positions:
        dmap = {}
        for i in range(4):
            mode = int(ghost_modes[i])
            if mode in (5, 6):        # RETURNING or ENJAILED — not dangerous
                continue
            gx, gy = pixel_to_dof(int(pos4[i][0]), int(pos4[i][1]))
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    d = abs(dx) + abs(dy)
                    if d <= radius:
                        key = (gx + dx, gy + dy)
                        dmap[key] = dmap.get(key, 0) + (radius + 1 - d) * mult
        dmaps.append(dmap)
    return dmaps


def static_danger_map(ghost_pixel_positions, ghost_modes, K, radius=3, mult=5):
    """
    Build K identical danger maps from current ghost positions (MF-style).
    
    Args:
        ghost_pixel_positions: (4, 2) array of current ghost pixel positions
        ghost_modes: array-like of 4 ghost mode integers
        K: number of steps to replicate
        radius, mult: same as build_danger_maps
    
    Returns:
        list[K] of identical dicts
    """
    import numpy as np
    cur = np.array([[ghost_pixel_positions[i][0].item(),
                     ghost_pixel_positions[i][1].item()] for i in range(4)])
    d0 = build_danger_maps([cur], ghost_modes, radius, mult)[0]
    return [d0] * K
