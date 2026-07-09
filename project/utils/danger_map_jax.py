"""
utils/danger_map_jax.py

Danger map construction using JAXAtari-native coordinates.
"""
from .coords_jax import pixel_to_dof

def build_danger_maps(predicted_pixel_positions, ghost_modes, radius=3, mult=5):
    dmaps = []
    for pos4 in predicted_pixel_positions:
        dmap = {}
        for i in range(4):
            mode = int(ghost_modes[i])
            if mode in (5, 6):
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
    import numpy as np
    cur = np.array([[ghost_pixel_positions[i][0].item(),
                     ghost_pixel_positions[i][1].item()] for i in range(4)])
    d0 = build_danger_maps([cur], ghost_modes, radius, mult)[0]
    return [d0] * K
