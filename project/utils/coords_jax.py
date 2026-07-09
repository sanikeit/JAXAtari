"""
utils/coords_jax.py

Coordinate system utilities matching JAXAtari's internal calculations.
Using: gx = (x + 5) // 4, gy = (y + 3) // 4
"""

def pixel_to_dof(px, py):
    """Convert pixel coordinates to DOF grid coordinates using JAX offset."""
    return (int(px) + 5) // 4, (int(py) + 3) // 4

def pellet_to_dof(pellx, pelly):
    """
    Convert pellet grid index to DOF grid coordinates.
    Derived from render_pellets: px = pellx*8+8 (approx), py = pelly*8+24.
    Resulting in JAX DOF: gx = pellx*2+3, gy = pelly*2+6
    """
    px = pellx * 8 + 5
    if pellx >= 9:
        px += 4
    py = pelly * 12 + 6
    return (px + 5) // 4, (py + 3) // 4

def is_walkable(dof, gx, gy):
    """Return True if DOF grid cell (gx, gy) is a navigable path cell."""
    return (0 <= gx < dof.shape[0] and
            0 <= gy < dof.shape[1] and
            bool(dof[gx, gy].any()))

def extract_pellet_targets(dof, pellets):
    """Return a set of (gx, gy) DOF grid positions for all live pellets."""
    targets = set()
    for pelly in range(pellets.shape[1]):
        for pellx in range(pellets.shape[0]):
            if pellets[pellx, pelly]:
                gx, gy = pellet_to_dof(pellx, pelly)
                if is_walkable(dof, gx, gy):
                    targets.add((gx, gy))
    return targets

def get_action_for_step(cur_dof, next_dof):
    """Convert a one-step move in DOF grid space to a JAXAtari action index."""
    dx = next_dof[0] - cur_dof[0]
    dy = next_dof[1] - cur_dof[1]
    if abs(dx) > abs(dy):
        return 3 if dx > 0 else 4   # RIGHT or LEFT
    elif abs(dy) > 0:
        return 5 if dy > 0 else 2   # DOWN or UP
    return 0

def get_action_for_path(cur_dof, path):
    """Return the first action to take along a planned path."""
    if len(path) < 2:
        return 0
    return get_action_for_step(cur_dof, path[1])
