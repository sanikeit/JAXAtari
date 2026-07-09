"""
utils/coords.py

Coordinate system utilities for the MsPacman A* planner project.

COORDINATE SYSTEM REFERENCE
============================
JAXAtari uses THREE different coordinate spaces. Confusing them was the root
cause of all Phase 1 / early Phase 2 bugs.

1. PIXEL SPACE
   - Range: x in [0,159], y in [0,175]
   - Used by: obs.player_position, obs.ghost_positions, state.ghosts.positions
   - Movement: 1 pixel per frame (4 frames to cross one DOF cell)

2. DOF GRID SPACE  (the correct navigation grid)
   - Range: gx in [0,39], gy in [0,43]
   - Used by: consts.DOF_MAZES[0]  ->  dof[gx, gy] = bool[4] allowed directions
   - Conversion: gx = pixel_x // 4,  gy = pixel_y // 4
   - Cell is walkable if: dof[gx, gy].any()

3. PELLET GRID SPACE
   - Range: pellx in [0,17], pelly in [0,13]
   - Used by: obs.pellets, state.level.pellets
   - Conversion to DOF: gx = pellx * 2 + 1,  gy = pelly * 3 + 2
   - Verified: 100% of 236 pellets land on walkable DOF cells

4. MsPacmanMaze.MAZES  (DO NOT USE FOR NAVIGATION)
   - Shape (44, 40), indexed [y, x] — DIFFERENT axis order from DOF!
   - This is a display/rendering grid, NOT the navigation grid.
   - Using MAZES[0] for A* was Bug A that invalidated all Phase 1 results.

CONFIRMED ACTION MAPPING  (verified empirically — do not change)
=================================================================
  0 = NOOP
  2 = UP    (pixel_y decreases)
  3 = RIGHT (pixel_x increases)
  4 = LEFT  (pixel_x decreases)
  5 = DOWN  (pixel_y increases)
"""


def pixel_to_dof(px, py):
    """Convert pixel coordinates to DOF grid coordinates."""
    return int(px) // 4, int(py) // 4


def pellet_to_dof(pellx, pelly):
    """
    Convert pellet grid index to DOF grid coordinates.
    
    Derived from the relationship between obs.pellets (18x14) and
    DOF grid (40x44):  gx = pellx*2+1,  gy = pelly*3+2
    Empirically verified: 100% of 236 pellets map to walkable cells.
    """
    return pellx * 2 + 1, pelly * 3 + 2


def is_walkable(dof, gx, gy):
    """Return True if DOF grid cell (gx, gy) is a navigable path cell."""
    return (0 <= gx < dof.shape[0] and
            0 <= gy < dof.shape[1] and
            bool(dof[gx, gy].any()))


def extract_pellet_targets(dof, pellets):
    """
    Return a set of (gx, gy) DOF grid positions for all live pellets.
    
    Args:
        dof: consts.DOF_MAZES[0]  — shape (40, 44, 4)
        pellets: obs.pellets or state.level.pellets  — shape (18, 14)
    
    Returns:
        set of (gx, gy) tuples in DOF grid space
    """
    targets = set()
    for pelly in range(pellets.shape[1]):
        for pellx in range(pellets.shape[0]):
            if pellets[pellx, pelly]:
                gx, gy = pellet_to_dof(pellx, pelly)
                if is_walkable(dof, gx, gy):
                    targets.add((gx, gy))
    return targets


def get_action_for_step(cur_dof, next_dof):
    """
    Convert a one-step move in DOF grid space to a JAXAtari action index.
    
    Args:
        cur_dof:  (gx, gy) current position
        next_dof: (gx, gy) next position (must be adjacent)
    
    Returns:
        int: JAXAtari action (2=UP, 3=RIGHT, 4=LEFT, 5=DOWN, 0=NOOP)
    """
    dx = next_dof[0] - cur_dof[0]
    dy = next_dof[1] - cur_dof[1]
    if abs(dx) > abs(dy):
        return 3 if dx > 0 else 4   # RIGHT or LEFT
    elif abs(dy) > 0:
        return 5 if dy > 0 else 2   # DOWN or UP
    return 0


def get_action_for_path(cur_dof, path):
    """
    Return the first action to take along a planned path.
    
    Args:
        cur_dof: (gx, gy) current position in DOF grid
        path: list of (gx, gy) tuples from A* — path[0] == cur_dof
    
    Returns:
        int: JAXAtari action
    """
    if len(path) < 2:
        return 0
    return get_action_for_step(cur_dof, path[1])
