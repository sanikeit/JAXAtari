#!/usr/bin/env python3
"""Minimal inspector for MsPacman observations and maze data.

Creates a MsPacman environment, prints the observation structure and
object information, and shows the maze/grid representation suitable
for A* planning (tile grid + DOF grid sample).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path


def _add_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"
    for candidate in (src_dir, repo_root):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


def _bootstrap_local_package() -> None:
    """Register a minimal `jaxatari` package so imports skip `__init__.py`."""
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"
    package_dir = src_dir / "jaxatari"
    _add_src_to_path()

    if "jaxatari" not in sys.modules:
        package = types.ModuleType("jaxatari")
        package.__path__ = [str(package_dir)]
        package.check_ownership = lambda: None
        sys.modules["jaxatari"] = package
    else:
        package = sys.modules["jaxatari"]
        if not hasattr(package, "check_ownership"):
            package.check_ownership = lambda: None


_bootstrap_local_package()

from jaxatari.core import make
from jaxatari.games.mspacman_mazes import MsPacmanMaze
from jaxatari.games.jax_mspacman import get_level_maze


def main() -> None:
    env = make("mspacman")

    # Reset returns (observation, state)
    observation, state = env.reset()

    print("Observation type:", type(observation))
    print("Available observation fields and example values:")
    print(" - player_position:", observation.player_position)
    print(" - ghost_positions:", observation.ghost_positions)
    print(" - pellets shape:", observation.pellets.shape, "dtype:", observation.pellets.dtype)
    print(" - power_pellets:", observation.power_pellets)

    # Maze / level info
    maze_idx = get_level_maze(state.level.id)
    mazes = MsPacmanMaze.MAZES
    tile_scale = MsPacmanMaze.TILE_SCALE
    dof_mazes = MsPacmanMaze.get_dof_mazes()

    print()
    print("Maze index for current level:", int(maze_idx))
    print("MAZES shape (levels, height, width):", mazes.shape)
    print("TILE_SCALE (pixels per logical tile):", int(tile_scale))
    print("DOF_MAZES shape (levels, width, height, 4 (U,R,L,D)):", dof_mazes.shape)

    print("\nSample logical maze (walkable=1, wall=0) for current level:")
    # Print as ints for readability
    print(mazes[int(maze_idx)].astype(int))

    # Map player pixel position -> DOF index and show DOF flags
    px, py = observation.player_position
    dof_x = (int(px) + 5) // tile_scale
    dof_y = (int(py) + 3) // tile_scale
    print()
    print("Player pixel position:", observation.player_position)
    print("Mapped DOF index (x, y):", (int(dof_x), int(dof_y)))
    print("DOF flags at that index (up, right, left, down):", dof_mazes[int(maze_idx)][int(dof_x), int(dof_y)])

    print("\nPellet grid (18 x 14) — 1 indicates pellet present:")
    print(observation.pellets.astype(int))


if __name__ == "__main__":
    main()
