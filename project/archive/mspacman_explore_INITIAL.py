import jax
from jaxatari.games.jax_mspacman import JaxPacman
from jaxatari.games.mspacman_mazes import MsPacmanMaze

def explore_mspacman():
    # 1. Create the environment
    print("--- Creating MsPacman Environment ---")
    env = JaxPacman()
    
    # 2. Reset the environment
    key = jax.random.PRNGKey(42)
    obs, state = env.reset(key)
    
    # 3. Print the structure of the observation
    print("\n--- Observation Structure ---")
    print(f"Observation type: {type(obs).__name__}")
    
    print("\nAvailable object-centric information:")
    print(f"Player Position (x, y): {obs.player_position}")
    print(f"Player Action: {obs.player_action}")
    
    print(f"Ghost Positions:\\n{obs.ghost_positions}")
    print(f"Ghost Actions: {obs.ghost_actions}")
    
    print(f"Fruit Position: {obs.fruit_position}, Type: {obs.fruit_type}")
    
    print(f"Pellets Grid Shape: {obs.pellets.shape}")
    print(f"Power Pellets State: {obs.power_pellets}")
    
    # 4. Print information about the Maze
    print("\n--- Maze \u0026 Internal State Information ---")
    print(f"Current Level ID: {state.level.id}")
    
    # Extract the true maze layout for this level
    maze_layouts = MsPacmanMaze.MAZES
    print(f"Total mazes available in codebase: {len(maze_layouts)}")
    print(f"Maze 0 layout shape (Graph Grid): {maze_layouts[0].shape}")
    print(f"Tile Scale factor (Pixels per Grid Cell): {MsPacmanMaze.TILE_SCALE}")
    
    # Ghost internal modes (useful for planning)
    # Ghost modes: 0: RANDOM, 1: CHASE, 2: SCATTER, 3: FRIGHTENED, etc.
    print(f"\nGhost Modes (Internal State): {state.ghosts.modes}")

if __name__ == "__main__":
    explore_mspacman()
