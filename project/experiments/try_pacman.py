import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import jax
from jaxatari.games.jax_pacman import JaxPacman as PacmanEnv, PacmanConstants
from project.utils.coords import extract_pellet_targets, is_walkable, pixel_to_dof

consts = PacmanConstants(RESET_LEVEL=1)   # maze 0
env = PacmanEnv(consts)
obs, state = env.reset(jax.random.PRNGKey(42))

# Pac-Man stores DOF on the level state, not on consts
dof = state.level.dofmaze

print("Level id:", state.level.id)
print("Player pos:", obs.player_position)
print("Pellets in obs:", int(obs.pellets.sum()), " shape:", obs.pellets.shape)
print("DOF shape:", dof.shape)

targets = extract_pellet_targets(dof, obs.pellets)
print("Targets extracted:", len(targets))
bad = [t for t in targets if not is_walkable(dof, t[0], t[1])]
print("Targets INSIDE walls (should be 0):", len(bad))

px, py = obs.player_position[0].item(), obs.player_position[1].item()
start = pixel_to_dof(px, py)
print("Player DOF cell:", start, " walkable?", is_walkable(dof, *start))