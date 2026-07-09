import sys
import os
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from jaxatari.games.jax_mspacman import JaxPacman
from project.planners.mf_baseline import MFPlanner
from project.utils.coords import pixel_to_dof, get_action_for_step

def test_traj():
    env = JaxPacman()
    step_fn = jax.jit(env.step)
    dof_maze = env.consts.DOF_MAZES[0]
    
    key = jax.random.PRNGKey(42)
    obs, state = env.reset(key)
    planner = MFPlanner(dof_maze)
    
    # Run 100 steps
    traj = []
    for _ in range(200):
        action = planner.plan(obs, state)
        obs, state, _, done, _ = step_fn(state, jnp.array(action, dtype=jnp.int32))
        traj.append({
            'action': action,
            'pos': (obs.player_position[0].item(), obs.player_position[1].item())
        })
        
    A_matches = 0
    B_matches = 0
    
    for i in range(1, len(traj)):
        p1 = traj[i-1]['pos']
        p2 = traj[i]['pos']
        act = traj[i-1]['action']
        
        if act == 0:
            continue
            
        # A
        gx_a1, gy_a1 = p1[0] // 4, p1[1] // 4
        gx_a2, gy_a2 = p2[0] // 4, p2[1] // 4
        
        # B
        gx_b1, gy_b1 = (p1[0] + 5) // 4, (p1[1] + 3) // 4
        gx_b2, gy_b2 = (p2[0] + 5) // 4, (p2[1] + 3) // 4
        
        act_A = get_action_for_step((gx_a1, gy_a1), (gx_a2, gy_a2))
        act_B = get_action_for_step((gx_b1, gy_b1), (gx_b2, gy_b2))
        
        if gx_a1 != gx_a2 or gy_a1 != gy_a2:
            if act_A == act: A_matches += 1
            else: A_matches -= 1
            
        if gx_b1 != gx_b2 or gy_b1 != gy_b2:
            if act_B == act: B_matches += 1
            else: B_matches -= 1

    print(f"Action consistency score (A): {A_matches}")
    print(f"Action consistency score (B): {B_matches}")
    
if __name__ == "__main__":
    test_traj()
