import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import jax, jax.numpy as jnp, numpy as np
import jaxatari

env = jaxatari.make("bankheist")
obs, state = env.reset(jax.random.PRNGKey(0))
step_fn = jax.jit(env.step)

for step in range(6):
    # dump ALL attributes of obs.banks
    b = obs.banks
    attrs = {a: np.array(getattr(b, a)).tolist() for a in dir(b)
             if not a.startswith("_") and hasattr(getattr(b, a), "shape")}
    print(f"step {step}: banks attrs = {attrs}")
    # internal truth
    print(f"         internal bank visibility = {np.array(state.bank_positions.visibility).tolist()}, "
          f"positions = {np.array(state.bank_positions.position).tolist()}")
    obs, state, _, done, _ = step_fn(state, jnp.array(0, dtype=jnp.int32))