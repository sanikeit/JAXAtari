import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import jax.numpy as jnp, numpy as np, os
import jaxatari
from jaxatari.games.jax_bankheist import JaxBankHeist

env = jaxatari.make("bankheist")

# 1. What does the game's own array contain, per city?
ccm = np.array(env.city_collision_maps)
print("city_collision_maps shape:", ccm.shape, "dtype:", ccm.dtype)
for i in range(ccm.shape[0]):
    u, c = np.unique(ccm[i], return_counts=True)
    print(f"  city {i}: uniques={dict(zip(u.tolist(), c.tolist()))}")

# 2. Where is the sprites dir, and does the raw npy have structure?
print("\nSPRITES_DIR:", env.consts.SPRITES_DIR)
f = os.path.join(env.consts.SPRITES_DIR, "map_1_collision.npy")
print("map_1_collision.npy exists?", os.path.exists(f))
if os.path.exists(f):
    raw = np.load(f)
    print("raw npy shape:", raw.shape, "dtype:", raw.dtype)
    print("raw[...,0] uniques:", np.unique(raw[...,0])[:10])