import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np, os
import jaxatari
env = jaxatari.make("bankheist")
f = os.path.join(env.consts.SPRITES_DIR, "map_1_collision.npy")
raw = np.load(f)
print("shape:", raw.shape)
for ch in range(raw.shape[-1]):
    u, c = np.unique(raw[..., ch], return_counts=True)
    frac0 = float((raw[..., ch] == 0).mean())
    print(f"channel {ch}: uniques={dict(zip(u.tolist()[:6], c.tolist()[:6]))}  fraction==0: {frac0:.3f}")