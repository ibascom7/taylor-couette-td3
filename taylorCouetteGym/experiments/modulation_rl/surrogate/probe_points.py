"""Fidelity probes: surrogate static rollouts vs known real-CFD numbers."""
import numpy as np
from surrogate_env import KNN, scan_static
data = np.load("dataset.npz")
knn = KNN(data["feats"], data["targs"])

def static_return(D, W, T):
    x, ret, r = 0.0, 0.0, 0.0
    whi = (300.0 - (1.0 - D) * W) / D
    for _ in range(5):
        xh, ph = knn.predict([whi, W, np.log(T), x])
        r = xh - ph / 31.94
        ret += r
        x = xh
    return ret, r

probes = [
    ("constant-300 (s0 corner D=.6 wlo=300)", 0.6, 300, 5.0, 0.851),
    ("constant-300 (other face D=.87)",       0.87, 300, 2.5, 0.851),
    ("champion D=.8 wlo=0 T=2.5",             0.8, 0, 2.5, None),
    ("v3 pinned corner D=.6 wlo=0 T=5",       0.6, 0, 5.0, 0.843),
]
print(f"{'probe':42s} {'surr ret':>9s} {'final R':>8s} {'real':>6s}")
for name, D, W, T, real in probes:
    ret, r = static_return(D, W, T)
    print(f"{name:42s} {ret:9.4f} {r:8.4f} {('%.3f' % real) if real else '   ~'}")
opt = scan_static()
print(f"\ngrid-scan optimum: D={opt['duty']:.3f} w_low={opt['w_low']:.0f} "
      f"T={opt['period']:.2f} -> return {opt['return']:.4f} finalR {opt['final_R']:.4f}")
