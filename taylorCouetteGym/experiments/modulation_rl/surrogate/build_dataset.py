"""Extract (action, X_prev) -> (X, P) block-samples from real CFD run logs.

Every logged block of every episode -- random OR policy phase -- is a valid
physics sample: features are the DECODED action (duty, w_low_rpm, log period)
plus the previous block's conversion X_prev (the first-order flow memory);
targets are the block conversion X and block-average motor power P [W].
Output: dataset.npz in this directory.

Run: python build_dataset.py  (any python with numpy)
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results", "td3")
RUNS = ["mod_wb300_s0", "mod_wb300_v3_s0"]   # add future runs here

feats, targs, meta = [], [], []
for run in RUNS:
    d = os.path.join(RESULTS, run)
    conv = np.load(os.path.join(d, "conv_per_step.npy"))     # [ep, 5]
    power = np.load(os.path.join(d, "power_per_step.npy"))
    par = np.load(os.path.join(d, "params_per_step.npy"))    # [ep, 5, (D, w_low, T, w_hi)]
    n_ep, n_bl = conv.shape
    for e in range(n_ep):
        for k in range(n_bl):
            D, wlo, T, whi = par[e, k]
            if not np.isfinite([D, wlo, T, whi, conv[e, k], power[e, k]]).all():
                continue
            if D <= 0 or T <= 0:
                continue
            x_prev = conv[e, k - 1] if k > 0 else 0.0
            # PHYSICAL wave coordinates: under the fixed-mean constraint,
            # (w_hi, w_low) determines duty, and both degenerate faces
            # (w_low=w_b, D=1) collapse to the point (w_b, w_b) -- exactly
            # like the physics. Raw box coords would spread one wave over a face.
            feats.append([whi, wlo, np.log(T), x_prev])
            targs.append([conv[e, k], power[e, k]])
            meta.append([hash(run) % 1000, e, k])
feats = np.asarray(feats, float)
targs = np.asarray(targs, float)
meta = np.asarray(meta, int)
out = os.path.join(HERE, "dataset.npz")
np.savez(out, feats=feats, targs=targs, meta=meta,
         feat_names=np.array(["w_hi_rpm", "w_low_rpm", "logT", "x_prev"]),
         targ_names=np.array(["x_block", "p_watt"]))
print(f"{len(feats)} samples -> {out}")
print("feature ranges:")
for i, n in enumerate(["w_hi", "w_low", "logT", "x_prev"]):
    print(f"  {n:6s}: {feats[:,i].min():8.3f} .. {feats[:,i].max():8.3f}")
print(f"targets: X {targs[:,0].min():.3f}..{targs[:,0].max():.3f}  "
      f"P {targs[:,1].min():.3f}..{targs[:,1].max():.3f} W")
