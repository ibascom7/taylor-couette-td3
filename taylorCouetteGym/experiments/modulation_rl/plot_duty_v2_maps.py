#!/usr/bin/env python3
"""Render the duty_v2 actors as policy maps -- the Zhou & Zhu Fig-2C analogue.

Each trained actor is a tiny (X, t/tau) -> (T+, T-) network, so unlike their
CNN it can be rendered EXHAUSTIVELY on a grid; no statistical joint PDF is
needed. For each seed: a T-(X, t) heatmap (the physics axis) with the visited
training states overlaid (dots = (X_block, t/tau) pairs actually reached, so
the map is only interpreted where the plant lives), plus a T+(X, t) heatmap.

Output: results/td3/duty_v2_policy_maps.png

USAGE: python3 plot_duty_v2_maps.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GYM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if GYM_ROOT not in sys.path:
    sys.path.insert(0, GYM_ROOT)
from train import make_policy  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TD3_DIR = os.path.join(HERE, "results", "td3")

SEEDS = [0, 1, 2, 3, 4]
T_SCALE = 26.0
X_RANGE = (0.30, 0.44)
T_RANGE = (0.0, 70.0)          # seconds since warmed start
N = 220

# raw [-1,1]^2 -> (T+, T-), matching TaylorCouetteDutyV2Env._decode defaults
def decode(a0, a1):
    t_plus = 1.0 + 0.5 * (a0 + 1.0) * (5.0 - 1.0)
    t_minus = 0.5 * (a1 + 1.0) * 5.0
    return t_plus, t_minus


def main():
    xs = np.linspace(*X_RANGE, N)
    ts = np.linspace(*T_RANGE, N)
    XX, TT = np.meshgrid(xs, ts)
    states = np.stack([XX.ravel(), (TT / T_SCALE).ravel()], axis=1).astype(np.float32)

    fig, axes = plt.subplots(2, len(SEEDS), figsize=(3.6 * len(SEEDS), 7.6),
                             sharex=True, sharey=True)
    fig.suptitle("duty_v2 — rendered policy maps (deterministic actor output over the "
                 "state plane; dots = visited training states, last 100 episodes)",
                 fontsize=13, x=0.5, y=0.99)
    for j, s in enumerate(SEEDS):
        d = os.path.join(TD3_DIR, f"duty_v2_s{s}")
        policy = make_policy("td3", state_dim=2, action_dim=2, max_action=1.0,
                             discount=0.99, tau=0.005)
        policy.load(os.path.join(d, "td3_tc_final"))
        acts = np.array([policy.select_action(st) for st in states])
        tp, tm = decode(acts[:, 0], acts[:, 1])
        tp = tp.reshape(N, N)
        tm = tm.reshape(N, N)

        # visited (X_block, t) pairs from the conv log (block b ends at (b+1)*10 s)
        cv = np.load(os.path.join(d, "conv_per_step.npy"))[-100:]
        vx, vt = [], []
        for row in cv:
            for b, x in enumerate(row):
                if not np.isnan(x):
                    vx.append(x)
                    vt.append((b + 1) * 10.0)

        for i, (Z, name, vmax) in enumerate([(tm, "$T_-$ (idle) [s]", 5.0),
                                             (tp, "$T_+$ (burst) [s]", 5.0)]):
            ax = axes[i, j]
            im = ax.pcolormesh(xs, ts, Z, cmap="viridis", vmin=0.0, vmax=vmax,
                               shading="auto")
            ax.scatter(vx, vt, s=4, c="white", alpha=0.5, lw=0)
            if i == 0:
                ax.set_title(f"seed {s}", fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{name}\n\ntime since warmed start [s]", fontsize=9)
            if i == 1:
                ax.set_xlabel("outlet conversion  X")
            fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    fig.text(0.5, 0.005,
             "reading: horizontal color bands (varying with t, flat across X) = "
             "time-scheduled policy; vertical bands = X-feedback; uniform = static "
             "waveform. Interpret only where the white dots (visited states) live.",
             ha="center", fontsize=9, color="0.35")
    fig.tight_layout(rect=[0, 0.015, 1, 0.965])
    out = os.path.join(TD3_DIR, "duty_v2_policy_maps.png")
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
