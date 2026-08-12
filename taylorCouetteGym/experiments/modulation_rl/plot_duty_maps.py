#!/usr/bin/env python3
"""Render the duty actors as policy maps -- the Zhou & Zhu Fig-2C analogue.

Each trained actor is a tiny (X, t/tau) -> (T+, T-) network, so unlike their
CNN it can be rendered EXHAUSTIVELY on a grid; no statistical joint PDF is
needed. For each seed: T-(X, t), T+(X, t) and duty D(X, t) heatmaps with the
visited training states overlaid (dots = (X_block, t) pairs actually reached,
so the map is only interpreted where the plant lives).

D is the row to read against the comparators: both static references are
indexed by duty (constant = D 1.0, champion = D 0.90).

Output: results/td3/duty_{version}_policy_maps.png

USAGE
    python3 plot_duty_maps.py                     # v3 (26 s tau-blocks)
    python3 plot_duty_maps.py --version v2        # the 10 s-block campaign
    python3 plot_duty_maps.py --seeds 1 2 --ckpt td3_tc_t40000
"""
import argparse
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
T_SCALE = 26.0                 # tau [s] -- the env's fixed time normalizer
X_RANGE = (0.30, 0.44)
N = 220

# per-campaign block bookkeeping: block_dt [s] and the plotted time span
# (max horizon 7 blocks + a margin), matching each run's slurm.
CAMPAIGNS = {
    "v2": dict(block_dt=10.0, t_max=75.0),
    "v3": dict(block_dt=26.0, t_max=190.0),
}


# raw [-1,1]^2 -> (T+, T-), matching TaylorCouetteDutyV2Env._decode defaults
def decode(a0, a1):
    t_plus = 1.0 + 0.5 * (a0 + 1.0) * (5.0 - 1.0)
    t_minus = 0.5 * (a1 + 1.0) * 5.0
    return t_plus, t_minus


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", choices=sorted(CAMPAIGNS), default="v3")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--ckpt", default="td3_tc_final",
                    help="checkpoint stem (e.g. td3_tc_t40000 for a run in flight)")
    args = ap.parse_args()
    cfg = CAMPAIGNS[args.version]
    block_dt = cfg["block_dt"]

    seeds = [s for s in args.seeds
             if os.path.isfile(os.path.join(TD3_DIR, f"duty_{args.version}_s{s}",
                                            f"{args.ckpt}_actor"))]
    missing = sorted(set(args.seeds) - set(seeds))
    if missing:
        print(f"[maps] no {args.ckpt} actor for seeds {missing} -- skipping those")
    if not seeds:
        sys.exit(f"[maps] no duty_{args.version} actors found under {TD3_DIR}")

    xs = np.linspace(*X_RANGE, N)
    ts = np.linspace(0.0, cfg["t_max"], N)
    XX, TT = np.meshgrid(xs, ts)
    states = np.stack([XX.ravel(), (TT / T_SCALE).ravel()], axis=1).astype(np.float32)

    fig, axes = plt.subplots(3, len(seeds), figsize=(3.6 * len(seeds), 11.0),
                             sharex=True, sharey=True, squeeze=False)
    fig.suptitle(f"duty_{args.version} — rendered policy maps (deterministic actor "
                 f"output over the state plane;\ndots = visited training states, "
                 f"last 100 episodes)", fontsize=13, x=0.5, y=0.99)
    for j, s in enumerate(seeds):
        d = os.path.join(TD3_DIR, f"duty_{args.version}_s{s}")
        policy = make_policy("td3", state_dim=2, action_dim=2, max_action=1.0,
                             discount=0.99, tau=0.005)
        policy.load(os.path.join(d, args.ckpt))
        acts = np.array([policy.select_action(st) for st in states])
        tp, tm = decode(acts[:, 0], acts[:, 1])
        duty = tp / (tp + tm)
        tp = tp.reshape(N, N)
        tm = tm.reshape(N, N)
        duty = duty.reshape(N, N)

        # visited (X_block, t) pairs from the conv log (block b ends at (b+1)*block_dt)
        cv = np.load(os.path.join(d, "conv_per_step.npy"))[-100:]
        vx, vt = [], []
        for row in cv:
            for b, x in enumerate(row):
                if not np.isnan(x):
                    vx.append(x)
                    vt.append((b + 1) * block_dt)

        panels = [(tm, "$T_-$ (idle) [s]", 0.0, 5.0, "viridis"),
                  (tp, "$T_+$ (burst) [s]", 0.0, 5.0, "viridis"),
                  (duty, "duty  $D = T_+/(T_++T_-)$", 0.5, 1.0, "magma")]
        for i, (Z, name, vmin, vmax, cmap) in enumerate(panels):
            ax = axes[i, j]
            im = ax.pcolormesh(xs, ts, Z, cmap=cmap, vmin=vmin, vmax=vmax,
                               shading="auto")
            ax.scatter(vx, vt, s=4, c="white", alpha=0.5, lw=0)
            if i == 0:
                ax.set_title(f"seed {s}", fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{name}\n\ntime since warmed start [s]", fontsize=9)
            if i == len(panels) - 1:
                ax.set_xlabel("outlet conversion  X")
            cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
            if i == 2:
                cb.ax.axhline(0.90, color="white", lw=1.4)
    fig.text(0.5, 0.005,
             "reading: horizontal color bands (varying with t, flat across X) = "
             "time-scheduled policy; vertical bands = X-feedback; uniform = static "
             "waveform. Interpret only where the white dots (visited states) live.\n"
             "white tick on the duty bar = the static champion D = 0.90; D = 1.0 is "
             "the constant-speed limit (no idle).",
             ha="center", fontsize=9, color="0.35")
    fig.tight_layout(rect=[0, 0.025, 1, 0.965])
    out = os.path.join(TD3_DIR, f"duty_{args.version}_policy_maps.png")
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
