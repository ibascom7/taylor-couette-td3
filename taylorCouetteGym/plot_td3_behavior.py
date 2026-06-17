#!/usr/bin/env python3
"""
Visualize a TD3 run's behavior on the reactor: learning curve + the policy it
actually executes (omega per step across training).

Reads a run dir written by train.py:
    episode_returns.npy   total return per completed episode
    omega_per_step.npy    chosen omega [rpm], shape [episode, step] (NaN-padded)
    reward_per_step.npy   reward,          shape [episode, step]

Writes <run_dir>/td3_behavior.png with three panels:
  1. learning curve (return vs episode, + trailing mean) -- is it improving?
  2. omega heatmap (episode x step) -- how the policy evolves; does it settle
     to a constant or keep modulating?
  3. omega trajectories for the first vs last episodes -- early exploration vs
     converged behavior.

Usage:  python plot_td3_behavior.py results/td3/full3d_s0
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="e.g. results/td3/full3d_s0")
    ap.add_argument("--last-n", type=int, default=3, help="how many final episodes to overlay")
    args = ap.parse_args()
    d = args.run_dir

    ret = np.load(os.path.join(d, "episode_returns.npy"))
    om = np.load(os.path.join(d, "omega_per_step.npy")).astype(float)  # [ep, step]
    if om.ndim == 1:
        om = om[None, :]
    n_ep = len(ret)
    om = om[:n_ep]                      # align to completed episodes
    steps = om.shape[1]

    fig = plt.figure(figsize=(13, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.3, 1.0])

    # ---- 1. learning curve ----
    ax = fig.add_subplot(gs[0])
    w = min(10, n_ep)
    mov = np.array([ret[max(0, i - w + 1):i + 1].mean() for i in range(n_ep)])
    ax.plot(np.arange(n_ep), ret, "-o", ms=3, lw=1, alpha=0.5, label="per-episode")
    ax.plot(np.arange(n_ep), mov, "-", lw=2.2, color="C3", label=f"trailing mean ({w})")
    ax.set_xlabel("episode"); ax.set_ylabel("return  ($-$mixing $-$ energy)")
    ax.set_title(f"Learning curve ({n_ep} episodes)")
    ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=8)

    # ---- 2. omega heatmap across training ----
    ax = fig.add_subplot(gs[1])
    im = ax.imshow(om, aspect="auto", origin="lower", cmap="viridis",
                   extent=[0, steps, 0, n_ep], interpolation="nearest")
    ax.set_xlabel("step within episode"); ax.set_ylabel("episode")
    ax.set_title("Policy: chosen $\\omega$ across training")
    fig.colorbar(im, ax=ax, label="$\\omega$ [rpm]")

    # ---- 3. early vs late omega trajectories ----
    ax = fig.add_subplot(gs[2])
    x = np.arange(steps)
    ax.plot(x, om[0], color="0.6", lw=1.5, label="episode 1 (explore)")
    for k in range(max(1, n_ep - args.last_n), n_ep):
        ax.plot(x, om[k], lw=1.6, label=f"episode {k+1}")
    ax.set_xlabel("step within episode"); ax.set_ylabel("$\\omega$ [rpm]")
    ax.set_title("Behavior: early vs final episodes")
    ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    out = os.path.join(d, "td3_behavior.png")
    fig.savefig(out, dpi=150)
    print(f"episodes={n_ep}  mean_return={ret.mean():+.3f}  "
          f"last{w}_mean={ret[-w:].mean():+.3f}")
    # describe the converged policy
    final = om[-args.last_n:]
    final = final[~np.isnan(final)]
    if final.size:
        print(f"final-episode omega: mean={final.mean():.0f} rpm  "
              f"min={final.min():.0f}  max={final.max():.0f}  "
              f"spread(std)={final.std():.0f} rpm  "
              f"-> {'~CONSTANT policy' if final.std() < 25 else 'MODULATING policy'}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
