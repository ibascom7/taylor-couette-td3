"""Overlay the TD3 hyperparameter-sweep return curves (run_carya_sweep.slurm).

Reads results/td3/<tag>/reward_per_step.npy for each sweep config and plots the
per-episode MEAN reward-per-step. We average reward *per step* (not the raw
episode return) so configs with different episode lengths -- e.g. ep120 -- are
directly comparable: a 120-step episode would otherwise look ~2x worse just for
being longer.

Safe to run mid-training: train.py re-writes these logs every save_every steps,
and missing / empty / partial logs are skipped or handled. Higher (less
negative) = better mixing + less energy.

Usage:
    python plot_sweep.py                 # all default tags
    python plot_sweep.py --tags base ep120
    python plot_sweep.py --smooth 11 --out sweep_progress.png
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless on Carya
import matplotlib.pyplot as plt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.join(SCRIPT_DIR, "results", "td3")
DEFAULT_TAGS = ["base", "expl0.3", "batch512", "ep120"]


def load_mean_reward_per_step(tag):
    """results/td3/<tag>/reward_per_step.npy -> (episode_idx, mean_reward_per_step).

    Returns (None, None) if the config hasn't produced a usable log yet.
    """
    path = os.path.join(RESULTS_ROOT, tag, "reward_per_step.npy")
    if not os.path.exists(path):
        return None, None
    grid = np.load(path)  # [episode, step], NaN-padded
    if grid.ndim != 2 or grid.size == 0:
        return None, None
    # Mean over steps within each episode, ignoring the NaN pad / partial tail.
    with np.errstate(invalid="ignore"):
        mean_per_ep = np.nanmean(grid, axis=1)
    mask = ~np.isnan(mean_per_ep)
    if not mask.any():
        return None, None
    episodes = np.arange(grid.shape[0])[mask]
    return episodes, mean_per_ep[mask]


def smooth(y, window):
    """Centered rolling mean; window<=1 or short series returns y unchanged."""
    if window <= 1 or len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", default=DEFAULT_TAGS,
                   help="config tags (results/td3/<tag>/) to overlay")
    p.add_argument("--smooth", type=int, default=9,
                   help="rolling-mean window (single-seed curves are noisy); 1 = off")
    p.add_argument("--out", default=os.path.join(SCRIPT_DIR, "sweep_progress.png"))
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = 0
    for tag in args.tags:
        episodes, mean_r = load_mean_reward_per_step(tag)
        if episodes is None:
            print(f"[skip] {tag}: no usable log yet")
            continue
        line = ax.plot(episodes, mean_r, alpha=0.25, linewidth=1)[0]
        ax.plot(episodes, smooth(mean_r, args.smooth), color=line.get_color(),
                linewidth=2, label=f"{tag}  (ep {episodes[-1]+1})")
        plotted += 1
        print(f"[ok]   {tag}: {episodes[-1]+1} episodes, "
              f"latest mean reward/step {mean_r[-1]:+.4f}")

    if plotted == 0:
        print("Nothing to plot yet -- no config has written reward_per_step.npy.")
        return

    ax.set_xlabel("episode")
    ax.set_ylabel("mean reward per step  (higher = better mixing + less energy)")
    ax.set_title("TD3 observation-expansion sweep on TaylorCouetteMixing")
    ax.legend(title="config")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
