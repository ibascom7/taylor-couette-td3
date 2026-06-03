"""Overlay TD3 hyperparameter-sweep curves (run_carya_sweep.slurm).

Reads results/td3/<tag>/{reward,omega}_per_step.npy for each sweep config and
plots the per-episode MEAN of the chosen metric. Averaging per step (not the
raw episode total) keeps configs with different episode lengths -- e.g. ep120
-- directly comparable.

Safe to run mid-training: train.py re-writes these logs every save_every steps,
and missing / empty / partial logs are skipped or handled.

Metrics:
    reward -- per-step reward (higher = better mixing + less energy)
    omega  -- chosen angular velocity (rpm); shows the policy's control signal
              over training. Mean omega near 0 = symmetric use of +/- spin.

Usage:
    python plot_sweep.py                          # reward, all default tags
    python plot_sweep.py --metric omega
    python plot_sweep.py --tags base ep120 --smooth 11 --out sweep_omega.png
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

METRIC_FILES = {
    "reward": "reward_per_step.npy",
    "omega":  "omega_per_step.npy",
}
METRIC_YLABEL = {
    "reward": "mean reward per step  (higher = better mixing + less energy)",
    "omega":  "mean omega per step  (rpm; sign = spin direction)",
}


def load_mean_per_step(tag, metric):
    """results/td3/<tag>/<metric>_per_step.npy -> (episode_idx, mean per ep).

    Returns (None, None) if the config hasn't produced a usable log yet.
    """
    path = os.path.join(RESULTS_ROOT, tag, METRIC_FILES[metric])
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
    p.add_argument("--metric", choices=list(METRIC_FILES), default="reward",
                   help="which per-step quantity to plot")
    p.add_argument("--smooth", type=int, default=9,
                   help="rolling-mean window (single-seed curves are noisy); 1 = off")
    p.add_argument("--out", default=None,
                   help="output path (default: sweep_<metric>.png next to this script)")
    args = p.parse_args()

    out_path = args.out or os.path.join(SCRIPT_DIR, f"sweep_{args.metric}.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = 0
    for tag in args.tags:
        episodes, mean_v = load_mean_per_step(tag, args.metric)
        if episodes is None:
            print(f"[skip] {tag}: no usable log yet")
            continue
        line = ax.plot(episodes, mean_v, alpha=0.25, linewidth=1)[0]
        ax.plot(episodes, smooth(mean_v, args.smooth), color=line.get_color(),
                linewidth=2, label=f"{tag}  (ep {episodes[-1]+1})")
        plotted += 1
        print(f"[ok]   {tag}: {episodes[-1]+1} episodes, "
              f"latest mean {args.metric}/step {mean_v[-1]:+.4f}")

    if plotted == 0:
        print(f"Nothing to plot yet -- no config has written {METRIC_FILES[args.metric]}.")
        return

    ax.set_xlabel("episode")
    ax.set_ylabel(METRIC_YLABEL[args.metric])
    ax.set_title(f"TD3 sweep on TaylorCouetteMixing -- {args.metric}")
    if args.metric == "omega":
        ax.axhline(0.0, color="k", linewidth=0.6, alpha=0.4)
    ax.legend(title="config")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
