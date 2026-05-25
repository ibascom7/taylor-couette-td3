"""3-panel TD3 vs DDPG comparison on TaylorCouetteMixingEnv.

Reads results/<algo>/seed<seed>/{episode_returns.npy, episode_end_steps.npy}
produced by train.py and renders, in the style of a HalfCheetah comparison:
  1. mean +/- std training return over seeds, per algorithm
  2. per-seed training-return curves (one color per seed; algo = line style)
  3. final-performance bars per seed

Usage:
  python plot_comparison.py
  python plot_comparison.py --results results --algos td3 ddpg \
                            --final_episodes 10 --out comparison.png
"""

import argparse
import glob
import os
import re

import numpy as np
import matplotlib.pyplot as plt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ALGO_COLOR = {"td3": "C0", "ddpg": "C1"}   # blue / orange, as in the template
ALGO_STYLE = {"td3": "-", "ddpg": "--"}    # solid TD3, dashed DDPG (per-seed panel)


def discover_seeds(results_root, algo):
    """Return {seed_int: run_dir} for every seed<N> dir under results/<algo>/."""
    seeds = {}
    for d in sorted(glob.glob(os.path.join(results_root, algo, "seed*"))):
        m = re.search(r"seed(\d+)$", d)
        if m and os.path.exists(os.path.join(d, "episode_returns.npy")):
            seeds[int(m.group(1))] = d
    return seeds


def load_run(run_dir):
    """Return (timesteps, returns) for a run; falls back to episode index for x."""
    returns = np.load(os.path.join(run_dir, "episode_returns.npy"))
    steps_path = os.path.join(run_dir, "episode_end_steps.npy")
    if os.path.exists(steps_path):
        steps = np.load(steps_path)
    else:
        steps = np.arange(1, len(returns) + 1)
    n = min(len(returns), len(steps))
    return steps[:n], returns[:n]


def common_grid(curves, n=200):
    """Interpolate (xs, ys) curves onto a shared x-grid; return (grid, Y[seed, x])."""
    curves = [(xs, ys) for xs, ys in curves if len(xs) >= 2]
    if not curves:
        return None, None
    x_lo = max(xs[0] for xs, _ in curves)
    x_hi = min(xs[-1] for xs, _ in curves)   # align to the shortest run, no extrapolation
    if x_hi <= x_lo:
        return None, None
    grid = np.linspace(x_lo, x_hi, n)
    Y = np.array([np.interp(grid, xs, ys) for xs, ys in curves])
    return grid, Y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=os.path.join(SCRIPT_DIR, "results"))
    parser.add_argument("--algos", nargs="+", default=["td3", "ddpg"])
    parser.add_argument("--final_episodes", type=int, default=10,
                        help="average the last N episode returns for the final-performance bars")
    parser.add_argument("--out", default=os.path.join(SCRIPT_DIR, "comparison.png"))
    args = parser.parse_args()

    # algo -> {seed: (timesteps, returns)}
    data = {}
    for algo in args.algos:
        seeds = discover_seeds(args.results, algo)
        if seeds:
            data[algo] = {s: load_run(d) for s, d in seeds.items()}
    if not data:
        raise SystemExit(f"No runs found under {args.results}. Expected results/<algo>/seed<N>/.")

    fig, (ax_mean, ax_seed, ax_bar) = plt.subplots(1, 3, figsize=(20, 5))

    # --- Panel 1: mean +/- std over seeds, per algorithm ---
    for algo, runs in data.items():
        grid, Y = common_grid(list(runs.values()))
        if grid is None:
            continue
        mean, std = Y.mean(axis=0), Y.std(axis=0)
        c = ALGO_COLOR.get(algo, None)
        ax_mean.plot(grid, mean, color=c, label=algo.upper())
        ax_mean.fill_between(grid, mean - std, mean + std, color=c, alpha=0.2)
    n_seeds = max(len(r) for r in data.values())
    ax_mean.set_title(f"TD3 vs DDPG on TaylorCouetteMixing (mean ± std, {n_seeds} seeds)")
    ax_mean.set_xlabel("Timesteps")
    ax_mean.set_ylabel("Episode return")
    ax_mean.grid(alpha=0.3)
    ax_mean.legend(loc="best")

    # --- Panel 2: per-seed curves (color = seed, line style = algo) ---
    for algo, runs in data.items():
        ls = ALGO_STYLE.get(algo, "-")
        for seed, (xs, ys) in sorted(runs.items()):
            ax_seed.plot(xs, ys, ls, color=f"C{seed}", alpha=0.9,
                         label=f"{algo.upper()} seed {seed}")
    ax_seed.set_title("TD3 vs DDPG on TaylorCouetteMixing (per seed)")
    ax_seed.set_xlabel("Timesteps")
    ax_seed.set_ylabel("Episode return")
    ax_seed.grid(alpha=0.3)
    ax_seed.legend(loc="best", fontsize=8)

    # --- Panel 3: final-performance bars per seed ---
    all_seeds = sorted({s for runs in data.values() for s in runs})
    x = np.arange(len(all_seeds))
    width = 0.8 / max(len(data), 1)
    for i, (algo, runs) in enumerate(data.items()):
        finals = []
        for s in all_seeds:
            if s in runs:
                _, ys = runs[s]
                k = min(args.final_episodes, len(ys))
                finals.append(np.mean(ys[-k:]) if k else np.nan)
            else:
                finals.append(np.nan)
        ax_bar.bar(x + i * width, finals, width, color=ALGO_COLOR.get(algo, None),
                   label=algo.upper())
    ax_bar.set_title("Final Performance Comparison")
    ax_bar.set_ylabel(f"Mean return (last {args.final_episodes} episodes)")
    ax_bar.set_xticks(x + width * (len(data) - 1) / 2)
    ax_bar.set_xticklabels([f"Seed {s}" for s in all_seeds])
    ax_bar.legend(loc="best")

    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"Saved comparison figure to {args.out}")
    for algo, runs in data.items():
        print(f"  {algo}: seeds {sorted(runs)}")


if __name__ == "__main__":
    main()
