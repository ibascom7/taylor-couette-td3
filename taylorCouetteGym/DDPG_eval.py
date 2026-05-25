"""Plot a single run's training returns saved by train.py.

Single-run learning curve for one results/<algo>/seed<seed>/ dir. For the
TD3-vs-DDPG multi-seed comparison figure, use plot_comparison.py instead.
"""

import os

import numpy as np
import matplotlib.pyplot as plt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RETURNS_PATH = os.path.join(SCRIPT_DIR, "results", "ddpg", "seed0", "episode_returns.npy")
PLOT_PATH = os.path.join(SCRIPT_DIR, "learning_curve.png")


def main():
    r = np.load(RETURNS_PATH)
    n = len(r)

    window = min(10, n)
    mov = np.array([r[max(0, i - window + 1):i + 1].mean() for i in range(n)])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(n), r, "-o", linewidth=1.2, markersize=4,
            label="Per-episode return")
    ax.plot(np.arange(n), mov, "-", linewidth=2,
            label=f"Trailing mean (up to {window} episodes)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode return")
    ax.set_title(f"DDPG on TaylorCouetteMixingEnv ({n} episodes)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=200)

    print(f"Episodes:     {n}")
    print(f"Mean return:  {r.mean():+.3f}")
    print(f"Last 10 mean: {r[-min(10, n):].mean():+.3f}")
    print(f"Min / max:    {r.min():+.3f} / {r.max():+.3f}")
    print(f"Saved plot to: {PLOT_PATH}")

    plt.show()


if __name__ == "__main__":
    main()
