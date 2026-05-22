"""Evaluate a trained TD3 or DDPG agent on TaylorCouetteMixingEnv and plot training curve.

Run from the repo root without installing tc_gym:
    python3 tc-gym/eval.py --algo td3 --run-dir runs/td3
    python3 tc-gym/eval.py --algo ddpg --run-dir runs/ddpg --episodes 3

Reads:
    <run-dir>/{td3_final.zip | ddpg_final.zip}   (or --model-path override)
    <run-dir>/monitor.monitor.csv                (for the learning curve plot)

Writes:
    <run-dir>/learning_curve.png
    <run-dir>/eval_trajectory.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# Make `tc_gym` importable without `pip install`.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

import matplotlib

matplotlib.use("Agg")  # headless save to PNG
import matplotlib.pyplot as plt

from stable_baselines3 import DDPG, TD3

from tc_gym import TaylorCouetteMixingEnv


DEFAULT_CASE_PATH = HERE / "cases" / "tc_mixing_case"

ALGO_CLASS = {"td3": TD3, "ddpg": DDPG}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=ALGO_CLASS.keys(), required=True)
    p.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Training save_dir. Used to find the model and monitor log.",
    )
    p.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Override path to the saved model .zip (default: <run-dir>/<algo>_final.zip).",
    )
    p.add_argument("--case-path", type=str, default=str(DEFAULT_CASE_PATH))
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-steps-per-ep", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def load_episode_returns(monitor_csv: Path) -> np.ndarray:
    """Read SB3 Monitor CSV. First line is a JSON header; rest is rl-csv."""
    if not monitor_csv.is_file():
        return np.array([])
    returns = []
    with monitor_csv.open() as f:
        first = f.readline()
        if not first.startswith("#"):
            f.seek(0)
        reader = csv.DictReader(f)
        for row in reader:
            try:
                returns.append(float(row["r"]))
            except (KeyError, ValueError):
                continue
    return np.array(returns)


def plot_learning_curve(returns: np.ndarray, out_path: Path, algo: str) -> None:
    if returns.size == 0:
        print(f"No episodes in monitor log; skipping learning curve.")
        return
    n = len(returns)
    window = min(10, n)
    mov = np.array(
        [returns[max(0, i - window + 1) : i + 1].mean() for i in range(n)]
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(n), returns, "-o", lw=1.2, ms=4, label="Per-episode return")
    ax.plot(np.arange(n), mov, "-", lw=2, label=f"Trailing mean (≤{window} eps)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title(f"{algo.upper()} on TaylorCouetteMixingEnv ({n} episodes)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved learning curve to: {out_path}")


def rollout(model, env: TaylorCouetteMixingEnv, episodes: int, seed: int):
    """Run `episodes` deterministic rollouts; yield per-step trajectory rows."""
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep, options={"reset_mode": "hard"})
        ep_return = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            done = terminated or truncated
            yield {
                "episode": ep,
                "step": info["step_count"],
                "action": float(action[0]),
                "omega_rpm": info["omega"],
                "mixing_index": info["mixing_index"],
                "energy_total": info["energy_consumption"],
                "reward": float(reward),
            }
        print(f"[eval ep {ep}] return={ep_return:+.4f}")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    model_path = (
        Path(args.model_path)
        if args.model_path
        else run_dir / f"{args.algo}_final.zip"
    )
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    env = TaylorCouetteMixingEnv(
        case_path=args.case_path,
        max_steps=args.max_steps_per_ep,
    )

    model_cls = ALGO_CLASS[args.algo]
    model = model_cls.load(str(model_path), env=env)
    print(f"Loaded {args.algo.upper()} model from {model_path}")

    traj_path = run_dir / "eval_trajectory.csv"
    fieldnames = [
        "episode", "step", "action", "omega_rpm",
        "mixing_index", "energy_total", "reward",
    ]
    with traj_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rollout(model, env, args.episodes, args.seed):
            writer.writerow(row)
            print(
                f"  ep={row['episode']} step={row['step']} "
                f"a={row['action']:+.3f} omega={row['omega_rpm']:+.2f} "
                f"I={row['mixing_index']:.4f} E={row['energy_total']:.3e} "
                f"r={row['reward']:+.4f}"
            )
    print(f"Saved eval trajectory to: {traj_path}")

    if not args.no_plot:
        monitor_csv = run_dir / "monitor.monitor.csv"
        returns = load_episode_returns(monitor_csv)
        plot_learning_curve(returns, run_dir / "learning_curve.png", args.algo)


if __name__ == "__main__":
    main()
