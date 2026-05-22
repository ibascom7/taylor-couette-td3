"""Train a TD3 agent on TaylorCouetteMixingEnv with stable-baselines3.

Run from the repo root without installing tc_gym:
    python3 tc-gym/train_TD3.py [--case-path PATH] [--total-timesteps N] ...

Artifacts (under --save-dir, default: runs/td3):
    - checkpoints/td3_<n>_steps.zip
    - td3_final.zip
    - monitor.csv          (per-episode returns/lengths, written by Monitor)
    - tb/                  (TensorBoard logs)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Make `tc_gym` importable without `pip install`.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from tc_gym import TaylorCouetteMixingEnv


DEFAULT_CASE_PATH = HERE / "cases" / "tc_mixing_case"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--case-path", type=str, default=str(DEFAULT_CASE_PATH))
    p.add_argument("--save-dir", type=str, default="runs/td3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-timesteps", type=int, default=3000)
    p.add_argument("--learning-starts", type=int, default=120)
    p.add_argument("--max-steps-per-ep", type=int, default=60)
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--expl-noise-sigma", type=float, default=0.1)
    p.add_argument("--save-freq", type=int, default=500)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    env = TaylorCouetteMixingEnv(
        case_path=args.case_path,
        max_steps=args.max_steps_per_ep,
    )
    env = Monitor(env, filename=str(save_dir / "monitor"))

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=args.expl_noise_sigma * np.ones(n_actions),
    )

    model = TD3(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        action_noise=action_noise,
        verbose=1,
        tensorboard_log=str(save_dir / "tb"),
        seed=args.seed,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=str(save_dir / "checkpoints"),
        name_prefix="td3",
    )

    t0 = time.time()
    model.learn(total_timesteps=args.total_timesteps, callback=ckpt_cb, progress_bar=False)
    elapsed = time.time() - t0

    final_path = save_dir / "td3_final"
    model.save(str(final_path))

    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    print(f"\nTraining time: {int(h)}h {int(m)}m {int(s)}s")
    print(f"Saved final model to: {final_path}.zip")
    print(f"Monitor log:         {save_dir / 'monitor.monitor.csv'}")
    print(f"TensorBoard logs:    {save_dir / 'tb'}")


if __name__ == "__main__":
    main()
