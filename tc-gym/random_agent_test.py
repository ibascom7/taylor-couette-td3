"""Random-agent smoke test for TaylorCouetteMixingEnv.

Usage:
    python random_agent_test.py --case-path /path/to/tc_mixing_case

Runs a few short episodes with random actions to confirm the env can
reset, step, and parse OpenFOAM output end-to-end.
"""

from __future__ import annotations

import argparse

from tc_gym import TaylorCouetteMixingEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-path", required=True, help="Path to the OpenFOAM case directory.")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=3)
    args = parser.parse_args()

    env = TaylorCouetteMixingEnv(case_path=args.case_path, max_steps=args.max_steps)

    for ep in range(args.episodes):
        obs, info = env.reset(options={"reset_mode": "hard"})
        print(f"[ep {ep}] reset -> obs={obs} info={info}")

        done = False
        total_reward = 0.0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            print(
                f"  step={info['step_count']} action={action} "
                f"omega={info['omega']:+.2f} I={info['mixing_index']:.4f} "
                f"E={info['energy_consumption']:.4e} r={reward:+.4f}"
            )

        print(f"[ep {ep}] done. total_reward={total_reward:+.4f}")


if __name__ == "__main__":
    main()
