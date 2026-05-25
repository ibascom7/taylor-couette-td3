"""Random-agent smoke test for TaylorCouetteMixingEnv."""

from taylor_couette_mixing.envs.taylor_couette_mixing import TaylorCouetteMixingEnv

CASE_PATH = "taylor_couette_mixing/cases/tc_mixing_case"
NUM_EPISODES = 10

env = TaylorCouetteMixingEnv(case_path=CASE_PATH, max_steps=5)

for ep in range(NUM_EPISODES):
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
            f"omega={obs['omega']:.2f} I={info['mixing_index']:.4f} "
            f"E={info['energy_consumption']:.4e} r={reward:.4f}"
        )

    print(f"[ep {ep}] done. total_reward={total_reward:.4f}")