"""Train a TD3 or DDPG agent on TaylorCouetteMixingEnv.

Select the algorithm with --algo {td3,ddpg}. Each run writes to
results/<algo>/seed<seed>/ so multiple seeds and both algorithms coexist and
can be compared by plot_comparison.py.

Each env.step() invokes pimpleFoam, so timesteps are budgeted carefully: no
separate eval env, small start_timesteps, modest max_timesteps. Episodes are
truncated at max_steps (never terminated), so done_bool=0 everywhere and the
critic always bootstraps past episode boundaries.

The env mutates its OpenFOAM case directory in place, so concurrent runs MUST
use distinct --case_path copies (e.g. one per Slurm array task).
"""

import argparse
import os
import time

import numpy as np
import torch

import DDPG
import TD3
from taylor_couette_mixing.envs.taylor_couette_mixing import TaylorCouetteMixingEnv


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.join(SCRIPT_DIR, "results")
DEFAULT_CASE_PATH = "taylor_couette_mixing/cases/tc_mixing_case"


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e5)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
        )


def make_obs_to_state(omega_max, energy_norm):
    """Build a Dict-obs -> flat normalized state adapter.

    The three observed quantities live on very different scales (raw omega
    ~±300 vs cumulative energy ~1e-2 J), so without rescaling the energy and
    mixing signals are swamped in the network. Each is mapped to ~[-1, 1]:
      omega              -> omega / omega_max        in [-1, 1]
      mixing_index       -> 2*I - 1   (I in [0, 1])  in [-1, 1]
      energy_consumption -> E / energy_norm          in ~[0, 1]
    """
    def obs_to_state(obs):
        return np.array(
            [
                float(obs["omega"]) / omega_max,
                2.0 * float(obs["mixing_index"]) - 1.0,
                float(obs["energy_consumption"]) / energy_norm,
            ],
            dtype=np.float32,
        )

    return obs_to_state


def make_policy(algo, state_dim, action_dim, max_action, discount, tau):
    """Build the requested agent. Both expose select_action/train/save/load."""
    if algo == "ddpg":
        return DDPG.DDPG(state_dim, action_dim, max_action, discount=discount, tau=tau)
    if algo == "td3":
        return TD3.TD3(state_dim, action_dim, max_action, discount=discount, tau=tau)
    raise ValueError(f"Unknown algo: {algo!r}")


def _pad_to_grid(episodes):
    """List of per-episode value lists -> 2D array [episode, step], NaN-padded.

    Rows are episodes, columns are step-within-episode. The final episode may be
    shorter (training stops mid-episode), so rows are right-padded with NaN to
    keep the grid rectangular. Average a metric over episodes with
    np.nanmean(grid, axis=0).
    """
    if not episodes:
        return np.empty((0, 0))
    width = max(len(ep) for ep in episodes)
    grid = np.full((len(episodes), width), np.nan)
    for i, ep in enumerate(episodes):
        grid[i, : len(ep)] = ep
    return grid


def save_logs(run_dir, episode_returns, episode_end_steps,
              omega_history, reward_history, ep_omegas, ep_rewards):
    """Persist all per-run logs read by plot_comparison.py / DDPG_eval.py.

      - episode_returns.npy:   total return of each completed episode
      - episode_end_steps.npy: global timestep at which each episode ended
                               (x-axis for return-vs-timesteps curves)
      - omega_per_step.npy:    chosen angular velocity (rpm), [episode, step]
      - reward_per_step.npy:   reward, [episode, step]
    The in-progress episode is included in the per-step grids so periodic saves
    capture the latest steps.
    """
    np.save(os.path.join(run_dir, "episode_returns.npy"), np.array(episode_returns))
    np.save(os.path.join(run_dir, "episode_end_steps.npy"), np.array(episode_end_steps))
    omega_grid = _pad_to_grid(omega_history + ([ep_omegas] if ep_omegas else []))
    reward_grid = _pad_to_grid(reward_history + ([ep_rewards] if ep_rewards else []))
    np.save(os.path.join(run_dir, "omega_per_step.npy"), omega_grid)
    np.save(os.path.join(run_dir, "reward_per_step.npy"), reward_grid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["td3", "ddpg"], default="td3")
    parser.add_argument("--case_path", default=DEFAULT_CASE_PATH,
                        help="OpenFOAM case dir (use a distinct copy per concurrent run)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps_per_ep", type=int, default=60)
    parser.add_argument("--max_timesteps", type=int, default=3_000)
    parser.add_argument("--start_timesteps", type=int, default=120)
    parser.add_argument("--expl_noise", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--tag", default=None,
                        help="Results subdir name under results/<algo>/. Defaults to "
                             "seed<seed>. Use a distinct tag per sweep config so concurrent "
                             "runs at the same seed don't overwrite each other's outputs.")
    args = parser.parse_args()

    algo = args.algo
    seed = args.seed
    max_steps_per_ep = args.max_steps_per_ep
    max_timesteps = args.max_timesteps
    start_timesteps = args.start_timesteps
    expl_noise = args.expl_noise
    batch_size = args.batch_size
    discount = args.discount
    tau = args.tau
    save_every = args.save_every

    run_subdir = args.tag if args.tag else f"seed{seed}"
    run_dir = os.path.join(RESULTS_ROOT, algo, run_subdir)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_prefix = os.path.join(run_dir, f"{algo}_tc")
    print(f"[{algo}] seed={seed} tag={args.tag} case={args.case_path} -> {run_dir}")

    env = TaylorCouetteMixingEnv(case_path=args.case_path, max_steps=max_steps_per_ep)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Normalize obs to ~[-1, 1]; cumulative energy is bounded by E_max_per_step * max_steps.
    obs_to_state = make_obs_to_state(
        omega_max=env.omega_max,
        energy_norm=env.E_max_per_step * max_steps_per_ep,
    )
    state_dim = 3   # omega, mixing_index, energy_consumption
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])  # 1.0

    policy = make_policy(algo, state_dim, action_dim, max_action, discount, tau)

    replay_buffer = ReplayBuffer(state_dim, action_dim)

    obs, info = env.reset(seed=seed, options={"reset_mode": "hard"})
    state = obs_to_state(obs)

    episode_reward = 0.0
    episode_timesteps = 0
    episode_num = 0
    episode_returns = []
    episode_end_steps = []  # global timestep (t+1) at which each episode ended

    # Per-step logs, grouped by episode for later [episode, step] grids.
    omega_history = []   # completed episodes' chosen omega (rpm) per step
    reward_history = []  # completed episodes' reward per step
    ep_omegas = []       # current (in-progress) episode
    ep_rewards = []

    total_start = time.time()

    for t in range(max_timesteps):
        episode_timesteps += 1

        if t < start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                policy.select_action(state)
                + np.random.normal(0, max_action * expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        step_start = time.time()
        next_obs, reward, terminated, truncated, info = env.step(action)
        step_wall = time.time() - step_start

        next_state = obs_to_state(next_obs)
        done = terminated or truncated
        # Truncation is not a real terminal -> always bootstrap.
        done_bool = float(terminated)

        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward
        ep_omegas.append(float(next_obs["omega"]))
        ep_rewards.append(float(reward))

        if t >= start_timesteps:
            policy.train(replay_buffer, batch_size)

        print(
            f"[{algo} s{seed}] t={t+1}/{max_timesteps} ep={episode_num} step={info['step_count']} "
            f"a={action[0]:+.3f} omega={next_obs['omega']:+.2f} "
            f"I={info['mixing_index']:.4f} E={info['energy_consumption']:.3e} "
            f"r={reward:+.4f} dt={step_wall:.1f}s"
        )

        if done:
            episode_returns.append(episode_reward)
            episode_end_steps.append(t + 1)
            omega_history.append(ep_omegas)
            reward_history.append(ep_rewards)
            print(
                f"--- episode {episode_num} done. "
                f"return={episode_reward:.3f} len={episode_timesteps} ---"
            )
            obs, info = env.reset(seed=seed, options={"reset_mode": "hard"})
            state = obs_to_state(obs)
            episode_reward = 0.0
            episode_timesteps = 0
            episode_num += 1
            ep_omegas = []
            ep_rewards = []

        if (t + 1) % save_every == 0:
            policy.save(f"{ckpt_prefix}_t{t+1}")
            save_logs(run_dir, episode_returns, episode_end_steps,
                      omega_history, reward_history, ep_omegas, ep_rewards)

    policy.save(f"{ckpt_prefix}_final")
    save_logs(run_dir, episode_returns, episode_end_steps,
              omega_history, reward_history, ep_omegas, ep_rewards)

    total_time = time.time() - total_start
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTotal training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Episodes completed: {episode_num}")
    if episode_returns:
        print(f"Last 10 episode returns mean: {np.mean(episode_returns[-10:]):.3f}")
