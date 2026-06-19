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
from taylor_couette_mixing.envs.taylor_couette_constant_omega import (
    TaylorCouetteConstantOmegaEnv,
)
from taylor_couette_mixing.envs.taylor_couette_catalysis import (
    TaylorCouetteCatalysisEnv,
)


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
    parser.add_argument("--env", choices=["stepwise", "constant", "catalysis"],
                        default="stepwise",
                        help="stepwise: pick delta_omega each second (original). "
                             "constant: pick ONE absolute omega per episode and run "
                             "it for --episode_duration seconds (a 1-D optimizer). "
                             "catalysis: pick an ABSOLUTE omega each second on the "
                             "catalytic-wall case; reward = conversion - energy.")
    parser.add_argument("--episode_duration", type=float, default=60.0,
                        help="[constant env] seconds the chosen omega runs before "
                             "mixing/energy are scored.")
    parser.add_argument("--capture_episodes", default="",
                        help="[constant env] comma-separated 1-based episode "
                             "indices whose OpenFOAM time dirs are saved under "
                             "<run_dir>/frames/ for ParaView (e.g. '1,5,final'; "
                             "'final' = --max_timesteps). Empty = capture none.")
    parser.add_argument("--case_path", default=DEFAULT_CASE_PATH,
                        help="OpenFOAM case dir (use a distinct copy per concurrent run)")
    parser.add_argument("--r_in", type=float, default=25.4,
                        help="Inner cylinder radius (mm) of the case geometry. "
                             "Wedge tc_mixing_case: 25.4. Full 3D full_tc_mixing_case: 38.0.")
    parser.add_argument("--r_out", type=float, default=31.75,
                        help="Outer cylinder radius (mm) of the case geometry. "
                             "Wedge tc_mixing_case: 31.75. Full 3D full_tc_mixing_case: 40.35.")
    parser.add_argument("--e_max_per_step", type=float, default=0.0011017031875434,
                        help="Per-step energy normalizer (J) for the reward's energy term. "
                             "Geometry-specific (full annulus burns far more than the wedge); "
                             "calibrate per case. Wedge default shown.")
    parser.add_argument("--warmup_duration", type=float, default=10.0,
                        help="Seconds to spin the case up (once, cached as 0.warmed) before "
                             "training, so dye has reached the bottom outlet. Must exceed the "
                             "advective residence time H/U_inlet (~13.7 s for the full 3D case, "
                             "so use ~18-20 there; the wedge default 10 s suffices for the wedge).")
    parser.add_argument("--warmup_omega_rpm", type=float, default=100.0,
                        help="Angular velocity (rpm) used during the warmup spin-up.")
    parser.add_argument("--conv_weight", type=float, default=1.0,
                        help="[catalysis env] weight on conversion in the reward "
                             "(maximized). reward = conv_weight*conv - energy_weight*E_norm.")
    parser.add_argument("--energy_weight", type=float, default=0.1,
                        help="[catalysis env] weight on per-step input energy in the "
                             "reward (penalized). THE knob: too high -> agent idles "
                             "omega->0; too low -> omega pins to omega_max. The paper's "
                             "'more conversion at less energy' lives in between.")
    parser.add_argument("--energy_model", choices=["motor", "mechanical"], default="motor",
                        help="[catalysis env] reward energy term. motor: the paper's "
                             "electric-motor model (Eqs. 18-23) -- bearing-dominated and "
                             "regenerative, so modulation can be cheap and the agent can "
                             "discover it. mechanical: viscous-drag work from the CFD "
                             "(convex, punishes bursts).")
    parser.add_argument("--ramp_time", type=float, default=0.05,
                        help="[catalysis env] seconds to ramp the wall from the "
                             "previous omega to the new one each step (finite "
                             "acceleration -> bounded Courant; mirrors make_case.py's "
                             "square-wave ramps). 0 = instantaneous jump.")
    parser.add_argument("--results_dir", default=None,
                        help="Override the results root (default: <gym>/results). Runs "
                             "land in <results_dir>/<algo>/<tag>/. Use to keep an "
                             "experiment's outputs in its own folder.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps_per_ep", type=int, default=60,
                        help="stepwise: simulated seconds per episode. "
                             "constant: independent omega trials per gym episode.")
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
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny end-to-end sanity run: a few short episodes with "
                             "frequent saves (and, for --env constant, short duration + "
                             "frame capture) to validate the whole pipeline in minutes "
                             "before committing a full run. Overrides the budget flags.")
    args = parser.parse_args()

    if args.smoke:
        # Shrink everything to the smallest run that still exercises the env
        # step loop, policy.train(), a periodic save, the final save, and (for
        # the constant env) frame snapshotting.
        args.max_timesteps = 3
        args.start_timesteps = 1   # < max_timesteps so policy.train() also runs
        args.save_every = 1        # so a periodic save fires too
        if args.env == "constant":
            args.episode_duration = 5.0          # short sim, still real pimpleFoam
            if not args.capture_episodes:
                args.capture_episodes = "1,final"  # exercise the snapshot path
        else:
            args.max_steps_per_ep = 3
        print(f"[smoke] env={args.env} max_timesteps={args.max_timesteps} "
              f"start_timesteps={args.start_timesteps} "
              f"episode_duration={args.episode_duration} "
              f"capture={args.capture_episodes!r}")

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

    results_root = args.results_dir if args.results_dir else RESULTS_ROOT
    run_subdir = args.tag if args.tag else f"seed{seed}"
    run_dir = os.path.join(results_root, algo, run_subdir)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_prefix = os.path.join(run_dir, f"{algo}_tc")
    print(f"[{algo}/{args.env}] seed={seed} tag={args.tag} case={args.case_path} -> {run_dir}")

    if args.env == "constant":
        # "final" resolves to the last episode (max_timesteps episodes, since
        # each step is one episode here).
        capture_episodes = [
            max_timesteps if tok.strip() == "final" else int(tok)
            for tok in args.capture_episodes.split(",")
            if tok.strip()
        ]
        capture_dir = os.path.join(run_dir, "frames") if capture_episodes else None
        if capture_dir:
            os.makedirs(capture_dir, exist_ok=True)
        env = TaylorCouetteConstantOmegaEnv(
            case_path=args.case_path,
            episode_duration=args.episode_duration,
            capture_episodes=capture_episodes,
            capture_dir=capture_dir,
            r_in=args.r_in,
            r_out=args.r_out,
            E_max_per_step=args.e_max_per_step,
            warmup_duration=args.warmup_duration,
            warmup_omega_rpm=args.warmup_omega_rpm,
        )
        # Energy is now the total over one constant-omega run.
        energy_norm = env.E_max_per_step * args.episode_duration
    else:
        # Frame capture: 'final' = the last episode that fully completes (and so
        # truncates -> gets snapshotted) within the step budget. An episode ends
        # the step AFTER step_count reaches max_steps, i.e. it spans
        # (max_steps_per_ep + 1) env steps, so divide by that. Indices 1-based.
        steps_per_episode = max_steps_per_ep + 1
        final_ep = max(1, max_timesteps // steps_per_episode)
        capture_episodes = [
            final_ep if tok.strip() == "final" else int(tok)
            for tok in args.capture_episodes.split(",")
            if tok.strip()
        ]
        capture_dir = os.path.join(run_dir, "frames") if capture_episodes else None
        if capture_dir:
            os.makedirs(capture_dir, exist_ok=True)
        if args.env == "catalysis":
            # Stepwise control on the catalytic-wall case: the agent picks an
            # absolute omega each second; reward = conversion - energy. Conversion
            # is carried in the obs/info "mixing_index" slot (see env docstring),
            # so obs_to_state and all logging below work unchanged.
            env = TaylorCouetteCatalysisEnv(
                case_path=args.case_path,
                max_steps=max_steps_per_ep,
                r_in=args.r_in,
                r_out=args.r_out,
                E_max_per_step=args.e_max_per_step,
                warmup_duration=args.warmup_duration,
                warmup_omega_rpm=args.warmup_omega_rpm,
                conv_weight=args.conv_weight,
                energy_weight=args.energy_weight,
                energy_model=args.energy_model,
                ramp_time=args.ramp_time,
                capture_episodes=capture_episodes,
                capture_dir=capture_dir,
            )
        else:
            env = TaylorCouetteMixingEnv(
                case_path=args.case_path,
                max_steps=max_steps_per_ep,
                r_in=args.r_in,
                r_out=args.r_out,
                E_max_per_step=args.e_max_per_step,
                warmup_duration=args.warmup_duration,
                warmup_omega_rpm=args.warmup_omega_rpm,
                capture_episodes=capture_episodes,
                capture_dir=capture_dir,
            )
        # Cumulative energy is bounded by E_max_per_step * max_steps.
        energy_norm = env.E_max_per_step * max_steps_per_ep

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Normalize obs to ~[-1, 1].
    obs_to_state = make_obs_to_state(
        omega_max=env.omega_max,
        energy_norm=energy_norm,
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
