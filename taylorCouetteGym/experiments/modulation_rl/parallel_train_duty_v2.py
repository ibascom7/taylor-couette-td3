"""Parallelized TD3 over the (T+, T-) duty env -- the duty_v2 run.

Same threaded harness as parallel_train_duty.py (which itself reuses
parallel_train.py's env-agnostic pieces; the WARMED template prep is imported
from there unchanged) but the env is TaylorCouetteDutyV2Env
(taylor_couette_mixing/envs/taylor_couette_duty_v2.py): 2-D action
(T_plus in [1, 5] s burst, T_minus in [0, 5] s idle; T = T+ + T-,
D = T+/T, w_hi solved for an EXACT commanded block mean of w_b = 300),
2-D obs (X_block, t/tau), warmed continuous-manufacturing episodes,
continuing-task truncation handling (bootstrap through done=0).

FOUR training changes vs duty_v1, motivated by its collapse-to-D=1 post-mortem
(results/duty_diag: sharp interior optimum at 0.5 s idle, cliff at the
constant corner, exploration noise doing the film renewal):

  1. REWARD CENTERING: rewards enter the REPLAY BUFFER as r - reward_baseline,
     with the baseline = the warmed constant-300 reward at t = 50 s
     (R50- = 0.2390, X over [47.5, 50] s, P over [0, 50] s; computed by
     modulation_vs_constant/t50_table.py from
     results_warmed_T2p5_D80/constant_wb300). With gamma = 0.99 this removes
     the ~24-unit constant component of Q, so the critic represents +-few-unit
     differential values from init instead of spending its early updates (the
     corner-pinning window) filling in baseline. A constant shift is exactly
     policy-invariant here (continuing task, never terminates, bootstraps
     through truncation). ALL LOGS STAY RAW -- episode_returns.npy and
     reward_per_step.npy are benchmark-comparable; only the buffer is centered.
  2. RANDOM-PHASE ACTION REPEAT (--random_action_repeat 3): each random action
     is HELD for 3 consecutive blocks (~ the tau = 2.6-block conversion delay),
     so the sustained-operation value of a waveform -- the thing duty_v1's
     uniform random phase essentially never demonstrated (P[3 consecutive
     draws near the optimum] ~ 0.2%) -- actually appears in the buffer. The
     random phase becomes a randomized static sweep.
  3. LOWER EXPLORATION NOISE (--expl_noise 0.1): raw std 0.1 ~ 0.22 s in T+
     and 0.25 s in T-. duty_v1's 0.2 put ~0.5 s of idle jitter on a peak whose
     whole width is ~0.5 s, blurring it into the cliff (and inflating the
     corner's apparent on-policy value).
  4. SEED DECORRELATION: worker rngs / env seeds use seed*1000 + wid (duty_v1
     and v5 used seed + wid, which made the random phases of different seeds
     largely IDENTICAL episodes -- the "shared exploration data" caveat).

Logging format is unchanged except params_per_step.npy rows are now
[duty, t_minus_s, period_s, w_hi_rpm] (the w_low slot -- pinned 0 anyway --
carries the idle duration; T+ = duty * period reconstructs the raw action).

Example (Carya, 44 workers -- see run_carya_duty_v2.slurm):
  python parallel_train_duty_v2.py --n_workers 44 \
      --worker_root /tmp/$USER/duty_v2_workers --tag duty_v2_s0 \
      --max_episodes 300 --start_timesteps 500 --grad_per_step 64
"""

import argparse
import faulthandler
import os
import sys
import threading
import time
import traceback

import numpy as np
import torch

GYM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if GYM_ROOT not in sys.path:
    sys.path.insert(0, GYM_ROOT)
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)

from parallel_train import (  # noqa: E402  (env-agnostic pieces, reused as-is)
    ThreadSafeReplayBuffer, Shared, _snapshot_logs, _save_buffer, learner_loop,
)
from parallel_train_duty import prepare_worker_cases  # noqa: E402  (warmed template)
from train import make_policy  # noqa: E402
from taylor_couette_mixing.envs.taylor_couette_duty_v2 import (  # noqa: E402
    TaylorCouetteDutyV2Env,
)

RESULTS_ROOT = os.path.join(EXP_DIR, "results")

# Warmed constant-300 reward at t = 50 s (see module docstring for provenance).
R50_CONSTANT_300 = 0.2390


def make_env(case_path, args):
    return TaylorCouetteDutyV2Env(
        case_path=case_path,
        w_b_rpm=args.w_b_rpm,
        episode_duration=args.blocks_max * args.block_dt,   # per-episode override below
        block_dt=args.block_dt,
        t_plus_min=args.t_plus_min, t_plus_max=args.t_plus_max,
        t_minus_min=args.t_minus_min, t_minus_max=args.t_minus_max,
        w_hi_cap_rpm=args.w_hi_cap_rpm,
        ramp_time=args.ramp_time,
        p_max_watt=args.p_max_watt,
        t_scale=args.t_scale,
        x_init=args.x_init,
        reward_mode=args.reward_mode,
        flux_to_conv=args.flux_to_conv,
    )


# --------------------------------------------------------------------------- #
# Collector: duty_v1's continuing-task loop + centering + random action repeat.
# --------------------------------------------------------------------------- #
def collector_loop(wid, env, policy, buffer, cfg, shared, stop_event):
    rng = np.random.default_rng(cfg["seed"] * 1000 + wid)   # decorrelated across seeds
    first = True
    consec_fail = 0
    rand_action, rand_left = None, 0    # random-phase action-repeat state
    try:
        while not stop_event.is_set() and consec_fail < cfg["max_fail"]:
            n_blocks = int(rng.integers(cfg["blocks_min"], cfg["blocks_max"] + 1))
            try:
                obs, info = env.reset(
                    seed=(cfg["seed"] * 1000 + wid) if first else None,
                    options={"reset_mode": "hard", "n_blocks": n_blocks},
                )
                if first:
                    env.action_space.seed(cfg["seed"] * 1000 + wid)
            except Exception as e:
                consec_fail += 1
                print(f"[w{wid:02d}] reset failed ({consec_fail}/{cfg['max_fail']}): "
                      f"{type(e).__name__}: {e}", flush=True)
                traceback.print_exc(); sys.stdout.flush()
                continue
            first = False
            state = np.asarray(obs, dtype=np.float32)
            ep_ret, ep_om, ep_rw, ep_cv, ep_pw, ep_pr = 0.0, [], [], [], [], []

            while not stop_event.is_set():
                with shared.lock:
                    gstep = shared.total_env_steps
                if gstep < cfg["start_timesteps"]:
                    # Hold each random action for random_action_repeat blocks so
                    # SUSTAINED waveform values appear in the buffer (see docstring).
                    if rand_left <= 0:
                        rand_action = env.action_space.sample()
                        rand_left = cfg["random_action_repeat"]
                    action = rand_action
                    rand_left -= 1
                else:
                    with shared.policy_lock:
                        a = policy.select_action(state)
                    action = (a + rng.normal(
                        0, cfg["max_action"] * cfg["expl_noise"], size=cfg["action_dim"]
                    )).clip(-cfg["max_action"], cfg["max_action"])

                try:
                    next_obs, reward, terminated, truncated, info = env.step(action)
                except Exception as e:
                    consec_fail += 1
                    print(f"[w{wid:02d}] step failed ({consec_fail}/{cfg['max_fail']}, "
                          f"pimpleFoam?): {type(e).__name__}: {e}", flush=True)
                    traceback.print_exc(); sys.stdout.flush()
                    break   # abandon this episode -> outer loop resets
                consec_fail = 0

                next_state = np.asarray(next_obs, dtype=np.float32)
                # Continuing task (done = terminated = False, bootstrap through
                # truncation); the buffer reward is CENTERED, logs stay raw.
                buffer.add(state, action, next_state,
                           reward - cfg["reward_baseline"], float(terminated))

                with shared.lock:
                    shared.total_env_steps += 1
                    gstep = shared.total_env_steps
                state = next_state
                ep_ret += reward
                ep_om.append(float(info["w_hi_rpm"]))
                ep_rw.append(float(reward))
                ep_cv.append(float(info["mixing_index"]))    # X_block
                ep_pw.append(float(info["power_watt"]))
                ep_pr.append([float(info["duty"]), float(info["t_minus_s"]),
                              float(info["period_s"]), float(info["w_hi_rpm"])])

                if terminated or truncated:
                    with shared.lock:
                        shared.episode_returns.append(ep_ret)
                        shared.episode_end_steps.append(gstep)
                        shared.omega_history.append(ep_om)
                        shared.reward_history.append(ep_rw)
                        shared.conv_history.append(ep_cv)
                        shared.power_history.append(ep_pw)
                        shared.params_history.append(ep_pr)
                        n_done = len(shared.episode_returns)
                        if cfg["max_episodes"] and n_done >= cfg["max_episodes"]:
                            stop_event.set()
                    print(f"[w{wid:02d}] ep {n_done} done ({n_blocks} blocks) "
                          f"ret={ep_ret:+.3f} (ctr {ep_ret - n_blocks * cfg['reward_baseline']:+.3f}) "
                          f"Xlast={ep_cv[-1]:.3f} "
                          f"T-={' '.join(f'{p[1]:.1f}' for p in ep_pr)} "
                          f"D={' '.join(f'{p[0]:.2f}' for p in ep_pr)} "
                          f"steps={gstep}/{cfg['max_timesteps']}", flush=True)
                    break
                if gstep >= cfg["max_timesteps"]:
                    stop_event.set()
                    break
            if shared.total_env_steps >= cfg["max_timesteps"]:
                stop_event.set()
        if consec_fail >= cfg["max_fail"]:
            print(f"[w{wid:02d}] giving up after {consec_fail} consecutive failures", flush=True)
    finally:
        with shared.lock:
            shared.live_workers -= 1
            remaining = shared.live_workers
            if remaining <= 0:
                stop_event.set()
        print(f"[w{wid:02d}] exited (live workers now {remaining})", flush=True)


# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # parallelism / case layout
    p.add_argument("--n_workers", type=int, default=16)
    p.add_argument("--worker_root", default=None,
                   help="dir for _template/ + worker_NN/. Defaults to "
                        "<results_dir>/<algo>/<tag>/workers. Cached (compile + "
                        "warmup) across re-runs unless --rebuild.")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--case_path",
                   default="taylor_couette_mixing/cases/side_outlet_grad_case",
                   help="master RL-drivable case (the template is warmed FROM its "
                        "pristine IC, so the base case needs no cached 0.warmed).")
    # agent / algo
    p.add_argument("--algo", choices=["td3", "ddpg"], default="td3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_timesteps", type=int, default=1800)
    p.add_argument("--max_episodes", type=int, default=300)
    p.add_argument("--start_timesteps", type=int, default=500,
                   help="random-action steps before the policy kicks in "
                        "(~100 random episodes at the mean 5-block horizon).")
    p.add_argument("--random_action_repeat", type=int, default=3,
                   help="hold each random-phase action this many consecutive "
                        "blocks (~ the tau = 2.6-block conversion delay) so "
                        "sustained waveform values appear in the buffer.")
    p.add_argument("--grad_per_step", type=float, default=64.0,
                   help="UTD -- the surrogate-validated v5 recipe carries over.")
    p.add_argument("--max_fail", type=int, default=3)
    p.add_argument("--expl_noise", type=float, default=0.1,
                   help="Gaussian std on the RAW 2-D action (0.1 raw ~ 0.22 s "
                        "in T+ and 0.25 s in T-; duty_v1's 0.2 blurred the "
                        "0.5 s-wide idle optimum into the constant cliff).")
    p.add_argument("--reward_baseline", type=float, default=R50_CONSTANT_300,
                   help="constant subtracted from rewards ENTERING THE BUFFER "
                        "(logs stay raw). Default = warmed constant-300 R50- "
                        "(t50_table.py, results_warmed_T2p5_D80/constant_wb300). "
                        "Policy-invariant; conditioning only. 0 disables.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--discount", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--save_every", type=int, default=4000)
    # env (the (T+, T-) design; see taylor_couette_duty_v2.py)
    p.add_argument("--w_b_rpm", type=float, default=300.0)
    p.add_argument("--block_dt", type=float, default=10.0)
    p.add_argument("--blocks_min", type=int, default=3,
                   help="episode horizon is drawn uniform in [blocks_min, blocks_max] "
                        "per episode (mean 5 = the nominal 50 s episode) -- "
                        "randomized truncation keeps late-t states grounded "
                        "and hides the horizon from the raw-time obs.")
    p.add_argument("--blocks_max", type=int, default=7)
    p.add_argument("--t_plus_min", type=float, default=1.0,
                   help="1 s floor caps the exact-mean solve at 1500 rpm in the "
                        "deep-idle corner (v1's proven ceiling, inside the "
                        "2500 rpm power-model calibration).")
    p.add_argument("--t_plus_max", type=float, default=5.0)
    p.add_argument("--t_minus_min", type=float, default=0.0)
    p.add_argument("--t_minus_max", type=float, default=5.0)
    p.add_argument("--w_hi_cap_rpm", type=float, default=2500.0,
                   help="burst guard rail (calibrated envelope); the default box "
                        "peaks at exactly 1500 rpm, so it never engages -- the "
                        "commanded block mean is w_b EVERYWHERE.")
    p.add_argument("--ramp_time", type=float, default=0.05)
    p.add_argument("--p_max_watt", type=float, default=31.94)
    p.add_argument("--t_scale", type=float, default=26.0,
                   help="FIXED physical normalizer for the time obs (tau = V/Q).")
    p.add_argument("--x_init", type=float, default=0.353,
                   help="warmed steady conversion (initial obs stand-in).")
    p.add_argument("--reward_mode", choices=["conv", "flux"], default="conv")
    p.add_argument("--flux_to_conv", type=float, default=4.2e7,
                   help="mass-balance k = X_ss/J_ss (flux reward mode only).")
    p.add_argument("--warm_duration", type=float, default=60.0,
                   help="one-time template spin-up at w_b [s] (60 ~ 2.3 tau).")
    # output
    p.add_argument("--results_dir", default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--smoke", action="store_true",
                   help="tiny end-to-end validation: 2 workers, 3 x 2 s blocks, "
                        "2 s warmup, tiny budget (~15 min of real CFD).")
    return p


def main():
    args = build_parser().parse_args()
    if args.smoke:
        args.n_workers = 2
        args.block_dt = 2.0
        args.blocks_min = args.blocks_max = 3
        args.t_plus_max = 2.0        # keep at least one full cycle in a 2 s block
        args.t_minus_max = 1.0
        args.warm_duration = 2.0     # plumbing test, not a steady state
        args.max_timesteps = 12
        args.max_episodes = None
        args.start_timesteps = 4
        args.random_action_repeat = 2
        args.batch_size = 4
        args.save_every = 2
        print("[smoke] n_workers=2 episodes=3x2s warm=2s budget tiny", flush=True)

    faulthandler.enable()
    if os.environ.get("PT_DEBUG_HANG"):
        faulthandler.dump_traceback_later(int(os.environ["PT_DEBUG_HANG"]), repeat=True)

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    results_root = args.results_dir if args.results_dir else RESULTS_ROOT
    tag = args.tag if args.tag else f"duty_v2_seed{args.seed}"
    run_dir = os.path.join(results_root, args.algo, tag)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_prefix = os.path.join(run_dir, f"{args.algo}_tc")
    worker_root = args.worker_root or os.path.join(run_dir, "workers")

    master = os.path.join(GYM_ROOT, args.case_path) if not os.path.isabs(args.case_path) \
        else args.case_path
    workers = prepare_worker_cases(master, worker_root, args.n_workers, args.rebuild,
                                   args.w_b_rpm, args.warm_duration)

    envs = [make_env(w, args) for w in workers]
    obs0, _ = envs[0].reset(seed=args.seed, options={"reset_mode": "hard"})
    state_dim = np.asarray(obs0).shape[0]
    action_dim = envs[0].action_space.shape[0]
    max_action = float(envs[0].action_space.high[0])
    print(f"[{args.algo}] tag={tag} workers={args.n_workers} state_dim={state_dim} "
          f"action_dim={action_dim} (T+,T-) fixed-mean w_b={args.w_b_rpm}rpm "
          f"T+=[{args.t_plus_min},{args.t_plus_max}]s T-=[{args.t_minus_min},{args.t_minus_max}]s "
          f"blocks {args.blocks_min}-{args.blocks_max} x {args.block_dt}s "
          f"reward={args.reward_mode} baseline={args.reward_baseline} -> {run_dir}", flush=True)

    policy = make_policy(args.algo, state_dim, action_dim, max_action,
                         args.discount, args.tau)
    buffer = ThreadSafeReplayBuffer(state_dim, action_dim)
    shared = Shared()
    stop_event = threading.Event()
    cfg = dict(seed=args.seed, start_timesteps=args.start_timesteps,
               max_timesteps=args.max_timesteps, expl_noise=args.expl_noise,
               max_action=max_action, action_dim=action_dim,
               batch_size=args.batch_size, grad_per_step=args.grad_per_step,
               save_every=args.save_every, max_fail=args.max_fail,
               max_episodes=args.max_episodes,
               blocks_min=args.blocks_min, blocks_max=args.blocks_max,
               reward_baseline=args.reward_baseline,
               random_action_repeat=max(1, args.random_action_repeat))
    shared.live_workers = args.n_workers

    threads = [
        threading.Thread(target=collector_loop, name=f"w{i:02d}",
                         args=(i, envs[i], policy, buffer, cfg, shared, stop_event),
                         daemon=True)
        for i in range(args.n_workers)
    ]
    t0 = time.time()
    for t in threads:
        t.start()

    grad_steps = learner_loop(policy, buffer, cfg, shared, stop_event, run_dir, ckpt_prefix)
    for t in threads:
        t.join(timeout=600)
    target = int(cfg["grad_per_step"] * max(0, shared.total_env_steps - args.start_timesteps))
    while grad_steps < target and buffer.size >= args.batch_size:
        policy.train(buffer, args.batch_size)
        grad_steps += 1
    policy.save(f"{ckpt_prefix}_final")
    _snapshot_logs(run_dir, shared)
    _save_buffer(run_dir, buffer)
    print(f"[done] collected={shared.total_env_steps} grad_steps={grad_steps} "
          f"episodes={len(shared.episode_returns)} wall={time.time()-t0:.0f}s -> {run_dir}",
          flush=True)


if __name__ == "__main__":
    main()
