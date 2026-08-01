"""Parallelized TD3 over the D-ONLY duty-cycle env (the interpretability run).

Same threaded harness as parallel_train.py (N workers, one shared replay
buffer, one learner; the reusable pieces are IMPORTED from there) but the env
is TaylorCouetteDutyEnv (taylor_couette_mixing/envs/taylor_couette_duty.py):
1-D action (duty D in [0.2, 1]), 2-D obs (X_block, t/t_scale), fixed mean
w_b = 300 with w_low = 0 and T = 5 s pinned, WARMED continuous-manufacturing
episodes. Design + rationale: the duty env's module docstring.

THREE deliberate differences from parallel_train.py:

  1. WARM-ONCE-AND-FAN-OUT (like parallelized_catalysis_rl): template prep
     runs blockMesh + the coded-FO compile, then spins the case w_b x
     --warm_duration (60 s ~ 2.3 tau, ~15 min one-time) and caches 0.warmed/.
     Workers copytree the template, so every hard reset lands on the warmed
     constant-300 steady state (X = 0.353). Cached via sentinel; --rebuild to
     redo. No dependence on any locally-built artifact -- a fresh Carya clone
     warms itself on-node.

  2. CONTINUING TASK: buffer.add stores done = terminated (ALWAYS False) --
     the critic BOOTSTRAPS through truncation (Pardo time-limit treatment).
     This is the deliberate OPPOSITE of parallel_train.py's fixed-horizon
     handling: that fix was specific to the fraction-clock obs, where clock=1
     states could never be source states. Here the horizon is RANDOMIZED
     (--blocks_min/--blocks_max, drawn per episode), so late-t states appear
     as source states in longer episodes and their Q-values stay grounded --
     and no single t value ever signals "episode about to end", which is what
     keeps the raw-time obs from becoming a batch-process clock (the v5-s2
     block-5 D->1 collapse was exactly that artifact).

  3. Reward mode passthrough: --reward_mode conv (default; X_block - P/P_max,
     benchmark-comparable) or flux (mass-balance equivalent, no tau-delay in
     credit assignment). wallFlux is logged either way.

Logging format is unchanged (save_logs + params_per_step.npy with
[duty, w_low_rpm=0, period_s, w_hi_rpm] rows), so the existing analysis
scripts read these runs too. Episodes have VARIABLE length; the per-step
grids are NaN-padded to the longest episode as before.

Example (Carya, 44 workers -- see run_carya_D_modulation.slurm):
  python parallel_train_duty.py --n_workers 44 \
      --worker_root /tmp/$USER/duty_workers --tag duty_v1_s0 \
      --max_episodes 300 --start_timesteps 500 --grad_per_step 64
"""

import argparse
import faulthandler
import os
import shutil
import subprocess
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
    ThreadSafeReplayBuffer, Shared, _copy_master, _blockmesh,
    _snapshot_logs, _save_buffer, learner_loop,
)
from train import make_policy  # noqa: E402
from taylor_couette_mixing.envs.helpers import Helpers  # noqa: E402
from taylor_couette_mixing.envs.taylor_couette_duty import (  # noqa: E402
    TaylorCouetteDutyEnv, RPM,
)

RESULTS_ROOT = os.path.join(EXP_DIR, "results")


# --------------------------------------------------------------------------- #
# One-time case preparation: compile + WARM a template, fan out to N workers.
# --------------------------------------------------------------------------- #
def prepare_worker_cases(master, worker_root, n_workers, rebuild,
                         w_b_rpm, warm_duration):
    """Like parallel_train.prepare_worker_cases but the template is WARMED:
    after the coded-FO compile it runs constant w_b x warm_duration once and
    caches 0.warmed/, so workers hard-reset onto the steady operating state."""
    os.makedirs(worker_root, exist_ok=True)
    template = os.path.join(worker_root, "_template")
    sentinel = os.path.join(template, ".warmed")

    if rebuild and os.path.isdir(template):
        shutil.rmtree(template)

    if not os.path.isfile(sentinel):
        print(f"[setup] building WARMED template at {template} (blockMesh + "
              f"coded-FO compile + {w_b_rpm:.0f} rpm x {warm_duration:.0f} s "
              f"spin-up; one-time) ...", flush=True)
        t0 = time.time()
        _copy_master(master, template)
        _blockmesh(template)
        helpers = Helpers(template)
        helpers._update_end_time(0.05)   # latest time is 0 -> endTime 0.05
        with open(os.path.join(template, "log.compile"), "w") as fh:
            r = subprocess.run(["pimpleFoam"], cwd=template,
                               stdout=fh, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            raise RuntimeError(
                f"template compile run failed in {template} (see log.compile)")
        helpers.reset_case(mode="hard")   # pristine again; dynamicCode/ stays

        # Spin up once and cache. _warmup_case = reset + do_simulation +
        # copytree(latest -> 0.warmed); idempotent if 0.warmed survived rebuild.
        helpers._warmup_case(w_b_rpm * RPM, warm_duration)
        helpers.reset_case(mode="hard")   # 0/ <- 0.warmed, stray time dirs gone
        open(sentinel, "w").close()
        print(f"[setup] template compiled + warmed in {time.time()-t0:.0f}s", flush=True)
    else:
        print(f"[setup] reusing cached warmed template at {template}", flush=True)

    workers = []
    for i in range(n_workers):
        wdir = os.path.join(worker_root, f"worker_{i:02d}")
        if rebuild and os.path.isdir(wdir):
            shutil.rmtree(wdir)
        if not os.path.isdir(wdir):
            shutil.copytree(template, wdir)
        workers.append(wdir)
    print(f"[setup] {n_workers} worker cases ready under {worker_root}", flush=True)
    return workers


def make_env(case_path, args):
    return TaylorCouetteDutyEnv(
        case_path=case_path,
        w_b_rpm=args.w_b_rpm,
        episode_duration=args.blocks_max * args.block_dt,   # per-episode override below
        block_dt=args.block_dt,
        duty_min=args.duty_min, duty_max=args.duty_max,
        period=args.period,
        ramp_time=args.ramp_time,
        p_max_watt=args.p_max_watt,
        t_scale=args.t_scale,
        x_init=args.x_init,
        reward_mode=args.reward_mode,
        flux_to_conv=args.flux_to_conv,
    )


# --------------------------------------------------------------------------- #
# Collector: differs from parallel_train's in the two CONTINUING-TASK points
# (randomized horizon per episode; done = terminated, never truncation).
# --------------------------------------------------------------------------- #
def collector_loop(wid, env, policy, buffer, cfg, shared, stop_event):
    rng = np.random.default_rng(cfg["seed"] + 1000 + wid)
    first = True
    consec_fail = 0
    try:
        while not stop_event.is_set() and consec_fail < cfg["max_fail"]:
            n_blocks = int(rng.integers(cfg["blocks_min"], cfg["blocks_max"] + 1))
            try:
                obs, info = env.reset(
                    seed=(cfg["seed"] + wid) if first else None,
                    options={"reset_mode": "hard", "n_blocks": n_blocks},
                )
                if first:
                    env.action_space.seed(cfg["seed"] + wid)
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
                    action = env.action_space.sample()
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
                # CONTINUING task: done = terminated (always False here), so the
                # critic bootstraps THROUGH truncation -- the deliberate opposite
                # of parallel_train.py (see module docstring). Randomized
                # horizons keep the bootstrapped late-t states grounded.
                buffer.add(state, action, next_state, reward, float(terminated))

                with shared.lock:
                    shared.total_env_steps += 1
                    gstep = shared.total_env_steps
                state = next_state
                ep_ret += reward
                ep_om.append(float(info["w_hi_rpm"]))
                ep_rw.append(float(reward))
                ep_cv.append(float(info["mixing_index"]))    # X_block
                ep_pw.append(float(info["power_watt"]))
                ep_pr.append([float(info["duty"]), float(info["w_low_rpm"]),
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
                          f"ret={ep_ret:+.3f} Xlast={ep_cv[-1]:.3f} "
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
                        "(500 ~ 100 random episodes at the mean 5-block horizon).")
    p.add_argument("--grad_per_step", type=float, default=64.0,
                   help="UTD -- the surrogate-validated v5 recipe carries over.")
    p.add_argument("--max_fail", type=int, default=3)
    p.add_argument("--expl_noise", type=float, default=0.2,
                   help="Gaussian std on the RAW 1-D action (0.2 raw ~ 0.08 duty).")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--discount", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--save_every", type=int, default=4000)
    # env (the settled D-only design; see taylor_couette_duty.py)
    p.add_argument("--w_b_rpm", type=float, default=300.0)
    p.add_argument("--block_dt", type=float, default=10.0)
    p.add_argument("--blocks_min", type=int, default=3,
                   help="episode horizon is drawn uniform in [blocks_min, blocks_max] "
                        "per episode -- randomized truncation keeps late-t states "
                        "grounded and hides the horizon from the raw-time obs.")
    p.add_argument("--blocks_max", type=int, default=7)
    p.add_argument("--duty_min", type=float, default=0.2,
                   help="w_hi = w_b/D <= 1500 rpm at the floor (envelope: 2500).")
    p.add_argument("--duty_max", type=float, default=1.0)
    p.add_argument("--period", type=float, default=5.0,
                   help="PINNED period; 5 s divides block_dt=10 -> exact block means.")
    p.add_argument("--ramp_time", type=float, default=0.05)
    p.add_argument("--p_max_watt", type=float, default=31.94)
    p.add_argument("--t_scale", type=float, default=26.0,
                   help="FIXED physical normalizer for the time obs (tau = V/Q). A "
                        "units choice, NOT episode information; 1.0 = literal seconds.")
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
        args.period = 1.0            # divides the 2 s smoke block
        args.warm_duration = 2.0     # plumbing test, not a steady state
        args.max_timesteps = 12
        args.max_episodes = None
        args.start_timesteps = 4
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
    tag = args.tag if args.tag else f"duty_seed{args.seed}"
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
          f"action_dim={action_dim} D-only fixed-mean w_b={args.w_b_rpm}rpm T={args.period}s "
          f"blocks {args.blocks_min}-{args.blocks_max} x {args.block_dt}s "
          f"reward={args.reward_mode} -> {run_dir}", flush=True)

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
               blocks_min=args.blocks_min, blocks_max=args.blocks_max)
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
