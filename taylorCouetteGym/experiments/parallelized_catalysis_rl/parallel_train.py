"""Parallelized TD3 for the side-outlet catalysis case.

N env workers -- each with its OWN OpenFOAM case copy -- run pimpleFoam rollouts
CONCURRENTLY in threads, pushing transitions into ONE shared replay buffer; a
single TD3 learner trains the actor/critic on the pooled data. TD3 is off-policy,
so this is exactly the AlphaZero pattern (many self-play games -> one buffer ->
one network), specialized to an expensive CFD environment.

Why threads (not processes): env.step() blocks inside subprocess.run(pimpleFoam),
which releases the GIL, so N worker threads run N pimpleFoam processes truly in
parallel. The actor/critic MLP is tiny, so the learner barely competes for CPU.

Warmup happens ONCE: a single spin-up is cached as 0.warmed/ in a template case
(the same run also compiles the coded functionObjects into dynamicCode/), and the
template is COPIED into every worker -- so all workers start from the identical
warmed IC, instantly, with no per-worker warmup and no dynamicCode compile race.
Re-running with the same --worker_root reuses the cached template + workers (pass
--rebuild to force a fresh warmup).

Speedup: wall-clock ~ (serial CFD time) / n_workers. The learner keeps the
gradient-steps : env-steps ratio at --grad_per_step (default 1.0, matching the
serial trainer), so learning dynamics stay comparable -- you just collect N x
faster.

Example (Carya, 16 workers):
  python parallel_train.py --n_workers 16 \
      --case_path taylor_couette_mixing/cases/side_outlet_cat_case \
      --worker_root /tmp/$USER/so_workers --tag so_parallel_s0 \
      --control_mode freeform --warmup_duration 130 --warmup_omega_rpm 500 \
      --max_timesteps 6000 --start_timesteps 600 --save_every 500
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

from train import ReplayBuffer, make_obs_to_state, make_policy, save_logs  # noqa: E402
from taylor_couette_mixing.envs.taylor_couette_catalysis import (  # noqa: E402
    TaylorCouetteCatalysisEnv,
)
from taylor_couette_mixing.envs.taylor_couette_waveform import (  # noqa: E402
    TaylorCouetteWaveformEnv,
)

RESULTS_ROOT = os.path.join(GYM_ROOT, "results")


# --------------------------------------------------------------------------- #
# Thread-safe replay buffer (collectors add, learner samples)
# --------------------------------------------------------------------------- #
class ThreadSafeReplayBuffer(ReplayBuffer):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._lock = threading.Lock()

    def add(self, *a, **k):
        with self._lock:
            super().add(*a, **k)

    def sample(self, *a, **k):
        with self._lock:
            return super().sample(*a, **k)


# --------------------------------------------------------------------------- #
# Shared, mutable run state guarded by locks
# --------------------------------------------------------------------------- #
class Shared:
    def __init__(self):
        self.lock = threading.Lock()          # guards counters + log lists
        self.policy_lock = threading.Lock()   # guards actor/critic (fwd + train)
        self.total_env_steps = 0
        self.live_workers = 0                 # set to n_workers before start; 0 => all done
        # completed-episode logs (append in completion order), fed to save_logs()
        self.episode_returns = []
        self.episode_end_steps = []
        self.omega_history = []
        self.reward_history = []
        self.conv_history = []
        self.power_history = []               # step-average power (W) per episode


# --------------------------------------------------------------------------- #
# One-time case preparation: warm a template, fan it out to N workers
# --------------------------------------------------------------------------- #
def _copy_master(master, dest):
    """Copy the pristine case (0.orig, constant, system) and seed 0/ from 0.orig."""
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.makedirs(dest)
    for sub in ("0.orig", "constant", "system"):
        src = os.path.join(master, sub)
        if not os.path.isdir(src):
            raise FileNotFoundError(f"master case missing {sub}/: {src}")
        shutil.copytree(src, os.path.join(dest, sub))
    shutil.copytree(os.path.join(dest, "0.orig"), os.path.join(dest, "0"))


def _blockmesh(case):
    with open(os.path.join(case, "log.blockMesh"), "w") as fh:
        r = subprocess.run(["blockMesh"], cwd=case, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"blockMesh failed in {case} (see log.blockMesh)")


def prepare_worker_cases(master, worker_root, n_workers, env_kwargs, rebuild):
    """Return a list of N ready-to-run worker case dirs, all seeded from one
    warmed template. Warms exactly once (cached); reuses on re-run unless rebuild."""
    os.makedirs(worker_root, exist_ok=True)
    template = os.path.join(worker_root, "_template")
    warmed = os.path.join(template, "0.warmed")

    if rebuild and os.path.isdir(template):
        shutil.rmtree(template)

    if not os.path.isdir(warmed):
        print(f"[setup] building + warming template at {template} "
              f"(one-time; ~warmup_duration s of pimpleFoam) ...", flush=True)
        _copy_master(master, template)
        _blockmesh(template)
        # Instantiate the env once: __init__ spins the case up and caches 0.warmed/
        # (this run also compiles the coded functionObjects into dynamicCode/).
        t0 = time.time()
        tmp = TaylorCouetteCatalysisEnv(case_path=template, **env_kwargs)
        if not os.path.isdir(warmed):
            raise RuntimeError("warmup did not produce 0.warmed/")
        # Leave the template clean (0/ restored from 0.warmed, stray time dirs gone)
        # so the fan-out copy is small and workers start from a pristine warmed IC.
        tmp.helpers.reset_case(mode="hard")
        del tmp
        print(f"[setup] template warmed in {time.time()-t0:.0f}s", flush=True)
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


def make_env(case_path, args, cat_kwargs):
    """Build a catalysis env on an already-warmed case (skips warmup)."""
    if args.control_mode in ("waveform_adaptive", "waveform_episode"):
        return TaylorCouetteWaveformEnv(
            per_episode=(args.control_mode == "waveform_episode"),
            control_dt=args.control_dt,
            episode_duration=args.episode_duration,
            period_min=args.period_min, period_max=args.period_max,
            duty_min=args.duty_min, duty_max=args.duty_max,
            max_steps=args.max_steps_per_ep, case_path=case_path, **cat_kwargs,
        )
    if args.control_mode == "freeform":
        return TaylorCouetteCatalysisEnv(
            max_steps=args.max_steps_per_ep, time_step=args.freeform_dt,
            clock_in_obs=True, case_path=case_path, **cat_kwargs,
        )
    return TaylorCouetteCatalysisEnv(
        max_steps=args.max_steps_per_ep, case_path=case_path, **cat_kwargs,
    )


# --------------------------------------------------------------------------- #
# Collector: one thread per env, pushes transitions into the shared buffer
# --------------------------------------------------------------------------- #
def collector_loop(wid, env, policy, buffer, obs_to_state, cfg, shared, stop_event):
    rng = np.random.default_rng(cfg["seed"] + 1000 + wid)
    first = True
    consec_fail = 0   # consecutive failed steps/resets; give up on this worker after max_fail
    try:
        while not stop_event.is_set() and consec_fail < cfg["max_fail"]:
            # Seed the env's RNG once (distinct per worker) then let the stream run,
            # so workers explore differently and each is reproducible.
            try:
                obs, info = env.reset(
                    seed=(cfg["seed"] + wid) if first else None,
                    options={"reset_mode": "hard"},
                )
            except Exception as e:
                consec_fail += 1
                print(f"[w{wid:02d}] reset failed ({consec_fail}/{cfg['max_fail']}): "
                      f"{type(e).__name__}: {e}", flush=True)
                traceback.print_exc(); sys.stdout.flush()
                continue
            first = False
            state = obs_to_state(obs)
            ep_ret, ep_om, ep_rw, ep_cv, ep_pw = 0.0, [], [], [], []

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

                # A pimpleFoam divergence raises here. Don't kill the worker: log it,
                # abandon this episode, and the outer loop hard-resets to the warmed IC
                # and starts fresh. Only give up after max_fail consecutive failures.
                try:
                    next_obs, reward, terminated, truncated, info = env.step(action)
                except Exception as e:
                    consec_fail += 1
                    print(f"[w{wid:02d}] step failed ({consec_fail}/{cfg['max_fail']}, "
                          f"pimpleFoam?): {type(e).__name__}: {e}", flush=True)
                    traceback.print_exc(); sys.stdout.flush()
                    break   # abandon this episode -> outer loop resets
                consec_fail = 0   # a good step clears the failure streak

                next_state = obs_to_state(next_obs)
                buffer.add(state, action, next_state, reward, float(terminated))

                with shared.lock:
                    shared.total_env_steps += 1
                    gstep = shared.total_env_steps
                state = next_state
                ep_ret += reward
                ep_om.append(float(next_obs["omega"]))
                ep_rw.append(float(reward))
                ep_cv.append(float(info["mixing_index"]))  # conversion (catalysis)
                ep_pw.append(float(info.get("power_watt", np.nan)))  # step-avg power (W)

                if terminated or truncated:
                    with shared.lock:
                        shared.episode_returns.append(ep_ret)
                        shared.episode_end_steps.append(gstep)
                        shared.omega_history.append(ep_om)
                        shared.reward_history.append(ep_rw)
                        shared.conv_history.append(ep_cv)
                        shared.power_history.append(ep_pw)
                        n_done = len(shared.episode_returns)
                        if cfg["max_episodes"] and n_done >= cfg["max_episodes"]:
                            stop_event.set()   # target episode count reached
                    print(f"[w{wid:02d}] ep {n_done} done ret={ep_ret:+.3f} "
                          f"convmean={np.mean(ep_cv):.3f} steps={gstep}/{cfg['max_timesteps']}",
                          flush=True)
                    break
                if gstep >= cfg["max_timesteps"]:
                    stop_event.set()
                    break
            if shared.total_env_steps >= cfg["max_timesteps"]:
                stop_event.set()
        if consec_fail >= cfg["max_fail"]:
            print(f"[w{wid:02d}] giving up after {consec_fail} consecutive failures", flush=True)
    finally:
        # A worker leaving must never strand the learner: when the last one exits,
        # signal stop so the learner drains its backlog and finalizes.
        with shared.lock:
            shared.live_workers -= 1
            remaining = shared.live_workers
            if remaining <= 0:
                stop_event.set()
        print(f"[w{wid:02d}] exited (live workers now {remaining})", flush=True)


# --------------------------------------------------------------------------- #
# Learner: paces gradient steps to collected transitions; saves periodically
# --------------------------------------------------------------------------- #
def _snapshot_logs(run_dir, shared):
    with shared.lock:
        args = (list(shared.episode_returns), list(shared.episode_end_steps),
                [list(e) for e in shared.omega_history],
                [list(e) for e in shared.reward_history])
        conv = [list(e) for e in shared.conv_history]
        power = [list(e) for e in shared.power_history]
    save_logs(run_dir, *args, [], [], conv_history=conv, ep_convs=[],
              power_history=power, ep_powers=[])


def learner_loop(policy, buffer, cfg, shared, stop_event, run_dir, ckpt_prefix):
    grad_steps, last_save, t0 = 0, 0, time.time()
    while True:
        with shared.lock:
            collected = shared.total_env_steps
        target = int(cfg["grad_per_step"] * max(0, collected - cfg["start_timesteps"]))
        did_work = False
        while grad_steps < target and buffer.size >= cfg["batch_size"]:
            with shared.policy_lock:
                policy.train(buffer, cfg["batch_size"])
            grad_steps += 1
            did_work = True
            if grad_steps - last_save >= cfg["save_every"]:
                policy.save(f"{ckpt_prefix}_t{grad_steps}")
                _snapshot_logs(run_dir, shared)
                rate = collected / max(time.time() - t0, 1e-9)
                print(f"[learner] grad={grad_steps} collected={collected} "
                      f"buf={buffer.size} rate={rate:.2f} env-steps/s saved t{grad_steps}",
                      flush=True)
                last_save = grad_steps
        # Exit once collection is finished AND either we've caught up to the target
        # OR there is not yet a full batch to train on (e.g. a tiny/smoke run) --
        # otherwise the learner would wait forever for grad steps it can never do.
        if stop_event.is_set() and (grad_steps >= target or buffer.size < cfg["batch_size"]):
            break
        if not did_work:
            time.sleep(0.05)   # waiting on collectors; don't busy-spin
    return grad_steps


# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # parallelism / case layout
    p.add_argument("--n_workers", type=int, default=16,
                   help="number of concurrent env workers (one pimpleFoam each). "
                        "Set to the cores you have.")
    p.add_argument("--worker_root", default=None,
                   help="dir to hold _template/ + worker_NN/ case copies. Defaults "
                        "to <results_dir>/<algo>/<tag>/workers. Reused across runs "
                        "(cached warmup) unless --rebuild.")
    p.add_argument("--rebuild", action="store_true",
                   help="delete + rebuild the template and worker cases (forces a "
                        "fresh warmup).")
    p.add_argument("--case_path",
                   default="taylor_couette_mixing/cases/side_outlet_cat_case",
                   help="master (pristine) OpenFOAM case to copy from.")
    # agent / algo
    p.add_argument("--algo", choices=["td3", "ddpg"], default="td3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_timesteps", type=int, default=6000,
                   help="TOTAL env steps to collect across ALL workers (upper safety "
                        "bound when --max_episodes is set).")
    p.add_argument("--max_episodes", type=int, default=None,
                   help="stop once this many episodes have COMPLETED (across all workers). "
                        "The primary stop when set; workers still in an episode at that "
                        "moment truncate, so you get ~this many completed episodes exactly.")
    p.add_argument("--start_timesteps", type=int, default=600,
                   help="random-action steps (total) before the policy + learning kick in.")
    p.add_argument("--grad_per_step", type=float, default=1.0,
                   help="gradient steps per collected env step (1.0 = serial-equivalent).")
    p.add_argument("--max_fail", type=int, default=3,
                   help="a worker recovers (hard reset) from a diverged pimpleFoam step; "
                        "after this many CONSECUTIVE failures it gives up and exits.")
    p.add_argument("--expl_noise", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--discount", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--save_every", type=int, default=500, help="save every N grad steps.")
    # env / reward (mirror train.py; side-outlet defaults)
    p.add_argument("--control_mode",
                   choices=["omega", "freeform", "waveform_adaptive", "waveform_episode"],
                   default="freeform")
    p.add_argument("--max_steps_per_ep", type=int, default=60)
    p.add_argument("--freeform_dt", type=float, default=1.0)
    p.add_argument("--warmup_duration", type=float, default=80.0,
                   help="one-time spin-up seconds. Shortened side-outlet @100 mL/min: "
                        "residence tau~26 s, so ~3 tau=80 s reaches steady state.")
    p.add_argument("--warmup_omega_rpm", type=float, default=500.0)
    p.add_argument("--r_in", type=float, default=25.4)
    p.add_argument("--r_out", type=float, default=31.75)
    p.add_argument("--feed_velocity", type=float, default=1.462e-3)  # Q0=100 mL/min (Lopez)
    p.add_argument("--wallflux_max", type=float, default=None)
    p.add_argument("--e_max_per_step", type=float, default=0.0011017031875434)
    p.add_argument("--conv_weight", type=float, default=1.0)
    p.add_argument("--energy_weight", type=float, default=0.1)
    p.add_argument("--energy_model", choices=["motor", "mechanical"], default="motor")
    p.add_argument("--ramp_time", type=float, default=0.05)
    # waveform-only
    p.add_argument("--control_dt", type=float, default=10.0)
    p.add_argument("--episode_duration", type=float, default=60.0)
    p.add_argument("--period_min", type=float, default=5.0)
    p.add_argument("--period_max", type=float, default=30.0)
    p.add_argument("--duty_min", type=float, default=0.1)
    p.add_argument("--duty_max", type=float, default=1.0)
    # output
    p.add_argument("--results_dir", default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--smoke", action="store_true",
                   help="tiny end-to-end validation: 2 workers, short warmup + budget.")
    return p


def main():
    args = build_parser().parse_args()
    if args.smoke:
        args.n_workers = 2
        args.warmup_duration = 3.0
        args.max_steps_per_ep = 3
        args.max_timesteps = 12
        args.start_timesteps = 4
        args.save_every = 2
        print("[smoke] n_workers=2 warmup=3s budget tiny", flush=True)

    faulthandler.enable()
    if os.environ.get("PT_DEBUG_HANG"):
        # every N s, dump ALL thread stacks to stderr so a hang is diagnosable
        faulthandler.dump_traceback_later(int(os.environ["PT_DEBUG_HANG"]), repeat=True)

    torch.set_num_threads(1)   # 16 pimpleFoam already saturate the cores
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    results_root = args.results_dir if args.results_dir else RESULTS_ROOT
    tag = args.tag if args.tag else f"seed{args.seed}"
    run_dir = os.path.join(results_root, args.algo, tag)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_prefix = os.path.join(run_dir, f"{args.algo}_tc")
    worker_root = args.worker_root or os.path.join(run_dir, "workers")

    cat_kwargs = dict(
        r_in=args.r_in, r_out=args.r_out,
        feed_velocity=args.feed_velocity, wallflux_max=args.wallflux_max,
        E_max_per_step=args.e_max_per_step,
        warmup_duration=args.warmup_duration, warmup_omega_rpm=args.warmup_omega_rpm,
        conv_weight=args.conv_weight, energy_weight=args.energy_weight,
        energy_model=args.energy_model, ramp_time=args.ramp_time,
    )
    # env kwargs for the one-time template warmup (freeform sets time_step so the
    # warmup writeInterval matches what the workers will use).
    warm_kwargs = dict(cat_kwargs, max_steps=args.max_steps_per_ep)
    if args.control_mode == "freeform":
        warm_kwargs.update(time_step=args.freeform_dt, clock_in_obs=True)

    master = os.path.join(GYM_ROOT, args.case_path) if not os.path.isabs(args.case_path) \
        else args.case_path
    workers = prepare_worker_cases(master, worker_root, args.n_workers,
                                   warm_kwargs, args.rebuild)

    # Build the N worker envs (all warmed -> cheap) and derive dims from one.
    envs = [make_env(w, args, cat_kwargs) for w in workers]
    obs0, _ = envs[0].reset(seed=args.seed, options={"reset_mode": "hard"})
    energy_norm = getattr(envs[0], "energy_obs_norm", envs[0].E_max_per_step) * envs[0].max_steps
    obs_to_state = make_obs_to_state(
        omega_max=envs[0].omega_max, energy_norm=energy_norm,
        period_logmin=getattr(envs[0], "_logTmin", None),
        period_logmax=getattr(envs[0], "_logTmax", None),
    )
    state_dim = obs_to_state(obs0).shape[0]
    action_dim = envs[0].action_space.shape[0]
    max_action = float(envs[0].action_space.high[0])
    print(f"[{args.algo}] tag={tag} workers={args.n_workers} state_dim={state_dim} "
          f"action_dim={action_dim} mode={args.control_mode} -> {run_dir}", flush=True)

    policy = make_policy(args.algo, state_dim, action_dim, max_action, args.discount, args.tau)
    buffer = ThreadSafeReplayBuffer(state_dim, action_dim)
    shared = Shared()
    stop_event = threading.Event()
    cfg = dict(seed=args.seed, start_timesteps=args.start_timesteps,
               max_timesteps=args.max_timesteps, expl_noise=args.expl_noise,
               max_action=max_action, action_dim=action_dim,
               batch_size=args.batch_size, grad_per_step=args.grad_per_step,
               save_every=args.save_every, max_fail=args.max_fail,
               max_episodes=args.max_episodes)
    shared.live_workers = args.n_workers

    threads = [
        threading.Thread(target=collector_loop, name=f"w{i:02d}",
                         args=(i, envs[i], policy, buffer, obs_to_state, cfg, shared, stop_event),
                         daemon=True)
        for i in range(args.n_workers)
    ]
    t0 = time.time()
    for t in threads:
        t.start()

    grad_steps = learner_loop(policy, buffer, cfg, shared, stop_event, run_dir, ckpt_prefix)
    # Collectors are daemons; join with a timeout so a single slow/stuck pimpleFoam
    # can't block finalization (any straggler dies when the process exits).
    for t in threads:
        t.join(timeout=600)
    # final catch-up + save (collection is done; drain any remaining grad steps)
    target = int(cfg["grad_per_step"] * max(0, shared.total_env_steps - args.start_timesteps))
    while grad_steps < target and buffer.size >= args.batch_size:
        policy.train(buffer, args.batch_size)
        grad_steps += 1
    policy.save(f"{ckpt_prefix}_final")
    _snapshot_logs(run_dir, shared)
    print(f"[done] collected={shared.total_env_steps} grad_steps={grad_steps} "
          f"episodes={len(shared.episode_returns)} wall={time.time()-t0:.0f}s -> {run_dir}",
          flush=True)


if __name__ == "__main__":
    main()
