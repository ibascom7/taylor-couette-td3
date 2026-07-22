"""Parallelized TD3 over block-wise waveform modulation.

Same harness as experiments/parallelized_catalysis_rl/parallel_train.py -- N env
workers, each with its OWN OpenFOAM case copy, run pimpleFoam rollouts
CONCURRENTLY in threads, pushing transitions into ONE shared replay buffer; a
single TD3 learner trains on the pooled data -- but the environment is
TaylorCouetteModulationEnv (see taylor_couette_mixing/envs/
taylor_couette_modulation.py and experiments/modulation_rl/README.md):
5 blocks x 10 s per 50 s episode, reward X_block - P_block/P_max. Two action
modes: FREE-MEAN (default, --w_b_rpm omitted): 3-D action (duty, nominal mean
w_nom, log-period) -- the agent picks its LOW/mean speed per block and the env
converts it to the burst w_hi = w_nom/D (idle pinned at 0). Fixed-mean
(--w_b_rpm <rpm>): 3-D action (duty, idle speed, log-period) with the burst
speed solved from the fixed-mean constraint.

TWO deliberate differences from the catalysis harness:
  1. NO warmup / no 0.warmed. Episodes start from the PRISTINE pre-filled IC
     (0.orig) so they are bit-comparable to the static-grid benchmark episodes
     (fig7_sweep_td3_prep: constant-300 X=0.3402, D=80 T=2.5 champion R=0.270).
     Template prep therefore only runs blockMesh + a 0.05 s throwaway pimpleFoam
     to compile the coded functionObjects into dynamicCode/ (fanned out to all
     workers, so there is no per-worker compile race).
  2. The env emits an already-flat, already-normalized 7-D observation, so
     obs_to_state is the identity (train.py's make_obs_to_state is not used).

Logging: the shared save_logs format is kept (episode_returns / reward_per_step
/ conv_per_step / power_per_step; conv = X_block, power = block-average motor W).
The omega_per_step slot records the decoded BURST speed w_hi [rpm] (the most
informative scalar; the commanded mean is w_b by construction). A additional
params_per_step.npy [episode, step, (duty, w_low_rpm, period_s, w_hi_rpm)]
records the full decoded action.

Example (Carya, 44 workers):
  python parallel_train.py --n_workers 44 \
      --case_path taylor_couette_mixing/cases/side_outlet_grad_case \
      --worker_root /tmp/$USER/mod_workers --tag mod_wb300_s0 \
      --max_episodes 1000 --max_timesteps 6000 --start_timesteps 500
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

from train import ReplayBuffer, make_policy, save_logs  # noqa: E402
from taylor_couette_mixing.envs.helpers import Helpers  # noqa: E402
from taylor_couette_mixing.envs.taylor_couette_modulation import (  # noqa: E402
    TaylorCouetteModulationEnv,
)

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.join(EXP_DIR, "results")


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
        # completed-episode logs (append in completion order)
        self.episode_returns = []
        self.episode_end_steps = []
        self.omega_history = []               # w_hi [rpm] per block
        self.reward_history = []
        self.conv_history = []                # X_block per block
        self.power_history = []               # block-average motor power [W]
        self.params_history = []              # [duty, w_low_rpm, period_s, w_hi_rpm] per block


# --------------------------------------------------------------------------- #
# One-time case preparation: compile a template, fan it out to N workers.
# NO warmup -- episodes run from the pristine pre-filled IC (see module doc).
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


def prepare_worker_cases(master, worker_root, n_workers, rebuild):
    """Return N ready-to-run worker case dirs, all copied from one compiled
    template. The template runs pimpleFoam for one 0.05 s throwaway step so the
    coded functionObjects compile into dynamicCode/ exactly once (no per-worker
    compile race), then hard-resets to the pristine IC. Cached across re-runs
    with the same --worker_root unless --rebuild."""
    os.makedirs(worker_root, exist_ok=True)
    template = os.path.join(worker_root, "_template")
    sentinel = os.path.join(template, ".compiled")

    if rebuild and os.path.isdir(template):
        shutil.rmtree(template)

    if not os.path.isfile(sentinel):
        print(f"[setup] building template at {template} "
              f"(blockMesh + one-time coded-FO compile) ...", flush=True)
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
        # Back to the pristine IC (time dirs wiped, endTime 0); dynamicCode/ stays.
        helpers.reset_case(mode="hard")
        open(sentinel, "w").close()
        print(f"[setup] template compiled in {time.time()-t0:.0f}s", flush=True)
    else:
        print(f"[setup] reusing cached compiled template at {template}", flush=True)

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
    return TaylorCouetteModulationEnv(
        case_path=case_path,
        w_b_rpm=args.w_b_rpm,
        episode_duration=args.episode_duration,
        block_dt=args.block_dt,
        duty_min=args.duty_min, duty_max=args.duty_max,
        idle_min_rpm=args.idle_min_rpm, idle_max_rpm=args.idle_max_rpm,
        nom_min_rpm=args.nom_min_rpm, nom_max_rpm=args.nom_max_rpm,
        period_min=args.period_min, period_max=args.period_max,
        ramp_time=args.ramp_time,
        p_max_watt=args.p_max_watt,
        wallflux_max=args.wallflux_max,
    )


# --------------------------------------------------------------------------- #
# Collector: one thread per env, pushes transitions into the shared buffer
# --------------------------------------------------------------------------- #
def collector_loop(wid, env, policy, buffer, cfg, shared, stop_event):
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

                # A pimpleFoam divergence raises here. Don't kill the worker: log it,
                # abandon this episode, and the outer loop hard-resets to the pristine
                # IC and starts fresh. Only give up after max_fail consecutive failures.
                try:
                    next_obs, reward, terminated, truncated, info = env.step(action)
                except Exception as e:
                    consec_fail += 1
                    print(f"[w{wid:02d}] step failed ({consec_fail}/{cfg['max_fail']}, "
                          f"pimpleFoam?): {type(e).__name__}: {e}", flush=True)
                    traceback.print_exc(); sys.stdout.flush()
                    break   # abandon this episode -> outer loop resets
                consec_fail = 0   # a good step clears the failure streak

                next_state = np.asarray(next_obs, dtype=np.float32)
                # Fixed-horizon task with the clock in the obs: the horizon end
                # ends the value chain. clock=1 states never occur as SOURCE
                # states, so bootstrapping into them (done=0) regresses toward
                # an ungrounded Q that drifts freely and drags the actor to an
                # arbitrary action corner (s0/v2/v3 all pinned raw (-1,+1,+1);
                # v3's actor even saturated its DEAD idle dim -- proven 2026-07-21).
                buffer.add(state, action, next_state, reward,
                           float(terminated or truncated))

                with shared.lock:
                    shared.total_env_steps += 1
                    gstep = shared.total_env_steps
                state = next_state
                ep_ret += reward
                ep_om.append(float(info["w_hi_rpm"]))
                ep_rw.append(float(reward))
                ep_cv.append(float(info["mixing_index"]))    # X_block
                ep_pw.append(float(info["power_watt"]))      # block-avg motor power (W)
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
                            stop_event.set()   # target episode count reached
                    print(f"[w{wid:02d}] ep {n_done} done ret={ep_ret:+.3f} "
                          f"Xlast={ep_cv[-1]:.3f} lastblock "
                          f"D={ep_pr[-1][0]:.2f} wlo={ep_pr[-1][1]:.0f} T={ep_pr[-1][2]:.2f} "
                          f"steps={gstep}/{cfg['max_timesteps']}",
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
def _save_params(run_dir, params_history):
    """params_per_step.npy: [episode, step, (duty, w_low_rpm, period_s, w_hi_rpm)],
    NaN-padded like the other per-step grids."""
    if not params_history:
        return
    width = max(len(ep) for ep in params_history)
    grid = np.full((len(params_history), width, 4), np.nan)
    for i, ep in enumerate(params_history):
        grid[i, :len(ep)] = ep
    np.save(os.path.join(run_dir, "params_per_step.npy"), grid)


def _snapshot_logs(run_dir, shared):
    with shared.lock:
        args = (list(shared.episode_returns), list(shared.episode_end_steps),
                [list(e) for e in shared.omega_history],
                [list(e) for e in shared.reward_history])
        conv = [list(e) for e in shared.conv_history]
        power = [list(e) for e in shared.power_history]
        params = [list(e) for e in shared.params_history]
    save_logs(run_dir, *args, [], [], conv_history=conv, ep_convs=[],
              power_history=power, ep_powers=[])
    _save_params(run_dir, params)


def _save_buffer(run_dir, buffer):
    """Persist the replay buffer so every run doubles as an offline dataset
    (surrogate refits, offline learner iteration). ~250 KB at this scale."""
    with buffer._lock:
        n = buffer.size
        arrs = {k: getattr(buffer, k)[:n].copy()
                for k in ("state", "action", "next_state", "reward", "not_done")}
    np.savez_compressed(os.path.join(run_dir, "replay_buffer.npz"), **arrs)


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
                _save_buffer(run_dir, buffer)
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
                        "(cached compile) unless --rebuild.")
    p.add_argument("--rebuild", action="store_true",
                   help="delete + rebuild the template and worker cases.")
    p.add_argument("--case_path",
                   default="taylor_couette_mixing/cases/side_outlet_grad_case",
                   help="master (pristine) RL-drivable case to copy from. Must emit "
                        "METRICS lines (side_outlet_grad_case = the RL twin of "
                        "side_outlet_case_sc1075_graded).")
    # agent / algo
    p.add_argument("--algo", choices=["td3", "ddpg"], default="td3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_timesteps", type=int, default=6000,
                   help="TOTAL env steps (blocks) to collect across ALL workers "
                        "(upper safety bound when --max_episodes is set).")
    p.add_argument("--max_episodes", type=int, default=1000,
                   help="stop once this many episodes have COMPLETED (across all "
                        "workers). The primary stop.")
    p.add_argument("--start_timesteps", type=int, default=500,
                   help="random-action steps (total) before the policy + learning "
                        "kick in. 500 = 100 fully-random episodes.")
    p.add_argument("--grad_per_step", type=float, default=1.0,
                   help="gradient steps per collected env step (1.0 = serial-equivalent).")
    p.add_argument("--max_fail", type=int, default=3,
                   help="a worker recovers (hard reset) from a diverged pimpleFoam "
                        "step; after this many CONSECUTIVE failures it gives up.")
    p.add_argument("--expl_noise", type=float, default=0.1,
                   help="Gaussian exploration noise (std) per RAW action dim. The "
                        "period dim is log-mapped, so this explores octaves of T "
                        "evenly.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--discount", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--save_every", type=int, default=500, help="save every N grad steps.")
    # env (modulation design; defaults = the settled README spec)
    p.add_argument("--w_b_rpm", type=float, default=None,
                   help="FIXED nominal (mean) speed [rpm]; the burst speed is then "
                        "solved from it every block. OMIT for the FREE-MEAN action "
                        "space (v5): a1 chooses the nominal mean w_nom per block "
                        "(the low speed; the env CONVERTS it to the burst "
                        "w_hi = w_nom/D), idle pinned at --idle_min_rpm, and the "
                        "X - P/P_max reward arbitrates power natively.")
    p.add_argument("--nom_min_rpm", type=float, default=0.0,
                   help="free-mean only: lower end of the a1 -> w_nom map.")
    p.add_argument("--nom_max_rpm", type=float, default=500.0,
                   help="free-mean only: upper end of the a1 -> w_nom map "
                        "(500 = benchmark-grid ceiling; bursts reach w_nom/D "
                        "<= 2500 rpm, the proven fig7 envelope = P_max).")
    p.add_argument("--episode_duration", type=float, default=50.0)
    p.add_argument("--block_dt", type=float, default=10.0,
                   help="seconds per control block (5 decisions per 50 s episode).")
    p.add_argument("--duty_min", type=float, default=0.6)
    p.add_argument("--duty_max", type=float, default=1.0)
    p.add_argument("--idle_min_rpm", type=float, default=0.0)
    p.add_argument("--idle_max_rpm", type=float, default=None,
                   help="default None -> w_b.")
    p.add_argument("--period_min", type=float, default=0.5,
                   help="T floor 0.5 s: below it the gap-scale vortices low-pass the "
                        "forcing and the 0.05 s ramps consume the waveform.")
    p.add_argument("--period_max", type=float, default=5.0)
    p.add_argument("--ramp_time", type=float, default=0.05)
    p.add_argument("--p_max_watt", type=float, default=31.94,
                   help="global reward normalizer: motor power at 2500 rpm.")
    p.add_argument("--wallflux_max", type=float, default=1.32e-8,
                   help="obs normalizer: steady resolved wallFlux at 2500 rpm "
                        "(mass balance Q_wedge * conv(2500) = 2.315e-8 * 0.568).")
    # output
    p.add_argument("--results_dir", default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--smoke", action="store_true",
                   help="tiny end-to-end validation: 2 workers, 6 s episodes of "
                        "3 x 2 s blocks, tiny budget (~15 min of real CFD).")
    return p


def main():
    args = build_parser().parse_args()
    if args.smoke:
        args.n_workers = 2
        args.episode_duration = 6.0
        args.block_dt = 2.0
        args.max_timesteps = 12
        args.max_episodes = None
        args.start_timesteps = 4
        args.batch_size = 4
        args.save_every = 2
        print("[smoke] n_workers=2 episodes=3x2s budget tiny", flush=True)

    faulthandler.enable()
    if os.environ.get("PT_DEBUG_HANG"):
        # every N s, dump ALL thread stacks to stderr so a hang is diagnosable
        faulthandler.dump_traceback_later(int(os.environ["PT_DEBUG_HANG"]), repeat=True)

    torch.set_num_threads(1)   # the pimpleFoam workers saturate the cores
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    results_root = args.results_dir if args.results_dir else RESULTS_ROOT
    tag = args.tag if args.tag else f"seed{args.seed}"
    run_dir = os.path.join(results_root, args.algo, tag)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_prefix = os.path.join(run_dir, f"{args.algo}_tc")
    worker_root = args.worker_root or os.path.join(run_dir, "workers")

    master = os.path.join(GYM_ROOT, args.case_path) if not os.path.isabs(args.case_path) \
        else args.case_path
    workers = prepare_worker_cases(master, worker_root, args.n_workers, args.rebuild)

    # Build the N worker envs and derive dims from one (obs is already flat).
    envs = [make_env(w, args) for w in workers]
    obs0, _ = envs[0].reset(seed=args.seed, options={"reset_mode": "hard"})
    state_dim = np.asarray(obs0).shape[0]
    action_dim = envs[0].action_space.shape[0]
    max_action = float(envs[0].action_space.high[0])
    mode = ("FREE-MEAN w_nom<=%grpm" % args.nom_max_rpm if args.w_b_rpm is None
            else f"fixed-mean w_b={args.w_b_rpm}rpm")
    print(f"[{args.algo}] tag={tag} workers={args.n_workers} state_dim={state_dim} "
          f"action_dim={action_dim} {mode} "
          f"blocks={envs[0].max_steps}x{args.block_dt}s -> {run_dir}", flush=True)

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
               max_episodes=args.max_episodes)
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
    _save_buffer(run_dir, buffer)
    print(f"[done] collected={shared.total_env_steps} grad_steps={grad_steps} "
          f"episodes={len(shared.episode_returns)} wall={time.time()-t0:.0f}s -> {run_dir}",
          flush=True)


if __name__ == "__main__":
    main()
