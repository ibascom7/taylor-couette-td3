#!/usr/bin/env python3
"""A -- FAIR STATIC CHAMPION SEARCH: Bayesian optimization over (T+, T-).

WHY
    The static champion used as the RL comparator so far (T = 5 s, D = 0.90,
    w_b = 300; sustained r = 0.2948) is ONE run from a 7-point duty grid at a
    hand-picked period. Claiming it is "the best fixed pulse" is not
    defensible. This search finds the best time-invariant waveform in the SAME
    family and under the SAME metric the duty_v3 policies were evaluated on,
    so the RL-vs-static comparison is fair in both directions:

      * family : square wave, trough w_low = 0, fixed commanded mean w_b = 300
                 rpm, burst T+ in [1, 5] s, idle T- in [0, 5] s (T- = 0 is
                 constant-300). Burst speed solved for the exact block mean,
                 exactly as TaylorCouetteDutyV2Env does for the RL runs.
      * metric : 130 s episode from the WARMED constant-300 steady state,
                 5 x 26 s tau-blocks, objective = LAST-BLOCK reward
                 r5 = X_[104,130] - P/31.94 (the duty_v3 gate convention).
                 X and P per block are logged so any other window is
                 recoverable.

    One decision per episode + one scalar reward = a 2-D noisy black-box
    problem, so the optimizer is a Gaussian process (Matern-5/2 ARD, numpy
    only -- no scipy/sklearn on Carya) with expected improvement, run
    ASYNCHRONOUSLY over N worker cases: whenever a worker frees, the GP is
    refit on everything finished and the pending points are fantasized at
    their posterior mean (Kriging believer), so no worker ever idles at a
    "round boundary". Budget = n_init space-filling points (anchored on the
    known grid points for cross-checks) + n_bo GP-guided points + replicates
    of the top-k for a measured noise floor.

    Replicates: the solver is deterministic and 300 rpm is calm TVF, so an
    identical input reproduces almost exactly. Each replicate therefore
    perturbs the commanded mean by +-rep_jitter_rpm (0.5 rpm default: a
    0.3 % power change, ~3e-4 in r) to seed any chaotic divergence -- the
    replicate spread is the run-to-run floor, the block-to-block spread the
    flutter floor; both are reported.

OUTPUTS (results/A/ by default)
    evaluations.csv          every finished episode (append-only; --resume
                             reloads it and continues the budget)
    episodes/ep####_blocks.csv, ep####_timeseries.csv   per-episode detail
    A_surface.png            GP posterior mean + sd over (T+, T-), samples,
                             posterior argmax, the anchors, the T- = 0.5 s
                             idle-hypothesis line
    A_slices.png             r vs T- at fixed T+ and r vs T+ at fixed T-
    A_idle_collapse.png      observed r vs idle duration / duty (the
                             off-diagonal test the idle hypothesis needs)
    A_convergence.png        best-so-far vs evaluation index
    top_table.csv, replicates.csv, summary.json, headline printed

USAGE
    # Carya (44 workers, ~20-24 h; submit FROM taylorCouetteGym/):
    sbatch experiments/endgame/run_carya_A_champion_search.slurm
    # local (12 workers, ~2 days):
    nohup python3 -u experiments/endgame/a_champion_search.py --n_workers 12 \
        --worker_root /tmp/$USER/endgame_A_workers > experiments/endgame/logs/A_local.log 2>&1 &
    # pipeline check (~15 min, real pimpleFoam, 2 workers, 4 s episodes):
    python3 experiments/endgame/a_champion_search.py --smoke --worker_root /tmp/$USER/endgame_A_smoke
    # replot / refit from evaluations.csv (no CFD, no torch):
    python3 experiments/endgame/a_champion_search.py --analyze-only
"""

import argparse
import csv
import json
import math
import os
import queue
import shutil
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
GYM_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
MOD_RL = os.path.join(GYM_ROOT, "experiments", "modulation_rl")
for _p in (GYM_ROOT, MOD_RL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from taylor_couette_mixing.envs.taylor_couette_duty_v2 import (  # noqa: E402
    TaylorCouetteDutyV2Env, solve_block_wave, RPM,
)

# ---- the duty_v3 environment, verbatim (run_carya_duty_v3.slurm) -----------
MASTER_CASE = os.path.join(GYM_ROOT, "taylor_couette_mixing", "cases", "side_outlet_grad_case")
W_B = 300.0
BLOCK_DT = 26.0                 # one residence time tau = V/Q
N_BLOCKS = 5                    # 130 s episodes
T_PLUS_MIN, T_PLUS_MAX = 1.0, 5.0
T_MINUS_MIN, T_MINUS_MAX = 0.0, 5.0
W_HI_CAP = 2500.0
RAMP = 0.05
P_MAX = 31.94
T_SCALE = 26.0
X_INIT = 0.353
WARM_DURATION = 60.0            # constant-300 spin-up cached as 0.warmed/

# Known grid points, re-run inside the search as cross-checks against
# results/duty_v3_refs (26 s windows) and results/duty_diag.
ANCHORS = [
    ("constant_300",   3.0,  0.0),    # T- = 0 -> constant w_b
    ("champ_D90_T5",   4.5,  0.5),    # the current champion (duty_v3_refs: r 0.2948)
    ("D80_T2p5",       2.0,  0.5),    # the 50 s "rapid shimmy" champion
    ("D95_T5",         4.75, 0.25),
    ("D80_T5",         4.0,  1.0),
    ("LGlike_D20_T5",  1.0,  4.0),    # Lopez-Guajardo's duty inside this box
]

REFS = dict(
    constant_r=0.24412, constant_x=0.36123,       # duty_v3_refs, [104,130] s
    champion_r=0.29476, champion_x=0.41092,       # T=5 D=0.90 (the n=1 champion)
    flutter_sigma=0.00231,                        # champion's 26 s window-to-window sd
    duty_v3_eval_r=[0.2793, 0.2834, 0.2840, 0.2729, 0.2819],   # deterministic seeds 0-4
)

EVAL_FIELDS = ["ep", "kind", "rep_of", "t_plus_s", "t_minus_s", "w_b_rpm",
               "duty", "period_s", "w_hi_rpm", "r_last", "X_last", "P_last_W",
               "r_mean", "X_mean", "X_blocks", "r_blocks", "wall_min", "worker",
               "status"]


# =========================================================================== #
# Gaussian process (numpy only): Matern-5/2 ARD on the unit square,
# standardized targets, hyperparameters by grid-search marginal likelihood.
# =========================================================================== #
def to_unit(tp, tm):
    return np.array([(tp - T_PLUS_MIN) / (T_PLUS_MAX - T_PLUS_MIN),
                     (tm - T_MINUS_MIN) / (T_MINUS_MAX - T_MINUS_MIN)], dtype=float)


def from_unit(u):
    u = np.asarray(u, dtype=float)
    return (T_PLUS_MIN + u[..., 0] * (T_PLUS_MAX - T_PLUS_MIN),
            T_MINUS_MIN + u[..., 1] * (T_MINUS_MAX - T_MINUS_MIN))


def _matern52(A, B, ls):
    d = np.sqrt((((A[:, None, :] - B[None, :, :]) / ls) ** 2).sum(-1))
    s = math.sqrt(5.0) * d
    return (1.0 + s + s * s / 3.0) * np.exp(-s)


_erf = np.vectorize(math.erf)


def _norm_cdf(z):
    return 0.5 * (1.0 + _erf(z / math.sqrt(2.0)))


def _norm_pdf(z):
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


class GP:
    LS_GRID = np.logspace(math.log10(0.05), math.log10(2.0), 9)
    NOISE_GRID = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1])

    def __init__(self):
        self.ls = np.array([0.3, 0.3])
        self.noise = 1e-2
        self.X = None

    def _chol(self, X, yn, ls, noise):
        K = _matern52(X, X, ls) + (noise + 1e-8) * np.eye(len(X))
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, yn))
        return L, alpha

    def _lml(self, X, yn, ls, noise):
        try:
            L, alpha = self._chol(X, yn, ls, noise)
        except np.linalg.LinAlgError:
            return -np.inf
        return float(-0.5 * yn @ alpha - np.log(np.diag(L)).sum()
                     - 0.5 * len(X) * math.log(2 * math.pi))

    def fit(self, X, y, refit_hyper=True):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float)
        self.ym = float(y.mean())
        self.ys = float(y.std()) if y.std() > 1e-12 else 1.0
        yn = (y - self.ym) / self.ys
        if refit_hyper and len(X) >= 4:
            best = (-np.inf, self.ls, self.noise)
            for l0 in self.LS_GRID:
                for l1 in self.LS_GRID:
                    ls = np.array([l0, l1])
                    for nz in self.NOISE_GRID:
                        v = self._lml(X, yn, ls, nz)
                        if v > best[0]:
                            best = (v, ls, nz)
            _, self.ls, self.noise = best
        self.X, self.yn = X, yn
        self.L, self.alpha = self._chol(X, yn, self.ls, self.noise)
        return self

    def predict(self, Xs):
        Xs = np.atleast_2d(np.asarray(Xs, dtype=float))
        ks = _matern52(Xs, self.X, self.ls)
        mu = ks @ self.alpha
        v = np.linalg.solve(self.L, ks.T)
        var = np.clip(1.0 - (v * v).sum(0), 1e-12, None)
        return mu * self.ys + self.ym, np.sqrt(var) * self.ys

    def noise_sigma(self):
        """Learned observation noise in reward units."""
        return math.sqrt(self.noise) * self.ys


def expected_improvement(mu, sd, best):
    z = (mu - best) / sd
    return (mu - best) * _norm_cdf(z) + sd * _norm_pdf(z)


def latin_hypercube(n, rng):
    u = np.empty((n, 2))
    for j in range(2):
        u[:, j] = (rng.permutation(n) + rng.random(n)) / n
    return u


def candidate_set(rng, n_random=6000, n_grid=51):
    g = np.linspace(0.0, 1.0, n_grid)
    grid = np.array([[a, b] for a in g for b in g])
    return np.vstack([grid, rng.random((n_random, 2))])


def propose_next(done_X, done_y, pending_X, rng, min_dist=0.015):
    """Kriging-believer EI: fantasize pending points at the posterior mean,
    refit with the same hyperparameters, maximize EI over a candidate set
    away from every evaluated/pending point."""
    gp = GP().fit(done_X, done_y)
    if len(pending_X):
        mu_p, _ = gp.predict(pending_X)
        X_f = np.vstack([done_X, pending_X])
        y_f = np.concatenate([done_y, mu_p])
        gp.fit(X_f, y_f, refit_hyper=False)
    else:
        X_f = done_X
    cand = candidate_set(rng)
    if len(X_f):
        d = np.sqrt(((cand[:, None, :] - X_f[None, :, :]) ** 2).sum(-1)).min(1)
        cand = cand[d > min_dist]
    mu, sd = gp.predict(cand)
    best = float(np.max(done_y))
    ei = expected_improvement(mu, sd, best)
    return cand[int(np.argmax(ei))], gp


# =========================================================================== #
# One episode
# =========================================================================== #
def raw_action(t_plus, t_minus):
    return np.array([
        2.0 * (t_plus - T_PLUS_MIN) / (T_PLUS_MAX - T_PLUS_MIN) - 1.0,
        2.0 * (t_minus - T_MINUS_MIN) / (T_MINUS_MAX - T_MINUS_MIN) - 1.0,
    ], dtype=float)


def evaluate_point(worker_dir, t_plus, t_minus, w_b, block_dt, n_blocks):
    env = TaylorCouetteDutyV2Env(
        worker_dir, w_b_rpm=w_b, episode_duration=n_blocks * block_dt,
        block_dt=block_dt, t_plus_min=T_PLUS_MIN, t_plus_max=T_PLUS_MAX,
        t_minus_min=T_MINUS_MIN, t_minus_max=T_MINUS_MAX,
        w_hi_cap_rpm=W_HI_CAP, ramp_time=RAMP, p_max_watt=P_MAX,
        t_scale=T_SCALE, x_init=X_INIT, reward_mode="conv")
    dense = []
    orig = env.helpers.do_simulation_table

    def tee(points, dt):
        res = orig(points, dt)
        dense.extend(res)
        return res
    env.helpers.do_simulation_table = tee

    a = raw_action(t_plus, t_minus)
    env.reset(options={"reset_mode": "hard", "n_blocks": n_blocks})
    rows = []
    for k in range(n_blocks):
        _, r, _, _, info = env.step(a)
        rows.append(dict(block=k + 1, t_plus_s=info["t_plus_s"], t_minus_s=info["t_minus_s"],
                         duty=info["duty"], period_s=info["period_s"],
                         w_hi_rpm=info["w_hi_rpm"], realized_mean_rpm=info["realized_mean_rpm"],
                         X_block=info["mixing_index"], wf_block=info["wf_block"],
                         P_block_W=info["power_watt"], reward=r))
    return rows, dense


def _task(worker_dir, ep, kind, rep_of, t_plus, t_minus, w_b, cfg):
    t0 = time.time()
    out = dict(ep=ep, kind=kind, rep_of=rep_of, t_plus_s=t_plus, t_minus_s=t_minus,
               w_b_rpm=w_b, worker=os.path.basename(worker_dir), status="ok")
    try:
        rows, dense = evaluate_point(worker_dir, t_plus, t_minus, w_b,
                                     cfg["block_dt"], cfg["n_blocks"])
        epdir = os.path.join(cfg["results_dir"], "episodes")
        os.makedirs(epdir, exist_ok=True)
        with open(os.path.join(epdir, f"ep{ep:04d}_blocks.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        with open(os.path.join(epdir, f"ep{ep:04d}_timeseries.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "conversion", "wallFlux"])
            for m in dense:
                w.writerow([f"{m['t']:.6g}", f"{m['conv']:.8g}", f"{m['wallFlux']:.6g}"])
        last = rows[-1]
        out.update(duty=last["duty"], period_s=last["period_s"], w_hi_rpm=last["w_hi_rpm"],
                   r_last=last["reward"], X_last=last["X_block"], P_last_W=last["P_block_W"],
                   r_mean=float(np.mean([r["reward"] for r in rows])),
                   X_mean=float(np.mean([r["X_block"] for r in rows])),
                   X_blocks=";".join(f"{r['X_block']:.6f}" for r in rows),
                   r_blocks=";".join(f"{r['reward']:.6f}" for r in rows))
    except Exception as e:  # pimpleFoam failure etc. -- keep the search alive
        out.update(status=f"failed: {type(e).__name__}", duty=float("nan"),
                   period_s=float("nan"), w_hi_rpm=float("nan"), r_last=float("nan"),
                   X_last=float("nan"), P_last_W=float("nan"), r_mean=float("nan"),
                   X_mean=float("nan"), X_blocks="", r_blocks="")
        print(f"[ep {ep:04d}] FAILED on {os.path.basename(worker_dir)}: "
              f"{type(e).__name__}: {str(e)[-400:]}", flush=True)
        traceback.print_exc()
    out["wall_min"] = (time.time() - t0) / 60.0
    return worker_dir, out


# =========================================================================== #
# Persistence
# =========================================================================== #
def eval_csv(results_dir):
    return os.path.join(results_dir, "evaluations.csv")


def load_evaluations(results_dir):
    p = eval_csv(results_dir)
    if not os.path.isfile(p):
        return []
    rows = []
    with open(p) as f:
        for r in csv.DictReader(f):
            for k in ("t_plus_s", "t_minus_s", "w_b_rpm", "duty", "period_s", "w_hi_rpm",
                      "r_last", "X_last", "P_last_W", "r_mean", "X_mean", "wall_min"):
                r[k] = float(r[k]) if r[k] not in ("", "nan") else float("nan")
            r["ep"] = int(r["ep"])
            r["rep_of"] = int(r["rep_of"]) if r["rep_of"] not in ("", "-1") else -1
            rows.append(r)
    return rows


def append_evaluation(results_dir, row, lock):
    p = eval_csv(results_dir)
    with lock:
        new = not os.path.isfile(p)
        with open(p, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=EVAL_FIELDS)
            if new:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in EVAL_FIELDS})


def ok_rows(rows):
    return [r for r in rows if r["status"] == "ok" and np.isfinite(r["r_last"])]


# =========================================================================== #
# Main search loop (asynchronous)
# =========================================================================== #
def run_search(args):
    from parallel_train_duty import prepare_worker_cases   # torch import; CFD path only
    os.makedirs(args.results_dir, exist_ok=True)
    cfg = dict(block_dt=args.block_dt, n_blocks=args.n_blocks, results_dir=args.results_dir)
    rng = np.random.default_rng(args.seed)
    lock = threading.Lock()

    workers = prepare_worker_cases(MASTER_CASE, args.worker_root, args.n_workers,
                                   args.rebuild, W_B, args.warm_duration)
    free = queue.Queue()
    for w in workers:
        free.put(w)

    rows = load_evaluations(args.results_dir) if args.resume else []
    if rows and not args.resume:
        sys.exit(f"{eval_csv(args.results_dir)} exists; pass --resume or a fresh --results_dir")
    next_ep = (max(r["ep"] for r in rows) + 1) if rows else 0
    n_init_done = sum(1 for r in rows if r["kind"] == "init")
    n_bo_done = sum(1 for r in rows if r["kind"] == "bo" and r["status"] == "ok")
    print(f"[A] resume: {len(rows)} evaluations on disk "
          f"({n_init_done} init, {n_bo_done} bo)" if rows else "[A] fresh search", flush=True)

    # Initial design: anchors first, then a seeded Latin hypercube.
    n_lhs = max(0, args.n_init - len(ANCHORS))
    design = [(nm, tp, tm) for nm, tp, tm in ANCHORS]
    for i, u in enumerate(latin_hypercube(n_lhs, rng) if n_lhs else []):
        tp, tm = from_unit(u)
        design.append((f"lhs{i:02d}", float(tp), float(tm)))
    design = design[n_init_done:]               # resume: skip what is done
    n_bo_left = max(0, args.n_bo - n_bo_done)

    print(f"[A] budget: {len(design)} design + {n_bo_left} BO + replicates "
          f"(top {args.replicate_top} x {args.n_rep}); {args.n_workers} workers; "
          f"episodes {args.n_blocks} x {args.block_dt:.0f} s from the warmed IC", flush=True)

    pending = {}      # future -> (unit point)
    ep_counter = [next_ep]
    n_fail = [0]

    def submit(ex, kind, rep_of, tp, tm, wb):
        wdir = free.get()
        ep = ep_counter[0]
        ep_counter[0] += 1
        fut = ex.submit(_task, wdir, ep, kind, rep_of, tp, tm, wb, cfg)
        pending[fut] = to_unit(tp, tm)
        print(f"[submit ep {ep:04d}] {kind:5s} T+={tp:.3f} T-={tm:.3f} w_b={wb:.1f} "
              f"-> {os.path.basename(wdir)}", flush=True)

    def harvest(done_futs):
        for fut in done_futs:
            wdir, out = fut.result()
            pending.pop(fut, None)
            free.put(wdir)
            append_evaluation(args.results_dir, out, lock)
            rows.append(out)
            if out["status"] == "ok":
                best = max(r["r_last"] for r in ok_rows(rows))
                print(f"[done   ep {out['ep']:04d}] {out['kind']:5s} T+={out['t_plus_s']:.3f} "
                      f"T-={out['t_minus_s']:.3f} D={out['duty']:.3f} whi={out['w_hi_rpm']:.0f} "
                      f"| r_last={out['r_last']:+.4f} X_last={out['X_last']:.4f} "
                      f"({out['wall_min']:.0f} min) | best so far {best:+.4f} "
                      f"[{len(ok_rows(rows))} ok]", flush=True)
            else:
                n_fail[0] += 1
                if n_fail[0] > args.max_fail:
                    raise RuntimeError(f"{n_fail[0]} failed episodes -- aborting")

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.n_workers) as ex:
        # ---- phase 1: design + BO, asynchronous --------------------------
        n_bo_submitted = 0
        while design or n_bo_submitted < n_bo_left or pending:
            while not free.empty() and (design or n_bo_submitted < n_bo_left):
                if design:
                    nm, tp, tm = design.pop(0)
                    submit(ex, "init", -1, tp, tm, W_B)
                else:
                    okr = ok_rows(rows)
                    if len(okr) < 4:          # nothing to model yet
                        u = rng.random(2)
                    else:
                        u, _ = propose_next(
                            np.array([to_unit(r["t_plus_s"], r["t_minus_s"]) for r in okr]),
                            np.array([r["r_last"] for r in okr]),
                            np.array(list(pending.values())) if pending else np.empty((0, 2)),
                            rng)
                    tp, tm = from_unit(u)
                    submit(ex, "bo", -1, float(tp), float(tm), W_B)
                    n_bo_submitted += 1
            if not pending:
                break
            done, _ = wait(list(pending.keys()), return_when=FIRST_COMPLETED)
            harvest(done)

        # ---- phase 2: replicates of the top-k (by GP posterior mean) -----
        okr = ok_rows(rows)
        if okr and args.replicate_top > 0 and args.n_rep > 1:
            gp = GP().fit(np.array([to_unit(r["t_plus_s"], r["t_minus_s"]) for r in okr]),
                          np.array([r["r_last"] for r in okr]))
            base = [r for r in okr if r["kind"] in ("init", "bo")]
            mu, _ = gp.predict(np.array([to_unit(r["t_plus_s"], r["t_minus_s"]) for r in base]))
            order = np.argsort(-mu)
            top = [base[i] for i in order[:args.replicate_top]]
            have = {}
            for r in rows:
                if r["kind"] == "rep":
                    have[r["rep_of"]] = have.get(r["rep_of"], 0) + 1
            jitters = [args.rep_jitter_rpm * s for s in (1, -1, 2, -2, 3, -3, 4, -4)]
            for r in top:
                for j in range(have.get(r["ep"], 0), args.n_rep - 1):
                    while free.empty():
                        done, _ = wait(list(pending.keys()), return_when=FIRST_COMPLETED)
                        harvest(done)
                    submit(ex, "rep", r["ep"], r["t_plus_s"], r["t_minus_s"], W_B + jitters[j])
            while pending:
                done, _ = wait(list(pending.keys()), return_when=FIRST_COMPLETED)
                harvest(done)

    print(f"[A] search finished in {(time.time()-t_start)/3600:.2f} h; "
          f"{len(ok_rows(rows))} ok episodes", flush=True)
    return rows


# =========================================================================== #
# Analysis
# =========================================================================== #
def analyze(args):
    rows = load_evaluations(args.results_dir)
    okr = ok_rows(rows)
    if len(okr) < 4:
        print(f"[A] only {len(okr)} ok evaluations in {args.results_dir}; nothing to fit")
        return
    X = np.array([to_unit(r["t_plus_s"], r["t_minus_s"]) for r in okr])
    y = np.array([r["r_last"] for r in okr])
    gp = GP().fit(X, y)

    g = np.linspace(0.0, 1.0, 121)
    U = np.array([[a, b] for a in g for b in g])
    mu, sd = gp.predict(U)
    i_star = int(np.argmax(mu))
    tp_star, tm_star = from_unit(U[i_star])
    i_obs = int(np.argmax(y))
    obs_best = okr[i_obs]
    # GP mean at the observed best, and the observed rows nearest the argmax
    mu_obs_best, _ = gp.predict(X[i_obs:i_obs + 1])

    # replicates -> run-to-run noise floor (pooled sd about each parent's mean)
    reps = {}
    for r in okr:
        if r["kind"] == "rep":
            reps.setdefault(r["rep_of"], []).append(r["r_last"])
    by_ep = {r["ep"]: r for r in okr}
    rep_rows, pooled = [], []
    for parent, vals in reps.items():
        if parent in by_ep:
            allv = [by_ep[parent]["r_last"]] + vals
            m, s = float(np.mean(allv)), float(np.std(allv, ddof=1)) if len(allv) > 1 else float("nan")
            p = by_ep[parent]
            rep_rows.append(dict(parent_ep=parent, t_plus_s=p["t_plus_s"], t_minus_s=p["t_minus_s"],
                                 duty=p["duty"], period_s=p["period_s"], n=len(allv),
                                 r_mean=m, r_sd=s, r_values=";".join(f"{v:.5f}" for v in allv)))
            if np.isfinite(s):
                pooled.append(s)
    rep_sigma = float(np.sqrt(np.mean(np.square(pooled)))) if pooled else float("nan")

    # ---- tables ----
    with open(os.path.join(args.results_dir, "replicates.csv"), "w", newline="") as f:
        if rep_rows:
            w = csv.DictWriter(f, fieldnames=list(rep_rows[0].keys()))
            w.writeheader()
            w.writerows(rep_rows)
    mu_all, sd_all = gp.predict(X)
    order = np.argsort(-mu_all)
    with open(os.path.join(args.results_dir, "top_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "ep", "kind", "t_plus_s", "t_minus_s", "duty", "period_s", "w_hi_rpm",
                    "r_last_obs", "X_last_obs", "gp_mean", "gp_sd"])
        for k, i in enumerate(order[:15]):
            r = okr[i]
            w.writerow([k + 1, r["ep"], r["kind"], f"{r['t_plus_s']:.3f}", f"{r['t_minus_s']:.3f}",
                        f"{r['duty']:.3f}", f"{r['period_s']:.3f}", f"{r['w_hi_rpm']:.1f}",
                        f"{r['r_last']:.5f}", f"{r['X_last']:.5f}", f"{mu_all[i]:.5f}", f"{sd_all[i]:.5f}"])

    tp_ob, tm_ob = obs_best["t_plus_s"], obs_best["t_minus_s"]
    summary = dict(
        n_evaluations=len(okr), n_failed=len(rows) - len(okr),
        objective="last-block reward r5 = X_[104,130] - P/31.94 (130 s warmed, 26 s blocks)",
        gp_hyper=dict(ls_unit=gp.ls.tolist(), noise_rel=float(gp.noise),
                      noise_sigma_r=gp.noise_sigma()),
        posterior_champion=dict(t_plus_s=float(tp_star), t_minus_s=float(tm_star),
                                duty=float(tp_star / (tp_star + tm_star)),
                                period_s=float(tp_star + tm_star),
                                w_hi_rpm_approx=float(W_B * (tp_star + tm_star) / tp_star),
                                gp_mean=float(mu[i_star]), gp_sd=float(sd[i_star])),
        observed_best=dict(ep=obs_best["ep"], t_plus_s=tp_ob, t_minus_s=tm_ob,
                           duty=obs_best["duty"], period_s=obs_best["period_s"],
                           w_hi_rpm=obs_best["w_hi_rpm"], r_last=obs_best["r_last"],
                           X_last=obs_best["X_last"], gp_mean=float(mu_obs_best[0])),
        replicate_sigma_r=rep_sigma, replicate_groups=len(rep_rows),
        refs=REFS,
        anchors_observed={nm: next((r["r_last"] for r in okr
                                    if abs(r["t_plus_s"] - tp) < 1e-6 and abs(r["t_minus_s"] - tm) < 1e-6
                                    and r["kind"] == "init"), None)
                          for nm, tp, tm in ANCHORS},
    )
    with open(os.path.join(args.results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- headline ----
    print("\n" + "=" * 78)
    print("A -- FAIR STATIC CHAMPION (T+, T-) SEARCH, fixed mean 300 rpm, warmed, 130 s")
    print("=" * 78)
    print(f"  evaluations            : {len(okr)} ok, {len(rows) - len(okr)} failed")
    print(f"  posterior champion     : T+={tp_star:.2f} s  T-={tm_star:.2f} s  "
          f"(D={tp_star/(tp_star+tm_star):.3f}, T={tp_star+tm_star:.2f} s)  "
          f"r = {mu[i_star]:.4f} +- {sd[i_star]:.4f}")
    print(f"  observed best          : ep {obs_best['ep']} T+={tp_ob:.2f} T-={tm_ob:.2f}  "
          f"r_last = {obs_best['r_last']:.4f}  (X_last {obs_best['X_last']:.4f})")
    print(f"  noise: GP {gp.noise_sigma():.4f} | replicates {rep_sigma:.4f} "
          f"({len(rep_rows)} groups) | flutter ref {REFS['flutter_sigma']}")
    print(f"  refs (duty_v3_refs)    : constant {REFS['constant_r']} | n=1 champion {REFS['champion_r']}")
    print(f"  duty_v3 deterministic  : {REFS['duty_v3_eval_r']}  (max {max(REFS['duty_v3_eval_r'])})")
    for nm, v in summary["anchors_observed"].items():
        if v is not None:
            print(f"  anchor {nm:14s}: r_last = {v:.4f}")
    print("=" * 78 + "\n")

    # ---- figures ----
    TP = (T_PLUS_MIN + g * (T_PLUS_MAX - T_PLUS_MIN))
    TM = (T_MINUS_MIN + g * (T_MINUS_MAX - T_MINUS_MIN))
    MU = mu.reshape(len(g), len(g)).T      # rows: T-, cols: T+
    SD = sd.reshape(len(g), len(g)).T
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, Z, ttl, cm in ((axes[0], MU, "GP posterior mean of r5", "viridis"),
                           (axes[1], SD, "GP posterior sd", "magma")):
        cf = ax.contourf(TP, TM, Z, levels=30, cmap=cm)
        fig.colorbar(cf, ax=ax, shrink=0.9)
        kinds = {"init": ("o", "white"), "bo": ("o", "#d1495b"), "rep": ("s", "#ffb703")}
        for kd, (mk, col) in kinds.items():
            sel = [r for r in okr if r["kind"] == kd]
            if sel:
                ax.plot([r["t_plus_s"] for r in sel], [r["t_minus_s"] for r in sel], mk,
                        color=col, mec="k", ms=5, mew=0.5, label=kd)
        ax.plot(tp_star, tm_star, "*", color="yellow", mec="k", ms=18, label="posterior argmax")
        ax.plot(tp_ob, tm_ob, "P", color="cyan", mec="k", ms=11, label="observed best")
        ax.axhline(0.5, color="w", ls="--", lw=0.8, alpha=0.7)
        for nm, tp, tm in ANCHORS:
            ax.annotate(nm, (tp, tm), fontsize=6.5, color="w", xytext=(3, 3),
                        textcoords="offset points")
        ax.set_xlabel("burst duration T+ [s]")
        ax.set_ylabel("idle duration T- [s]   (T- = 0 is constant 300 rpm)")
        ax.set_title(ttl)
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"A: static (T+, T-) landscape, fixed mean 300 rpm, warmed 130 s, "
                 f"objective r5 = X_[104,130] - P/Pmax  |  n = {len(okr)}", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(args.results_dir, "A_surface.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for tp in (1.0, 2.0, 3.0, 4.0, 5.0):
        Us = np.array([to_unit(tp, tm) for tm in TM])
        m, s = gp.predict(Us)
        axes[0].plot(TM, m, label=f"T+ = {tp:.0f} s")
        axes[0].fill_between(TM, m - s, m + s, alpha=0.12)
    axes[0].axhline(REFS["champion_r"], color="k", ls=":", lw=1, label="n=1 champion 0.2948")
    axes[0].axhline(REFS["constant_r"], color="gray", ls=":", lw=1, label="constant 0.2441")
    axes[0].set_xlabel("idle T- [s]")
    axes[0].set_ylabel("r5")
    axes[0].set_title("r vs idle duration at fixed burst duration")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)
    for tm in (0.25, 0.5, 1.0, 2.0, 4.0):
        Us = np.array([to_unit(tp, tm) for tp in TP])
        m, s = gp.predict(Us)
        axes[1].plot(TP, m, label=f"T- = {tm:g} s")
        axes[1].fill_between(TP, m - s, m + s, alpha=0.12)
    axes[1].set_xlabel("burst T+ [s]")
    axes[1].set_ylabel("r5")
    axes[1].set_title("r vs burst duration at fixed idle duration")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.results_dir, "A_slices.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    sc = axes[0].scatter([r["t_minus_s"] for r in okr], y, c=[r["t_plus_s"] for r in okr],
                         cmap="plasma", s=28, edgecolor="k", linewidth=0.3)
    fig.colorbar(sc, ax=axes[0], label="T+ [s]")
    axes[0].axhline(REFS["champion_r"], color="k", ls=":", lw=1)
    axes[0].axhline(REFS["constant_r"], color="gray", ls=":", lw=1)
    axes[0].set_xlabel("idle duration T- [s]")
    axes[0].set_ylabel("observed r5")
    axes[0].set_title("Idle-duration hypothesis test: r vs T-, colored by T+")
    axes[0].grid(True, alpha=0.3)
    sc = axes[1].scatter([r["duty"] for r in okr], y, c=[r["period_s"] for r in okr],
                         cmap="viridis", s=28, edgecolor="k", linewidth=0.3)
    fig.colorbar(sc, ax=axes[1], label="period T [s]")
    axes[1].set_xlabel("duty D")
    axes[1].set_ylabel("observed r5")
    axes[1].set_title("r vs duty, colored by period (Lopez-Guajardo Fig. 6 axes)")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.results_dir, "A_idle_collapse.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    seq = sorted(okr, key=lambda r: r["ep"])
    best_so_far = np.maximum.accumulate([r["r_last"] for r in seq])
    ax.plot([r["ep"] for r in seq], [r["r_last"] for r in seq], ".", color="#999", label="episode r5")
    ax.plot([r["ep"] for r in seq], best_so_far, "-", color="#d1495b", lw=2, label="best so far")
    ax.axhline(REFS["champion_r"], color="k", ls=":", lw=1, label="n=1 champion")
    ax.axhline(max(REFS["duty_v3_eval_r"]), color="#2e6f95", ls="--", lw=1, label="best duty_v3 seed")
    ax.set_xlabel("evaluation index")
    ax.set_ylabel("r5")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("A: search convergence")
    fig.tight_layout()
    fig.savefig(os.path.join(args.results_dir, "A_convergence.png"), dpi=150)
    plt.close(fig)
    print(f"[A] outputs -> {args.results_dir}", flush=True)
    return summary


# =========================================================================== #
def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_workers", type=int, default=44)
    p.add_argument("--worker_root", default=None,
                   help="scratch dir for the warmed template + worker cases "
                        "(default $TMPDIR/$USER/endgame_A_workers)")
    p.add_argument("--results_dir", default=os.path.join(HERE, "results", "A"))
    p.add_argument("--n_init", type=int, default=44,
                   help="space-filling design size (anchors included)")
    p.add_argument("--n_bo", type=int, default=170, help="GP-guided evaluations")
    p.add_argument("--replicate_top", type=int, default=3)
    p.add_argument("--n_rep", type=int, default=3, help="evaluations per replicated point (incl. original)")
    p.add_argument("--rep_jitter_rpm", type=float, default=0.5)
    p.add_argument("--block_dt", type=float, default=BLOCK_DT)
    p.add_argument("--n_blocks", type=int, default=N_BLOCKS)
    p.add_argument("--warm_duration", type=float, default=WARM_DURATION)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_fail", type=int, default=20)
    p.add_argument("--resume", action="store_true", help="continue from evaluations.csv")
    p.add_argument("--rebuild", action="store_true", help="rebuild the warmed template + workers")
    p.add_argument("--smoke", action="store_true",
                   help="2 workers, 2 x 2 s blocks, 2 s warm-up, 3 init + 2 bo + 1x2 rep (~15 min)")
    p.add_argument("--analyze-only", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    if args.worker_root is None:
        args.worker_root = os.path.join(os.environ.get("TMPDIR", "/tmp"),
                                        os.environ.get("USER", "user"), "endgame_A_workers")
    if args.smoke:
        args.n_workers, args.block_dt, args.n_blocks, args.warm_duration = 2, 2.0, 2, 2.0
        args.n_init, args.n_bo, args.replicate_top, args.n_rep = 3, 2, 1, 2
        args.results_dir = os.path.join(HERE, "results", "A_smoke")
        args.worker_root = args.worker_root.rstrip("/") + "_smoke"
        print("[A] SMOKE: 2 workers, 2 x 2 s blocks, 7 tiny episodes", flush=True)
    if args.analyze_only:
        analyze(args)
        return
    if shutil.which("pimpleFoam") is None or shutil.which("foamDictionary") is None:
        sys.exit("ERROR: OpenFOAM not on PATH (need pimpleFoam, foamDictionary).")
    run_search(args)
    analyze(args)


if __name__ == "__main__":
    main()
