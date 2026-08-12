#!/usr/bin/env python3
"""Deterministic 130 s WARMED evals + learning curves for the duty_v3 seeds.

Scores the PRE-REGISTERED duty_v3 gate (run_carya_duty_v3.slurm header,
amended 2026-08-05 before any v3 data existed):

    success = deterministic sustained last-block reward matches or beats the
    best measured static comparator within the measured noise floor; the
    converged waveform (T+, T-) is REPORTED, not preconditioned.

Per seed (results/td3/duty_v3_s{n}/td3_tc_final), runs the final actor
noise-free for 5 x 26 s tau-blocks from the warmed constant-300 steady state
-- the SAME 130 s / 26 s windows the comparators were measured on
(duty_v3_refs.py) -- and writes to results/warmed_eval/duty_v3_s{n}/:

    blocks.csv, timeseries.csv
    eval_conversion_vs_time.png -- 1 Hz conversion trace + block-mean dashes
        + the warmed reference lines
    eval_omega_vs_time.png -- the commanded waveform, rebuilt with the env's
        own solve_block_wave from the logged (T+, T-)
    case/ -- ParaView-ready snapshot of the episode fields

Plus results/td3/duty_v3_learning_curves.png: per-seed return per episode.
Episodes have RANDOMIZED horizons (3-7 blocks), so returns are normalized to
the 130 s equivalent (5 x mean block reward) to be comparable across
episodes; rewards in the logs are RAW (uncentered) by trainer contract.

DO NOT reuse the duty_v2 / t50-table reference numbers here: v3 blocks are
26 s, so its constant and champion plateaus differ (see REF_* below).

USAGE
    python3 eval_duty_v3.py                     # curves + all 5 CFD evals (threaded)
    python3 eval_duty_v3.py --analyze-only      # learning curves only (no CFD)
    python3 eval_duty_v3.py --seeds 1 2         # subset
    python3 eval_duty_v3.py --ckpt td3_tc_t40000 --seeds 1   # mid-run checkpoint
"""

import argparse
import csv
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GYM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if GYM_ROOT not in sys.path:
    sys.path.insert(0, GYM_ROOT)

from train import make_policy  # noqa: E402
from taylor_couette_mixing.envs.taylor_couette_duty_v2 import (  # noqa: E402
    TaylorCouetteDutyV2Env, solve_block_wave, RPM,
)

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(EXP_DIR, "results", "warmed_grad300", "side_outlet_grad_case")
TD3_DIR = os.path.join(EXP_DIR, "results", "td3")
OUT_ROOT = os.path.join(EXP_DIR, "results", "warmed_eval")

SEEDS = [0, 1, 2, 3, 4]

# ---- env config: must match run_carya_duty_v3.slurm exactly -----------------
N_BLOCKS = 5                # the nominal (mean) horizon; 5 x 26 s = 130 s
BLOCK_DT = 26.0             # ONE residence time tau = V/Q
W_B, RAMP, W_HI_CAP = 300.0, 0.05, 2500.0
T_PLUS_MIN, T_PLUS_MAX = 1.0, 5.0
T_MINUS_MIN, T_MINUS_MAX = 0.0, 5.0
P_MAX, T_SCALE, X_INIT = 31.94, 26.0, 0.353
BASELINE = 0.2390           # trainer centering zero (policy-invariant constant)
START_TIMESTEPS = 1000      # ~200-episode random phase (repeat x2)

# ---- comparators, measured on the SAME 26 s windows (duty_v3_refs) ----------
# results/duty_v3_refs/{constant_wb300,champion_D0.90}/timeseries.csv,
# block-averaged over [104, 130] s = the v3 sustained (last-block) window.
REF_CONST_X, REF_CONST_R = 0.36123, 0.24412     # constant w_b = 300 (D = 1.0)
REF_CHAMP_X, REF_CHAMP_R = 0.41092, 0.29476     # static champion T = 5, D = 0.90
# Measured plateau flutter: 1 sigma over the champion's three settled 26 s
# windows ([52,78], [78,104], [104,130] -> 0.29307 / 0.29763 / 0.29476).
NOISE_SIGMA = 0.00231
GATE_R = REF_CHAMP_R - 2.0 * NOISE_SIGMA        # 0.2902: "matches within noise"
RUN_MEAN_W = 15


# --------------------------------------------------------------- CFD evals --
def run_seed(seed, ckpt):
    tag = f"duty_v3_s{seed}"
    run_dir = os.path.join(OUT_ROOT, tag)
    workdir = os.path.join(run_dir, "case_work")
    os.makedirs(run_dir, exist_ok=True)
    if os.path.isdir(workdir):
        shutil.rmtree(workdir)
    shutil.copytree(TEMPLATE, workdir)

    policy = make_policy("td3", state_dim=2, action_dim=2, max_action=1.0,
                         discount=0.99, tau=0.005)
    policy.load(os.path.join(TD3_DIR, tag, ckpt))

    env = TaylorCouetteDutyV2Env(
        workdir, w_b_rpm=W_B, episode_duration=N_BLOCKS * BLOCK_DT,
        block_dt=BLOCK_DT, t_plus_min=T_PLUS_MIN, t_plus_max=T_PLUS_MAX,
        t_minus_min=T_MINUS_MIN, t_minus_max=T_MINUS_MAX,
        w_hi_cap_rpm=W_HI_CAP, ramp_time=RAMP, p_max_watt=P_MAX,
        t_scale=T_SCALE, x_init=X_INIT, reward_mode="conv")
    dense = []
    orig_table = env.helpers.do_simulation_table

    def tee(points, dt):
        res = orig_table(points, dt)
        dense.extend(res)
        return res
    env.helpers.do_simulation_table = tee

    obs, _ = env.reset(options={"reset_mode": "hard"})
    state = np.asarray(obs, dtype=np.float32)
    rows = []
    t0 = time.time()
    for k in range(N_BLOCKS):
        action = policy.select_action(state)
        obs, reward, _, _, info = env.step(action)
        state = np.asarray(obs, dtype=np.float32)
        rows.append(dict(block=k + 1, t_plus_s=info["t_plus_s"],
                         t_minus_s=info["t_minus_s"], duty=info["duty"],
                         period_s=info["period_s"], w_hi_rpm=info["w_hi_rpm"],
                         realized_mean_rpm=info["realized_mean_rpm"],
                         X_block=info["mixing_index"], wf_block=info["wf_block"],
                         P_block_W=info["power_watt"], reward=reward,
                         reward_centered=reward - BASELINE))

    with open(os.path.join(run_dir, "blocks.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(run_dir, "timeseries.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "conversion", "wallFlux"])
        for m in dense:
            w.writerow([f"{m['t']:.6g}", f"{m['conv']:.8g}", f"{m['wallFlux']:.6g}"])

    fig_conversion(seed, run_dir, rows, dense)
    fig_omega(seed, run_dir, rows)

    env.helpers.snapshot_frames(os.path.join(run_dir, "case"))
    shutil.rmtree(workdir)
    last = rows[-1]
    print(f"  [{tag}] "
          f"T-={' '.join(f'{r['t_minus_s']:.2f}' for r in rows)} | "
          f"T+={' '.join(f'{r['t_plus_s']:.2f}' for r in rows)} | "
          f"D={' '.join(f'{r['duty']:.2f}' for r in rows)} | "
          f"X_last={last['X_block']:.4f} last-r={last['reward']:+.4f} "
          f"(ctr {last['reward_centered']:+.4f}) "
          f"({(time.time()-t0)/60:.0f} min) -> {run_dir}", flush=True)
    return dict(seed=seed, last_r=last["reward"], x_last=last["X_block"],
                t_minus=[r["t_minus_s"] for r in rows],
                t_plus=[r["t_plus_s"] for r in rows],
                duty=[r["duty"] for r in rows],
                mean_r=float(np.mean([r["reward"] for r in rows])))


# ------------------------------------------------------------------ figures --
def fig_conversion(seed, run_dir, rows, dense):
    """~1 Hz trace + 26 s block-mean dashes against the warmed references."""
    t = np.array([m["t"] for m in dense])
    c = np.array([m["conv"] for m in dense])
    keep = np.searchsorted(t, np.arange(0.0, t[-1] + 0.5, 1.0))
    keep = np.unique(np.clip(keep, 0, len(t) - 1))
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.plot(t[keep], c[keep], "-o", color="forestgreen", ms=3, lw=1.4)
    for r in rows:
        x0, x1 = (r["block"] - 1) * BLOCK_DT, r["block"] * BLOCK_DT
        ax.hlines(r["X_block"], x0 + 0.6, x1 - 0.6, color="black", ls="--", lw=1.6)
    ax.axhline(REF_CHAMP_X, color="0.45", ls=":", lw=1.2)
    ax.axhline(REF_CONST_X, color="0.45", ls=":", lw=1.2)
    ax.text(1.0, REF_CHAMP_X + 0.002,
            f"warmed static champion (T=5, D=0.90), sustained X = {REF_CHAMP_X:.4f}",
            fontsize=9, color="0.35")
    ax.text(1.0, REF_CONST_X + 0.002,
            f"warmed constant-300, sustained X = {REF_CONST_X:.4f}",
            fontsize=9, color="0.35")
    tminus = " ".join(f"{r['t_minus_s']:.2f}" for r in rows)
    last_r = rows[-1]["reward"]
    ax.set_title(f"duty_v3 seed {seed} (deterministic) — outlet conversion, "
                 f"130 s WARMED episode\n[$T_-$ per block: {tminus} s;  "
                 f"last-block r = {last_r:+.4f}  "
                 f"(gate {GATE_R:.4f} = champion − 2σ)]", loc="left", fontsize=11)
    ax.set_xlabel("time since warmed start  [s]")
    ax.set_ylabel("outlet conversion  X")
    ax.grid(alpha=0.3)
    ax.set_xlim(-1, N_BLOCKS * BLOCK_DT + 1)
    fig.text(0.99, 0.01, "line: ~1 Hz METRICS samples (±0.02 outlet flutter); "
             "dashes: 26 s tau-block averages (reward basis)",
             ha="right", fontsize=8, color="0.4")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "eval_conversion_vs_time.png"), dpi=140)
    plt.close(fig)


def fig_omega(seed, run_dir, rows):
    """The commanded waveform, rebuilt from the logged (T+, T-)."""
    tt, ww = [], []
    for r in rows:
        t0 = (r["block"] - 1) * BLOCK_DT
        pts, _, _, _, _ = solve_block_wave(
            t0, BLOCK_DT, r["t_plus_s"], r["t_minus_s"], W_B * RPM,
            RAMP, W_HI_CAP * RPM)
        kept = [(t, w / RPM) for t, w in pts if t <= t0 + BLOCK_DT + 1e-9]
        kept.append((t0 + BLOCK_DT, kept[-1][1]))
        tt.extend(p[0] for p in kept)
        ww.extend(p[1] for p in kept)
    w_top = max(max(ww), 320.0)
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.plot(tt, ww, color="tab:blue", lw=1.2)
    ax.axhline(W_B, color="0.35", ls="--", lw=1.1)
    for x in np.arange(BLOCK_DT, N_BLOCKS * BLOCK_DT, BLOCK_DT):
        ax.axvline(x, color="0.88", lw=1.0, zorder=0)
    for r in rows:
        xc = (r["block"] - 0.5) * BLOCK_DT
        ax.text(xc, w_top * 1.35, f"b{int(r['block'])}\n"
                f"$T_+$={r['t_plus_s']:.2f}s\n$T_-$={r['t_minus_s']:.2f}s\n"
                f"D={r['duty']:.2f}\n$w_{{hi}}$={r['w_hi_rpm']:.0f}",
                ha="center", va="top", fontsize=8.5, color="0.35")
    ax.text(0.5, W_B + 0.02 * w_top, "$w_b$ = 300 rpm (commanded block mean)",
            ha="left", fontsize=8.5, color="0.35")
    ax.set_title(f"duty_v3 seed {seed} (deterministic) — commanded waveform, "
                 f"130 s WARMED episode (26 s tau-blocks)", loc="left", fontsize=12)
    ax.set_xlabel("time since warmed start  [s]")
    ax.set_ylabel("inner-wall angular velocity  [rpm]")
    ax.grid(alpha=0.25)
    ax.set_xlim(-0.5, N_BLOCKS * BLOCK_DT + 0.5)
    ax.set_ylim(-0.05 * w_top, 1.38 * w_top)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "eval_omega_vs_time.png"), dpi=140)
    plt.close(fig)


# ---------------------------------------------------------- learning curves --
def fig_learning_curves(seeds):
    have = [s for s in seeds
            if os.path.isfile(os.path.join(TD3_DIR, f"duty_v3_s{s}",
                                           "reward_per_step.npy"))]
    if not have:
        print("[curves] no duty_v3 training logs found yet -- skipping")
        return
    n = len(have)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.1 * n), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    fig.suptitle("duty_v3 — TD3 on the (T+, T-) duty env, 26 s tau-blocks: "
                 "return per episode\n(returns normalized to the 130 s / "
                 "5-block equivalent — episodes have randomized 3–7-block "
                 "horizons; rewards RAW)", fontsize=12.5, x=0.5, y=0.995)
    for s, ax in zip(have, axes):
        d = os.path.join(TD3_DIR, f"duty_v3_s{s}")
        rw = np.load(os.path.join(d, "reward_per_step.npy"))
        ends = np.load(os.path.join(d, "episode_end_steps.npy"))
        mean_block = np.array([row[~np.isnan(row)].mean() for row in rw])
        ret130 = N_BLOCKS * mean_block
        ep = np.arange(1, len(ret130) + 1)
        rand_eps = int(np.searchsorted(ends, START_TIMESTEPS))
        ax.axvspan(0, rand_eps, color="0.92", zorder=0)
        ax.scatter(ep, ret130, s=6, alpha=0.35, color="tab:blue", lw=0)
        if len(ret130) >= RUN_MEAN_W:
            run = np.convolve(ret130, np.ones(RUN_MEAN_W) / RUN_MEAN_W, mode="valid")
            ax.plot(ep[RUN_MEAN_W - 1:], run, color="tab:blue", lw=1.8)
        ax.axhline(N_BLOCKS * REF_CONST_R, color="0.4", ls="--", lw=1.1)
        ax.axhline(N_BLOCKS * REF_CHAMP_R, color="0.4", ls=":", lw=1.1)
        lastblk = np.array([row[~np.isnan(row)][-1] for row in rw])
        n_late = min(30, len(lastblk))
        ax.set_title(f"seed {s} — {len(ret130)} eps; last-40 mean "
                     f"{ret130[-40:].mean():.3f}; late-{n_late} final-block r = "
                     f"{lastblk[-n_late:].mean():.4f} "
                     f"(champion plateau {REF_CHAMP_R})", loc="left", fontsize=10.5)
        ax.set_ylabel("return (130 s-normalized)")
        ax.grid(alpha=0.25)
        if s == have[0]:
            ax.text(len(ret130), N_BLOCKS * REF_CONST_R - 0.02,
                    f"warmed constant-300 sustained ×5 = "
                    f"{N_BLOCKS * REF_CONST_R:.3f}",
                    ha="right", va="top", fontsize=8, color="0.35",
                    bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1))
            ax.text(len(ret130), N_BLOCKS * REF_CHAMP_R + 0.008,
                    f"static champion plateau ×5 = {N_BLOCKS * REF_CHAMP_R:.3f}",
                    ha="right", fontsize=8, color="0.35",
                    bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1))
            ax.text(rand_eps / 2, N_BLOCKS * REF_CONST_R - 0.09,
                    "random phase\n(repeat ×2)", ha="center", fontsize=8, color="0.4")
    axes[-1].set_xlabel("episode")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out = os.path.join(TD3_DIR, "duty_v3_learning_curves.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[curves] wrote {out}")


# --------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--ckpt", default="td3_tc_final",
                    help="checkpoint stem inside results/td3/duty_v3_s{n} "
                         "(e.g. td3_tc_t40000 to score a run still in flight)")
    ap.add_argument("--analyze-only", action="store_true",
                    help="learning curves only; skip the CFD evals")
    args = ap.parse_args()

    fig_learning_curves(args.seeds)
    if args.analyze_only:
        return
    if not os.path.isdir(os.path.join(TEMPLATE, "0.warmed")):
        sys.exit(f"ERROR: warmed template missing: {TEMPLATE}")
    todo = [s for s in args.seeds
            if os.path.isfile(os.path.join(TD3_DIR, f"duty_v3_s{s}",
                                           f"{args.ckpt}_actor"))]
    missing = sorted(set(args.seeds) - set(todo))
    if missing:
        print(f"[eval] no {args.ckpt} actor yet for seeds {missing} -- skipping those")
    if not todo:
        return
    os.makedirs(OUT_ROOT, exist_ok=True)
    print(f"deterministic warmed evals: duty_v3 seeds {todo} @ {args.ckpt}, "
          f"{N_BLOCKS}x{BLOCK_DT:.0f} s each\n"
          f"references (26 s windows, duty_v3_refs): constant r {REF_CONST_R} | "
          f"champion r {REF_CHAMP_R} ± {NOISE_SIGMA:.5f} | "
          f"pre-registered gate {GATE_R:.4f}", flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=len(todo)) as ex:
        futs = [ex.submit(run_seed, s, args.ckpt) for s in todo]
        for fut in as_completed(futs):
            results.append(fut.result())
    results.sort(key=lambda r: r["seed"])
    print("\n===== DUTY_V3 DETERMINISTIC EVAL SUMMARY =====")
    for r in results:
        z = (r["last_r"] - REF_CHAMP_R) / NOISE_SIGMA
        if r["last_r"] >= GATE_R:
            verdict = "PASS (matches/beats champion within noise)"
        elif r["last_r"] > REF_CONST_R + 2 * NOISE_SIGMA:
            verdict = "FAIL — modulates, but short of the static champion"
        else:
            verdict = "FAIL — collapsed to constant"
        print(f"  seed {r['seed']}: last-r={r['last_r']:+.4f} ({z:+.1f}σ vs champion) "
              f"X_last={r['x_last']:.4f} mean-r={r['mean_r']:+.4f}")
        print(f"           T-={' '.join(f'{v:.2f}' for v in r['t_minus'])}  "
              f"T+={' '.join(f'{v:.2f}' for v in r['t_plus'])}  "
              f"D={' '.join(f'{v:.2f}' for v in r['duty'])}")
        print(f"           {verdict}")
    n_pass = sum(1 for r in results if r["last_r"] >= GATE_R)
    print(f"\n  gate: last-r >= {GATE_R:.4f} (champion {REF_CHAMP_R} − 2σ, "
          f"σ={NOISE_SIGMA:.5f} measured plateau flutter)")
    print(f"  refs: constant sustained {REF_CONST_R} | static champion "
          f"{REF_CHAMP_R} | v2 best deterministic 0.2713")
    print(f"  {n_pass}/{len(results)} seeds pass. Converged T- is REPORTED, "
          f"not preconditioned (see the slurm header).")


if __name__ == "__main__":
    main()
