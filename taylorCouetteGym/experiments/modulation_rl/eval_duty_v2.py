#!/usr/bin/env python3
"""Deterministic 50 s WARMED evals + learning curves for the duty_v2 seeds.

Per seed (results/td3/duty_v2_s{n}/td3_tc_final), runs the final actor
noise-free for 5 x 10 s blocks from the warmed constant-300 steady state and
writes to results/warmed_eval/duty_v2_s{n}/:

    blocks.csv, timeseries.csv
    eval_conversion_vs_time.png -- 1 Hz conversion trace + block-mean dashes +
        the warmed reference lines (style of results/td3/eval_conversion_vs_time.png)
    eval_omega_vs_time.png -- the commanded waveform, rebuilt with the env's
        own solve_block_wave from the logged (T+, T-) so the plot shows exactly
        what the BC ran (style of results/td3/eval_omega_vs_time.png)
    case/ -- ParaView-ready snapshot of the episode fields

Plus results/td3/duty_v2_learning_curves.png: per-seed return per episode.
Episodes have RANDOMIZED horizons (3-7 blocks), so returns are normalized to
the 50 s equivalent (5 x mean block reward) to be comparable across episodes;
rewards in the logs are RAW (uncentered) by trainer contract.

USAGE
    python3 eval_duty_v2.py                    # curves + all 5 CFD evals (~40 min, threaded)
    python3 eval_duty_v2.py --analyze-only     # learning curves only (no CFD)
    python3 eval_duty_v2.py --seeds 0 2        # subset
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
N_BLOCKS = 5
BLOCK_DT = 10.0
W_B, RAMP, W_HI_CAP = 300.0, 0.05, 2500.0   # must match the env defaults
BASELINE = 0.2390           # warmed constant-300 R50- (training's centering zero)
REF_CONST_X = 0.3608        # warmed constant-300 last-block X (duty_diag)
REF_CHAMP_X = 0.4048        # warmed static champion T=5/D=0.90 (duty_diag)
REF_CHAMP_R = 0.2886        # its last-block reward (the static plateau)
START_TIMESTEPS = 500
RUN_MEAN_W = 15


# --------------------------------------------------------------- CFD evals --
def run_seed(seed):
    run_dir = os.path.join(OUT_ROOT, f"duty_v2_s{seed}")
    workdir = os.path.join(run_dir, "case_work")
    os.makedirs(run_dir, exist_ok=True)
    if os.path.isdir(workdir):
        shutil.rmtree(workdir)
    shutil.copytree(TEMPLATE, workdir)

    policy = make_policy("td3", state_dim=2, action_dim=2, max_action=1.0,
                         discount=0.99, tau=0.005)
    policy.load(os.path.join(TD3_DIR, f"duty_v2_s{seed}", "td3_tc_final"))

    env = TaylorCouetteDutyV2Env(workdir, episode_duration=N_BLOCKS * BLOCK_DT)
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
    print(f"  [duty_v2_s{seed}] "
          f"T-={' '.join(f'{r['t_minus_s']:.2f}' for r in rows)} | "
          f"D={' '.join(f'{r['duty']:.2f}' for r in rows)} | "
          f"X_last={last['X_block']:.4f} last-r={last['reward']:+.4f} "
          f"(ctr {last['reward_centered']:+.4f}) "
          f"({(time.time()-t0)/60:.0f} min) -> {run_dir}", flush=True)
    return dict(seed=seed, last_r=last["reward"], x_last=last["X_block"])


# ------------------------------------------------------------------ figures --
def fig_conversion(seed, run_dir, rows, dense):
    """eval_conversion_vs_time.png style: ~1 Hz trace + block-mean dashes."""
    t = np.array([m["t"] for m in dense])
    c = np.array([m["conv"] for m in dense])
    keep = np.searchsorted(t, np.arange(0.0, t[-1] + 0.5, 1.0))
    keep = np.unique(np.clip(keep, 0, len(t) - 1))
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.plot(t[keep], c[keep], "-o", color="forestgreen", ms=4, lw=1.6)
    for r in rows:
        x0, x1 = (r["block"] - 1) * BLOCK_DT, r["block"] * BLOCK_DT
        ax.hlines(r["X_block"], x0 + 0.3, x1 - 0.3, color="black", ls="--", lw=1.6)
    ax.axhline(REF_CHAMP_X, color="0.45", ls=":", lw=1.2)
    ax.axhline(REF_CONST_X, color="0.45", ls=":", lw=1.2)
    ax.text(0.5, REF_CHAMP_X + 0.002,
            f"warmed static champion (T=5, D=0.90), last-block X = {REF_CHAMP_X:.4f}",
            fontsize=9, color="0.35")
    ax.text(0.5, REF_CONST_X + 0.002,
            f"warmed constant-300, last-block X = {REF_CONST_X:.4f}",
            fontsize=9, color="0.35")
    tminus = " ".join(f"{r['t_minus_s']:.2f}" for r in rows)
    last_r = rows[-1]["reward"]
    ax.set_title(f"duty_v2 seed {seed} (deterministic) — outlet conversion, "
                 f"50 s WARMED episode\n[$T_-$ per block: {tminus} s;  "
                 f"last-block r = {last_r:+.4f}]", loc="left", fontsize=11)
    ax.set_xlabel("time since warmed start  [s]")
    ax.set_ylabel("outlet conversion  X")
    ax.grid(alpha=0.3)
    ax.set_xlim(-1, N_BLOCKS * BLOCK_DT + 1)
    fig.text(0.99, 0.01, "line: ~1 Hz METRICS samples (±0.02 outlet flutter); "
             "dashes: 10 s block averages (reward basis)",
             ha="right", fontsize=8, color="0.4")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "eval_conversion_vs_time.png"), dpi=140)
    plt.close(fig)


def fig_omega(seed, run_dir, rows):
    """eval_omega_vs_time.png style: the commanded waveform, per-block labels."""
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
    ax.plot(tt, ww, color="tab:blue", lw=1.8)
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
    ax.set_title(f"duty_v2 seed {seed} (deterministic) — commanded waveform, "
                 f"50 s WARMED episode", loc="left", fontsize=12)
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
            if os.path.isfile(os.path.join(TD3_DIR, f"duty_v2_s{s}",
                                           "episode_returns.npy"))]
    if not have:
        print("[curves] no duty_v2 training logs found yet -- skipping")
        return
    n = len(have)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.1 * n), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    fig.suptitle("duty_v2 — TD3 on the (T+, T-) duty env: return per episode\n"
                 "(returns normalized to the 50 s / 5-block equivalent — "
                 "episodes have randomized 3–7-block horizons; rewards RAW)",
                 fontsize=12.5, x=0.5, y=0.995)
    for s, ax in zip(have, axes):
        d = os.path.join(TD3_DIR, f"duty_v2_s{s}")
        rw = np.load(os.path.join(d, "reward_per_step.npy"))
        ends = np.load(os.path.join(d, "episode_end_steps.npy"))
        mean_block = np.array([row[~np.isnan(row)].mean() for row in rw])
        ret50 = N_BLOCKS * mean_block
        ep = np.arange(1, len(ret50) + 1)
        rand_eps = int(np.searchsorted(ends, START_TIMESTEPS))
        ax.axvspan(0, rand_eps, color="0.92", zorder=0)
        ax.scatter(ep, ret50, s=6, alpha=0.35, color="tab:blue", lw=0)
        if len(ret50) >= RUN_MEAN_W:
            run = np.convolve(ret50, np.ones(RUN_MEAN_W) / RUN_MEAN_W, mode="valid")
            ax.plot(ep[RUN_MEAN_W - 1:], run, color="tab:blue", lw=1.8)
        ax.axhline(N_BLOCKS * BASELINE, color="0.4", ls="--", lw=1.1)
        ax.axhline(N_BLOCKS * REF_CHAMP_R, color="0.4", ls=":", lw=1.1)
        lastblk = np.array([row[~np.isnan(row)][-1] for row in rw])
        ax.set_title(f"seed {s} — last-40 mean {ret50[-40:].mean():.3f}; "
                     f"late-30 final-block r = {lastblk[-30:].mean():.4f} "
                     f"(champion plateau {REF_CHAMP_R})", loc="left", fontsize=10.5)
        ax.set_ylabel("return (50 s-normalized)")
        ax.grid(alpha=0.25)
        if s == have[0]:
            ax.text(len(ret50), N_BLOCKS * BASELINE - 0.02,
                    f"warmed constant-300 t50 baseline ×5 = {N_BLOCKS * BASELINE:.3f}",
                    ha="right", va="top", fontsize=8, color="0.35",
                    bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1))
            ax.text(len(ret50), N_BLOCKS * REF_CHAMP_R + 0.008,
                    f"static champion plateau ×5 = {N_BLOCKS * REF_CHAMP_R:.3f}",
                    ha="right", fontsize=8, color="0.35",
                    bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1))
            ax.text(rand_eps / 2, N_BLOCKS * BASELINE - 0.09, "random phase\n(repeat ×3)",
                    ha="center", fontsize=8, color="0.4")
    axes[-1].set_xlabel("episode")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out = os.path.join(TD3_DIR, "duty_v2_learning_curves.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[curves] wrote {out}")


# --------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--analyze-only", action="store_true",
                    help="learning curves only; skip the CFD evals")
    args = ap.parse_args()

    fig_learning_curves(args.seeds)
    if args.analyze_only:
        return
    if not os.path.isdir(os.path.join(TEMPLATE, "0.warmed")):
        sys.exit(f"ERROR: warmed template missing: {TEMPLATE}")
    todo = [s for s in args.seeds
            if os.path.isfile(os.path.join(TD3_DIR, f"duty_v2_s{s}", "td3_tc_final_actor"))]
    missing = sorted(set(args.seeds) - set(todo))
    if missing:
        print(f"[eval] no final actor yet for seeds {missing} -- skipping those")
    if not todo:
        return
    os.makedirs(OUT_ROOT, exist_ok=True)
    print(f"deterministic warmed evals: duty_v2 seeds {todo}, {N_BLOCKS}x10 s each\n"
          f"references: const X {REF_CONST_X} | champ X {REF_CHAMP_X} | "
          f"champ r {REF_CHAMP_R} | centering zero {BASELINE}", flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=len(todo)) as ex:
        futs = [ex.submit(run_seed, s) for s in todo]
        for fut in as_completed(futs):
            results.append(fut.result())
    results.sort(key=lambda r: r["seed"])
    print("\n===== DUTY_V2 DETERMINISTIC EVAL SUMMARY =====")
    for r in results:
        verdict = "BEATS static plateau" if r["last_r"] > REF_CHAMP_R else (
            "beats constant" if r["last_r"] > 0.2437 else "<= constant")
        print(f"  seed {r['seed']}: last-r={r['last_r']:+.4f} "
              f"X_last={r['x_last']:.4f}  {verdict}")
    print(f"  refs: constant sustained 0.2437 | static plateau {REF_CHAMP_R} | "
          f"v5-s2 3-knob 0.2908")


if __name__ == "__main__":
    main()
