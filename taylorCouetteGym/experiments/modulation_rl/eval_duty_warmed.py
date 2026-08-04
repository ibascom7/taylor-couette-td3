#!/usr/bin/env python3
"""Deterministic 50 s WARMED evals of the duty_v1 policies, one figure per
seed in the style of results/td3/eval_conversion_vs_time.png (1 Hz conversion
trace + 10 s block-mean dashes + labeled reference lines).

Each seed's final actor runs noise-free for 5 x 10 s blocks from the warmed
constant-300 steady state (the training IC). References on the plot are the
WARMED comparators measured in the same pipeline (results/duty_diag):
constant-300 X = 0.3608 and the static champion T=5/D=0.90 X = 0.4048.

Outputs per seed -> results/warmed_eval/duty_v1_s{n}/:
    blocks.csv, timeseries.csv, eval_conversion_vs_time.png
plus a printed summary row (last-block reward vs the warmed benchmarks).

USAGE
    nohup python3 -u eval_duty_warmed.py > results/warmed_eval/duty_evals.log 2>&1 &
"""

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
from taylor_couette_mixing.envs.taylor_couette_duty import TaylorCouetteDutyEnv  # noqa: E402

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(EXP_DIR, "results", "warmed_grad300", "side_outlet_grad_case")
OUT_ROOT = os.path.join(EXP_DIR, "results", "warmed_eval")

SEEDS = [0, 1, 2]
N_BLOCKS = 5
REF_CONST = 0.3608          # warmed constant-300, last-block X (duty_diag)
REF_CHAMP = 0.4048          # warmed static champion T=5/D=0.90 (duty_diag)
P_MAX = 31.94


def run_seed(seed):
    run_dir = os.path.join(OUT_ROOT, f"duty_v1_s{seed}")
    workdir = os.path.join(run_dir, "case_work")   # snapshot lands in case/
    os.makedirs(run_dir, exist_ok=True)
    if os.path.isdir(workdir):
        shutil.rmtree(workdir)
    shutil.copytree(TEMPLATE, workdir)

    policy = make_policy("td3", state_dim=2, action_dim=1, max_action=1.0,
                         discount=0.99, tau=0.005)
    policy.load(os.path.join(EXP_DIR, "results", "td3", f"duty_v1_s{seed}",
                             "td3_tc_final"))

    env = TaylorCouetteDutyEnv(workdir, episode_duration=N_BLOCKS * 10.0)
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
        rows.append(dict(block=k + 1, duty=info["duty"], w_hi_rpm=info["w_hi_rpm"],
                         X_block=info["mixing_index"], P_block_W=info["power_watt"],
                         reward=reward))

    with open(os.path.join(run_dir, "blocks.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(run_dir, "timeseries.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "conversion", "wallFlux"])
        for m in dense:
            w.writerow([f"{m['t']:.6g}", f"{m['conv']:.8g}", f"{m['wallFlux']:.6g}"])

    # ---- figure, eval_conversion_vs_time.png style ----------------------
    t = np.array([m["t"] for m in dense])
    c = np.array([m["conv"] for m in dense])
    # ~1 Hz subsample for the dotted-line look of the reference figure
    keep = np.searchsorted(t, np.arange(0.0, t[-1] + 0.5, 1.0))
    keep = np.unique(np.clip(keep, 0, len(t) - 1))
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.plot(t[keep], c[keep], "-o", color="forestgreen", ms=4, lw=1.6)
    for r in rows:
        x0, x1 = (r["block"] - 1) * 10.0, r["block"] * 10.0
        ax.hlines(r["X_block"], x0 + 0.3, x1 - 0.3, color="black", ls="--", lw=1.6)
    ax.axhline(REF_CHAMP, color="0.45", ls=":", lw=1.2)
    ax.axhline(REF_CONST, color="0.45", ls=":", lw=1.2)
    ax.text(0.5, REF_CHAMP + 0.002,
            f"warmed static champion (T=5, D=0.90), last-block X = {REF_CHAMP:.4f}",
            fontsize=9, color="0.35")
    ax.text(0.5, REF_CONST + 0.002,
            f"warmed constant-300, last-block X = {REF_CONST:.4f}",
            fontsize=9, color="0.35")
    duties = " ".join(f"{r['duty']:.2f}" for r in rows)
    last_r = rows[-1]["reward"]
    ax.set_title(f"duty_v1 seed {seed} (deterministic) — outlet conversion, "
                 f"50 s WARMED episode   [D per block: {duties};  "
                 f"last-block r = {last_r:+.4f}]", loc="left", fontsize=12)
    ax.set_xlabel("time since warmed start  [s]")
    ax.set_ylabel("outlet conversion  X")
    ax.grid(alpha=0.3)
    ax.set_xlim(-1, N_BLOCKS * 10 + 1)
    fig.text(0.99, 0.01, "line: ~1 Hz METRICS samples (±0.02 outlet flutter); "
             "dashes: 10 s block averages (reward basis)",
             ha="right", fontsize=8, color="0.4")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "eval_conversion_vs_time.png"), dpi=140)
    plt.close(fig)

    # Keep a ParaView-ready copy of the episode fields (mesh + every 1 s time
    # dir + a .foam stub, like full_tc_eval's case/); drop the working clutter
    # (0.orig, 0.warmed, dynamicCode, logs).
    env.helpers.snapshot_frames(os.path.join(run_dir, "case"))
    shutil.rmtree(workdir)
    print(f"  [duty_v1_s{seed}] D={duties} | X_last={rows[-1]['X_block']:.4f} "
          f"last-r={last_r:+.4f} mean-r={np.mean([r['reward'] for r in rows]):+.4f} "
          f"({(time.time()-t0)/60:.0f} min) -> {run_dir}", flush=True)
    return dict(seed=seed, last_r=last_r, x_last=rows[-1]["X_block"])


def main():
    if not os.path.isdir(os.path.join(TEMPLATE, "0.warmed")):
        sys.exit(f"ERROR: warmed template missing: {TEMPLATE}")
    os.makedirs(OUT_ROOT, exist_ok=True)
    print(f"deterministic warmed evals: duty_v1 seeds {SEEDS}, "
          f"{N_BLOCKS}x10 s each\n"
          f"references: warmed const {REF_CONST} | warmed champ {REF_CHAMP}", flush=True)
    with ThreadPoolExecutor(max_workers=len(SEEDS)) as ex:
        futs = [ex.submit(run_seed, s) for s in SEEDS]
        for fut in as_completed(futs):
            fut.result()
    print("done.", flush=True)


if __name__ == "__main__":
    main()
