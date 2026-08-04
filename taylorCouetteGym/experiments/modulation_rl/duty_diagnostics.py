#!/usr/bin/env python3
"""Duty-campaign diagnostics: the static-D landscape + deterministic actor evals.

Two questions the duty_v1 training logs cannot answer by themselves:

1. WHAT IS THE TRUE 1-D LANDSCAPE? The replay-buffer reward-vs-D curve is
   monotone toward D=1, but same-block binning is confounded by the tau ~ 2.6
   block conversion delay and by on-policy history bias. Run STEADY constant-D
   episodes (the planned step-1 static sweep): D grid x 50 s warmed episodes,
   fixed action every block. The last-block reward is the sustained-operation
   value of that duty -- directly comparable to the warmed fig7 t50 rows
   (constant 0.2390, T=2.5/D=80% 0.2714, T=10/D=80% 0.2651).

2. IS THE AGENTS' ~0.26 FROM THE ACTOR OR FROM THE NOISE? All three duty_v1
   actors converged to D ~ 0.93-0.97 yet their late training block rewards
   (~0.254-0.276) sit well above the warmed constant level (0.2390) -- either
   micro-pulsing near D=1 genuinely pays, or the 0.08-duty exploration jitter
   was doing the film renewal (the ew0.8 lesson: training pulsing can be
   exploration noise). Deterministic (noise-free) rollouts of each seed's
   final actor answer it.

All runs: warmed template clones, TaylorCouetteDutyEnv, 5 x 10 s blocks,
T=5 s, w_b=300, conv reward. 10 concurrent runs ~ 1 h wall.

USAGE
    nohup python3 -u duty_diagnostics.py > results/duty_diag/run.log 2>&1 &
"""

import csv
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

GYM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if GYM_ROOT not in sys.path:
    sys.path.insert(0, GYM_ROOT)

from train import make_policy  # noqa: E402
from taylor_couette_mixing.envs.taylor_couette_duty import TaylorCouetteDutyEnv  # noqa: E402

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(EXP_DIR, "results", "warmed_grad300", "side_outlet_grad_case")
OUT_ROOT = os.path.join(EXP_DIR, "results", "duty_diag")

D_GRID = [0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0]
SEEDS = [0, 1, 2]
N_BLOCKS = 5
BENCH_NOTE = "warmed t50 rows: const 0.2390 | T2.5/D80 0.2714 | T10/D80 0.2651 | v5-s2 eval 0.2908"


def duty_to_action(d):
    return 2.0 * (d - 0.2) / 0.8 - 1.0


def make_env(workdir):
    return TaylorCouetteDutyEnv(workdir, episode_duration=N_BLOCKS * 10.0)


def run_one(tag, actor):
    """actor: callable state -> action array. Returns summary row."""
    run_dir = os.path.join(OUT_ROOT, tag)
    workdir = os.path.join(run_dir, "case")
    os.makedirs(run_dir, exist_ok=True)
    if os.path.isdir(workdir):
        shutil.rmtree(workdir)
    shutil.copytree(TEMPLATE, workdir)
    env = make_env(workdir)
    obs, _ = env.reset(options={"reset_mode": "hard"})
    state = np.asarray(obs, dtype=np.float32)
    rows = []
    t0 = time.time()
    for k in range(N_BLOCKS):
        action = np.asarray(actor(state), dtype=float).ravel()
        obs, reward, term, trunc, info = env.step(action)
        state = np.asarray(obs, dtype=np.float32)
        rows.append(dict(block=k + 1, duty=info["duty"], w_hi_rpm=info["w_hi_rpm"],
                         X_block=info["mixing_index"], wf_block=info["wf_block"],
                         P_block_W=info["power_watt"], reward=reward))
    with open(os.path.join(run_dir, "blocks.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    mean_r = float(np.mean([r["reward"] for r in rows]))
    last_r = rows[-1]["reward"]
    print(f"  [{tag:16s}] D={' '.join(f'{r['duty']:.2f}' for r in rows)} | "
          f"X_last={rows[-1]['X_block']:.4f} last-r={last_r:+.4f} "
          f"mean-r={mean_r:+.4f} ({(time.time()-t0)/60:.0f} min)", flush=True)
    shutil.rmtree(workdir)   # keep the diag dir small; CSVs carry the data
    return dict(tag=tag, last_r=last_r, mean_r=mean_r,
                x_last=rows[-1]["X_block"], p_last=rows[-1]["P_block_W"])


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    if not os.path.isdir(os.path.join(TEMPLATE, "0.warmed")):
        sys.exit(f"ERROR: warmed template missing: {TEMPLATE}")

    jobs = []
    for d in D_GRID:
        a = duty_to_action(d)
        jobs.append((f"static_D{d:.2f}", lambda s, a=a: [a]))
    for seed in SEEDS:
        prefix = os.path.join(EXP_DIR, "results", "td3", f"duty_v1_s{seed}",
                              "td3_tc_final")
        policy = make_policy("td3", state_dim=2, action_dim=1, max_action=1.0,
                             discount=0.99, tau=0.005)
        policy.load(prefix)
        jobs.append((f"actor_s{seed}", policy.select_action))

    print(f"{len(jobs)} runs ({len(D_GRID)} static + {len(SEEDS)} actors), "
          f"{N_BLOCKS}x10 s each, warmed IC\n{BENCH_NOTE}", flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        futs = {ex.submit(run_one, tag, fn): tag for tag, fn in jobs}
        for fut in as_completed(futs):
            results.append(fut.result())

    results.sort(key=lambda r: r["tag"])
    out = os.path.join(OUT_ROOT, "summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\n===== SUMMARY (last-block reward = sustained value) =====")
    for r in results:
        print(f"  {r['tag']:16s} last-r={r['last_r']:+.4f} mean-r={r['mean_r']:+.4f} "
              f"X_last={r['x_last']:.4f}")
    print(f"  {BENCH_NOTE}\n  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
