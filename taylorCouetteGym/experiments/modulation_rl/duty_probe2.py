#!/usr/bin/env python3
"""Follow-up static probes: localize the D peak + test the IDLE-DURATION law.

duty_diagnostics found a sharp interior optimum: T=5 D=0.90 -> last-block
r = 0.2886 (vs 0.2735 at D=0.80 and 0.2437 at D=1.0 -- a cliff over the last
0.1 of duty). Two things to pin down, same pipeline (warmed, 5 x 10 s blocks):

1. Peak localization: D = 0.85 and 0.95 at T = 5.
2. The idle-duration hypothesis: T=5/D=0.9 idles 0.5 s -- exactly the idle of
   the T=2.5/D=0.8 champion and ~ the 0.6 s swirl-decay time. If enhancement
   is set by idle length (film renewal) rather than period, then matching
   idle=0.5 s at OTHER periods should score comparably:
       T=2.5, D=0.80 (idle 0.5 s)  and  T=10, D=0.95 (idle 0.5 s).
   Same duty-env window as the T=5 grid, so the three are directly comparable.

USAGE
    nohup python3 -u duty_probe2.py > results/duty_diag/probe2.log 2>&1 &
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

from taylor_couette_mixing.envs.taylor_couette_duty import TaylorCouetteDutyEnv  # noqa: E402

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(EXP_DIR, "results", "warmed_grad300", "side_outlet_grad_case")
OUT_ROOT = os.path.join(EXP_DIR, "results", "duty_diag")

# (tag, duty D, period T)
PROBES = [
    ("static_D0.85_T5", 0.85, 5.0),
    ("static_D0.95_T5", 0.95, 5.0),
    ("static_D0.80_T2p5", 0.80, 2.5),   # idle 0.5 s
    ("static_D0.95_T10", 0.95, 10.0),   # idle 0.5 s
]


def run_one(tag, duty, period):
    run_dir = os.path.join(OUT_ROOT, tag)
    workdir = os.path.join(run_dir, "case")
    os.makedirs(run_dir, exist_ok=True)
    if os.path.isdir(workdir):
        shutil.rmtree(workdir)
    shutil.copytree(TEMPLATE, workdir)
    env = TaylorCouetteDutyEnv(workdir, episode_duration=50.0, period=period)
    a = 2.0 * (duty - 0.2) / 0.8 - 1.0
    env.reset(options={"reset_mode": "hard"})
    rows = []
    t0 = time.time()
    for k in range(5):
        _, reward, _, _, info = env.step([a])
        rows.append(dict(block=k + 1, duty=info["duty"], period_s=info["period_s"],
                         w_hi_rpm=info["w_hi_rpm"], X_block=info["mixing_index"],
                         wf_block=info["wf_block"], P_block_W=info["power_watt"],
                         reward=reward))
    with open(os.path.join(run_dir, "blocks.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  [{tag:18s}] idle={(1-duty)*period:.2f}s whi={rows[0]['w_hi_rpm']:.0f} | "
          f"X_last={rows[-1]['X_block']:.4f} last-r={rows[-1]['reward']:+.4f} "
          f"mean-r={float(np.mean([r['reward'] for r in rows])):+.4f} "
          f"({(time.time()-t0)/60:.0f} min)", flush=True)
    shutil.rmtree(workdir)
    return dict(tag=tag, duty=duty, period=period, idle_s=(1 - duty) * period,
                last_r=rows[-1]["reward"], x_last=rows[-1]["X_block"])


def main():
    if not os.path.isdir(os.path.join(TEMPLATE, "0.warmed")):
        sys.exit(f"ERROR: warmed template missing: {TEMPLATE}")
    print(f"{len(PROBES)} static probes (peak localization + idle-0.5s law)", flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=len(PROBES)) as ex:
        futs = [ex.submit(run_one, *p) for p in PROBES]
        for fut in as_completed(futs):
            results.append(fut.result())
    results.sort(key=lambda r: r["tag"])
    out = os.path.join(OUT_ROOT, "probe2_summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print("\n===== PROBE2 SUMMARY =====")
    print("  reference: T5/D0.90 idle 0.5s -> 0.2886 | T5/D0.80 idle 1.0s -> 0.2735 | T5/D1.0 -> 0.2437")
    for r in results:
        print(f"  {r['tag']:18s} idle={r['idle_s']:.2f}s last-r={r['last_r']:+.4f} X_last={r['x_last']:.4f}")
    print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
