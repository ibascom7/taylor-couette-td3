#!/usr/bin/env python3
"""130 s warmed comparators for the duty_v3 tau-block campaign.

duty_v3 scores 26 s blocks over 130 s episodes, so its constant / champion
references must be measured on the SAME windows -- the 50/60 s t50-table and
duty_diag numbers do not transfer. This runs the two comparators for 130 s
from the warmed constant-300 steady state and reports X-bar and the reward
X-bar - P/31.94 on every 26 s window, [104, 130] being the v3 last-block
(sustained) reference.

Runs (both via the v1 duty env, T = 5 s pinned, 13 x 10 s blocks for a fine
conversion series; the WAVEFORM is what matters, block bookkeeping here is
just logging):
    constant_wb300  -- D = 1.0  (the constant baseline)
    champion_D0.90  -- D = 0.9  (the static idle law, T=5)

Outputs -> results/duty_v3_refs/{tag}/blocks.csv + timeseries.csv and a
printed summows summary. ~100 min each, run concurrently.

USAGE
    nohup python3 -u duty_v3_refs.py > results/duty_v3_refs/run.log 2>&1 &
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
from taylor_couette_mixing import motor_power  # noqa: E402
from taylor_couette_mixing.envs.taylor_couette_duty import RPM  # noqa: E402

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(EXP_DIR, "results", "warmed_grad300", "side_outlet_grad_case")
OUT_ROOT = os.path.join(EXP_DIR, "results", "duty_v3_refs")

DURATION = 130.0
P_MAX = 31.94
RUNS = [("constant_wb300", 1.0), ("champion_D0.90", 0.9)]


def duty_to_action(d):
    return 2.0 * (d - 0.2) / 0.8 - 1.0


def run_one(tag, duty):
    run_dir = os.path.join(OUT_ROOT, tag)
    workdir = os.path.join(run_dir, "case")
    os.makedirs(run_dir, exist_ok=True)
    if os.path.isdir(workdir):
        shutil.rmtree(workdir)
    shutil.copytree(TEMPLATE, workdir)
    env = TaylorCouetteDutyEnv(workdir, episode_duration=DURATION)
    dense = []
    orig = env.helpers.do_simulation_table

    def tee(points, dt):
        res = orig(points, dt)
        dense.extend(res)
        return res
    env.helpers.do_simulation_table = tee

    env.reset(options={"reset_mode": "hard"})
    a = duty_to_action(duty)
    rows = []
    t0 = time.time()
    for k in range(int(DURATION / 10.0)):
        _, r, _, _, info = env.step([a])
        rows.append(dict(block10=k + 1, duty=info["duty"], X=info["mixing_index"],
                         P_W=info["power_watt"], r=r))
    with open(os.path.join(run_dir, "blocks.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(run_dir, "timeseries.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "conversion", "wallFlux"])
        for m in dense:
            w.writerow([f"{m['t']:.6g}", f"{m['conv']:.8g}", f"{m['wallFlux']:.6g}"])
    shutil.rmtree(workdir)

    # 26 s-window metrics (P from the commanded waveform's block power is
    # constant across windows for these steady waveforms; use the run mean)
    t = np.array([m["t"] for m in dense])
    c = np.array([m["conv"] for m in dense])
    phys = (c >= -0.02) & (c <= 1.02)
    t, c = t[phys], c[phys]
    p_mean = float(np.mean([r_["P_W"] for r_ in rows]))
    out = []
    for k in range(5):
        w0, w1 = k * 26.0, (k + 1) * 26.0
        xw = float(np.mean(c[(t >= w0) & (t <= w1)]))
        out.append((k + 1, w0, w1, xw, xw - p_mean / P_MAX))
    print(f"  [{tag}] {(time.time()-t0)/60:.0f} min, P={p_mean:.3f} W")
    for k, w0, w1, xw, rw in out:
        print(f"     block {k} [{w0:5.0f},{w1:5.0f}] s: X={xw:.4f}  r={rw:.4f}")
    return tag, out[-1]


def main():
    if not os.path.isdir(os.path.join(TEMPLATE, "0.warmed")):
        sys.exit(f"ERROR: warmed template missing: {TEMPLATE}")
    os.makedirs(OUT_ROOT, exist_ok=True)
    print(f"130 s warmed comparators for duty_v3 (26 s windows): "
          f"{[t for t, _ in RUNS]}", flush=True)
    finals = {}
    with ThreadPoolExecutor(max_workers=len(RUNS)) as ex:
        futs = [ex.submit(run_one, t, d) for t, d in RUNS]
        for fut in as_completed(futs):
            tag, last = fut.result()
            finals[tag] = last
    print("\n===== DUTY_V3 SUSTAINED REFERENCES (block 5, [104,130] s) =====")
    for tag, (_, _, _, xw, rw) in sorted(finals.items()):
        print(f"  {tag}: X={xw:.4f}  r={rw:.4f}")


if __name__ == "__main__":
    main()
