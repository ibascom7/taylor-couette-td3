#!/usr/bin/env python3
"""Deterministic eval of a trained modulation TD3 policy on the WARMED
(continuous-manufacturing) IC.

Same geometry, env, and 50 s / 5-block horizon the v5 policy trained on; the
ONLY change is the initial condition: the episode starts from the constant-300
steady state (0.warmed/ in the local warmed template built by
warm_template.py -- steady X = 0.353, wallFlux = 8.4e-9) instead of the
pristine pre-filled state. TaylorCouetteModulationEnv.reset(hard) prefers
0.warmed/ wherever it exists, so no env changes are needed.

NB the env's reset observation is all zeros by construction (_zero_state), so
the policy's FIRST block action is identical to its pristine-IC first action;
the warmed IC enters through the flow observables from block 2 onward. Worth
remembering when reading the block table.

Metrics match the warmed fig7 table (experiments/modulation_vs_constant/
REPORT_warmed_benchmarks.md): X_last = final-block mean conversion (== the
sweep's last-full-period window, block_dt = 10 s), P_ep = episode-average
commanded motor power, R- = X_last - P_ep/31.94. Warmed w_b=300 rows to beat:
pulsed T=2.5/D=80% R- = 0.2844, pulsed T=10/D=80% R- = 0.2674, constant
R- = 0.2374.

COST: ~1 h (5 blocks x 10 s at ~60 s CPU per sim-second, 1 core).

USAGE
    mkdir -p results/warmed_eval
    nohup python3 -u eval_warmed.py > results/warmed_eval/s2_warmed.log 2>&1 &
    python3 eval_warmed.py --checkpoint results/td3/mod_wb300_v5_s2/td3_tc_t28000
"""

import argparse
import csv
import json
import os
import shutil
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GYM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if GYM_ROOT not in sys.path:
    sys.path.insert(0, GYM_ROOT)

from train import make_policy  # noqa: E402
from taylor_couette_mixing.envs.taylor_couette_modulation import (  # noqa: E402
    TaylorCouetteModulationEnv,
)

EXP_DIR = os.path.dirname(os.path.abspath(__file__))

# Warmed w_b=300 rows AT t=50 (summary_table_t50.csv -- the 50 s RL episodes
# must be compared against 50 s-windowed benchmarks, not the 60 s table).
BENCH = {"pulsed T=2.5 D=80%": 0.2714, "pulsed T=10 D=80%": 0.2651,
         "constant 300": 0.2390}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint",
                   default=os.path.join(EXP_DIR, "results", "td3",
                                        "mod_wb300_v5_s2", "td3_tc_final"),
                   help="policy prefix (needs _actor/_critic + _optimizer files)")
    p.add_argument("--template",
                   default=os.path.join(EXP_DIR, "results", "warmed_grad300",
                                        "side_outlet_grad_case"),
                   help="warmed template with 0.warmed/ + compiled dynamicCode/")
    p.add_argument("--tag", default="s2_warmed")
    # v5 env spec (run_carya_modulation.slurm; do not change for the s2 eval)
    p.add_argument("--w_b_rpm", type=float, default=300.0)
    p.add_argument("--episode_duration", type=float, default=50.0)
    p.add_argument("--block_dt", type=float, default=10.0)
    p.add_argument("--duty_min", type=float, default=0.6)
    p.add_argument("--duty_max", type=float, default=1.0)
    p.add_argument("--idle_min_rpm", type=float, default=0.0)
    p.add_argument("--period_min", type=float, default=0.5)
    p.add_argument("--period_max", type=float, default=5.0)
    p.add_argument("--ramp_time", type=float, default=0.05)
    p.add_argument("--p_max_watt", type=float, default=31.94)
    p.add_argument("--wallflux_max", type=float, default=1.32e-8)
    return p


def load_policy(prefix):
    for suffix in ("_actor", "_actor_optimizer", "_critic", "_critic_optimizer"):
        if not os.path.isfile(prefix + suffix):
            raise FileNotFoundError(f"checkpoint file missing: {prefix + suffix}")
    policy = make_policy("td3", state_dim=7, action_dim=3, max_action=1.0,
                         discount=0.99, tau=0.005)
    policy.load(prefix)
    return policy


BLOCK_FIELDS = ["block", "t0_s", "duty", "w_low_rpm", "period_s", "w_hi_rpm",
                "X_block", "wf_norm", "P_block_W", "reward", "wall_min"]


def main():
    args = build_parser().parse_args()
    run_dir = os.path.join(EXP_DIR, "results", "warmed_eval", args.tag)
    os.makedirs(run_dir, exist_ok=True)
    workdir = os.path.join(run_dir, "case")

    if not os.path.isdir(os.path.join(args.template, "0.warmed")):
        sys.exit(f"ERROR: template has no 0.warmed/: {args.template} "
                 "(run warm_template.py first)")
    if os.path.isdir(workdir):
        shutil.rmtree(workdir)
    shutil.copytree(args.template, workdir)   # inherits 0.warmed + dynamicCode

    policy = load_policy(args.checkpoint)
    print(f"[eval] checkpoint={args.checkpoint}")
    print(f"[eval] warmed case={workdir}")
    a0 = policy.select_action(np.zeros(7, dtype=np.float32))
    print(f"[eval] block-1 action on the zero reset-obs (IC-blind): {a0}")

    env = TaylorCouetteModulationEnv(
        case_path=workdir,
        w_b_rpm=args.w_b_rpm,
        episode_duration=args.episode_duration,
        block_dt=args.block_dt,
        duty_min=args.duty_min, duty_max=args.duty_max,
        idle_min_rpm=args.idle_min_rpm,
        period_min=args.period_min, period_max=args.period_max,
        ramp_time=args.ramp_time,
        p_max_watt=args.p_max_watt,
        wallflux_max=args.wallflux_max,
    )

    # Tee the raw METRICS samples for a dense conversion-vs-time trace.
    dense = []
    orig_table = env.helpers.do_simulation_table

    def tee(points, dt):
        res = orig_table(points, dt)
        dense.extend(res)
        return res
    env.helpers.do_simulation_table = tee

    obs, _ = env.reset(options={"reset_mode": "hard"})
    state = np.asarray(obs, dtype=np.float32)

    blocks_path = os.path.join(run_dir, "blocks.csv")
    with open(blocks_path, "w", newline="") as f:
        csv.writer(f).writerow(BLOCK_FIELDS)

    ep_ret, rows = 0.0, []
    for k in range(env.max_steps):
        t_blk = time.time()
        action = policy.select_action(state)
        obs, reward, terminated, truncated, info = env.step(action)
        state = np.asarray(obs, dtype=np.float32)
        ep_ret += reward
        row = dict(block=k + 1, t0_s=k * args.block_dt,
                   duty=info["duty"], w_low_rpm=info["w_low_rpm"],
                   period_s=info["period_s"], w_hi_rpm=info["w_hi_rpm"],
                   X_block=info["mixing_index"], wf_norm=info["wf_norm"],
                   P_block_W=info["power_watt"], reward=reward,
                   wall_min=(time.time() - t_blk) / 60.0)
        rows.append(row)
        with open(blocks_path, "a", newline="") as f:
            csv.writer(f).writerow([row[k2] for k2 in BLOCK_FIELDS])
        print(f"[eval] block {k+1}/{env.max_steps}: D={row['duty']:.2f} "
              f"wlo={row['w_low_rpm']:.0f} T={row['period_s']:.2f} "
              f"whi={row['w_hi_rpm']:.0f} | X={row['X_block']:.4f} "
              f"P={row['P_block_W']:.3f}W r={reward:+.4f} "
              f"({row['wall_min']:.1f} min)", flush=True)

    # Dense trace CSV (fig7-comparable).
    ts_path = os.path.join(run_dir, "timeseries.csv")
    with open(ts_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "conversion", "wallFlux"])
        for m in dense:
            w.writerow([f"{m['t']:.6g}", f"{m['conv']:.8g}", f"{m['wallFlux']:.6g}"])

    x_last = rows[-1]["X_block"]
    p_ep = float(np.mean([r["P_block_W"] for r in rows]))
    r_minus = x_last - p_ep / args.p_max_watt
    summary = dict(tag=args.tag, checkpoint=args.checkpoint,
                   episode_return=ep_ret, x_last=x_last, p_ep_watt=p_ep,
                   r_minus=r_minus, benchmarks_warmed_wb300=BENCH)
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n================ WARMED EVAL SUMMARY ================")
    print(f"  episode return (sum of 5 block rewards): {ep_ret:+.4f}")
    print(f"  X_last={x_last:.4f}  P_ep={p_ep:.3f} W  R- = {r_minus:.4f}")
    for name, r in BENCH.items():
        verdict = "BEATS" if r_minus > r else "below"
        print(f"    vs {name:22s} R-={r:.4f}  -> {verdict}")

    # Conversion-vs-time with block duty annotations.
    t = np.array([m["t"] for m in dense])
    c = np.array([m["conv"] for m in dense])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, c, lw=1.2)
    for r in rows:
        ax.axvline(r["t0_s"], color="grey", lw=0.5, ls=":")
        ax.text(r["t0_s"] + 0.3, ax.get_ylim()[0] + 0.01,
                f"D={r['duty']:.2f}\nT={r['period_s']:.1f}", fontsize=7)
    ax.set_xlabel("time since warmed start [s]")
    ax.set_ylabel("outlet conversion")
    ax.set_title(f"v5-s2 policy on WARMED IC ({args.tag}): "
                 f"X_last={x_last:.3f}, R-={r_minus:.3f}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "conversion_vs_time.png"), dpi=140)
    print(f"\n[eval] outputs -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
