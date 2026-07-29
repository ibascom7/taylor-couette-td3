#!/usr/bin/env python3
"""Deterministic TRANSFER eval of a trained modulation TD3 policy on the
FULL-HEIGHT (Gamma = 30) reactor.

The v5 seed-2 policy (results/td3/mod_wb300_v5_s2, fixed-mean w_b = 300) was
trained on the SHORT graded case (side_outlet_grad_case, Gamma = 6, tau ~ 26 s,
50 s episodes of 5 x 10 s blocks). The fig7 reference sweep
(experiments/modulation_vs_constant/results_full_tc) showed conversion behaves
far better on the full Lopez-Guajardo geometry (full_tc_cat_case, H = 190.5 mm,
tau ~ 130 s, 300 s episodes): constant-300 X = 0.787, static pulsed D=80%/T=10s
X = 0.876, both at ~3.7 W. This script runs the LEARNED policy on that same
full-height case (via cases/full_tc_grad_case, the RL-drivable twin -- identical
physics, rlMetrics controlDict) and reports the same metrics, so the three rows
are directly comparable:

    X_last  = final-block mean conversion (= the sweep's last-full-period
              window [290, 300] s, since block_dt == the sweep's T = 10 s)
    P_ep    = episode-average commanded motor power (paper Eqs 18-23; the
              fixed-mean constraint pins every block's commanded mean at
              w_b = 300, so P_ep ~ 3.7 W by construction -- equal-power test)
    R-      = X_last - P_ep / 31.94   (the fig7 table's reward convention)

CLOCK MODES (the one training/deploy mismatch is the episode clock, obs[3]):
    stretch (default) -- feed the env's native clock t/300. The policy's
              schedule (trained against t/50) is stretched 6x; all clock values
              stay inside the trained [0, 1) range.
    wrap    -- feed (t mod 50)/50: replay the policy's 50 s program cyclically
              6x, letting the flow state carry over. Tests the policy as a
              periodic controller instead of a stretched schedule.

COST: ~260 CPU-s per sim-second on the 16,650-cell mesh (fig7-measured
~1300 min per 300 s episode, 1 core). Expect ~22 h per eval run. --smoke runs
a 4 s / 2-block episode (~20 min) to validate the whole pipeline first.

USAGE
    python3 eval_full_tc.py --smoke                       # pipeline check
    nohup python3 -u eval_full_tc.py --clock stretch > eval_stretch.log 2>&1 &
    nohup python3 -u eval_full_tc.py --clock wrap    > eval_wrap.log    2>&1 &
    python3 eval_full_tc.py --analyze-only --tag s2_stretch   # replot

Progress is written incrementally to results/full_tc_eval/<tag>/blocks.csv
after every 10 s block, so a run can be monitored (and salvaged) mid-flight.
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
from parallel_train import _copy_master, _blockmesh  # noqa: E402
from taylor_couette_mixing.envs.helpers import Helpers  # noqa: E402
from taylor_couette_mixing.envs.taylor_couette_modulation import (  # noqa: E402
    TaylorCouetteModulationEnv,
)
from taylor_couette_mixing.envs.taylor_couette_waveform import square_wave_points  # noqa: E402

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
RPM = 2.0 * np.pi / 60.0

REF_DIR = os.path.join(GYM_ROOT, "experiments", "modulation_vs_constant",
                       "results_full_tc")


# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint",
                   default=os.path.join(EXP_DIR, "results", "td3",
                                        "mod_wb300_v5_s2", "td3_tc_final"),
                   help="TD3 checkpoint PREFIX (needs _actor/_critic and both "
                        "_optimizer files; TD3.load requires all four).")
    p.add_argument("--case_path",
                   default="taylor_couette_mixing/cases/full_tc_grad_case",
                   help="master RL-drivable case (relative to gym root).")
    p.add_argument("--clock", choices=["stretch", "wrap"], default="stretch",
                   help="obs[3] mapping: 'stretch' = native t/episode_duration; "
                        "'wrap' = (t mod train_episode)/train_episode (cyclic "
                        "replay of the trained 50 s program).")
    p.add_argument("--episode_duration", type=float, default=300.0,
                   help="episode length [s] (fig7 full-TC reference = 300).")
    p.add_argument("--block_dt", type=float, default=10.0)
    p.add_argument("--train_episode_duration", type=float, default=50.0,
                   help="episode length the policy was TRAINED on (wrap period).")
    # env design -- MUST match training (run_carya_modulation.slurm v5)
    p.add_argument("--w_b_rpm", type=float, default=300.0)
    p.add_argument("--duty_min", type=float, default=0.6)
    p.add_argument("--duty_max", type=float, default=1.0)
    p.add_argument("--idle_min_rpm", type=float, default=0.0)
    p.add_argument("--period_min", type=float, default=0.5)
    p.add_argument("--period_max", type=float, default=5.0)
    p.add_argument("--ramp_time", type=float, default=0.05)
    p.add_argument("--p_max_watt", type=float, default=31.94)
    p.add_argument("--wallflux_max", type=float, default=1.32e-8,
                   help="obs normalizer -- keep the TRAINING value (the policy's "
                        "wf_norm scale must match training; on the tall reactor "
                        "wallFlux can exceed it, wf_norm > 1 -- that is part of "
                        "the transfer test, not an error).")
    p.add_argument("--tag", default=None,
                   help="results subdir name; default s2_<clock>[_smoke].")
    p.add_argument("--results_dir",
                   default=os.path.join(EXP_DIR, "results", "full_tc_eval"))
    p.add_argument("--keep_case", action="store_true",
                   help="keep the run case's time dirs for ParaView (~1 GB).")
    p.add_argument("--smoke", action="store_true",
                   help="4 s episode of 2 x 2 s blocks (~20 min): validates case "
                        "twin, coded-FO compile, METRICS, policy load, logging.")
    p.add_argument("--analyze-only", action="store_true",
                   help="regenerate plots/summary from an existing blocks.csv.")
    return p


# --------------------------------------------------------------------------- #
def prepare_case(master, workdir):
    """Fresh case copy (ALWAYS fresh -- a reused case's leftover time dirs
    contaminate the rollout), blockMesh, one throwaway pimpleFoam step to
    compile the coded rlMetrics FO into dynamicCode/, hard reset to pristine."""
    print(f"[setup] building fresh eval case at {workdir}", flush=True)
    t0 = time.time()
    _copy_master(master, workdir)
    _blockmesh(workdir)
    helpers = Helpers(workdir)
    helpers._update_end_time(0.05)
    import subprocess
    with open(os.path.join(workdir, "log.compile"), "w") as fh:
        r = subprocess.run(["pimpleFoam"], cwd=workdir,
                           stdout=fh, stderr=subprocess.STDOUT,
                           env=dict(os.environ, OMP_NUM_THREADS="1"))
    if r.returncode != 0:
        raise RuntimeError(f"compile run failed in {workdir} (see log.compile)")
    helpers.reset_case(mode="hard")
    print(f"[setup] case ready in {time.time()-t0:.0f}s", flush=True)


def load_policy(prefix):
    for suffix in ("_actor", "_actor_optimizer", "_critic", "_critic_optimizer"):
        if not os.path.isfile(prefix + suffix):
            raise FileNotFoundError(f"checkpoint file missing: {prefix + suffix}")
    policy = make_policy("td3", state_dim=7, action_dim=3, max_action=1.0,
                         discount=0.99, tau=0.005)
    policy.load(prefix)
    return policy


# --------------------------------------------------------------------------- #
BLOCK_FIELDS = ["block", "t0_s", "duty", "w_low_rpm", "period_s", "w_hi_rpm",
                "X_block", "wf_norm", "P_block_W", "reward", "clock_fed",
                "wall_min"]


def run_episode(args, run_dir, workdir):
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

    # Tee the raw METRICS samples (~0.1 s cadence) out of the env's step calls
    # for a dense conversion-vs-time trace comparable to the fig7 CSVs.
    dense = []
    orig_table = env.helpers.do_simulation_table

    def tee(points, dt):
        res = orig_table(points, dt)
        dense.extend(res)
        return res
    env.helpers.do_simulation_table = tee

    policy = load_policy(args.checkpoint)
    wrap_blocks = max(1, int(round(args.train_episode_duration / args.block_dt)))

    obs, _ = env.reset(options={"reset_mode": "hard"})
    state = np.asarray(obs, dtype=np.float32)

    blocks_path = os.path.join(run_dir, "blocks.csv")
    with open(blocks_path, "w", newline="") as f:
        csv.writer(f).writerow(BLOCK_FIELDS)

    n_blocks = env.max_steps
    ep_ret = 0.0
    rows = []
    t_start = time.time()
    for k in range(n_blocks):
        s = state.copy()
        if args.clock == "wrap":
            s[3] = (k % wrap_blocks) / wrap_blocks
        t_blk = time.time()
        action = policy.select_action(s)
        obs, reward, terminated, truncated, info = env.step(action)
        state = np.asarray(obs, dtype=np.float32)
        ep_ret += reward
        wall_min = (time.time() - t_blk) / 60.0
        row = dict(block=k + 1, t0_s=k * args.block_dt,
                   duty=info["duty"], w_low_rpm=info["w_low_rpm"],
                   period_s=info["period_s"], w_hi_rpm=info["w_hi_rpm"],
                   X_block=info["mixing_index"], wf_norm=info["wf_norm"],
                   P_block_W=info["power_watt"], reward=reward,
                   clock_fed=float(s[3]), wall_min=wall_min)
        rows.append(row)
        with open(blocks_path, "a", newline="") as f:
            csv.writer(f).writerow([f"{row[c]:.6g}" if isinstance(row[c], float)
                                    else row[c] for c in BLOCK_FIELDS])
        done_frac = (k + 1) / n_blocks
        eta_h = (time.time() - t_start) / done_frac * (1 - done_frac) / 3600
        print(f"[block {k+1:2d}/{n_blocks}] D={info['duty']:.2f} "
              f"wlo={info['w_low_rpm']:5.1f} T={info['period_s']:.2f}s "
              f"whi={info['w_hi_rpm']:5.1f} | X={info['mixing_index']:.4f} "
              f"P={info['power_watt']:.2f}W r={reward:+.4f} | "
              f"{wall_min:.1f} min, ETA {eta_h:.1f} h", flush=True)
        if terminated or truncated:
            break

    # Dense trace (reject unphysical scalar-boundedness blips like the sweep).
    with open(os.path.join(run_dir, "dense_timeseries.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "conversion", "wallFlux"])
        for r in dense:
            w.writerow([f"{r['t']:.6g}", f"{r['conv']:.8g}", f"{r['wallFlux']:.8g}"])

    # Exact commanded waveform, rebuilt per block (phase resets each block).
    with open(os.path.join(run_dir, "waveform_points.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "omega_cmd_rad_s"])
        for row in rows:
            pts, _ = square_wave_points(
                row["t0_s"], args.block_dt, row["w_hi_rpm"] * RPM,
                row["w_low_rpm"] * RPM, row["period_s"], row["duty"],
                args.ramp_time, phase0=0.0)
            pts = Helpers.sanitize_table_points(pts)
            for t, om in pts:
                if t <= row["t0_s"] + args.block_dt + 1e-9:
                    w.writerow([f"{t:.6f}", f"{om:.6f}"])

    if args.keep_case:
        open(os.path.join(workdir, "case.foam"), "w").close()   # ParaView opener

    return rows, ep_ret


# --------------------------------------------------------------------------- #
def load_rows(run_dir):
    rows = []
    with open(os.path.join(run_dir, "blocks.csv")) as f:
        for r in csv.DictReader(f):
            rows.append({k: (int(v) if k == "block" else float(v))
                         for k, v in r.items()})
    return rows


def load_reference():
    """(summary rows, timeseries dict) from the fig7 full-TC sweep, if present."""
    ref_rows, ref_ts = [], {}
    summ = os.path.join(REF_DIR, "summary_table.csv")
    if os.path.isfile(summ):
        with open(summ) as f:
            ref_rows = list(csv.DictReader(f))
    for tag in ("constant_wb300", "pulsed_wb300"):
        p = os.path.join(REF_DIR, f"{tag}_timeseries.csv")
        if os.path.isfile(p):
            t, c = [], []
            with open(p) as f:
                for r in csv.DictReader(f):
                    t.append(float(r["time_s"]))
                    c.append(float(r["conversion"]))
            ref_ts[tag] = (np.array(t), np.array(c))
    return ref_rows, ref_ts


def analyze(args, run_dir, rows, ep_ret=None):
    x_last = rows[-1]["X_block"]
    p_ep = float(np.mean([r["P_block_W"] for r in rows]))
    r_minus = x_last - p_ep / args.p_max_watt
    r_final_block = rows[-1]["reward"]

    ref_rows, ref_ts = load_reference()
    ref = {(r["mode"], int(r["wb_rpm"])): r for r in ref_rows}

    summary = dict(
        tag=args.tag, clock=args.clock, checkpoint=args.checkpoint,
        episode_duration=args.episode_duration, block_dt=args.block_dt,
        n_blocks=len(rows), X_last=x_last, P_ep_W=p_ep,
        R_minus=r_minus, R_final_block=r_final_block,
        episode_return=ep_ret,
        wallflux_max=args.wallflux_max, p_max_watt=args.p_max_watt,
    )
    for mode in ("constant", "pulsed"):
        r = ref.get((mode, 300))
        if r:
            summary[f"ref_{mode}300_X"] = float(r["X_conv_lastperiod"])
            summary[f"ref_{mode}300_R_minus"] = float(r["R_minus (X - P/Pmax)"])
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 74)
    print(f"TD3 v5-s2 TRANSFER EVAL -- full-height Gamma=30 reactor "
          f"({args.clock} clock)")
    print("=" * 74)
    print(f"  X_last (final {args.block_dt:.0f}s block) : {x_last:.4f}")
    print(f"  P_ep (episode-avg motor power)  : {p_ep:.3f} W "
          f"(P/Pmax = {p_ep/args.p_max_watt:.4f})")
    print(f"  R- = X_last - P_ep/Pmax         : {r_minus:.4f}")
    if ep_ret is not None:
        print(f"  episode return ({len(rows)} blocks)      : {ep_ret:.3f}")
    for mode in ("constant", "pulsed"):
        r = ref.get((mode, 300))
        if r:
            print(f"  reference {mode:9s} wb300      : X={float(r['X_conv_lastperiod']):.4f}  "
                  f"R-={float(r['R_minus (X - P/Pmax)']):.4f}")
    print("=" * 74 + "\n")

    # ---- plots ----
    t_blk = np.array([r["t0_s"] + args.block_dt for r in rows])
    x_blk = np.array([r["X_block"] for r in rows])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    dense_p = os.path.join(run_dir, "dense_timeseries.csv")
    if os.path.isfile(dense_p):
        td, cd = [], []
        with open(dense_p) as f:
            for r in csv.DictReader(f):
                tt, cc = float(r["time_s"]), float(r["conversion"])
                if -0.02 <= cc <= 1.02:
                    td.append(tt), cd.append(cc)
        ax.plot(td, cd, "-", color="#d1495b", lw=1.4,
                label=f"TD3 v5-s2 ({args.clock} clock)")
    ax.plot(t_blk, x_blk, "o", color="#d1495b", ms=5, mfc="white",
            label="TD3 block means")
    ref_styles = {"constant_wb300": ("#2e6f95", "constant 300 rpm (ref)"),
                  "pulsed_wb300": ("#666666", "pulsed D80/T10 @300 (ref)")}
    for tag2, (col, lab) in ref_styles.items():
        if tag2 in ref_ts:
            tt, cc = ref_ts[tag2]
            keep = (cc >= -0.02) & (cc <= 1.02)
            ax.plot(tt[keep], cc[keep], "-", color=col, lw=1.2, alpha=0.8, label=lab)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("outlet conversion  (1 - cup c / c0)")
    ax.set_title(f"TD3 modulation policy on the full-height (Gamma=30) reactor\n"
                 f"trained on Gamma=6 / 50 s episodes; {args.clock} clock, "
                 f"fixed-mean w_b=300")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "conversion_vs_time.png"), dpi=140)
    plt.close(fig)

    wf_p = os.path.join(run_dir, "waveform_points.csv")
    if os.path.isfile(wf_p):
        tt, om = [], []
        with open(wf_p) as f:
            for r in csv.DictReader(f):
                tt.append(float(r["time_s"])), om.append(float(r["omega_cmd_rad_s"]))
        fig, ax = plt.subplots(figsize=(11, 3.2))
        ax.plot(tt, np.array(om) / RPM, "-", color="#30323a", lw=0.9)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("commanded omega [rpm]")
        ax.set_title("Commanded inner-wall speed (per-block TD3 waveforms, "
                     "commanded mean = 300 rpm every block)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "omega_command.png"), dpi=140)
        plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
    bl = [r["block"] for r in rows]
    axes[0].step(bl, [r["duty"] for r in rows], where="mid", color="#2e6f95")
    axes[0].set_ylabel("duty D")
    axes[0].set_ylim(0.55, 1.05)
    axes[1].step(bl, [r["w_low_rpm"] for r in rows], where="mid",
                 color="#2e6f95", label="trough $w_{lo}$")
    axes[1].step(bl, [r["w_hi_rpm"] for r in rows], where="mid",
                 color="#d1495b", label="burst $w_{hi}$")
    axes[1].set_ylabel("rpm")
    axes[1].legend(fontsize=8)
    axes[2].step(bl, [r["period_s"] for r in rows], where="mid", color="#2e6f95")
    axes[2].set_ylabel("period T [s]")
    axes[2].set_xlabel("block")
    for ax2 in axes:
        ax2.grid(True, alpha=0.3)
    axes[0].set_title("Per-block decoded actions (deterministic policy)")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "actions_per_block.png"), dpi=140)
    plt.close(fig)

    print(f"outputs -> {run_dir}", flush=True)
    return summary


# --------------------------------------------------------------------------- #
def main():
    args = build_parser().parse_args()
    if args.smoke:
        args.episode_duration = 4.0
        args.block_dt = 2.0
        print("[smoke] 4 s episode, 2 x 2 s blocks", flush=True)
    if args.tag is None:
        args.tag = f"s2_{args.clock}" + ("_smoke" if args.smoke else "")
    run_dir = os.path.join(args.results_dir, args.tag)

    if args.analyze_only:
        rows = load_rows(run_dir)
        analyze(args, run_dir, rows)
        return

    os.makedirs(run_dir, exist_ok=True)
    master = (args.case_path if os.path.isabs(args.case_path)
              else os.path.join(GYM_ROOT, args.case_path))
    workdir = os.path.join(run_dir, "case")

    prepare_case(master, workdir)
    t0 = time.time()
    rows, ep_ret = run_episode(args, run_dir, workdir)
    print(f"[episode done] {len(rows)} blocks in {(time.time()-t0)/3600:.2f} h",
          flush=True)
    analyze(args, run_dir, rows, ep_ret)
    if not args.keep_case:
        # keep 0/, constant/, system/, logs; drop the ~1 GB of time dirs
        import subprocess  # noqa: F401  (clean_run_artifacts equivalent inline)
        for name in os.listdir(workdir):
            p = os.path.join(workdir, name)
            if os.path.isdir(p):
                try:
                    if float(name) != 0.0:
                        shutil.rmtree(p)
                except ValueError:
                    pass


if __name__ == "__main__":
    main()
