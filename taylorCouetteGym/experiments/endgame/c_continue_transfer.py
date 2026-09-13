#!/usr/bin/env python3
"""C -- CONTINUE the Gamma=30 transfer eval past its trained horizon.

WHY
    The stretch-clock transfer run (modulation_rl/results/full_tc_eval/
    s2_stretch) beats the static D80/T10 pulse by +0.025 in X, but the whole
    margin appears in the last four blocks, where the policy glides its duty
    from 0.63 to 0.97 as the clock approaches 1 -- an end-of-run HARVEST of
    the film it renewed during the shimmy phase (blocks 17-26 sit at 0.880-
    0.885, a tie with the static pulse at 0.876). The v5 trainer treated the
    episode end as terminal, so harvesting was the optimal thing to learn.
    This continues the SAME simulation from its kept 300 s state for several
    more residence times with the policy still in the loop, and asks whether
    conversion holds (a genuinely better sustained state: shimmy-then-hold)
    or decays to the 0.88 plateau or below (the margin was borrowed, and the
    honest claim is "matches in continuous operation, exceeds in batch").

HOW
    The source eval kept its case (every 1 s time folder to 300 s, git-
    tracked). A MINIMAL copy (0.orig, 0, constant, system, the latest time
    folder; the coded FOs recompile on start-up) is made under
    results/C/<tag>/case; the modulation
    env is built on it WITHOUT reset() and its internal state (X, dX,
    wallFlux, previous action, block counter) is reconstructed from the
    source blocks.csv; then blocks 31, 32, ... run through the same
    Helpers.do_simulation_table path (omega table re-stamped into the
    restart U file, omegaCoeffs cleared -- the freeze-bug guard).

    --clock hold      (stretch source) feed obs[3] at the last value the policy
                      saw (29/30 = 0.9667; it never acted at exactly 1.0 in
                      training), i.e. "keep doing your endgame"
    --clock saturate  feed 1.0
    --clock wrap      (wrap source) keep cycling (t mod 50)/50
    --mode frozen     open-loop: repeat the final block's action forever
                      (asks whether the endgame WAVEFORM is sustainable,
                      independent of the policy)

OUTPUTS (results/C/<tag>/)
    blocks.csv (block numbers continue from the source), dense_timeseries.csv,
    summary.json (X per residence-time window after 300 s vs the references),
    conversion_vs_time.png (source 0-300 s + continuation + references),
    actions_per_block.png. --resume continues a killed run from its own
    blocks.csv + latest time folder.

USAGE
    python3 c_continue_transfer.py --source s2_stretch --clock hold --smoke   # 1 block, ~40 min
    nohup python3 -u c_continue_transfer.py --source s2_stretch --clock hold > logs/C_stretch.log 2>&1 &
    nohup python3 -u c_continue_transfer.py --source s2_wrap    --clock wrap > logs/C_wrap.log 2>&1 &
    python3 c_continue_transfer.py --source s2_stretch --clock hold --analyze-only
"""

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time

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

from taylor_couette_mixing.envs.taylor_couette_modulation import (  # noqa: E402
    TaylorCouetteModulationEnv, RPM,
)
from taylor_couette_mixing.envs.taylor_couette_waveform import square_wave_points  # noqa: E402
from taylor_couette_mixing.envs.helpers import Helpers  # noqa: E402

EVAL_DIR = os.path.join(MOD_RL, "results", "full_tc_eval")
REF_DIR = os.path.join(GYM_ROOT, "experiments", "modulation_vs_constant", "results_full_tc")
DEFAULT_CKPT = os.path.join(MOD_RL, "results", "td3", "mod_wb300_v5_s2", "td3_tc_final")

# env design -- MUST match training / eval_full_tc.py (fixed-mean v5)
ENV = dict(w_b_rpm=300.0, block_dt=10.0, duty_min=0.6, duty_max=1.0, idle_min_rpm=0.0,
           period_min=0.5, period_max=5.0, ramp_time=0.05, p_max_watt=31.94,
           wallflux_max=1.32e-8)
TRAIN_EPISODE = 50.0
TAU = 130.0
REFS = dict(constant300_X=0.786883, pulsed_D80T10_X=0.87645,
            stretch_X_last=0.901486, wrap_X_last=0.881342)

BLOCK_FIELDS = ["block", "t0_s", "duty", "w_low_rpm", "period_s", "w_hi_rpm",
                "X_block", "wf_norm", "P_block_W", "reward", "clock_fed", "wall_min"]


# --------------------------------------------------------------------------- #
def invert_action(duty, w_low, period):
    """Decoded (duty, w_low, period) -> the raw [-1,1]^3 action that produced it
    (inverse of TaylorCouetteModulationEnv._decode in fixed-mean mode)."""
    a0 = 2.0 * (duty - ENV["duty_min"]) / (ENV["duty_max"] - ENV["duty_min"]) - 1.0
    a1 = 2.0 * (w_low - ENV["idle_min_rpm"]) / (ENV["w_b_rpm"] - ENV["idle_min_rpm"]) - 1.0
    lt0, lt1 = math.log(ENV["period_min"]), math.log(ENV["period_max"])
    a2 = 2.0 * (math.log(max(period, 1e-9)) - lt0) / (lt1 - lt0) - 1.0
    return np.clip(np.array([a0, a1, a2], dtype=float), -1.0, 1.0)


def load_blocks(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({k: (int(v) if k == "block" else float(v)) for k, v in r.items()})
    return rows


def _time_dirs(case):
    out = []
    for n in os.listdir(case):
        p = os.path.join(case, n)
        if os.path.isdir(p):
            try:
                out.append((float(n), n))
            except ValueError:
                pass
    return sorted(out)


def copy_minimal_case(src, dst):
    """Everything needed to continue: static dirs + latest state. dynamicCode/
    (the compiled coded-FO binaries) is deliberately NOT copied: the kept
    case's binaries may come from another host (the evals ran locally, the
    slurm runs on Carya) and pimpleFoam recompiles coded function objects
    once at start-up when the cache is absent (~30 s)."""
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    for sub in ("0.orig", "0", "constant", "system"):
        s = os.path.join(src, sub)
        if os.path.isdir(s):
            shutil.copytree(s, os.path.join(dst, sub))
    tds = _time_dirs(src)
    if not tds:
        raise RuntimeError(f"no time folders in {src}")
    t_latest, latest = tds[-1]
    if latest != "0":
        shutil.copytree(os.path.join(src, latest), os.path.join(dst, latest))
    open(os.path.join(dst, "case.foam"), "w").close()
    return t_latest, latest


def load_policy(prefix):
    from train import make_policy   # torch import only on the CFD path
    for suffix in ("_actor", "_actor_optimizer", "_critic", "_critic_optimizer"):
        if not os.path.isfile(prefix + suffix):
            raise FileNotFoundError(f"checkpoint file missing: {prefix + suffix}")
    policy = make_policy("td3", state_dim=7, action_dim=3, max_action=1.0, discount=0.99, tau=0.005)
    policy.load(prefix)
    return policy


def ensure_source(source_dir, clock):
    """If the source eval is missing (fresh Carya clone without the kept case),
    run eval_full_tc.py --keep_case to produce it (~22 h)."""
    case = os.path.join(source_dir, "case")
    if os.path.isfile(os.path.join(source_dir, "blocks.csv")) and _time_dirs(case) \
            and _time_dirs(case)[-1][0] > 0:
        return
    print(f"[C] source {source_dir} incomplete -- running eval_full_tc.py --keep_case "
          f"--clock {clock} first (~22 h)", flush=True)
    r = subprocess.run([sys.executable, os.path.join(MOD_RL, "eval_full_tc.py"), "--clock", clock,
                        "--keep_case", "--tag", os.path.basename(source_dir)], cwd=MOD_RL)
    if r.returncode != 0:
        raise RuntimeError("eval_full_tc.py failed; cannot continue")


# --------------------------------------------------------------------------- #
def run_continuation(args, run_dir):
    source_dir = os.path.join(EVAL_DIR, args.source)
    src_case = os.path.join(source_dir, "case")
    src_rows = load_blocks(os.path.join(source_dir, "blocks.csv"))
    n_src = len(src_rows)
    t_src_end = n_src * args.block_dt
    tds = _time_dirs(src_case)
    if not tds or abs(tds[-1][0] - t_src_end) > 1e-6:
        raise RuntimeError(f"source case latest time {tds[-1][0] if tds else None} != "
                           f"{t_src_end} (blocks.csv has {n_src} blocks); "
                           f"re-run eval_full_tc.py --keep_case or pass --ensure_source")

    case = os.path.join(run_dir, "case")
    blocks_path = os.path.join(run_dir, "blocks.csv")
    dense_path = os.path.join(run_dir, "dense_timeseries.csv")
    c_rows = load_blocks(blocks_path) if (args.resume and os.path.isfile(blocks_path)) else []
    if args.resume and os.path.isdir(case) and c_rows:
        t_have = _time_dirs(case)[-1][0]
        if abs(t_have - (t_src_end + len(c_rows) * args.block_dt)) > 1e-6:
            raise RuntimeError(f"resume mismatch: case at t={t_have}, blocks.csv has {len(c_rows)} rows")
        print(f"[C] resuming after {len(c_rows)} continuation blocks (t={t_have:.0f}s)", flush=True)
    else:
        c_rows = []
        t_latest, _ = copy_minimal_case(src_case, case)
        print(f"[C] minimal case copied from {src_case} at t={t_latest:.0f}s", flush=True)
        with open(blocks_path, "w", newline="") as f:
            csv.writer(f).writerow(BLOCK_FIELDS)
        with open(dense_path, "w", newline="") as f:
            csv.writer(f).writerow(["time_s", "conversion", "wallFlux"])

    n_more = args.n_more
    env = TaylorCouetteModulationEnv(case_path=case, episode_duration=(n_src + n_more) * args.block_dt,
                                     **ENV)
    # ---- reconstruct the env's internal state from the logged blocks (no reset!) ----
    all_rows = src_rows + c_rows
    last, prev = all_rows[-1], all_rows[-2]
    env.step_count = len(all_rows)
    env.max_steps = n_src + n_more
    env.x_block = last["X_block"]
    env.delta_x = last["X_block"] - prev["X_block"]
    env.wf_block = last["wf_norm"] * ENV["wallflux_max"]
    env.p_block = last["P_block_W"]
    env.prev_action = invert_action(last["duty"], last["w_low_rpm"], last["period_s"])
    env.last_params = (last["duty"], last["w_low_rpm"], last["period_s"], last["w_hi_rpm"])
    env.episode_count = 1

    dense = []
    orig = env.helpers.do_simulation_table

    def tee(points, dt):
        res = orig(points, dt)
        dense.extend(res)
        return res
    env.helpers.do_simulation_table = tee

    policy = None if args.mode == "frozen" else load_policy(args.checkpoint)
    frozen_action = env.prev_action.copy()
    hold_value = args.hold_value if args.hold_value is not None else src_rows[-1]["clock_fed"]
    wrap_blocks = int(round(TRAIN_EPISODE / args.block_dt))

    print(f"[C] source {args.source}: {n_src} blocks to t={t_src_end:.0f}s; continuing "
          f"{n_more - len(c_rows)} more blocks ({(n_more - len(c_rows)) * args.block_dt:.0f} s, "
          f"{(n_more - len(c_rows)) * args.block_dt / TAU:.1f} tau); clock={args.clock} "
          f"(hold value {hold_value:.4f}); mode={args.mode}", flush=True)
    t_start = time.time()
    for k in range(len(c_rows), n_more):
        idx = n_src + k                       # 0-based global block index
        s = np.asarray(env._get_obs(), dtype=np.float32)
        if args.clock == "hold":
            s[3] = hold_value
        elif args.clock == "saturate":
            s[3] = 1.0
        elif args.clock == "wrap":
            s[3] = (idx % wrap_blocks) / wrap_blocks
        t_blk = time.time()
        action = frozen_action if policy is None else policy.select_action(s)
        dense.clear()
        obs, reward, _, _, info = env.step(action)
        wall_min = (time.time() - t_blk) / 60.0
        row = dict(block=idx + 1, t0_s=idx * args.block_dt, duty=info["duty"],
                   w_low_rpm=info["w_low_rpm"], period_s=info["period_s"], w_hi_rpm=info["w_hi_rpm"],
                   X_block=info["mixing_index"], wf_norm=info["wf_norm"], P_block_W=info["power_watt"],
                   reward=reward, clock_fed=float(s[3]), wall_min=wall_min)
        c_rows.append(row)
        with open(blocks_path, "a", newline="") as f:
            csv.writer(f).writerow([f"{row[c]:.6g}" if isinstance(row[c], float) else row[c]
                                    for c in BLOCK_FIELDS])
        with open(dense_path, "a", newline="") as f:
            w = csv.writer(f)
            for m in dense:
                w.writerow([f"{m['t']:.6g}", f"{m['conv']:.8g}", f"{m['wallFlux']:.8g}"])
        done_frac = (k + 1 - 0) / n_more
        eta_h = (time.time() - t_start) / max(k + 1 - len(c_rows) + (len(c_rows) and 0), 1) \
            * (n_more - k - 1) / 3600
        print(f"[block {idx+1:3d}] t={idx*args.block_dt:.0f}s clock={s[3]:.3f} D={info['duty']:.2f} "
              f"wlo={info['w_low_rpm']:5.1f} T={info['period_s']:.2f}s whi={info['w_hi_rpm']:5.1f} "
              f"| X={info['mixing_index']:.4f} P={info['power_watt']:.2f}W r={reward:+.4f} "
              f"| {wall_min:.1f} min, ETA {eta_h:.1f} h", flush=True)

    # commanded waveform of the continuation, for the omega figure
    with open(os.path.join(run_dir, "waveform_points.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "omega_cmd_rad_s"])
        for row in c_rows:
            pts, _ = square_wave_points(row["t0_s"], args.block_dt, row["w_hi_rpm"] * RPM,
                                        row["w_low_rpm"] * RPM, row["period_s"], row["duty"],
                                        ENV["ramp_time"], phase0=0.0)
            for t, om in Helpers.sanitize_table_points(pts):
                if t <= row["t0_s"] + args.block_dt + 1e-9:
                    w.writerow([f"{t:.6f}", f"{om:.6f}"])
    if not args.keep_case:
        for t, n in _time_dirs(case)[:-1]:       # keep 0/ and the final state only
            if n != "0":
                shutil.rmtree(os.path.join(case, n), ignore_errors=True)
    return src_rows, c_rows


# --------------------------------------------------------------------------- #
def _read_ts(path):
    t, c = [], []
    if os.path.isfile(path):
        with open(path) as f:
            for r in csv.DictReader(f):
                tt, cc = float(r["time_s"]), float(r["conversion"])
                if -0.02 <= cc <= 1.02:
                    t.append(tt)
                    c.append(cc)
    return np.array(t), np.array(c)


def analyze(args, run_dir):
    source_dir = os.path.join(EVAL_DIR, args.source)
    src_rows = load_blocks(os.path.join(source_dir, "blocks.csv"))
    c_rows = load_blocks(os.path.join(run_dir, "blocks.csv"))
    if not c_rows:
        print("[C] no continuation blocks yet")
        return
    n_src = len(src_rows)
    t_src_end = n_src * args.block_dt
    ts, cs = _read_ts(os.path.join(source_dir, "dense_timeseries.csv"))
    tc, cc = _read_ts(os.path.join(run_dir, "dense_timeseries.csv"))
    t_end = float(tc[-1]) if len(tc) else t_src_end

    windows = []
    k = 0
    while t_src_end + (k + 1) * TAU <= t_end + 1e-6:
        w0, w1 = t_src_end + k * TAU, t_src_end + (k + 1) * TAU
        sel = (tc >= w0) & (tc <= w1)
        windows.append(dict(window=f"[{w0:.0f},{w1:.0f}]", tau_index=k + 1,
                            X_mean=float(np.mean(cc[sel])) if sel.sum() > 2 else float("nan")))
        k += 1
    x_last_block = c_rows[-1]["X_block"]
    x_src_last = src_rows[-1]["X_block"]
    x_src_plateau = float(np.mean([r["X_block"] for r in src_rows[16:26]])) if n_src >= 26 else float("nan")
    p_mean = float(np.mean([r["P_block_W"] for r in c_rows]))
    summary = dict(
        source=args.source, clock=args.clock, mode=args.mode, n_source_blocks=n_src,
        n_continuation_blocks=len(c_rows), t_end_s=t_end,
        X_source_last_block=x_src_last, X_source_plateau_blocks17_26=x_src_plateau,
        X_continuation_last_block=x_last_block, X_continuation_windows=windows,
        P_continuation_mean_W=p_mean, R_minus_last_block=x_last_block - p_mean / ENV["p_max_watt"],
        refs=REFS,
        verdict=("HOLDS above the static pulse" if x_last_block > REFS["pulsed_D80T10_X"] + 0.01
                 else "TIES the static pulse (within 0.01)" if x_last_block > REFS["pulsed_D80T10_X"] - 0.01
                 else "DECAYS below the static pulse"),
    )
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 76)
    print(f"C -- continuation of {args.source} ({args.clock} clock, {args.mode})")
    print("=" * 76)
    print(f"  source last block X      : {x_src_last:.4f}   (plateau blocks 17-26: {x_src_plateau:.4f})")
    for w in windows:
        print(f"  continuation tau {w['tau_index']} {w['window']:>10s} : X = {w['X_mean']:.4f}")
    print(f"  continuation last block  : X = {x_last_block:.4f}  ->  {summary['verdict']}")
    print(f"  refs: static pulse D80/T10 {REFS['pulsed_D80T10_X']:.4f} | constant 300 "
          f"{REFS['constant300_X']:.4f} | stretch peak {REFS['stretch_X_last']:.4f} | "
          f"wrap plateau {REFS['wrap_X_last']:.4f}")
    print("=" * 76 + "\n")

    # ---- figure: conversion vs time, source + continuation + refs ----
    fig, ax = plt.subplots(figsize=(11, 5.5))
    if len(ts):
        ax.plot(ts, cs, "-", color="#d1495b", lw=1.3, label=f"policy, {args.source} (0-{t_src_end:.0f} s)")
    if len(tc):
        ax.plot(tc, cc, "-", color="#8e1b2c", lw=1.3, label=f"continuation ({args.clock} clock, {args.mode})")
    ax.plot([r["t0_s"] + args.block_dt for r in src_rows + c_rows],
            [r["X_block"] for r in src_rows + c_rows], "o", color="k", ms=3, mfc="white", label="block means")
    for tag, col, lab in (("constant_wb300", "#2e6f95", "constant 300 (ref)"),
                          ("pulsed_wb300", "#666666", "pulsed D80/T10 @300 (ref)")):
        tr, cr = _read_ts(os.path.join(REF_DIR, f"{tag}_timeseries.csv"))
        if len(tr):
            ax.plot(tr, cr, "-", color=col, lw=1.0, alpha=0.8, label=lab)
            ax.hlines(REFS["constant300_X"] if "constant" in tag else REFS["pulsed_D80T10_X"],
                      tr[-1], t_end, color=col, ls=":", lw=1.0)
    ax.axvline(t_src_end, color="k", lw=1.0, ls="--", label="trained horizon (300 s)")
    for w in windows:
        ax.axvline(t_src_end + w["tau_index"] * TAU, color="k", lw=0.4, alpha=0.3)
    ax.set_xlabel("time [s]  (thin lines: residence times after the horizon)")
    ax.set_ylabel("outlet conversion")
    ax.set_title("C: does the transfer policy's end-of-run gain survive continuous operation?\n"
                 f"{args.source} continued {len(c_rows)} blocks past its 300 s horizon on the Gamma=30 reactor")
    ax.set_ylim(0.6, 0.95)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "conversion_vs_time.png"), dpi=140)
    plt.close(fig)

    rows = src_rows + c_rows
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    bl = [r["block"] for r in rows]
    axes[0].step(bl, [r["duty"] for r in rows], where="mid", color="#2e6f95")
    axes[0].set_ylabel("duty D")
    axes[1].step(bl, [r["w_hi_rpm"] for r in rows], where="mid", color="#d1495b", label="burst")
    axes[1].step(bl, [r["w_low_rpm"] for r in rows], where="mid", color="#2e6f95", label="trough")
    axes[1].set_ylabel("rpm")
    axes[1].legend(fontsize=8)
    axes[2].step(bl, [r["period_s"] for r in rows], where="mid", color="#2e6f95")
    axes[2].set_ylabel("period T [s]")
    axes[2].set_xlabel("block (10 s)")
    for ax2 in axes:
        ax2.axvline(n_src + 0.5, color="k", ls="--", lw=1)
        ax2.grid(True, alpha=0.3)
    axes[0].set_title("Per-block decoded actions: source run, then the continuation (right of the dashed line)")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "actions_per_block.png"), dpi=140)
    plt.close(fig)
    print(f"[C] outputs -> {run_dir}", flush=True)
    return summary


# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="s2_stretch",
                   help="subdir of modulation_rl/results/full_tc_eval with blocks.csv + case/")
    p.add_argument("--clock", choices=["hold", "saturate", "wrap"], default="hold")
    p.add_argument("--hold_value", type=float, default=None,
                   help="obs[3] value for --clock hold (default: the source's last clock_fed)")
    p.add_argument("--mode", choices=["policy", "frozen"], default="policy")
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--extra_duration", type=float, default=390.0, help="seconds to add (3 tau)")
    p.add_argument("--n_more", type=int, default=None, help="blocks to add (overrides --extra_duration)")
    p.add_argument("--block_dt", type=float, default=ENV["block_dt"])
    p.add_argument("--tag", default=None)
    p.add_argument("--results_dir", default=os.path.join(HERE, "results", "C"))
    p.add_argument("--keep_case", action="store_true", help="keep every continuation time folder")
    p.add_argument("--ensure_source", action="store_true",
                   help="run eval_full_tc.py --keep_case first if the source eval is missing")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true", help="one 10 s block (~40 min)")
    p.add_argument("--analyze-only", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    ENV["block_dt"] = args.block_dt
    if args.n_more is None:
        args.n_more = int(round(args.extra_duration / args.block_dt))
    if args.smoke:
        args.n_more = 1
    if args.tag is None:
        args.tag = f"{args.source}_{args.clock}_{args.mode}" + ("_smoke" if args.smoke else "")
    run_dir = os.path.join(args.results_dir, args.tag)
    os.makedirs(run_dir, exist_ok=True)
    if args.analyze_only:
        analyze(args, run_dir)
        return
    if shutil.which("pimpleFoam") is None or shutil.which("foamDictionary") is None:
        sys.exit("ERROR: OpenFOAM not on PATH (need pimpleFoam, foamDictionary).")
    if args.ensure_source:
        ensure_source(os.path.join(EVAL_DIR, args.source),
                      "wrap" if args.source.endswith("wrap") else "stretch")
    t0 = time.time()
    run_continuation(args, run_dir)
    print(f"[C] continuation done in {(time.time()-t0)/3600:.2f} h", flush=True)
    analyze(args, run_dir)


if __name__ == "__main__":
    main()
