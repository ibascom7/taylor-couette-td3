#!/usr/bin/env python3
"""
Compare a trained TD3 agent against the constant and pulsating (square-wave)
baselines on the catalysis wedge, the way Lopez-Guajardo et al. (CEJ 489 (2024)
151174) do -- i.e. AT EQUAL CONVERSION, using their electric-motor power model
(Eqs. 18-23, see taylor_couette_mixing/motor_power.py), NOT raw viscous-drag
energy. Conversion comes from the CFD env; power comes from the commanded
omega(t) through the motor model. Everything is driven through the SAME
TaylorCouetteCatalysisEnv (same warmed IC, same 1 s cadence, same 0.05 s ramps),
so the comparison is apples-to-apples.

It runs two speed sweeps (constant and pulsating at duty --duty, period --period)
plus the trained agent, and produces:

  fig_conversion_vs_speed.png  (Fig. 7): conversion vs MEAN angular speed, constant
        vs pulsating -- shows you must spin a CONSTANT shaft faster to match a
        slower PULSATING one. The TD3 agent is placed at its mean speed.
  fig_power_vs_conversion.png  (Fig. 11): avg motor power vs conversion, constant
        vs pulsating, with the EQUAL-CONVERSION power gap annotated (pulsating
        should sit below constant over a conversion band). TD3 placed at its point.
  fig_omega_ts.png             (Fig. 3 style): the omega(t) waveform each
        controller ran (with ramps) -- including the agent's LEARNED waveform.
  fig_conversion_ts.png        conversion vs time for the canonical three.
  fig_duty.png                 (Fig. 6 echo, only with --duty-sweep): conversion
        vs duty cycle at fixed mean speed -- low duty (~20%) should win.
  summary.txt                  tables + equal-conversion gaps + TD3 verdict.
  frames_{constant,squarewave,td3}/  the canonical three for ParaView.

The baseline sweeps are policy-independent and CACHED to baseline_sweep.npz, so
re-running after retraining the agent only re-runs the agent (use
--refresh-baselines to force a fresh sweep).

Usage (see run_carya_compare_catalysis.slurm for the Carya wrapper):
  python compare_catalysis.py --case <catalysis_case> \
      --policy <...>/td3_tc_final --out <...>/comparison \
      --const-rpm 300,500,700,900,1200 --puls-rpm 300,400,500 \
      --eval-seconds 60 --duty 0.2 --period 30
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make the gym root importable (this file lives in experiments/catalysis_rl/).
GYM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if GYM_ROOT not in sys.path:
    sys.path.insert(0, GYM_ROOT)

import TD3
from taylor_couette_mixing.envs.taylor_couette_catalysis import (
    TaylorCouetteCatalysisEnv,
)
from taylor_couette_mixing import motor_power

RPM = 2.0 * np.pi / 60.0
C_CONST, C_PULS, C_TD3 = "#1f77b4", "#d62728", "#2ca02c"


# --------------------------------------------------------------------------- #
# state adapter (identical to train.py; conversion sits in the mixing slot)
# --------------------------------------------------------------------------- #
def make_obs_to_state(omega_max, energy_norm):
    def obs_to_state(obs):
        return np.array(
            [
                float(obs["omega"]) / omega_max,
                2.0 * float(obs["mixing_index"]) - 1.0,    # = 2*conversion - 1
                float(obs["energy_consumption"]) / energy_norm,
            ],
            dtype=np.float32,
        )
    return obs_to_state


def omega_to_action(omega_rpm, omega_min, omega_max):
    """Invert the env's action->omega map so a prescribed omega is realized."""
    a = 2.0 * (omega_rpm - omega_min) / (omega_max - omega_min) - 1.0
    return float(np.clip(a, -1.0, 1.0))


def squarewave_omega(sec, mean_rpm, duty, period):
    """Per-second square wave: active (mean/duty) for the first duty*period of each
    period, idle (0) otherwise; mean == mean_rpm (matches make_case.py)."""
    return (mean_rpm / duty) if (sec % period) < (duty * period) else 0.0


def active_spans(t_max, period, duty):
    return [(k * period, k * period + duty * period)
            for k in range(int(t_max // period) + 1)]


# --------------------------------------------------------------------------- #
# rollout + power
# --------------------------------------------------------------------------- #
def rollout(env, obs_to_state, kind, eval_seconds, mean_rpm=0.0,
            duty=0.2, period=30.0, policy=None, seed=0, snapshot_dir=None):
    """Drive the env eval_seconds steps under one controller (kind in
    {constant, squarewave, td3}); return per-second t, omega(rpm), conv arrays.
    All controllers start from the same warmed IC (hard reset)."""
    obs, info = env.reset(seed=seed, options={"reset_mode": "hard"})
    state = obs_to_state(obs)
    t, omega, conv = [], [], []
    for sec in range(eval_seconds):
        if kind == "td3":
            action = policy.select_action(state)
        else:
            target = (mean_rpm if kind == "constant"
                      else squarewave_omega(sec, mean_rpm, duty, period))
            action = np.array([omega_to_action(target, env.omega_min, env.omega_max)])
        obs, reward, terminated, truncated, info = env.step(action)
        state = obs_to_state(obs)
        t.append(sec + 1)
        omega.append(float(obs["omega"]))
        conv.append(float(info["mixing_index"]))     # conversion slot
    if snapshot_dir is not None:
        env.helpers.snapshot_frames(snapshot_dir)
    return dict(t=np.array(t, float), omega=np.array(omega, float),
                conv=np.array(conv, float))


def reconstruct_omega(omega_steps, omega_start, ramp_time, nsub=200):
    """Reconstruct the fine-time omega(t) [rpm] the env actually commanded: each
    1 s step ramps from the previous value to omega_steps[k] over ramp_time s,
    then holds. Needed so the motor model sees the real ramp dynamics (inertia)."""
    t, w = [], []
    prev = omega_start
    for k, target in enumerate(omega_steps):
        for j in range(nsub):
            tau = j / nsub
            w.append(prev + (tau / ramp_time) * (target - prev)
                     if tau < ramp_time else target)
            t.append(k + tau)
        prev = target
    t.append(float(len(omega_steps)))
    w.append(prev)
    return np.array(t), np.array(w)


def summarize(res, w0, w1, omega_start, ramp_time):
    """Window-averaged conversion + avg motor power + avg drag-only power +
    mean omega for one rollout."""
    t, om, cv = res["t"], res["omega"], res["conv"]
    m = (t >= w0) & (t <= w1)
    conv = float(cv[m].mean()) if m.any() else float(cv.mean())
    omega_mean = float(om[m].mean()) if m.any() else float(om.mean())

    tf, wf = reconstruct_omega(om, omega_start, ramp_time)   # rpm
    mf = (tf >= w0) & (tf <= w1)
    tf_w, wf_w = tf[mf], wf[mf] * RPM                        # -> rad/s in window
    motor_W = motor_power.average_power(tf_w, wf_w)
    drag_W = motor_power.average_drag_power(tf_w, wf_w)
    return dict(conv=conv, motor_W=motor_W, drag_W=drag_W, omega_mean=omega_mean)


def equal_conversion_gap(const_conv, const_power, q_conv, q_power):
    """For each pulsating point, interpolate the constant power at the SAME
    conversion and return savings% = (P_const - P_puls)/P_const*100 (positive =
    pulsating cheaper). Points outside the constant conversion range are flagged."""
    order = np.argsort(const_conv)
    cc, cp = np.asarray(const_conv)[order], np.asarray(const_power)[order]
    out = []
    for qc, qp in zip(q_conv, q_power):
        in_range = cc.min() <= qc <= cc.max()
        p_const = float(np.interp(qc, cc, cp))
        sav = (p_const - qp) / p_const * 100.0 if p_const else float("nan")
        out.append(dict(conv=float(qc), p_puls=float(qp), p_const=p_const,
                        savings=sav, in_range=bool(in_range)))
    return out


# --------------------------------------------------------------------------- #
# baseline-sweep cache (policy-independent -> reuse across agent retrains)
# --------------------------------------------------------------------------- #
def cache_valid(npz, params):
    for k, v in params.items():
        if k not in npz:
            return False
        if not np.array_equal(np.asarray(npz[k]), np.asarray(v)):
            return False
    return True


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--policy", required=True, help="TD3 checkpoint prefix, e.g. .../td3_tc_final")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-seconds", type=int, default=60)
    ap.add_argument("--const-rpm", default="300,500,700,900,1200",
                    help="constant mean speeds to sweep (rpm)")
    ap.add_argument("--puls-rpm", default="300,400,500",
                    help="pulsating mean speeds to sweep (rpm); peak = mean/duty, "
                         "keep peak <= ~2500 (the validated range)")
    ap.add_argument("--duty", type=float, default=0.2)
    ap.add_argument("--period", type=float, default=30.0)
    ap.add_argument("--duty-sweep", default="",
                    help="optional duty cycles to sweep at mean 500 rpm for a Fig.6 "
                         "echo (e.g. '0.2,0.4,0.6,0.8'); empty = skip")
    ap.add_argument("--refresh-baselines", action="store_true",
                    help="recompute the baseline sweeps even if a valid cache exists")
    # env params -- MUST match training.
    ap.add_argument("--r_in", type=float, default=25.4)
    ap.add_argument("--r_out", type=float, default=31.75)
    ap.add_argument("--e_max_per_step", type=float, default=0.0011017031875434)
    ap.add_argument("--warmup_duration", type=float, default=20.0)
    ap.add_argument("--warmup_omega_rpm", type=float, default=500.0)
    ap.add_argument("--omega_max", type=float, default=2500.0)
    ap.add_argument("--ramp_time", type=float, default=0.05)
    ap.add_argument("--window", type=float, nargs=2, default=None,
                    help="TMIN TMAX to average over (default: last full --period)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    const_means = sorted({float(x) for x in args.const_rpm.split(",") if x.strip()} | {500.0})
    puls_means = sorted({float(x) for x in args.puls_rpm.split(",") if x.strip()} | {500.0})
    duty_vals = sorted({float(x) for x in args.duty_sweep.split(",") if x.strip()})

    env = TaylorCouetteCatalysisEnv(
        case_path=args.case, omega_max=args.omega_max, max_steps=args.eval_seconds,
        r_in=args.r_in, r_out=args.r_out, E_max_per_step=args.e_max_per_step,
        warmup_duration=args.warmup_duration, warmup_omega_rpm=args.warmup_omega_rpm,
        ramp_time=args.ramp_time,
    )
    obs_to_state = make_obs_to_state(env.omega_max, env.E_max_per_step * args.eval_seconds)
    omega_start = args.warmup_omega_rpm

    w0, w1 = (args.window if args.window else
              (max(0.0, args.eval_seconds - args.period), float(args.eval_seconds)))

    # ---- baseline sweeps (cached) --------------------------------------
    cache = os.path.join(args.out, "baseline_sweep.npz")
    params = dict(
        const_means=const_means, puls_means=puls_means, duty_vals=duty_vals,
        eval_seconds=args.eval_seconds, duty=args.duty, period=args.period,
        ramp_time=args.ramp_time, warmup_duration=args.warmup_duration,
        warmup_omega_rpm=args.warmup_omega_rpm, omega_max=args.omega_max,
        w0=w0, w1=w1, seed=args.seed,
    )
    sweep = None
    if os.path.exists(cache) and not args.refresh_baselines:
        npz = np.load(cache, allow_pickle=True)
        if cache_valid(npz, params):
            sweep = {k: npz[k] for k in npz.files}
            print(f"[compare] using cached baseline sweep {cache}")
        else:
            print(f"[compare] cache params changed -> recomputing baselines")

    if sweep is None:
        def sweep_constant(m):
            print(f"=== constant {m:.0f} rpm ===")
            r = rollout(env, obs_to_state, "constant", args.eval_seconds, mean_rpm=m, seed=args.seed)
            return r, summarize(r, w0, w1, omega_start, args.ramp_time)
        def sweep_puls(m, duty):
            print(f"=== pulsating mean {m:.0f} rpm (peak {m/duty:.0f}, D={duty}) ===")
            r = rollout(env, obs_to_state, "squarewave", args.eval_seconds, mean_rpm=m,
                        duty=duty, period=args.period, seed=args.seed)
            return r, summarize(r, w0, w1, omega_start, args.ramp_time)

        const = {m: sweep_constant(m) for m in const_means}
        puls = {m: sweep_puls(m, args.duty) for m in puls_means}
        duty = {d: sweep_puls(500.0, d) for d in duty_vals}

        # NB: const_means/puls_means/duty_vals come from **params (below); don't
        # also pass them explicitly here or dict() raises "multiple values".
        sweep = dict(
            const_conv=np.array([const[m][1]["conv"] for m in const_means]),
            const_motor=np.array([const[m][1]["motor_W"] for m in const_means]),
            const_drag=np.array([const[m][1]["drag_W"] for m in const_means]),
            puls_conv=np.array([puls[m][1]["conv"] for m in puls_means]),
            puls_motor=np.array([puls[m][1]["motor_W"] for m in puls_means]),
            puls_drag=np.array([puls[m][1]["drag_W"] for m in puls_means]),
            duty_conv=np.array([duty[d][1]["conv"] for d in duty_vals]),
            duty_motor=np.array([duty[d][1]["motor_W"] for d in duty_vals]),
            # canonical mean-500 traces for the time-series + omega(t) plots
            cc_t=const[500.0][0]["t"], cc_omega=const[500.0][0]["omega"], cc_conv=const[500.0][0]["conv"],
            cp_t=puls[500.0][0]["t"], cp_omega=puls[500.0][0]["omega"], cp_conv=puls[500.0][0]["conv"],
            **{k: np.asarray(v) for k, v in params.items()},
        )
        np.savez(cache, **sweep)

    # ---- canonical baseline frames for ParaView ------------------------
    # Same style as the prescribed oscillation_vs_constant runs: per-second time
    # dirs + a .foam file (snapshot_frames). Each is a full eval_seconds CFD
    # rollout, so only (re)generate when the frames dir is missing -- this keeps
    # the baseline_sweep.npz cache's compute savings on re-runs (e.g. after
    # retraining) while still GUARANTEEING the frames exist after any run.
    for kind, fdir in [("constant", "frames_constant"),
                       ("squarewave", "frames_squarewave")]:
        dpath = os.path.join(args.out, fdir)
        if os.path.isdir(dpath):
            print(f"[compare] {fdir}/ exists -> keeping (skip CFD rollout)")
            continue
        print(f"[compare] writing ParaView frames -> {fdir}/")
        rollout(env, obs_to_state, kind, args.eval_seconds, mean_rpm=500.0,
                duty=args.duty, period=args.period, seed=args.seed,
                snapshot_dir=dpath)

    # ---- TD3 agent (always fresh) --------------------------------------
    policy = TD3.TD3(3, env.action_space.shape[0], 1.0)
    policy.load(args.policy)
    print(f"[compare] loaded policy {args.policy}")
    print("=== td3 agent ===")
    td3 = rollout(env, obs_to_state, "td3", args.eval_seconds, policy=policy, seed=args.seed,
                  snapshot_dir=os.path.join(args.out, "frames_td3"))
    td3s = summarize(td3, w0, w1, omega_start, args.ramp_time)

    # ---- report --------------------------------------------------------
    L = [f"CATALYSIS comparison (paper-style: equal conversion, motor power model)",
         f"window t in [{w0:.1f}, {w1:.1f}] s   duty={args.duty}  period={args.period}s\n",
         "constant sweep:",
         "  rpm     conv    motor_P[W]  drag_P[W]"]
    for m, c, mo, dr in zip(sweep["const_means"], sweep["const_conv"],
                            sweep["const_motor"], sweep["const_drag"]):
        L.append(f"  {m:6.0f}  {c:6.3f}  {mo:9.2f}  {dr:9.4f}")
    L += ["", "pulsating sweep (D=%.2f, T=%.0fs):" % (args.duty, args.period),
          "  mean    conv    motor_P[W]  drag_P[W]"]
    for m, c, mo, dr in zip(sweep["puls_means"], sweep["puls_conv"],
                            sweep["puls_motor"], sweep["puls_drag"]):
        L.append(f"  {m:6.0f}  {c:6.3f}  {mo:9.2f}  {dr:9.4f}")

    L += ["", "EQUAL-CONVERSION power gap (pulsating vs constant at the SAME conversion):"]
    for g in equal_conversion_gap(sweep["const_conv"], sweep["const_motor"],
                                  sweep["puls_conv"], sweep["puls_motor"]):
        flag = "" if g["in_range"] else "  (extrapolated)"
        L.append(f"  conv={g['conv']:.3f}: pulsating {g['p_puls']:6.2f} W vs "
                 f"constant {g['p_const']:6.2f} W  ->  {g['savings']:+5.1f}% power{flag}")

    # TD3 verdict
    cc_order = np.argsort(sweep["const_conv"])
    p_const_at_td3 = float(np.interp(td3s["conv"], np.asarray(sweep["const_conv"])[cc_order],
                                     np.asarray(sweep["const_motor"])[cc_order]))
    L += ["", "TD3 agent:",
          f"  conv={td3s['conv']:.3f}  mean_omega={td3s['omega_mean']:.0f} rpm  "
          f"motor_P={td3s['motor_W']:.2f} W  drag_P={td3s['drag_W']:.4f} W",
          f"  vs constant at equal conversion ({p_const_at_td3:.2f} W): "
          f"{(p_const_at_td3 - td3s['motor_W'])/p_const_at_td3*100:+.1f}% power"]
    report = "\n".join(L)
    print("\n" + report)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(report + "\n")

    # ---- Fig. 7: conversion vs mean speed ------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    oc = np.argsort(sweep["const_means"]); op = np.argsort(sweep["puls_means"])
    ax.plot(np.asarray(sweep["const_means"])[oc], np.asarray(sweep["const_conv"])[oc],
            "o-", color=C_CONST, label="constant")
    ax.plot(np.asarray(sweep["puls_means"])[op], np.asarray(sweep["puls_conv"])[op],
            "^:", color=C_PULS, label=f"pulsating (D={args.duty}, T={args.period:.0f}s)")
    ax.scatter([td3s["omega_mean"]], [td3s["conv"]], marker="*", s=260, color=C_TD3,
               zorder=5, label="TD3 (at mean omega)")
    ax.set_xlabel("mean angular speed [rpm]"); ax.set_ylabel("conversion (1 - cup outlet C)")
    ax.set_title("Conversion vs mean speed (Fig. 7 style)\n"
                 "pulsating reaches a given conversion at LOWER mean speed")
    ax.grid(alpha=0.3); ax.legend()
    fig.savefig(os.path.join(args.out, "fig_conversion_vs_speed.png"), dpi=140,
                bbox_inches="tight"); plt.close(fig)

    # ---- Fig. 11: power vs conversion ----------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(np.asarray(sweep["const_conv"])[cc_order], np.asarray(sweep["const_motor"])[cc_order],
            "o-", color=C_CONST, label="constant")
    pp = np.argsort(sweep["puls_conv"])
    ax.plot(np.asarray(sweep["puls_conv"])[pp], np.asarray(sweep["puls_motor"])[pp],
            "^:", color=C_PULS, label="pulsating")
    ax.scatter([td3s["conv"]], [td3s["motor_W"]], marker="*", s=260, color=C_TD3,
               zorder=5, label="TD3")
    ax.set_xlabel("conversion (1 - cup outlet C)"); ax.set_ylabel("avg motor power [W]")
    ax.set_title("Power vs conversion (Fig. 11 style)\n"
                 "lower at equal conversion = cheaper; pulsating should sit below")
    ax.grid(alpha=0.3); ax.legend()
    fig.savefig(os.path.join(args.out, "fig_power_vs_conversion.png"), dpi=140,
                bbox_inches="tight"); plt.close(fig)

    # ---- Fig. 3 style: omega(t) waveforms (with ramps) -----------------
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for label, om, color in [("constant 500", sweep["cc_omega"], C_CONST),
                             ("square mean 500", sweep["cp_omega"], C_PULS),
                             ("TD3", td3["omega"], C_TD3)]:
        tf, wf = reconstruct_omega(np.asarray(om), omega_start, args.ramp_time)
        ax.plot(tf, wf, color=color, lw=1.4, label=label)
    ax.set_xlabel("time [s]"); ax.set_ylabel("inner-cylinder omega [rpm]")
    ax.set_title("Control signal omega(t): the agent's LEARNED waveform vs baselines")
    ax.grid(alpha=0.3); ax.legend()
    fig.savefig(os.path.join(args.out, "fig_omega_ts.png"), dpi=140,
                bbox_inches="tight"); plt.close(fig)

    # ---- conversion vs time --------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, (a, b) in enumerate(active_spans(args.eval_seconds, args.period, args.duty)):
        ax.axvspan(a, b, color="0.88", label="square-wave active" if i == 0 else None)
    ax.plot(sweep["cc_t"], sweep["cc_conv"], color=C_CONST, lw=1.4, label="constant 500")
    ax.plot(sweep["cp_t"], sweep["cp_conv"], color=C_PULS, lw=1.4, label="square mean 500")
    ax.plot(td3["t"], td3["conv"], color=C_TD3, lw=1.4, label="TD3")
    ax.axvspan(w0, w1, color="gold", alpha=0.15, label="averaging window")
    ax.set_xlabel("time [s]"); ax.set_ylabel("conversion (1 - cup outlet C)")
    ax.set_title("Conversion vs time"); ax.grid(alpha=0.3); ax.legend()
    fig.savefig(os.path.join(args.out, "fig_conversion_ts.png"), dpi=140,
                bbox_inches="tight"); plt.close(fig)

    # ---- Fig. 6 echo: conversion vs duty -------------------------------
    if len(sweep["duty_vals"]):
        fig, ax = plt.subplots(figsize=(7, 5))
        od = np.argsort(sweep["duty_vals"])
        ax.plot(np.asarray(sweep["duty_vals"])[od] * 100, np.asarray(sweep["duty_conv"])[od],
                "s-", color=C_PULS)
        ax.set_xlabel("duty cycle [%]"); ax.set_ylabel("conversion (1 - cup outlet C)")
        ax.set_title("Conversion vs duty cycle at mean 500 rpm (Fig. 6 echo)")
        ax.grid(alpha=0.3)
        fig.savefig(os.path.join(args.out, "fig_duty.png"), dpi=140,
                    bbox_inches="tight"); plt.close(fig)

    print(f"\nwrote figures + summary.txt + frames_*/ to {args.out}")


if __name__ == "__main__":
    main()
