#!/usr/bin/env python3
"""Evaluate a trained MODULATION policy (freeform OR parameterized waveform) and
ask the thesis question: does the learned omega(t) beat the best CONSTANT at equal
conversion? compare_catalysis.py can't (it builds a no-clock action_dim=1 env).

Two agent types (--agent):
  freeform  -- TaylorCouetteCatalysisEnv with clock_in_obs=True, time_step=
               --freeform_dt: absolute omega each short step, state has a phase
               clock (action_dim=1, state +phase). THE waveform agent.
  waveform  -- TaylorCouetteWaveformEnv (per_episode=False): action=(peak,duty,
               period) every --control_dt s (action_dim=3).

For either we: (1) roll out the deterministic policy in the SAME env it trained in,
recording omega(t) and the settled-tail mean conversion + motor power; (2) run a
constant-omega sweep with the IDENTICAL rollout methodology; (3) interpolate the
constant-speed power at the policy's conversion -> the equal-conversion power saving.

The obs->state mapping and energy normalizer come straight from train.py so the
policy sees exactly the state scaling it trained with. EVAL FLAGS MUST MATCH the
training slurm (warmup, --freeform_dt or --control_dt, and --eval_steps =
MAX_STEPS_PER_EP, which keeps the energy-obs normalizer identical).
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GYM_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if GYM_DIR not in sys.path:
    sys.path.insert(0, GYM_DIR)

from taylor_couette_mixing.envs.taylor_couette_catalysis import TaylorCouetteCatalysisEnv
from taylor_couette_mixing.envs.taylor_couette_waveform import TaylorCouetteWaveformEnv
from train import make_obs_to_state          # identical state scaling as training
import TD3


def build_env(args):
    """Build the env exactly as the agent trained. max_steps = eval_steps so the
    rollout is one training-length episode (keeps the energy-obs normalizer matched)."""
    common = dict(
        case_path=args.case_path, r_in=args.r_in, r_out=args.r_out,
        E_max_per_step=args.e_max_per_step, warmup_duration=args.warmup_duration,
        warmup_omega_rpm=args.warmup_omega_rpm, energy_model=args.energy_model,
        max_steps=args.eval_steps,
    )
    if args.agent == "freeform":
        return TaylorCouetteCatalysisEnv(time_step=args.freeform_dt, clock_in_obs=True, **common)
    return TaylorCouetteWaveformEnv(
        per_episode=False, control_dt=args.control_dt, episode_duration=args.episode_duration,
        period_min=args.period_min, period_max=args.period_max,
        duty_min=args.duty_min, duty_max=args.duty_max, **common)


def action_for_constant(env, rpm):
    """A constant-speed action, shaped for the env: action_dim=3 waveform ->
    (peak=rpm, duty=1, period=any); action_dim=1 freeform -> just absolute omega."""
    a0 = float(np.clip(2.0 * (rpm - env.omega_min) / (env.omega_max - env.omega_min) - 1.0, -1.0, 1.0))
    if env.action_space.shape[0] == 3:
        return np.array([a0, 1.0, 0.0])
    return np.array([a0])


def rollout(env, obs_to_state, action_fn, n_steps, settle_frac, seed):
    """Run n_steps from the warmed IC; return settled-tail mean (conv, power) and the
    per-step trajectory. Power per step = cumulative-energy diff / step duration.
    env.omega is the reported speed (absolute for freeform, mean for waveform)."""
    obs, _ = env.reset(seed=seed, options={"reset_mode": "hard"})
    e_prev = float(env.E_current)                 # 0 after reset; captured for safety
    has_wave = hasattr(env, "peak")               # parameterized waveform exposes (peak,duty,period)
    rows = []
    for _ in range(n_steps):
        obs, _, terminated, truncated, info = env.step(action_fn(obs_to_state(obs)))
        e_cum = float(info["energy_consumption"])
        row = {"omega_rpm": float(env.omega), "conv": float(info["mixing_index"]),
               "power_W": (e_cum - e_prev) / env.time_step}
        if has_wave:
            row.update(peak_rpm=float(env.peak), duty=float(env.duty), period_s=float(env.period))
        rows.append(row)
        e_prev = e_cum
        if terminated or truncated:
            break
    tail = rows[int(len(rows) * settle_frac):] or rows
    conv = float(np.mean([r["conv"] for r in tail]))
    power = float(np.mean([r["power_W"] for r in tail]))
    return conv, power, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=["freeform", "waveform"], default="freeform",
                    help="freeform (action_dim=1 + clock) or parameterized waveform (action_dim=3).")
    ap.add_argument("--case_path", required=True)
    ap.add_argument("--policy", required=True, help="checkpoint prefix, e.g. .../td3_tc_final")
    ap.add_argument("--out", required=True)
    # env flags -- MUST match run_carya_train_catalysis_td3.slurm
    ap.add_argument("--r_in", type=float, default=25.4)
    ap.add_argument("--r_out", type=float, default=31.75)
    ap.add_argument("--e_max_per_step", type=float, default=0.0011017031875434)
    ap.add_argument("--warmup_duration", type=float, default=100.0)
    ap.add_argument("--warmup_omega_rpm", type=float, default=500.0)
    ap.add_argument("--energy_model", default="motor")
    ap.add_argument("--freeform_dt", type=float, default=1.0, help="[freeform] s per control step (match training).")
    ap.add_argument("--control_dt", type=float, default=10.0, help="[waveform] s per control update.")
    ap.add_argument("--episode_duration", type=float, default=120.0)
    ap.add_argument("--period_min", type=float, default=5.0)
    ap.add_argument("--period_max", type=float, default=30.0)
    ap.add_argument("--duty_min", type=float, default=0.1)
    ap.add_argument("--duty_max", type=float, default=1.0)
    ap.add_argument("--eval_steps", type=int, default=None,
                    help="control steps per rollout. MUST equal training MAX_STEPS_PER_EP "
                         "(default: 120 freeform, 12 waveform) so the energy-obs norm matches.")
    ap.add_argument("--settle_frac", type=float, default=0.5,
                    help="average conv/power over the last (1-settle_frac) of the rollout.")
    ap.add_argument("--const_sweep", default="0,300,500,800,1200,1600,2000,2500")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.eval_steps is None:
        args.eval_steps = 60 if args.agent == "freeform" else 12
    os.makedirs(args.out, exist_ok=True)

    env = build_env(args)
    # obs->state EXACTLY as in train.py: energy_norm = energy_obs_norm * MAX_STEPS_PER_EP.
    energy_norm = getattr(env, "energy_obs_norm", env.E_max_per_step) * args.eval_steps
    obs_to_state = make_obs_to_state(env.omega_max, energy_norm,
                                     getattr(env, "_logTmin", None), getattr(env, "_logTmax", None))
    sample = obs_to_state(env._get_obs())
    policy = TD3.TD3(sample.shape[0], env.action_space.shape[0], 1.0)
    policy.load(args.policy)
    print(f"[eval] agent={args.agent}  loaded {args.policy}  "
          f"state_dim={sample.shape[0]} action_dim={env.action_space.shape[0]}")

    # ---- learned policy (deterministic rollout) ----
    pol_conv, pol_power, pol_rows = rollout(
        env, obs_to_state, lambda s: policy.select_action(s),
        args.eval_steps, args.settle_frac, args.seed)
    settled = pol_rows[int(len(pol_rows) * args.settle_frac):] or pol_rows
    if args.agent == "freeform":
        om = np.array([r["omega_rpm"] for r in settled])
        strat = {"omega_mean_rpm": float(om.mean()), "omega_std_rpm": float(om.std()),
                 "omega_min_rpm": float(om.min()), "omega_max_rpm": float(om.max())}
        strat_str = (f"mean omega={strat['omega_mean_rpm']:.0f} (std {strat['omega_std_rpm']:.0f}, "
                     f"range {strat['omega_min_rpm']:.0f}-{strat['omega_max_rpm']:.0f} rpm)")
    else:
        strat = {k: float(np.mean([r[k] for r in settled])) for k in ("peak_rpm", "duty", "period_s")}
        strat_str = f"peak={strat['peak_rpm']:.0f} D={strat['duty']:.2f} T={strat['period_s']:.1f}"
    print(f"[eval] policy: conv={pol_conv:.4f} power={pol_power:.3e}W  settled: {strat_str}")

    # ---- constant-omega sweep (same rollout methodology) ----
    const_pts = []
    for rpm in [float(x) for x in args.const_sweep.split(",") if x.strip()]:
        c, p, _ = rollout(env, obs_to_state,
                          lambda s, rpm=rpm: action_for_constant(env, rpm),
                          args.eval_steps, args.settle_frac, args.seed)
        const_pts.append((rpm, c, p))
        print(f"[eval] const {rpm:5.0f} rpm: conv={c:.4f} power={p:.3e}W")
    const_pts.sort(key=lambda x: x[1])
    cc = np.array([p[1] for p in const_pts]); cp = np.array([p[2] for p in const_pts])
    const_at = float(np.interp(pol_conv, cc, cp)) if cc.min() <= pol_conv <= cc.max() else float("nan")
    saving = const_at - pol_power
    saving_pct = 100.0 * saving / const_at if const_at == const_at and const_at > 0 else float("nan")

    summary = {
        "agent": args.agent, "policy": args.policy,
        "learned_strategy_settled": strat,
        "policy_performance": {"conv": pol_conv, "power_W": pol_power},
        "equal_conversion": {
            "constant_power_at_policy_conv_W": const_at, "waveform_power_W": pol_power,
            "power_saving_W": saving, "power_saving_pct": saving_pct,
            "beats_constant": bool(saving == saving and saving > 0),
        },
        "constant_sweep": [{"rpm": r, "conv": c, "power_W": p} for r, c, p in const_pts],
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, "policy_trajectory.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pol_rows[0].keys()))
        w.writeheader(); w.writerows(pol_rows)

    print(f"\n=== {args.agent}-TD3 vs constant (equal conversion) ===")
    print(f"learned: conv={pol_conv:.4f}  power={pol_power:.3e} W  ({strat_str})")
    if const_at == const_at:
        verdict = (f"BEATS constant: saves {saving:.3e} W ({saving_pct:.1f}%)" if saving > 0
                   else f"does NOT beat constant ({saving:.3e} W)")
        print(f"constant at the same conversion ({pol_conv:.3f}): {const_at:.3e} W  ==> {verdict}")
    else:
        print("policy conversion is outside the constant sweep range -- widen --const_sweep")
    _plot(args.out, args.agent, const_pts, pol_rows, pol_conv, pol_power)


def _plot(out, agent, const_pts, pol_rows, pol_conv, pol_power):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[eval] (skip plot: {e})")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    cp = sorted(const_pts, key=lambda x: x[1])
    ax1.plot([p[1] for p in cp], [p[2] for p in cp], "o-", color="#1f77b4", label="constant speed")
    ax1.scatter([pol_conv], [pol_power], s=160, marker="*", color="#2ca02c",
                label=f"learned {agent}", zorder=3)
    ax1.axvline(pol_conv, ls="--", c="k", lw=0.8)
    ax1.set_xlabel("conversion"); ax1.set_ylabel("motor power [W]")
    ax1.set_title("Learned modulation vs constant (lower = better)"); ax1.legend()
    t = np.arange(len(pol_rows))
    ax2.plot(t, [r["omega_rpm"] for r in pol_rows], "-o", ms=3, color="#2ca02c", label="omega(t)")
    if "peak_rpm" in pol_rows[0]:
        ax2.plot(t, [r["peak_rpm"] for r in pol_rows], "--", color="#d62728", label="burst peak")
    ax2.set_xlabel("control step"); ax2.set_ylabel("rpm")
    ax2.set_title(f"Learned {agent} omega(t)"); ax2.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out, "waveform_vs_constant.png"), dpi=130)
    print(f"[eval] wrote {os.path.join(out, 'waveform_vs_constant.png')}")


if __name__ == "__main__":
    main()
