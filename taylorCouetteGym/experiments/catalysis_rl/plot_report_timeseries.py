#!/usr/bin/env python3
"""Wang's three report panels for the CONVERGED policy: omega(t), energy/power(t),
mixing-index(=conversion)(t) -- stacked on a shared time axis.

Source of truth is the per-step rollout of the FINAL deterministic policy, i.e.
`policy_trajectory.csv` written by eval_waveform_policy.py (columns omega_rpm,
conv, power_W). That is the only place all three live physically consistently:
energy/power is NOT in the training npy (the reward is wall-flux-driven, not
conversion-driven -- see taylor_couette_catalysis.py:245 -- so it can't be backed
out of reward_per_step + conv_per_step).

Fallback (--from-npy): plot omega(t) and conversion(t) from a training run dir's
omega_per_step.npy / conv_per_step.npy (final episode). The energy panel is then
omitted, because it genuinely isn't recorded -- run eval to get it.

Usage:
  # preferred: after eval_waveform_policy.py has written policy_trajectory.csv
  python plot_report_timeseries.py --csv results/comparison_s0/freeform_eval/policy_trajectory.csv

  # quick, no CFD: omega + conversion only, from the training logs (no energy)
  python plot_report_timeseries.py --from-npy results/td3/catalysis_freeform_s0
"""
import argparse
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _final_episode_row(grid):
    """Last episode row of a [ep, step] NaN-padded grid, trimmed of trailing NaNs."""
    g = np.asarray(grid, float)
    if g.ndim == 1:
        g = g[None, :]
    row = g[-1]
    row = row[~np.isnan(row)]
    return row


def from_csv(path, dt):
    omega, conv, power = [], [], []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            omega.append(float(r["omega_rpm"]))
            conv.append(float(r["conv"]))
            power.append(float(r["power_W"]))
    t = np.arange(len(omega)) * dt
    return t, np.array(omega), np.array(conv), np.array(power)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="policy_trajectory.csv from eval_waveform_policy.py")
    src.add_argument("--from-npy", help="training run dir (omega/conv per step; NO energy)")
    ap.add_argument("--dt", type=float, default=1.0, help="seconds per control step (freeform_dt).")
    ap.add_argument("--out", default=None, help="output png (default: beside the source).")
    ap.add_argument("--title", default="Learned freeform policy (converged)")
    args = ap.parse_args()

    if args.csv:
        t, omega, conv, power = from_csv(args.csv, args.dt)
        out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.csv)),
                                       "report_timeseries.png")
        panels = 3
    else:
        d = args.from_npy
        omega = _final_episode_row(np.load(os.path.join(d, "omega_per_step.npy")))
        conv = _final_episode_row(np.load(os.path.join(d, "conv_per_step.npy")))
        n = min(len(omega), len(conv))
        omega, conv = omega[:n], conv[:n]
        t = np.arange(n) * args.dt
        power = None
        out = args.out or os.path.join(d, "report_timeseries.png")
        panels = 2

    fig, axes = plt.subplots(panels, 1, figsize=(8.5, 2.4 * panels + 0.6), sharex=True)
    ax_om = axes[0]
    ax_om.plot(t, omega, "-", lw=1.8, color="#2ca02c")
    ax_om.set_ylabel(r"$\omega$ [rpm]")
    ax_om.set_title(args.title)
    ax_om.grid(alpha=0.3)

    if power is not None:
        ax_e = axes[1]
        ax_e.plot(t, power, "-", lw=1.8, color="#d62728")
        ax_e.axhline(0.0, ls=":", lw=0.8, color="0.5")  # regen braking dips below 0
        ax_e.set_ylabel("motor power [W]")
        ax_e.grid(alpha=0.3)
        ax_c = axes[2]
    else:
        ax_c = axes[1]

    ax_c.plot(t, conv, "-", lw=1.8, color="#1f77b4")
    ax_c.set_ylabel("mixing index\n(conversion)")
    ax_c.set_xlabel("time [s]")
    ax_c.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"steps={len(t)}  omega mean={omega.mean():.0f} (std {omega.std():.0f}) rpm  "
          f"conv mean={conv.mean():.4f}"
          + (f"  power mean={power.mean():.3e} W" if power is not None else "  (no energy panel)"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
