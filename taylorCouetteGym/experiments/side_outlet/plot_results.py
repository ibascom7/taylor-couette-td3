"""Plots for Yuhe's side-outlet flow-through reactor case (no RL).

Produces, for the meeting deliverable:
  1. omega vs time          -- the Lopez modulated waveform driving the inner wall
  2. energy vs time         -- Lopez motor-model power P_e(t) and cumulative energy E(t)
  3. outlet conversion / cOut vs time  -- reactor performance + dye-breakthrough timing
  4. a combined 3-panel figure

The omega waveform is reproduced EXACTLY from 0/U (codedFixedValue
`lopezModulatedInnerWallVelocity`): omega_b = 500 rpm, duty D = 0.2, period
T = 20 s, smoothed transition = 0.05 s. High-phase omega = omega_b/D = 2500 rpm.

Energy uses the paper's full motor model (motor_power.py, Lopez et al. 2024
Eqs 18-23): inertia + bearing dry friction + drag -> current -> winding ->
electrical power with regenerative braking. This is the "energy index" the
slides reference; it is a pure function of the commanded omega(t).

Usage:  python plot_results.py [CASE_DIR] [OUT_DIR]
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reach the repo's motor_power.py (paper Eqs 18-23) WITHOUT triggering the
# package __init__ (which imports gymnasium). Load the standalone module by path.
import importlib.util
THIS = Path(__file__).resolve()
REPO = THIS.parents[2]                       # .../taylorCouetteGym
_mp_path = REPO / "taylor_couette_mixing" / "motor_power.py"
_spec = importlib.util.spec_from_file_location("motor_power", _mp_path)
mp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mp)

# ---- Yuhe waveform parameters (must match 0/U) ------------------------------
OMEGA_B_RPM = 500.0
DUTY        = 0.2
PERIOD      = 20.0
TRANSITION  = 0.05
C0          = 50.0          # inlet concentration [mmol/m3]


def omega_rad(t):
    """Lopez modulated inner-wall angular speed [rad/s], vectorized over t.
    Byte-for-byte the logic in 0/U's codedFixedValue."""
    t = np.asarray(t, float)
    if DUTY >= 0.999999:
        return np.full_like(t, OMEGA_B_RPM * 2 * np.pi / 60.0)
    high = (OMEGA_B_RPM / DUTY) * 2 * np.pi / 60.0
    Tplus = DUTY * PERIOD
    phase = np.mod(np.maximum(t, 0.0), PERIOD)
    gate = np.zeros_like(phase)
    # ramp up
    up = phase < TRANSITION
    s = np.clip(phase / TRANSITION, 0, 1)
    gate = np.where(up, s * s * (3 - 2 * s), gate)
    # ramp down
    dn = (phase > Tplus - TRANSITION) & (phase < Tplus)
    s2 = np.clip((Tplus - phase) / TRANSITION, 0, 1)
    gate = np.where(dn, s2 * s2 * (3 - 2 * s2), gate)
    # plateau
    plateau = (phase >= TRANSITION) & (phase <= Tplus - TRANSITION)
    gate = np.where(plateau, 1.0, gate)
    return high * gate


def load_dat(path):
    """Load a whitespace .dat written by the coded functionObjects (skip # header,
    keep only numeric columns)."""
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        num = []
        for p in parts:
            try:
                num.append(float(p))
            except ValueError:
                pass
        if num:
            rows.append(num)
    return np.array(rows)


def main():
    case = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else THIS.parent
    out.mkdir(parents=True, exist_ok=True)

    conv_f = case / "postProcessing/sideOutletConversion/0/outletConversion.dat"
    pow_f = case / "postProcessing/rotationalPower_innerWall/0/rotationalPower.dat"

    # endTime from the data (or default 200)
    tmax = 200.0
    if conv_f.exists():
        d = load_dat(conv_f)
        if len(d):
            tmax = float(d[:, 0].max())

    # ---- fine analytic waveform + motor model -------------------------------
    t = np.linspace(0, tmax, int(tmax * 50) + 1)      # 50 Hz: resolves 0.05 s ramps
    om = omega_rad(t)
    rpm = om * 60 / (2 * np.pi)
    P_e = mp.electrical_power(t, om)                  # W (motor model, Eqs 18-23)
    E = np.concatenate([[0.0], np.cumsum(0.5 * (P_e[1:] + P_e[:-1]) * np.diff(t))])  # J

    # ---- figure 1: omega vs time -------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(t, rpm, lw=1.2, color="C0")
    ax.set_xlabel("time  [s]"); ax.set_ylabel("inner-wall speed  [rpm]")
    ax.set_title(f"Inner-wall angular speed  (Lopez modulated: $\\omega_b$={OMEGA_B_RPM:.0f} rpm, "
                 f"D={DUTY}, T={PERIOD:.0f} s)")
    ax.grid(alpha=0.3)
    ax.axhline(OMEGA_B_RPM, ls="--", color="grey", lw=0.8)
    ax.text(tmax * 0.5, OMEGA_B_RPM + 60, f"time-avg = $\\omega_b$ = {OMEGA_B_RPM:.0f} rpm",
            color="grey", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "omega_vs_time.png", dpi=140); plt.close(fig)

    # ---- figure 2: energy vs time ------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(t, P_e, lw=1.0, color="C3", label="electrical power $P_e(t)$  (motor model)")
    ax.set_xlabel("time  [s]"); ax.set_ylabel("$P_e$  [W]", color="C3")
    ax.tick_params(axis="y", labelcolor="C3")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(t, E, lw=1.6, color="C2", label="cumulative energy $E(t)$")
    ax2.set_ylabel("cumulative energy $E$  [J]", color="C2")
    ax2.tick_params(axis="y", labelcolor="C2")
    ax.set_title(f"Energy index (Lopez motor model, Eqs 18-23):  total $E$ = {E[-1]:.2f} J, "
                 f"avg $P_e$ = {E[-1]/tmax:.3f} W")
    fig.tight_layout(); fig.savefig(out / "energy_vs_time.png", dpi=140); plt.close(fig)

    # ---- figure 3: conversion / outlet concentration -----------------------
    bt = ss = None
    if conv_f.exists() and len(load_dat(conv_f)):
        d = load_dat(conv_f)
        tc, cout, conv = d[:, 0], d[:, 1], d[:, 2]

        # per-cycle (T=20s) mean conversion -> periodic steady-state detection
        ncyc = int(np.ceil(tc.max() / PERIOD))
        cyc_t, cyc_mean = [], []
        for k in range(ncyc):
            m = (tc > k * PERIOD) & (tc <= (k + 1) * PERIOD)
            if m.any():
                cyc_t.append((k + 0.5) * PERIOD)
                cyc_mean.append(conv[m].mean())
        cyc_t, cyc_mean = np.array(cyc_t), np.array(cyc_mean)
        # periodic steady state = first cycle from which every later cycle-mean
        # stays inside plateau ± tol (plateau = avg of the last few cycles). This
        # ignores the residual cycle-to-cycle oscillation and catches the flattening.
        plateau = cyc_mean[-min(4, len(cyc_mean) // 2 or 1):].mean()
        tol = 0.025
        for i in range(len(cyc_mean)):
            if np.all(np.abs(cyc_mean[i:] - plateau) < tol):
                ss = cyc_t[i]
                break

        fig, ax = plt.subplots(figsize=(9, 3.6))
        ax.plot(tc, cout, "o-", ms=3, color="C0", label="outlet conc. $c_{out}$")
        ax.set_xlabel("time  [s]"); ax.set_ylabel("$c_{out}$  [mmol/m³]", color="C0")
        ax.tick_params(axis="y", labelcolor="C0"); ax.grid(alpha=0.3)
        ax.axhline(C0, ls=":", color="grey", lw=0.8)
        ax.text(tmax * 0.02, C0 * 0.93, f"inlet $c_0$={C0:.0f}", color="grey", fontsize=8)
        ax2 = ax.twinx()
        ax2.plot(tc, conv, "s-", ms=2.5, color="C1", alpha=0.5, label="conversion $X$ (per write)")
        ax2.plot(cyc_t, cyc_mean, "D-", ms=6, color="C3", label="cycle-mean $X$")
        ax2.set_ylabel("conversion $X = 1-c_{out}/c_0$", color="C1")
        ax2.tick_params(axis="y", labelcolor="C1"); ax2.set_ylim(0, 1)

        # breakthrough = first write where cout exceeds 1% of c0
        idx = np.where(cout > 0.01 * C0)[0]
        if len(idx):
            bt = tc[idx[0]]
            ax.axvline(bt, color="green", ls="--", lw=1)
            ax.text(bt + 2, C0 * 0.55, f"dye at outlet\nby t≈{bt:.0f} s\n(1st burst)",
                    color="green", fontsize=8)
        if ss is not None:
            ax.axvline(ss, color="purple", ls="--", lw=1)
            ax.text(ss + 2, C0 * 0.18, f"periodic steady\nstate t≈{ss:.0f} s\n"
                    f"(X̄≈{cyc_mean[cyc_t >= ss].mean():.2f})", color="purple", fontsize=8)
        ax.set_title("Dye breakthrough & outlet conversion at side_outlet (bottom of outer wall)")
        ax2.legend(loc="center right", fontsize=8, framealpha=0.9)
        fig.tight_layout(); fig.savefig(out / "conversion_vs_time.png", dpi=140); plt.close(fig)

    print(f"omega: peak {rpm.max():.0f} rpm, mean {rpm.mean():.1f} rpm")
    print(f"energy: total {E[-1]:.3f} J over {tmax:.0f} s, avg {E[-1]/tmax:.4f} W")
    if bt is not None:
        print(f"breakthrough (cout>1% c0): t ~ {bt:.1f} s")
    if ss is not None:
        print(f"approx steady state (conc stops changing): t ~ {ss:.1f} s")
    print(f"plots written to {out}")


if __name__ == "__main__":
    main()
