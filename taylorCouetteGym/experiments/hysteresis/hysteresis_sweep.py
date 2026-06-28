#!/usr/bin/env python3
"""
Hysteresis / bistability sweep on the catalysis wedge: look for a LOOP in the
conversion-vs-power (and conversion-vs-Re) plane by ramping the inner-cylinder
speed UP through a ladder of Reynolds numbers and then back DOWN, settling at
every level. Where the up- and down-branches differ, the system is path-dependent
-- and a non-convex / hysteretic conversion-vs-power curve is exactly what lets a
TIME-MODULATED speed beat a CONSTANT one at the same mean power.

What this measures, and what it proves
--------------------------------------
At each level we hold omega for --settle-seconds and record the SETTLED values of
  * conv     -- outlet conversion (1 - cup C). Lags omega by ~the residence time
                tau_res = H/u_ax (~26 s here), so it needs a long settle to be
                truly steady.
  * wf_norm  -- catalytic wall consumption / feed rate, a conversion-EQUIVALENT
                that tracks the FLOW STATE on the boundary-layer timescale
                (seconds), so it reveals the loop with a much shorter settle.
  * motor_P  -- the paper's electric-motor power (motor_power.py). A pure function
                of omega -> SINGLE-VALUED in Re. So in the conv-vs-motor_P plane a
                loop appears as two conversions at the SAME power (vertical gap).
  * mech_P   -- mechanical (viscous-drag) power rho*|Mz|*omega from the CFD torque.
                Depends on the flow state, so THIS can be two-valued in Re -- the
                clearest fingerprint of genuine flow-state bistability.

Up vs down differing is necessary but not sufficient for *equilibrium* bistability:
a finite ramp rate always makes a DYNAMIC loop (conversion lagging omega) that
shrinks as the settle time grows. To tell them apart, run this at several settle
times (the slurm does an array over --settle-seconds) and watch the loop AREA:
  * area -> a nonzero limit as settle grows  => TRUE (equilibrium) hysteresis.
  * area -> 0 as settle grows                => DYNAMIC hysteresis (transient).
Either way modulation can beat constant; the convex-hull analysis below quantifies
the achievable gain from this run's quasi-static curve.

Optional --probe-rpm runs the rigorous fixed-Re test: settle the SAME target speed
from a COLD state and from a HOT (pre-spun) state; if they land on different
conversions, that speed is genuinely bistable (two attractors at one Re).

CAVEAT (geometry): the wedge is 2D-axisymmetric (1 azimuthal cell). The first
Couette->Taylor transition at eta=0.8 is SUPERCRITICAL (no equilibrium hysteresis),
so what you most likely see here is dynamic hysteresis + convexity. Equilibrium
multistability that needs azimuthal (wavy-vortex) structure requires the full 3D
annulus -- run this on full_tc_mixing_case (--azimuthal-fraction 1.0, that case's
--r-in/--r-out) for those regimes.

Also doubles as the flow-regime VISUALIZATION run: every level logs Re_rot, the
Taylor number Ta and the axial Re, writes per-second ParaView frames, and emits the
Couette/Taylor regime-band figure + the Re staircase. For a cheap regime montage
without the down branch, run with --direction up (this subsumes the old standalone
reynolds_sweep experiment).

Usage (see run_carya_hysteresis.slurm):
  python hysteresis_sweep.py --case <configured_case> --out results/h40 \
      --rpm 40,60,80,100,120,160,220,300,500,800,1200,1800,2500 \
      --settle-seconds 40
  # cheap regime-visualization only (Couette->Taylor montage, no loop):
  python hysteresis_sweep.py --case <configured_case> --out results/regimes \
      --direction up --settle-seconds 20
"""
import argparse
import csv
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GYM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if GYM_ROOT not in sys.path:
    sys.path.insert(0, GYM_ROOT)

from taylor_couette_mixing.envs.helpers import Helpers
from taylor_couette_mixing import motor_power

RPM = 2.0 * math.pi / 60.0
RHO = 930.0
DEFAULT_RPM = "40,60,80,100,120,160,220,300,500,800,1200,1800,2500"


def reynolds(omega_rad, r_in_m, d_m, nu):
    return omega_rad * r_in_m * d_m / nu


def taylor_number(re_rot, d_m, r_in_m):
    """Taylor number Ta = Re^2 * (d / r_in) (gap-based, narrow-gap form)."""
    return re_rot ** 2 * (d_m / r_in_m)


RE_C = 120.0   # approx onset of axisymmetric Taylor vortices at eta=0.8 (for the eye)


def steady_motor_power(omega_rad):
    """Time-average motor power at CONSTANT omega (inertia/di-dt terms vanish)."""
    t = np.linspace(0.0, 1.0, 50)
    w = np.full_like(t, omega_rad)
    return motor_power.average_power(t, w)


def settled_stats(metrics, frac=0.5):
    """Mean over the settled tail (last `frac` of the level's METRICS samples)."""
    if not metrics:
        return {}
    tail = metrics[max(0, int(len(metrics) * (1.0 - frac))):]
    keys = set().union(*(m.keys() for m in tail))
    return {k: float(np.mean([m[k] for m in tail if k in m]))
            for k in keys if any(k in m for m in tail)}


def upper_convex_hull(points):
    """Upper convex hull of (x, y, *payload) points, returned left->right.
    These are the (mean power, conversion) operating points a time-share between
    two speeds can reach; where the hull is above the pointwise curve, modulation
    wins."""
    pts = sorted(set(points))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    hull = []
    for p in pts:
        while len(hull) >= 2 and cross(hull[-2], hull[-1], p) >= 0:
            hull.pop()
        hull.append(p)
    return hull


# --------------------------------------------------------------------------- #
# named-state snapshots for the fixed-Re two-IC bistability probe
# --------------------------------------------------------------------------- #
def freeze_state(case, name):
    """Copy the latest numeric time dir into case/<name> (non-numeric -> ignored
    by _get_latest_time) so it can be restored as an initial condition later."""
    case = Path(case)
    latest = max((p for p in case.iterdir() if p.is_dir() and _is_num(p.name)),
                 key=lambda p: float(p.name))
    dest = case / name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(latest, dest)


def restore_state(case, name):
    """Wipe numeric time dirs and seed 0/ from case/<name>, so a fresh run
    continues from that frozen state (controlDict startFrom latestTime)."""
    case = Path(case)
    for p in list(case.iterdir()):
        if p.is_dir() and _is_num(p.name):
            shutil.rmtree(p)
    shutil.copytree(case / name, case / "0")


def _is_num(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def snapshot_state(case, dest):
    """Copy mesh + system + ONLY the latest time dir into dest as a standalone,
    labeled single-state ParaView case (one .foam per (branch, Re) level) -- ideal
    for regime montages and matched-Re up-vs-down stills without scrubbing."""
    case = Path(case)
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    shutil.copytree(case / "constant", dest / "constant")
    shutil.copytree(case / "system", dest / "system")
    latest = max((p for p in case.iterdir() if p.is_dir() and _is_num(p.name)),
                 key=lambda p: float(p.name))
    shutil.copytree(latest, dest / latest.name)
    (dest / f"{dest.name}.foam").touch()


def write_frames_manifest(out, rows):
    """Map each (branch, Re) level to its [t_start, t_end] window in frames_loop/
    so the ParaView time slider on the continuous trajectory is interpretable
    (and you can jump straight to a branch's settled frame = t_end)."""
    with open(os.path.join(out, "frames_manifest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["branch", "rpm", "Re_rot", "t_start_s", "t_end_s",
                    "settled_frame_s"])
        for r in rows:
            w.writerow([r["branch"], f"{r['rpm']:.0f}", f"{r['Re_rot']:.1f}",
                        f"{r['t_start']:.0f}", f"{r['t_end']:.0f}", f"{r['t_end']:.0f}"])


# --------------------------------------------------------------------------- #
def run_branch(helpers, rpms, settle, ramp_time, r_in_m, d_m, nu, wallflux_ref,
               prev_rad, branch, capture=None):
    """Drive one monotone pass (up or down) through `rpms`, settling each. Returns
    a list of per-level records and the final omega (rad/s)."""
    rows = []
    for rpm in rpms:
        omega_rad = rpm * RPM
        re = reynolds(omega_rad, r_in_m, d_m, nu)
        t0 = float(helpers._get_latest_time())
        print(f"  [{branch}] {rpm:6.0f} rpm  Re={re:6.0f}  settle {settle:.0f}s ...",
              flush=True)
        metrics = helpers.do_simulation(omega_rad, settle,
                                        ramp_from=prev_rad, ramp_time=ramp_time)
        prev_rad = omega_rad
        t1 = float(helpers._get_latest_time())
        st = settled_stats(metrics)
        mz = st.get("Mz_kin", float("nan"))
        rows.append(dict(
            branch=branch, rpm=rpm, omega_rad=omega_rad, Re_rot=re,
            Ta=taylor_number(re, d_m, r_in_m),
            t_start=t0, t_end=t1,
            conv=st.get("conv", float("nan")),
            wf_norm=st.get("wallFlux", float("nan")) / wallflux_ref,
            wallFlux=st.get("wallFlux", float("nan")),
            Mz_kin=mz,
            mech_P=abs(mz) * RHO * omega_rad,        # viscous power (flow-state dep.)
            motor_P=steady_motor_power(omega_rad),   # paper motor power (omega only)
        ))
        c = rows[-1]["conv"]
        print(f"        conv={c:.3f}  wf_norm={rows[-1]['wf_norm']:.3f}  "
              f"motor_P={rows[-1]['motor_P']:.2f} W  mech_P={rows[-1]['mech_P']:.3e} W",
              flush=True)
        if capture:
            try:
                snapshot_state(helpers.case_path, os.path.join(
                    capture, f"{branch}_Re{int(round(re)):05d}_rpm{int(round(rpm)):04d}"))
            except Exception as e:   # a frame hiccup must never kill the sweep
                print(f"        [warn] per-level frame snapshot failed: {e}", flush=True)
    return rows, prev_rad


def modulation_gain(up_rows, all_rows):
    """Quasi-static estimate of how much a two-speed time-share beats CONSTANT.

    Constant operation (ramped from rest) follows the UP branch, so C_const(P) is
    interpolated from it. The convex hull of ALL (motor_P, conv) points is the
    modulation frontier. Returns the max conversion gain at matched power, the
    power where it peaks, and the tie-line endpoints (the two speeds + duty)."""
    up = sorted({(r["motor_P"], r["conv"], r["rpm"]) for r in up_rows
                 if np.isfinite(r["motor_P"]) and np.isfinite(r["conv"])})
    if len(up) < 2:
        return None
    upP = np.array([p[0] for p in up]); upC = np.array([p[1] for p in up])
    hull = upper_convex_hull({(r["motor_P"], r["conv"], r["rpm"]) for r in all_rows
                              if np.isfinite(r["motor_P"]) and np.isfinite(r["conv"])})
    hP = np.array([p[0] for p in hull]); hC = np.array([p[1] for p in hull])
    Pgrid = np.linspace(max(upP.min(), hP.min()), min(upP.max(), hP.max()), 400)
    gain = np.interp(Pgrid, hP, hC) - np.interp(Pgrid, upP, upC)
    k = int(np.argmax(gain))
    Pstar = float(Pgrid[k]); best = float(gain[k])
    # tie-line = the hull edge spanning Pstar
    j = int(np.searchsorted(hP, Pstar)) - 1
    j = max(0, min(j, len(hull) - 2))
    (Plo, Clo, rlo), (Phi, Chi, rhi) = hull[j], hull[j + 1]
    duty = (Pstar - Plo) / (Phi - Plo) if Phi > Plo else float("nan")
    return dict(Pstar=Pstar, gain=best, hull=hull,
                rpm_lo=rlo, rpm_hi=rhi, duty_high=duty,
                C_const=float(np.interp(Pstar, upP, upC)),
                C_mod=float(np.interp(Pstar, hP, hC)))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rpm", default=DEFAULT_RPM,
                    help="speed ladder (rpm), low->high; the down pass reverses it")
    ap.add_argument("--settle-seconds", type=float, default=40.0,
                    help="hold each level this long. Loop area vs this value is the "
                         "true(equilibrium)-vs-dynamic-hysteresis diagnostic.")
    ap.add_argument("--direction", choices=["updown", "up", "down"], default="updown")
    ap.add_argument("--ramp-time", type=float, default=0.10)
    ap.add_argument("--r-in", type=float, default=25.4)
    ap.add_argument("--r-out", type=float, default=31.75)
    ap.add_argument("--nu", type=float, default=1.075e-5)
    ap.add_argument("--feed-velocity", type=float, default=0.001462)
    ap.add_argument("--azimuthal-fraction", type=float, default=5.0 / 360.0,
                    help="wedge slice (5/360); use 1.0 for the full 360 annulus")
    ap.add_argument("--probe-rpm", default="",
                    help="comma-separated speeds for the fixed-Re two-IC bistability "
                         "test (cold-start vs hot-start). Empty = skip.")
    ap.add_argument("--probe-hot-rpm", type=float, default=2500.0,
                    help="speed used to pre-spin the HOT initial condition")
    ap.add_argument("--probe-prep-seconds", type=float, default=40.0)
    ap.add_argument("--no-frames", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rpms = [float(x) for x in args.rpm.split(",") if x.strip()]
    r_in_m, r_out_m = args.r_in * 1e-3, args.r_out * 1e-3
    d_m = r_out_m - r_in_m
    inlet_area = args.azimuthal_fraction * math.pi * (r_out_m**2 - r_in_m**2)
    wallflux_ref = (args.feed_velocity * inlet_area) or 1.0
    re_per_rpm = reynolds(RPM, r_in_m, d_m, args.nu)        # Re_rot = re_per_rpm * rpm
    re_ax = args.feed_velocity * d_m / args.nu              # axial (through-flow) Re

    print(f"[hysteresis] eta={r_in_m/r_out_m:.3f}  Re_rot={re_per_rpm:.3f}*rpm  "
          f"axial Re={re_ax:.3f} (negligible)  settle={args.settle_seconds:.0f}s  "
          f"dir={args.direction}")

    helpers = Helpers(args.case)
    helpers.set_write_interval(1.0)
    freeze_state(args.case, "0.cold")          # for the two-IC probe / reference

    # Labeled single-state snapshot per (branch, Re) level, written live during the
    # loop (one .foam each) for regime montages + matched-Re up/down stills.
    capture = None if args.no_frames else os.path.join(args.out, "frames_by_level")

    # ---- up / down quasi-static loop -----------------------------------
    rows, prev = [], 0.0
    if args.direction in ("updown", "up"):
        up, prev = run_branch(helpers, rpms, args.settle_seconds, args.ramp_time,
                              r_in_m, d_m, args.nu, wallflux_ref, prev, "up",
                              capture=capture)
        rows += up
    if args.direction in ("updown", "down"):
        # descend through the same ladder; skip the top point if we just did it.
        down_ladder = list(reversed(rpms))
        if args.direction == "updown":
            down_ladder = down_ladder[1:]
        start = prev if args.direction == "updown" else 0.0
        down, prev = run_branch(helpers, down_ladder, args.settle_seconds, args.ramp_time,
                                r_in_m, d_m, args.nu, wallflux_ref, start, "down",
                                capture=capture)
        rows += down

    # ---- ParaView: full up/down trajectory ------------------------------
    # Snapshot the continuous animation NOW, before the optional two-IC probe
    # wipes the case's time dirs (restore_state). Every simulated second is a
    # frame (writeInterval=1); frames_manifest.csv maps the time slider -> Re.
    if not args.no_frames:
        loopdir = os.path.join(args.out, "frames_loop")
        print(f"[hysteresis] snapshotting loop trajectory -> {loopdir}")
        helpers.snapshot_frames(loopdir)
        write_frames_manifest(args.out, rows)

    # ---- CSV -----------------------------------------------------------
    for r in rows:
        r["Re_axial"] = re_ax       # constant; logged for context (regime is rotation-set)
    csv_path = os.path.join(args.out, "hysteresis_branches.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[hysteresis] wrote {csv_path}")

    up_rows = [r for r in rows if r["branch"] == "up"]
    down_rows = [r for r in rows if r["branch"] == "down"]

    # ---- modulation-gain (convex-hull) analysis ------------------------
    mg = modulation_gain(up_rows or rows, rows) if up_rows else None
    if mg:
        with open(os.path.join(args.out, "modulation_estimate.txt"), "w") as f:
            f.write(
                "Quasi-static modulation-vs-constant estimate (conv vs motor power)\n"
                f"  max conversion gain : {mg['gain']*100:+.2f} percentage points\n"
                f"  at mean motor power : {mg['Pstar']:.2f} W\n"
                f"  constant conversion : {mg['C_const']:.3f}\n"
                f"  modulated conversion: {mg['C_mod']:.3f}\n"
                f"  optimal time-share  : {mg['duty_high']*100:.0f}% at "
                f"{mg['rpm_hi']:.0f} rpm / {100-mg['duty_high']*100:.0f}% at "
                f"{mg['rpm_lo']:.0f} rpm\n"
                "  (validate by running a square wave at this mean power: "
                "compare_catalysis.py / make_case.py --mode squarewave)\n")
        print(f"[hysteresis] modulation gain ~{mg['gain']*100:+.2f} pp at "
              f"{mg['Pstar']:.1f} W: {mg['duty_high']*100:.0f}% @{mg['rpm_hi']:.0f} / "
              f"{100-mg['duty_high']*100:.0f}% @{mg['rpm_lo']:.0f} rpm")

    # ---- figures -------------------------------------------------------
    _plots(args.out, up_rows, down_rows, rows, mg)
    _regime_figures(args.out, rows, re_per_rpm)

    # ---- fixed-Re two-IC bistability probe (optional) ------------------
    # Snapshots each cold-start and hot-start settling as its own trajectory under
    # frames_probe/ -- load the two at a given Re side by side to SEE bistability.
    probes = [float(x) for x in args.probe_rpm.split(",") if x.strip()]
    if probes:
        _two_ic_probe(helpers, args.case, args.out, probes, args.probe_hot_rpm,
                      args.probe_prep_seconds, args.settle_seconds, args.ramp_time,
                      r_in_m, d_m, args.nu, wallflux_ref, capture=not args.no_frames)

    print("[hysteresis] done.")


def _two_ic_probe(helpers, case, out, probes, hot_rpm, prep_secs, settle,
                  ramp_time, r_in_m, d_m, nu, wallflux_ref, capture=False):
    """For each probe speed, settle it from a COLD IC and from a HOT IC; differing
    settled conversions => two attractors at one Re (genuine bistability)."""
    print("\n[hysteresis] fixed-Re two-IC bistability probe")
    # Build the HOT reference state: spin cold up to hot_rpm and hold.
    restore_state(case, "0.cold")
    helpers.do_simulation(hot_rpm * RPM, prep_secs, ramp_from=0.0, ramp_time=ramp_time)
    freeze_state(case, "0.hot")

    out_rows = []
    for rpm in probes:
        rad = rpm * RPM
        re = reynolds(rad, r_in_m, d_m, nu)
        rec = {"rpm": rpm, "Re_rot": re}
        for ic, ic_rad in [("cold", 0.0), ("hot", hot_rpm * RPM)]:
            restore_state(case, f"0.{ic}")
            m = helpers.do_simulation(rad, settle, ramp_from=ic_rad, ramp_time=ramp_time)
            st = settled_stats(m)
            rec[f"conv_from_{ic}"] = st.get("conv", float("nan"))
            rec[f"wf_from_{ic}"] = st.get("wallFlux", float("nan")) / wallflux_ref
            if capture:   # full settling trajectory of each start -> side-by-side viz
                try:
                    helpers.snapshot_frames(os.path.join(
                        out, "frames_probe", f"rpm{int(round(rpm)):04d}_{ic}"))
                except Exception as e:
                    print(f"    [warn] probe frame snapshot failed: {e}", flush=True)
        rec["conv_split"] = rec["conv_from_hot"] - rec["conv_from_cold"]
        out_rows.append(rec)
        print(f"  {rpm:6.0f} rpm  Re={re:6.0f}  conv cold={rec['conv_from_cold']:.3f} "
              f"hot={rec['conv_from_hot']:.3f}  split={rec['conv_split']:+.3f}", flush=True)

    with open(os.path.join(out, "bistability_probe.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)

    fig, ax = plt.subplots(figsize=(7, 5))
    re = [r["Re_rot"] for r in out_rows]
    ax.plot(re, [r["conv_from_cold"] for r in out_rows], "o-", color="#1f77b4",
            label="settled from COLD start")
    ax.plot(re, [r["conv_from_hot"] for r in out_rows], "s-", color="#d62728",
            label="settled from HOT start")
    ax.set_xlabel(r"$Re_{rot}$"); ax.set_ylabel("settled conversion")
    ax.set_title("Fixed-Re two-IC bistability probe\n"
                 "vertical gap at a Re = two coexisting attractors (bistable)")
    ax.grid(alpha=0.3); ax.legend()
    fig.savefig(os.path.join(out, "fig_bistability_probe.png"), dpi=140,
                bbox_inches="tight"); plt.close(fig)


def _regime_figures(out, rows, re_per_rpm):
    """Flow-regime context figures (ported from the old reynolds_sweep): Re vs rpm
    with the Couette / Taylor-vortex bands + catalysis landmarks, and the Re-vs-time
    staircase showing the up-then-down protocol."""
    rpm = np.array([r["rpm"] for r in rows])
    re = np.array([r["Re_rot"] for r in rows])

    # Re vs rpm with regime bands + landmarks.
    fig, ax = plt.subplots(figsize=(7.5, 5))
    line_rpm = np.linspace(rpm.min(), rpm.max(), 200)
    ax.plot(line_rpm, re_per_rpm * line_rpm, "-", color="#1f77b4", zorder=3)
    ax.scatter(rpm, re, s=18, color="#1f77b4", zorder=4)
    top = max(re.max(), RE_C) * 1.05
    ax.axhline(RE_C, color="0.4", ls="--", lw=1)
    ax.axhspan(0, RE_C, color="#cfe8ff", alpha=0.5, zorder=0)
    ax.axhspan(RE_C, top, color="#ffe0cc", alpha=0.5, zorder=0)
    ax.set_ylim(0, top)
    ax.text(rpm.max(), RE_C * 0.5, "circular Couette\n(no vortices)",
            ha="right", va="center", fontsize=9, color="0.25")
    ax.text(rpm.max(), RE_C + (top - RE_C) * 0.45, "(axisymmetric) Taylor vortices",
            ha="right", va="center", fontsize=9, color="0.25")
    for mark_rpm, lab in [(80, "const. optimum ~80"), (500, "baseline 500"),
                          (2500, "peak 2500")]:
        if rpm.min() <= mark_rpm <= rpm.max():
            ax.axvline(mark_rpm, color="0.7", ls=":", lw=1)
            ax.text(mark_rpm, re.min(), f" {lab}", rotation=90, va="bottom",
                    ha="left", fontsize=7.5, color="0.4")
    ax.set_xlabel("inner-cylinder speed [rpm]")
    ax.set_ylabel(r"rotational Reynolds number  $Re=\omega r_i d/\nu$")
    ax.set_title(f"Re vs speed  (Re = {re_per_rpm:.3f} x rpm)\n"
                 r"$\eta=0.8$, Taylor-vortex onset $Re_c\approx120$")
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(out, "fig_Re_vs_rpm.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Re vs simulated time -- the up-then-down staircase (the protocol).
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for r in rows:
        c = "#1f77b4" if r["branch"] == "up" else "#d62728"
        ax.hlines(r["Re_rot"], r["t_start"], r["t_end"], color=c, lw=2.5)
    ax.plot([], [], color="#1f77b4", lw=2.5, label="up")
    ax.plot([], [], color="#d62728", lw=2.5, label="down")
    ax.axhline(RE_C, color="0.5", ls="--", lw=1, label=r"$Re_c\approx120$")
    ax.set_xlabel("simulated time [s]"); ax.set_ylabel(r"$Re_{rot}$")
    ax.set_title("Quasi-static Re staircase (up then down) -- the hysteresis protocol")
    ax.grid(alpha=0.3); ax.legend()
    fig.savefig(os.path.join(out, "fig_Re_staircase.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plots(out, up_rows, down_rows, rows, mg):
    def arr(rs, k):
        return np.array([r[k] for r in rs])

    # conv vs Re loop
    fig, ax = plt.subplots(figsize=(7.5, 5))
    if up_rows:
        ax.plot(arr(up_rows, "Re_rot"), arr(up_rows, "conv"), "o-", color="#1f77b4",
                label="up (increasing Re)")
    if down_rows:
        ax.plot(arr(down_rows, "Re_rot"), arr(down_rows, "conv"), "s--", color="#d62728",
                label="down (decreasing Re)")
    ax.set_xlabel(r"$Re_{rot}$"); ax.set_ylabel("settled conversion (1 - cup C)")
    ax.set_title("Conversion vs Re -- up vs down\n(separation = hysteresis loop)")
    ax.grid(alpha=0.3); ax.legend()
    fig.savefig(os.path.join(out, "fig_conv_vs_Re_loop.png"), dpi=140,
                bbox_inches="tight"); plt.close(fig)

    # wf_norm vs Re loop (fast flow-state metric)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    if up_rows:
        ax.plot(arr(up_rows, "Re_rot"), arr(up_rows, "wf_norm"), "o-", color="#1f77b4",
                label="up")
    if down_rows:
        ax.plot(arr(down_rows, "Re_rot"), arr(down_rows, "wf_norm"), "s--",
                color="#d62728", label="down")
    ax.set_xlabel(r"$Re_{rot}$"); ax.set_ylabel("wall-flux / feed  (conv-equivalent)")
    ax.set_title("Wall flux vs Re -- up vs down\n(flow-state metric; resolves the loop faster than conv)")
    ax.grid(alpha=0.3); ax.legend()
    fig.savefig(os.path.join(out, "fig_wf_vs_Re_loop.png"), dpi=140,
                bbox_inches="tight"); plt.close(fig)

    # mechanical power vs Re (torque hysteresis)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    if up_rows:
        ax.plot(arr(up_rows, "Re_rot"), arr(up_rows, "mech_P"), "o-", color="#1f77b4",
                label="up")
    if down_rows:
        ax.plot(arr(down_rows, "Re_rot"), arr(down_rows, "mech_P"), "s--",
                color="#d62728", label="down")
    ax.set_xlabel(r"$Re_{rot}$"); ax.set_ylabel("mechanical (viscous) power [W]")
    ax.set_title("Viscous power vs Re -- up vs down\n(two-valued here = genuine flow-state bistability)")
    ax.grid(alpha=0.3); ax.legend()
    fig.savefig(os.path.join(out, "fig_mechP_vs_Re_loop.png"), dpi=140,
                bbox_inches="tight"); plt.close(fig)

    # THE figure: conversion vs motor power, with convex hull + modulation tie-line
    fig, ax = plt.subplots(figsize=(8, 5.5))
    if up_rows:
        ax.plot(arr(up_rows, "motor_P"), arr(up_rows, "conv"), "o-", color="#1f77b4",
                label="up (= constant operation)")
    if down_rows:
        ax.plot(arr(down_rows, "motor_P"), arr(down_rows, "conv"), "s--",
                color="#d62728", label="down")
    if mg:
        hP = [p[0] for p in mg["hull"]]; hC = [p[1] for p in mg["hull"]]
        ax.plot(hP, hC, "-", color="#2ca02c", lw=2, alpha=0.7,
                label="modulation frontier (convex hull)")
        ax.plot([mg["Pstar"]], [mg["C_mod"]], "*", color="#2ca02c", ms=16, zorder=5)
        ax.annotate(f"+{mg['gain']*100:.1f} pp\n{mg['duty_high']*100:.0f}%@{mg['rpm_hi']:.0f}rpm",
                    (mg["Pstar"], mg["C_mod"]), textcoords="offset points",
                    xytext=(8, -4), fontsize=9, color="#1b7837")
        ax.plot([mg["Pstar"], mg["Pstar"]], [mg["C_const"], mg["C_mod"]],
                ":", color="0.4")
    ax.set_xlabel("mean motor power [W]"); ax.set_ylabel("conversion (1 - cup C)")
    ax.set_title("Conversion vs power: where the modulation frontier rises above\n"
                 "the constant curve, modulation beats constant at equal power")
    ax.grid(alpha=0.3); ax.legend(loc="lower right")
    fig.savefig(os.path.join(out, "fig_conv_vs_power.png"), dpi=140,
                bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
