#!/usr/bin/env python3
"""Mechanism experiment 4 (PLAN.md): measure the clocks from EXISTING traces.

No CFD. Reads the duty_v3 reference traces (warmed short reactor, 130 s,
maxCo 1.8, dense wall-flux + outlet-conversion samples):
    experiments/modulation_rl/results/duty_v3_refs/champion_D0.90/timeseries.csv
    experiments/modulation_rl/results/duty_v3_refs/constant_wb300/timeseries.csv

The stops are found from the wall-flux trace itself (the commanded wave of
those runs restarted every 26 s, so it is not reconstructed): each stop shows
as a dip of the catalytic-wall consumption. Per stop after t = 30 s:
    decay      fraction of wall flux left at the bottom of the dip, the dip
               duration, and an e-folding time of the fall
    overshoot  peak after restart over the burst plateau, minus 1
    fade       e-folding time of the excess decaying back to the plateau
and a gain decomposition of the last-26-s mean wall flux (= conversion x Q):
    champion - constant = (plateau excess) + (overshoot area) - (dip deficit)
which says whether the pulsing gain is a transient after each restart or a
sustained elevation through the burst.

The swirl clock is not in these traces; the new runs (mechanism_sweep.py)
log mean swirl and secondary-flow energy every 0.02 s.

USAGE (from taylorCouetteGym/):  python3 experiments/mechanism/exp4_champion_clocks.py
Outputs: experiments/mechanism/results/exp4/{exp4_report.json, exp4_stops.csv, exp4_traces.png}
"""

import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
GYM_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
REFS = os.path.join(GYM_ROOT, "experiments", "modulation_rl", "results", "duty_v3_refs")
OUT = os.path.join(HERE, "results", "exp4")

Q_WEDGE = 100e-6 / 60.0 / 72.0      # inlet flow per 5-degree wedge [m^3/s]
T_FROM = 30.0
WINDOW = (104.0, 130.0)
GAP = 0.00635
NU0 = 1.0752688172043011e-05


def load(name):
    d = np.genfromtxt(os.path.join(REFS, name, "timeseries.csv"), delimiter=",", names=True)
    t, i = np.unique(d["time_s"], return_index=True)
    return t, d["wallFlux"][i], d["conversion"][i]


def smooth(y, n):
    if n <= 1:
        return y
    k = np.ones(n) / n
    return np.convolve(np.pad(y, (n // 2, n - 1 - n // 2), mode="edge"), k, mode="valid")


def fit_tau(t, y):
    ok = np.isfinite(y) & (y > 0)
    if ok.sum() < 4:
        return float("nan")
    k = np.polyfit(t[ok] - t[ok][0], np.log(y[ok]), 1)[0]
    return float(-1.0 / k) if k < 0 else float("nan")


def find_dips(t, wf, plateau):
    """Stops = contiguous stretches with smoothed flux below 90 % of the plateau.
    Returns [(onset, bottom, recovered)] times; onset = last time at >= 97 %."""
    dt = float(np.median(np.diff(t)))
    s = smooth(wf, max(1, int(round(0.02 / dt))))
    low = s < 0.90 * plateau
    out = []
    i = 0
    n = len(t)
    while i < n:
        if not low[i]:
            i += 1
            continue
        j = i
        while j < n and low[j]:
            j += 1
        k0 = i
        while k0 > 0 and s[k0] < 0.97 * plateau:
            k0 -= 1
        kb = i + int(np.argmin(s[i:j]))
        out.append((t[k0], t[kb], t[min(j, n - 1)], k0, kb, j))
        i = j
    return out, s


def main():
    os.makedirs(OUT, exist_ok=True)
    tc, wc, xc = load("champion_D0.90")
    tk, wk, xk = load("constant_wb300")
    a, b = WINDOW
    selc, selk = (tc >= a) & (tc <= b), (tk >= a) & (tk <= b)
    wf_const = float(np.mean(wk[selk]))
    plateau0 = float(np.percentile(wc[tc >= T_FROM], 75))
    dips, s = find_dips(tc, wc, plateau0)
    dips = [d for d in dips if d[0] >= T_FROM]

    rows = []
    for idx, (t_on, t_bot, t_rec, k0, kb, j) in enumerate(dips):
        nxt = dips[idx + 1][0] if idx + 1 < len(dips) else tc[-1]
        # burst plateau: the second half of the stretch between this dip's
        # recovery and the next dip's onset
        pl_sel = (tc >= t_rec + 0.5 * (nxt - t_rec)) & (tc <= nxt - 0.05)
        if pl_sel.sum() < 10:
            continue
        plateau = float(np.mean(wc[pl_sel]))
        pre = float(np.mean(wc[(tc >= t_on - 0.5) & (tc <= t_on)]))
        fall = (tc >= t_on) & (tc <= t_bot)
        post = (tc >= t_bot) & (tc <= min(t_bot + 2.0, nxt))
        ip = int(np.argmax(s[post]))
        tp, wp = tc[post], s[post]
        ex = wp[ip:] - plateau
        npos = int(np.argmax(ex <= 0)) if (ex <= 0).any() else len(ex)
        rows.append(dict(onset=t_on, bottom=t_bot, dip_s=t_bot - t_on,
                         left=float(s[kb] / pre), tau_fall=fit_tau(tc[fall], s[fall]),
                         plateau=plateau, overshoot=float(wp[ip] / plateau - 1.0),
                         t_peak_after_restart=float(tp[ip] - t_bot),
                         tau_fade=fit_tau(tp[ip:ip + npos], ex[:npos]),
                         plateau_over_const=plateau / wf_const))
    with open(os.path.join(OUT, "exp4_stops.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- gain decomposition over the scoring window --------------------------
    win_rows = [r for r in rows if a <= r["onset"] <= b]
    plateau_w = float(np.mean([r["plateau"] for r in win_rows]))
    wfc = wc[selc]
    tt = tc[selc]
    excess = wfc - plateau_w
    dtt = np.gradient(tt)
    dip_mask = np.zeros_like(tt, bool)
    over_mask = np.zeros_like(tt, bool)
    for r in win_rows:
        dip_mask |= (tt >= r["onset"]) & (tt <= r["bottom"] + 0.5 * (r["t_peak_after_restart"]))
        over_mask |= (tt > r["bottom"] + 0.5 * r["t_peak_after_restart"]) & \
                     (tt <= r["bottom"] + 1.5) & (excess > 0)
    L = b - a
    deficit = float(np.sum(excess[dip_mask] * dtt[dip_mask]) / L)
    over = float(np.sum(excess[over_mask] * dtt[over_mask]) / L)
    mean_c = float(np.sum(wfc * dtt) / np.sum(dtt))
    decomp = dict(
        wf_champion=mean_c, wf_constant=wf_const, plateau=plateau_w,
        X_champion_budget=mean_c / Q_WEDGE, X_constant_budget=wf_const / Q_WEDGE,
        X_champion_outlet=float(np.mean(xc[selc])), X_constant_outlet=float(np.mean(xk[selk])),
        gain_X=(mean_c - wf_const) / Q_WEDGE,
        from_plateau_X=(plateau_w - wf_const) / Q_WEDGE,
        from_overshoot_X=over / Q_WEDGE,
        from_dips_X=deficit / Q_WEDGE,
        speed_only_expectation=(333.33 / 300.0) ** 0.7 - 1.0,
    )
    med = {k: float(np.nanmedian([r[k] for r in rows])) for k in
           ("dip_s", "left", "tau_fall", "overshoot", "t_peak_after_restart", "tau_fade", "plateau_over_const")}
    report = dict(n_stops=len(rows), median=med, decomposition=decomp,
                  swirl_clock_pred_s=GAP ** 2 / (np.pi ** 2 * NU0), film_clock_pred_s=0.8,
                  note="maxCo 1.8 traces: outlet X carries the outlet-jet artifact; the wall-flux "
                       "numbers are interior and unaffected")
    json.dump(report, open(os.path.join(OUT, "exp4_report.json"), "w"), indent=2)

    print(f"exp4: {len(rows)} stops after t={T_FROM:g} s in the champion trace")
    for k, v in med.items():
        print(f"   median {k:22s} {v:.4f}")
    d = decomp
    print(f"   window X: champion {d['X_champion_budget']:.4f} (budget) / {d['X_champion_outlet']:.4f} (outlet), "
          f"constant {d['X_constant_budget']:.4f} / {d['X_constant_outlet']:.4f}")
    print(f"   gain {d['gain_X']:+.4f} = plateau {d['from_plateau_X']:+.4f} + overshoot "
          f"{d['from_overshoot_X']:+.4f} + dips {d['from_dips_X']:+.4f}  "
          f"(sum {d['from_plateau_X'] + d['from_overshoot_X'] + d['from_dips_X']:+.4f})")
    print(f"   burst plateau is {100 * (med['plateau_over_const'] - 1):.1f} % above constant 300; "
          f"speed alone (333 vs 300 rpm, Sh ~ Re^0.7) predicts {100 * d['speed_only_expectation']:.1f} %")

    # ---- figure ---------------------------------------------------------------
    fig, ax = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    ax[0].plot(tc[selc], wc[selc] / Q_WEDGE, color="#c8102e", lw=0.8, label="champion D0.90 T5")
    ax[0].plot(tk[selk], wk[selk] / Q_WEDGE, color="#1f5f8b", lw=0.8, label="constant 300 rpm")
    ax[0].axhline(plateau_w / Q_WEDGE, color="#c8102e", ls=":", lw=1, label="burst plateau")
    ax[0].set_ylabel("wall consumption / inflow")
    ax[0].legend(fontsize=9)
    ax[1].plot(tc[selc], xc[selc], color="#c8102e", lw=0.8)
    ax[1].plot(tk[selk], xk[selk], color="#1f5f8b", lw=0.8)
    ax[1].set_ylabel("outlet conversion")
    ax[1].set_xlabel("time [s]")
    for r in win_rows:
        for a_ in ax:
            a_.axvspan(r["onset"], r["bottom"], color="0.85", lw=0)
    for a_ in ax:
        a_.grid(True, alpha=0.3)
    fig.suptitle("Existing duty_v3 traces, last 26 s (grey = wall-flux dips during stops)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "exp4_traces.png"), dpi=180)
    plt.close(fig)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
