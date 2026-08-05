#!/usr/bin/env python3
"""Recompute the warmed-sweep metrics AT t = 50 s (no CFD; log re-parse).

The warmed sweeps ran 60 s episodes (matching the pristine 60 s families),
but the RL episodes -- v5 modulation training, the duty_v1 campaign, and the
50 s pristine benchmarks (results_td3_prep, results_frag_*) -- are 50 s. For
cross-comparability this script re-windows every warmed episode at t = 50:

    X50  = mean conversion over the last full period ENDING at 50 s
           (window [50-T, 50]; T=10 -> [40,50], the td3_prep window)
    P50  = commanded motor power averaged over [0, 50] s
    R50- = X50 - P50/31.94

Writes summary_table_t50.csv into each warmed results dir and prints the
tables. The 60 s numbers in summary_table.csv stay the canonical
warmed-vs-pristine-60s comparison; these are the RL-comparable rows.

USAGE
    python3 t50_table.py
"""
import csv
import os

import numpy as np

import fig7_sweep as F

T_CUT = 50.0
SWEEPS = [
    ("results_warmed_T2p5_D80", 2.5, {"pulsed": 0.8, "constant": 1.0}),
    ("results_warmed_T10_D80", 10.0, {"pulsed": 0.8, "constant": 1.0}),
]


def x_at_cut(t_conv, conv, period):
    """Blip-rejected mean conversion over [T_CUT - period, T_CUT]."""
    phys = (conv >= -0.02) & (conv <= 1.02)
    tcp, ccp = t_conv[phys], conv[phys]
    win = ccp[(tcp >= T_CUT - period) & (tcp <= T_CUT)]
    return float(np.mean(win)) if len(win) >= 3 else float("nan")


def main():
    for dirname, period, modes in SWEEPS:
        F.PERIOD = period
        F.MODES = modes
        F.RESULTS_DIR = os.path.join(F.HERE, dirname)
        results = F.load_results_from_logs(60.0, period)
        rows = []
        for r in results:
            x50 = x_at_cut(r["t_conv"], r["conv"], period)
            p50 = F.motor_power_avg(r["pts"], T_CUT)
            rows.append(dict(mode=r["mode"], wb=r["wb"], duty=r["duty"],
                             X50=x50, P50_W=p50, P50_over_Pmax=p50 / F.P_MAX,
                             R50_minus=x50 - p50 / F.P_MAX))
        out = os.path.join(F.RESULTS_DIR, "summary_table_t50.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        print(f"\n=== {dirname}  (X over [{T_CUT-period:g},{T_CUT:g}] s, "
              f"P over [0,{T_CUT:g}] s) ===")
        print("   w_b | const X50 | pulsed X50 | const R50- | pulsed R50-")
        by = {(r["mode"], r["wb"]): r for r in rows}
        for wb in F.WBS:
            c, p = by.get(("constant", wb)), by.get(("pulsed", wb))
            g = lambda r, k: f"{r[k]:.4f}" if r else "--"
            print(f"  {wb:>4d} | {g(c,'X50'):>9s} | {g(p,'X50'):>10s} | "
                  f"{g(c,'R50_minus'):>10s} | {g(p,'R50_minus'):>11s}")
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
