#!/usr/bin/env python3
"""Figures that walk through how the mechanism conclusion was reached.

Reads experiments/mechanism/results/main (the Carya run) and results/exp4,
writes results/main/explained/fig1..fig7 (+ README.md in that folder).

    fig1  the pre-registered test: conversion vs stop length per viscosity, each
          point coloured and labelled by its Taylor-cell count
    fig2  conversion and burst wall flux follow the cell count, at every viscosity
    fig3  one cycle: the restart re-forms more cells; the wall flux follows
    fig4  without repeated stops the extra cells merge back (hold / kick runs)
    fig5  the vortices must nearly die during the stop (stop-level sweep)
    fig6  the gain is a sustained burst plateau, not the restart overshoot (exp4)
    fig7  the clocks behave as predicted, but do not set the best stop

USAGE (from taylorCouetteGym/):  python3 experiments/mechanism/explain_figures.py
"""

import importlib.util
import json
import os

import numpy as np
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("mechanism_sweep", os.path.join(HERE, "mechanism_sweep.py"))
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)

RES = os.path.join(HERE, "results", "main")
OUT = os.path.join(RES, "explained")
EXP4 = os.path.join(HERE, "results", "exp4", "exp4_report.json")

# validated palette (dataviz reference): 3 categorical slots, all-pairs safe
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
NU_COL = {"nu0p7": "#2a78d6", "nu1": "#eb6834", "nu2": "#1baf7a"}
NU_LAB = {"nu0p7": "0.7× viscosity", "nu1": "nominal viscosity", "nu2": "2× viscosity"}
# ordinal ramp for cell count (one hue, light -> dark)
CELL_COL = {2: "#cde2fb", 4: "#86b6ef", 6: "#2a78d6", 8: "#104281", 10: "#0d366b"}
RUN_COL = {"constant": "#52514e", "pulse": "#2a78d6", "hold": "#eb6834", "kick": "#1baf7a"}

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})


def cell_col(n):
    n = int(round(n))
    return CELL_COL.get(n, CELL_COL[min(CELL_COL, key=lambda k: abs(k - n))])


def roll_median(y, n):
    if n <= 1 or len(y) < n:
        return y
    pad = np.pad(y, (n // 2, n - 1 - n // 2), mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(pad, n), axis=1)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = [r for r in M.load_all(RES, False) if r["ok"]]
    by = {r["tag"]: r for r in rows}
    const = {nt: next(r for r in rows if r["nu_tag"] == nt and r["exp"] in ("e1", "e2") and r["stop"] == 0)
             for nt in M.NU}

    # ---- fig1: the pre-registered test -----------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
    for ax, nt in zip(axes, ("nu0p7", "nu1", "nu2")):
        rs = sorted([r for r in rows if r["nu_tag"] == nt and r["exp"] in ("e1", "e2") and r["stop"] > 0],
                    key=lambda r: r["stop"])
        c = const[nt]
        ax.axhline(c["x_budget"], color=INK2, ls="--", lw=1.2)
        ax.text(rs[-1]["stop"] * 1.05, c["x_budget"], f"constant 300 rpm\n({c['nvort_burst']:.0f} cells)",
                va="center", fontsize=9, color=INK2)
        ax.plot([r["stop"] for r in rs], [r["x_budget"] for r in rs], "-", color=MUTED, lw=1, zorder=1)
        for r in rs:
            ax.scatter(r["stop"], r["x_budget"], s=90, color=cell_col(r["nvort_burst"]), edgecolor="#fcfcfb",
                       linewidth=2, zorder=3)
            ax.annotate(f"{r['nvort_burst']:.0f}", (r["stop"], r["x_budget"]), xytext=(0, 9),
                        textcoords="offset points", ha="center", fontsize=9, color=INK)
        ax.set_xscale("log")
        ax.set_xticks([0.1, 0.2, 0.3, 0.5, 1.0, 2.0])
        ax.set_xticklabels(["0.1", "0.2", "0.3", "0.5", "1", "2"])
        ax.set_xlim(0.08, 4.5)
        ax.set_title(NU_LAB[nt])
        ax.set_xlabel("stop length [s]  (run 4.5 s, mean 300 rpm)")
    axes[0].set_ylabel("conversion (wall-flux budget, last ~26 s)")
    for n, lab in ((4, "4 cells"), (6, "6 cells"), (8, "8 cells")):
        axes[2].scatter([], [], s=90, color=CELL_COL[n], label=lab)
    axes[2].legend(title="Taylor cells during bursts", loc="lower left", fontsize=9)
    fig.suptitle("1. The best stop does not move smoothly with viscosity: conversion jumps between "
                 "cell-count levels (numbers = cells)", x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    save(fig, "fig1_stop_curves_by_cell_count.png")

    # ---- fig2: conversion follows cell count --------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    rng = np.random.default_rng(0)
    off = {"nu0p7": -0.22, "nu1": 0.0, "nu2": 0.22}
    for nt in ("nu0p7", "nu1", "nu2"):
        rs = [r for r in rows if r["nu_tag"] == nt and r["exp"] in ("e1", "e2") and r["stop"] > 0]
        c = const[nt]
        xs = np.array([r["nvort_burst"] - c["nvort_burst"] for r in rs]) + off[nt] + \
            rng.uniform(-0.08, 0.08, len(rs))
        lab = f"{NU_LAB[nt]} (constant: {c['nvort_burst']:.0f} cells)"
        ax[0].scatter(xs, [r["wall_burst"] / c["wall_mean"] for r in rs], s=55, color=NU_COL[nt],
                      edgecolor="#fcfcfb", linewidth=1.5, label=lab, zorder=3)
        ax[1].scatter(xs, [r["x_budget"] - c["x_budget"] for r in rs], s=55, color=NU_COL[nt],
                      edgecolor="#fcfcfb", linewidth=1.5, label=lab, zorder=3)
    ax[0].axhline(1.0, color=INK2, ls="--", lw=1)
    ax[1].axhline(0.0, color=INK2, ls="--", lw=1)
    ax[0].set_ylabel("wall consumption during bursts\n÷ that viscosity's constant-300 value")
    ax[1].set_ylabel("conversion gain over that viscosity's constant 300")
    for a in ax:
        a.set_xticks([0, 2, 4])
        a.set_xticklabels(["0", "+2", "+4"])
        a.set_xlabel("Taylor cells added over steady rotation (pulse − constant)")
        a.legend(fontsize=9, loc="lower right")
    ax[0].set_title("Cells added -> more wall consumption")
    ax[1].set_title("...and more conversion; no cells added, almost no gain")
    fig.suptitle("2. Every pulsed run: the gain appears when the restart adds cells over steady rotation",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    save(fig, "fig2_conversion_follows_cells.png")

    # ---- fig3: one cycle ----------------------------------------------------------
    fig, ax = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True)
    cases = [("e1_nu1_s0p00", "constant 300 rpm", RUN_COL["constant"]),
             ("e1_nu1_s0p50", "stop 0.5 s (8-cell window)", "#104281"),
             ("e1_nu1_s0p70", "stop 0.7 s (6-cell window)", "#86b6ef")]
    for tag, lab, col in cases:
        r = by[tag]
        m = M.read_mech(r["workdir"])
        t0 = 120.0 if r["stop"] == 0 else r["win_b"] - 2 * r["period"]
        sel = (m["t"] >= t0) & (m["t"] <= t0 + 10.4)
        tt = m["t"][sel] - t0
        ax[0].plot(tt, m["Omega"][sel] / M.RPM, color=col, lw=1.8, label=lab)
        ax[1].step(tt, m["nVort"][sel], where="post", color=col, lw=1.5)
        ax[2].plot(tt, m["wall"][sel] / const["nu1"]["wall_mean"], color=col, lw=1.8)
    ax[0].set_ylabel("wall speed [rpm]")
    ax[0].legend(fontsize=9, loc="center", ncol=3, bbox_to_anchor=(0.5, 0.45))
    ax[1].set_ylabel("Taylor cells\n(instantaneous)")
    ax[1].set_yticks([2, 4, 6, 8, 10, 12])
    ax[2].axhline(1.0, color=INK2, ls="--", lw=1)
    ax[2].set_ylabel("wall consumption\n÷ constant 300")
    ax[2].set_xlabel("time [s] (two periods near the end of each run)")
    ax[1].set_title("Each restart re-forms the vortices: a 0.5 s stop lands in 8 cells every cycle; "
                    "a 0.7 s stop lands in 6 or 8 (mostly 6)",
                    fontsize=11, fontweight="normal", loc="left")
    ax[2].set_title("More cells keep the wall consumption high for the whole burst, not just after restart",
                    fontsize=11, fontweight="normal", loc="left")
    fig.suptitle("3. Inside one cycle (nominal viscosity)", x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    save(fig, "fig3_one_cycle.png")

    # ---- fig4: hold / kick -------------------------------------------------------
    fig, ax = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)
    cases = [("e1_nu1_s0p00", "constant 300 rpm", RUN_COL["constant"]),
             ("e1_nu1_s0p50", "pulsing throughout (stop 0.5 s)", RUN_COL["pulse"]),
             ("e5_nu1_hold", "hold: pulsing until 60 s, then constant", RUN_COL["hold"]),
             ("e5_nu1_kick", "kick: constant with one 0.5 s stop at 60 s", RUN_COL["kick"])]
    for tag, lab, col in cases:
        m = M.read_mech(by[tag]["workdir"])
        sel = m["t"] >= 30
        ax[0].plot(m["t"][sel], roll_median(m["nVort"], 101)[sel], color=col, lw=1.8, label=lab)
        ax[1].plot(m["t"][sel], roll_median(m["xOut"], 51)[sel], color=col, lw=1.6, label=lab)
    for a in ax:
        a.axvline(60, color=INK2, lw=1, ls=":")
        a.axvspan(104, 130, color=GRID, alpha=0.5, lw=0)
    ax[0].text(60.5, 9.2, "switch at 60 s", fontsize=9, color=INK2)
    ax[1].text(104.5, ax[1].get_ylim()[1], "scoring window", fontsize=9, color=INK2, va="top")
    ax[0].set_ylabel("Taylor cells\n(2 s running median)")
    ax[0].set_yticks([4, 6, 8, 10])
    ax[1].set_ylabel("outlet conversion\n(1 s running median)")
    ax[1].set_xlabel("time [s]")
    ax[0].legend(fontsize=9, loc="upper right")
    ax[0].set_title("Under steady rotation the extra cells merge back to 4 within about 10-15 s",
                    fontsize=11, fontweight="normal", loc="left")
    ax[1].set_title("...and conversion falls back to the constant-rotation value",
                    fontsize=11, fontweight="normal", loc="left")
    fig.suptitle("4. The gain needs repeated stops (nominal viscosity)", x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    save(fig, "fig4_hold_and_kick.png")

    # ---- fig5: stop level --------------------------------------------------------
    e3 = sorted([r for r in rows if r["nu_tag"] == "nu1" and (r["exp"] == "e3" or
                 (r["exp"] == "e1" and abs(r["stop"] - 0.5) < 1e-9 and not r["rep"]))], key=lambda r: r["w_lo"])
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    ax[0].axhline(const["nu1"]["x_budget"], color=INK2, ls="--", lw=1.2)
    ax[0].text(152, const["nu1"]["x_budget"] + 0.002, "constant 300", fontsize=9, color=INK2, ha="right")
    ax[0].plot([r["w_lo"] for r in e3], [r["x_budget"] for r in e3], "-", color=MUTED, lw=1, zorder=1)
    for r in e3:
        ax[0].scatter(r["w_lo"], r["x_budget"], s=90, color=cell_col(r["nvort_burst"]), edgecolor="#fcfcfb",
                      linewidth=2, zorder=3)
        ax[0].annotate(f"{r['nvort_burst']:.0f}", (r["w_lo"], r["x_budget"]), xytext=(0, 9),
                       textcoords="offset points", ha="center", fontsize=9)
    ax[0].axvline(61, color=INK2, lw=1, ls=":")
    ax[0].text(63, ax[0].get_ylim()[1] - 0.004, "vortex onset ~61 rpm", fontsize=9, color=INK2)
    ax[0].set_xlabel("speed during the 0.5 s low period [rpm]")
    ax[0].set_ylabel("conversion (wall-flux budget)")
    ax[0].set_title("A full stop is needed (numbers = cells)")
    ax[1].plot([100 * r["esec_left"] for r in e3], [r["x_budget"] for r in e3], "-", color=MUTED, lw=1, zorder=1)
    for r in e3:
        ax[1].scatter(100 * r["esec_left"], r["x_budget"], s=90, color=cell_col(r["nvort_burst"]),
                      edgecolor="#fcfcfb", linewidth=2, zorder=3)
        ax[1].annotate(f"{r['w_lo']:.0f} rpm", (100 * r["esec_left"], r["x_budget"]), xytext=(6, 4),
                       textcoords="offset points", fontsize=9)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("vortex (secondary-flow) energy left at the end of the low period [% of burst level]")
    ax[1].set_title("The gain needs the vortices to fall below ~1%")
    fig.suptitle("5. Stop-level sweep: the vortices must nearly die before the restart",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    save(fig, "fig5_stop_level.png")

    # ---- fig6: gain decomposition (exp4) ------------------------------------------
    d = json.load(open(EXP4))["decomposition"]
    parts = [("higher wall consumption\nthrough each burst", d["from_plateau_X"]),
             ("overshoot right\nafter restart", d["from_overshoot_X"]),
             ("loss during\nthe stops", d["from_dips_X"])]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    run = 0.0
    for i, (lab, v) in enumerate(parts):
        ax.barh(i, v, left=run, height=0.55,
                color="#2a78d6" if v >= 0 else "#e34948", edgecolor="#fcfcfb", linewidth=2)
        ax.text(max(run, run + v) + 0.002, i, f"{v:+.3f}", va="center", fontsize=10)
        run += v
    ax.barh(len(parts), d["gain_X"], height=0.55, color="#52514e")
    ax.text(d["gain_X"] + 0.002, len(parts), f"{d['gain_X']:+.3f}  total gain over constant", va="center",
            fontsize=10)
    ax.set_yticks(range(len(parts) + 1))
    ax.set_yticklabels([p[0] for p in parts] + ["champion − constant"])
    ax.invert_yaxis()
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlabel("contribution to conversion (wall-flux budget), last 26 s")
    ax.set_xlim(-0.01, 0.115)
    ax.set_axisbelow(True)
    resid = d["gain_X"] - sum(v for _, v in parts)
    fig.text(0.01, -0.04, f"Parts sum to {sum(v for _, v in parts):+.3f}; the {resid:+.3f} difference "
             "comes from how the burst plateau is defined.", fontsize=9, color=INK2)
    fig.suptitle("6. Where the champion's gain comes from (exp 4, existing duty_v3 traces)",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    save(fig, "fig6_gain_decomposition.png")

    # ---- fig7: clocks ------------------------------------------------------------
    rep = json.load(open(os.path.join(RES, "mechanism_report.json")))
    cl = rep["clocks"]
    nts = [n for n in ("nu0p7", "nu1", "nu2") if n in cl]
    nf = np.array([cl[n]["nu_f"] for n in nts])
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    ax[0].plot(nf, [cl[n]["tau_swirl"] for n in nts], "-o", color="#2a78d6", lw=2, ms=8, label="measured")
    ax[0].plot(nf, [cl[n]["pred_swirl"] for n in nts], "--", color=INK2, lw=1.2, label="prediction d²/(π²ν)")
    ax[0].set_title("Swirl decay time follows 1/viscosity, as predicted")
    ax[0].set_ylabel("swirl e-folding time during stops [s]")
    best_cells = {}
    for n in nts:
        rs = [r for r in rows if r["nu_tag"] == n and r["exp"] in ("e1", "e2") and r["stop"] > 0]
        best_cells[n] = max(rs, key=lambda r: r["x_budget"])
    for n in nts:
        b = best_cells[n]
        ax[1].scatter(cl[n]["nu_f"], b["stop"], s=110, color=NU_COL[n], edgecolor="#fcfcfb", linewidth=2,
                      zorder=3)
        ax[1].annotate(f"{b['nvort_burst']:.0f} cells", (cl[n]["nu_f"], b["stop"]), xytext=(8, -3),
                       textcoords="offset points", fontsize=9)
    t1 = best_cells["nu1"]["stop"]
    ff = np.linspace(0.65, 2.1, 30)
    ax[1].plot(ff, t1 / ff, "--", color=INK2, lw=1.2, label="swirl-clock prediction (∝ 1/ν)")
    ax[1].fill_between(ff, t1 * ff ** 0.33, t1 * ff ** 0.69, color="#eda100", alpha=0.25, lw=0,
                       label="wall-layer prediction (∝ ν^0.33-0.69)")
    ax[1].set_title("The best stop follows neither")
    ax[1].set_ylabel("best stop tested [s]")
    ax[1].annotate("= lowest stop tested (edge of grid)", (0.7, best_cells["nu0p7"]["stop"]), xytext=(8, -16),
                   textcoords="offset points", fontsize=8.5, color=INK2)
    ax[1].annotate("= highest stop tested (edge of grid)", (2.0, best_cells["nu2"]["stop"]), xytext=(10, -18),
                   textcoords="offset points", fontsize=8.5, color=INK2, ha="right")
    ax[1].set_ylim(0.12, 3.0)
    for a in ax:
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xticks([0.7, 1.0, 2.0])
        a.set_xticklabels(["0.7×", "1×", "2×"])
        a.set_xlabel("viscosity (D fixed)")
        a.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        a.legend(fontsize=9, loc="upper left")
    ax[0].set_yticks([0.15, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax[1].set_yticks([0.15, 0.2, 0.3, 0.5, 1.0, 2.0])
    fig.suptitle("7. The pre-registered clocks: one behaves as expected, but neither sets the best stop",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    save(fig, "fig7_clocks.png")


if __name__ == "__main__":
    main()
