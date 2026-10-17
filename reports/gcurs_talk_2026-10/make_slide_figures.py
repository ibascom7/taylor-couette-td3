#!/usr/bin/env python3
"""Slide figures for the GCURS talk (12-min slideshow, 2026-10).

No titles inside the figures (the slide heading carries the message), large
fonts, wide 16:9-friendly shapes. Reads committed results:
    taylorCouetteGym/experiments/mechanism/results/main            (Carya job 8814106)
    taylorCouetteGym/experiments/mechanism/results/exp4
    taylorCouetteGym/experiments/mechanism_mesh/results/local       (mesh check)
and, for the vortex snapshot, the saved fine-mesh fields at t = 90 s of
mechanism_mesh/results/local/runs/{f1.41_s0p00, f1.41_s0p40} (needs OpenFOAM
on PATH for postProcess -func writeCellCentres; skipped if unavailable).

USAGE (from the repo root):  python3 reports/gcurs_talk_2026-10/make_slide_figures.py
"""

import importlib.util
import json
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
GYM = os.path.join(REPO, "taylorCouetteGym")
MECH = os.path.join(GYM, "experiments", "mechanism")
MESH = os.path.join(GYM, "experiments", "mechanism_mesh")
OUT = os.path.join(HERE, "figures")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load("mechanism_sweep", os.path.join(MECH, "mechanism_sweep.py"))
MC = _load("mesh_check", os.path.join(MESH, "mesh_check.py"))

# validated palette (dataviz reference)
INK, INK2, MUTED, GRID, BG = "#1b1b19", "#52514e", "#8a8984", "#e4e3df", "#fbfbf8"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
CELL = {4: "#86b6ef", 6: "#2a78d6", 8: "#104281"}
NEUTRAL = "#f0efec"

plt.rcParams.update({
    "font.size": 22, "axes.labelsize": 24, "xtick.labelsize": 21, "ytick.labelsize": 21,
    "legend.fontsize": 20, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "lines.linewidth": 3,
})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


def roll_median(y, n):
    pad = np.pad(y, (n // 2, n - 1 - n // 2), mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(pad, n), axis=1)


def label_cells(ax, x, y, n, color=INK, dy=14):
    ax.annotate(f"{n:.0f}", (x, y), xytext=(0, dy), textcoords="offset points", ha="center",
                fontsize=20, color=color, fontweight="bold")


# --------------------------------------------------------------------------- #
# 1. vortex snapshot: 4 cells (constant) vs 8 cells (pulsed, mid-burst)
# --------------------------------------------------------------------------- #
def read_vec(path):
    txt = open(path).read()
    i = txt.index("internalField")
    m = re.search(r"nonuniform\s+List<vector>\s*\n?(\d+)\s*\(", txt[i:])
    n = int(m.group(1))
    body = txt[i + m.end():]
    return np.array(re.findall(r"\(([-\deE.+]+)\s+([-\deE.+]+)\s+([-\deE.+]+)\)", body[:n * 90])[:n], float)


def snapshot():
    runs = os.path.join(MESH, "results", "local", "runs")
    cases = [("f1.41_s0p00", "Steady rotation, 300 rpm"), ("f1.41_s0p40", "Pulsed (0.4 s stops), mid-burst")]
    if shutil.which("postProcess") is None or not all(os.path.isfile(os.path.join(runs, c, "90", "U")) for c, _ in cases):
        print("  snapshot skipped (needs OpenFOAM and the saved t=90 fields)")
        return
    tmp = tempfile.mkdtemp()
    src = os.path.join(runs, cases[0][0])
    for d in ("constant", "system"):
        shutil.copytree(os.path.join(src, d), os.path.join(tmp, d))
    os.makedirs(os.path.join(tmp, "90"))
    shutil.copy(os.path.join(src, "90", "U"), os.path.join(tmp, "90", "U"))
    subprocess.run(["postProcess", "-func", "writeCellCentres", "-time", "90"], cwd=tmp,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    C = read_vec(os.path.join(tmp, "90", "C"))
    shutil.rmtree(tmp)
    r = np.hypot(C[:, 0], C[:, 1])
    z = C[:, 2]
    rs = np.unique(np.round(r, 9))
    zs = np.unique(np.round(z, 9))
    ir = np.searchsorted(rs, np.round(r, 9))
    iz = np.searchsorted(zs, np.round(z, 9))
    er = np.c_[C[:, 0] / r, C[:, 1] / r]
    Ri, Ro, z0 = 0.0254, 0.03175, -0.01905
    fields = []
    for tag, _ in cases:
        U = read_vec(os.path.join(runs, tag, "90", "U"))
        ur = U[:, 0] * er[:, 0] + U[:, 1] * er[:, 1]
        UR = np.full((len(zs), len(rs)), np.nan)
        UZ = np.full_like(UR, np.nan)
        UR[iz, ir] = ur
        UZ[iz, ir] = U[:, 2]
        fields.append((UR, UZ))
    vmax = np.nanpercentile(np.abs(np.concatenate([f[0].ravel() for f in fields])), 98)
    cmap = LinearSegmentedColormap.from_list("div", [BLUE, NEUTRAL, RED])
    # uniform grid for streamlines: x = height along the reactor, y = across the gap
    xg = np.linspace(zs[0], zs[-1], 400)
    yg = np.linspace(rs[0], rs[-1], 60)
    fig, axes = plt.subplots(2, 1, figsize=(18, 7.6), sharex=True)
    for ax, (tag, title), (UR, UZ) in zip(axes, cases, fields):
        X, Y = (zs - z0) * 1e3, (rs - Ri) * 1e3
        ax.grid(False)
        pc = ax.pcolormesh(X, Y, UR.T * 1e3, cmap=cmap, vmin=-vmax * 1e3, vmax=vmax * 1e3, shading="nearest",
                           rasterized=True)
        # interpolate (r rows) onto the uniform grid for streamplot
        URi = np.array([np.interp(yg, rs, UR[j]) for j in range(len(zs))])
        UZi = np.array([np.interp(yg, rs, UZ[j]) for j in range(len(zs))])
        URg = np.array([np.interp(xg, zs, URi[:, k]) for k in range(len(yg))])
        UZg = np.array([np.interp(xg, zs, UZi[:, k]) for k in range(len(yg))])
        ax.streamplot((xg - z0) * 1e3, (yg - Ri) * 1e3, UZg, URg, color=INK, linewidth=0.9, density=(4.5, 0.9),
                      arrowsize=0.9)
        ax.set_aspect("equal")
        ax.set_xlim(0, (zs[-1] - z0) * 1e3 + 0.2)
        ax.set_ylim(0, (Ro - Ri) * 1e3)
        ax.grid(False)
        ax.set_yticks([0, 6.35])
        ax.set_yticklabels(["inner\nwall", "catalytic\nwall"], fontsize=17)
        n = 4 if "s0p00" in tag else 8
        ax.set_title(f"{title}:  {n} Taylor cells", loc="left", fontsize=24, color=INK, pad=10)
    axes[1].set_xlabel("height along the reactor [mm]   (outlet at left, inlet at right)")
    cb = fig.colorbar(pc, ax=axes, shrink=0.85, pad=0.015)
    cb.set_label("radial velocity [mm/s]\n(red: toward catalytic wall)", fontsize=19)
    cb.ax.tick_params(labelsize=17)
    save(fig, "s_cells_snapshot.png")


# --------------------------------------------------------------------------- #
# data for the rest
# --------------------------------------------------------------------------- #
RES = os.path.join(MECH, "results", "main")
rows = [r for r in M.load_all(RES, False) if r["ok"]]
by = {r["tag"]: r for r in rows}
const = {nt: next(r for r in rows if r["nu_tag"] == nt and r["exp"] in ("e1", "e2") and r["stop"] == 0)
         for nt in M.NU}


def cell_color(n):
    return CELL.get(int(round(n)), CELL[min(CELL, key=lambda k: abs(k - n))])


def stop_curve():
    rs = sorted([r for r in rows if r["nu_tag"] == "nu1" and r["exp"] == "e1" and r["stop"] > 0 and not r["rep"]],
                key=lambda r: r["stop"])
    fig, ax = plt.subplots(figsize=(16, 7.5))
    c = const["nu1"]
    ax.axhline(c["x_budget"], color=INK2, ls="--", lw=2)
    ax.text(2.05, c["x_budget"] + 0.0015, "steady rotation (4 cells)", fontsize=20, color=INK2, ha="right")
    ax.plot([r["stop"] for r in rs], [r["x_budget"] for r in rs], "-", color=MUTED, lw=1.5, zorder=1)
    for r in rs:
        ax.scatter(r["stop"], r["x_budget"], s=260, color=cell_color(r["nvort_burst"]), edgecolor=BG, lw=2.5,
                   zorder=3)
        label_cells(ax, r["stop"], r["x_budget"], r["nvort_burst"])
    for n in (6, 8):
        ax.scatter([], [], s=260, color=CELL[n], label=f"{n} cells during bursts")
    ax.legend(loc="upper right")
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0])
    ax.set_xticklabels(["0.1", "0.2", "0.3", "0.5", "0.7", "1", "2"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("stop length [s]   (4.5 s bursts, 300 rpm average)")
    ax.set_ylabel("conversion")
    ax.set_ylim(0.335, 0.395)
    save(fig, "s_stop_curve_cells.png")


def cycle():
    fig, ax = plt.subplots(2, 1, figsize=(16, 8.5), sharex=True)
    for tag, lab, col in (("e1_nu1_s0p00", "steady 300 rpm", INK2), ("e1_nu1_s0p50", "0.5 s stops", BLUE)):
        r = by[tag]
        m = M.read_mech(r["workdir"])
        t0 = 120.0 if r["stop"] == 0 else r["win_b"] - 2 * r["period"]
        sel = (m["t"] >= t0) & (m["t"] <= t0 + 10.0)
        tt = m["t"][sel] - t0
        ax[0].step(tt, m["nVort"][sel], where="post", color=col, lw=2.5, label=lab)
        ax[1].plot(tt, m["wall"][sel] / const["nu1"]["wall_mean"], color=col, lw=3, label=lab)
    ax[0].set_ylabel("Taylor cells")
    ax[0].set_yticks([2, 4, 6, 8, 10, 12])
    ax[0].legend(loc="upper right", ncol=2)
    ax[1].set_ylabel("wall reaction rate\n(÷ steady)")
    ax[1].set_xlabel("time [s]")
    for x in (4.5, 9.5):
        for a in ax:
            a.axvspan(x, x + 0.5, color=GRID, alpha=0.8, lw=0)
    ax[0].text(4.75, 11.3, "stop", ha="center", fontsize=19, color=INK2)
    save(fig, "s_one_cycle.png")


def hold_kick():
    fig, ax = plt.subplots(2, 1, figsize=(16, 8.5), sharex=True)
    for tag, lab, col in (("e1_nu1_s0p50", "pulsing throughout", BLUE),
                          ("e5_nu1_hold", "pulsing, then steady after 60 s", ORANGE),
                          ("e5_nu1_kick", "steady with one stop at 60 s", AQUA),
                          ("e1_nu1_s0p00", "steady throughout", INK2)):
        m = M.read_mech(by[tag]["workdir"])
        sel = m["t"] >= 40
        ax[0].plot(m["t"][sel], roll_median(m["nVort"], 101)[sel], color=col, lw=3, label=lab)
        ax[1].plot(m["t"][sel], roll_median(m["xOut"], 101)[sel], color=col, lw=2.5, label=lab)
    for a in ax:
        a.axvline(60, color=INK2, lw=1.5, ls=":")
    ax[0].set_ylabel("Taylor cells")
    ax[0].set_yticks([4, 6, 8])
    ax[0].legend(loc="center right", bbox_to_anchor=(1.0, 0.36), fontsize=18)
    ax[1].set_ylabel("conversion\n(2 s median)")
    ax[1].set_xlabel("time [s]")
    save(fig, "s_hold_kick.png")


def stop_level():
    e3 = sorted([r for r in rows if r["nu_tag"] == "nu1" and (r["exp"] == "e3" or
                 (r["exp"] == "e1" and abs(r["stop"] - 0.5) < 1e-9 and not r["rep"]))], key=lambda r: r["w_lo"])
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axhline(const["nu1"]["x_budget"], color=INK2, ls="--", lw=2)
    ax.text(150, const["nu1"]["x_budget"] + 0.002, "steady rotation", fontsize=20, color=INK2, ha="right")
    ax.plot([r["w_lo"] for r in e3], [r["x_budget"] for r in e3], "-", color=MUTED, lw=1.5, zorder=1)
    for r in e3:
        ax.scatter(r["w_lo"], r["x_budget"], s=260, color=cell_color(r["nvort_burst"]), edgecolor=BG, lw=2.5,
                   zorder=3)
        label_cells(ax, r["w_lo"], r["x_budget"], r["nvort_burst"])
    ax.axvline(61, color=INK2, lw=1.5, ls=":")
    ax.text(64, 0.392, "vortices form\nabove ~61 rpm", fontsize=19, color=INK2, va="top")
    ax.set_xlabel("speed during the 0.5 s \"stop\" [rpm]")
    ax.set_ylabel("conversion")
    ax.set_ylim(0.315, 0.395)
    save(fig, "s_stop_level.png")


def gain_breakdown():
    d = json.load(open(os.path.join(MECH, "results", "exp4", "exp4_report.json")))["decomposition"]
    parts = [("higher reaction rate\nthroughout each burst", d["from_plateau_X"], BLUE),
             ("spike right after restart", d["from_overshoot_X"], BLUE),
             ("lost during the stops", d["from_dips_X"], RED)]
    fig, ax = plt.subplots(figsize=(15, 6))
    run = 0.0
    for i, (lab, v, col) in enumerate(parts):
        ax.barh(i, v, left=run, height=0.6, color=col)
        ax.text(max(run, run + v) + 0.002, i, f"{v:+.3f}", va="center", fontsize=22)
        run += v
    ax.barh(3, d["gain_X"], height=0.6, color=INK2)
    ax.text(d["gain_X"] + 0.002, 3, f"{d['gain_X']:+.3f}", va="center", fontsize=22, fontweight="bold")
    ax.set_yticks(range(4))
    ax.set_yticklabels([p[0] for p in parts] + ["net gain over steady"], fontsize=21)
    ax.invert_yaxis()
    ax.axvline(0, color=INK2, lw=1.5)
    ax.set_xlabel("change in conversion")
    ax.set_xlim(-0.005, 0.1)
    ax.set_axisbelow(True)
    ax.grid(axis="y", visible=False)
    save(fig, "s_gain_breakdown.png")


def clocks():
    rep = json.load(open(os.path.join(RES, "mechanism_report.json")))["clocks"]
    nts = ["nu0p7", "nu1", "nu2"]
    nf = np.array([rep[n]["nu_f"] for n in nts])
    best = {n: max([r for r in rows if r["nu_tag"] == n and r["exp"] in ("e1", "e2") and r["stop"] > 0],
                   key=lambda r: r["x_budget"]) for n in nts}
    fig, ax = plt.subplots(1, 2, figsize=(18, 7))
    ax[0].plot(nf, [rep[n]["tau_swirl"] for n in nts], "-o", color=BLUE, ms=12, label="measured in CFD")
    ax[0].plot(nf, [rep[n]["pred_swirl"] for n in nts], "--", color=INK2, lw=2, label="theory  d²/(π²ν)")
    ax[0].set_ylabel("swirl decay time [s]")
    ax[0].legend(loc="upper right")
    t1 = best["nu1"]["stop"]
    ff = np.linspace(0.65, 2.1, 30)
    ax[1].plot(ff, t1 / ff, "--", color=INK2, lw=2, label="if swirl decay set it")
    ax[1].fill_between(ff, t1 * ff ** 0.33, t1 * ff ** 0.69, color="#eda100", alpha=0.3, lw=0,
                       label="if the wall layer set it")
    for n, col in zip(nts, (BLUE, ORANGE, AQUA)):
        ax[1].scatter(rep[n]["nu_f"], best[n]["stop"], s=300, color=col, edgecolor=BG, lw=2.5, zorder=3)
        ax[1].annotate(f"{best[n]['nvort_burst']:.0f} cells", (rep[n]["nu_f"], best[n]["stop"]), xytext=(14, -6),
                       textcoords="offset points", fontsize=20)
    ax[1].set_ylabel("best stop length [s]")
    ax[1].set_ylim(0.12, 3.0)
    ax[1].legend(loc="upper left", fontsize=19)
    for a in ax:
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xticks([0.7, 1.0, 2.0])
        a.set_xticklabels(["0.7×", "1×", "2×"])
        a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        a.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        a.set_xlabel("liquid viscosity")
    ax[0].set_yticks([0.2, 0.3, 0.4, 0.5])
    ax[1].set_yticks([0.2, 0.3, 0.5, 1.0, 2.0])
    save(fig, "s_clocks.png")


def mesh():
    res = os.path.join(MESH, "results", "local")
    allr = MC.reference_runs() + MC.load_rows(res, False)
    fig, ax = plt.subplots(figsize=(16, 7.5))
    styles = {"f0.71": ("coarse mesh (1.7k cells)", "#86b6ef"), "f1.00": ("production mesh (3.3k)", INK2),
              "f1.41": ("fine mesh (6.6k)", ORANGE)}
    for m, (lab, col) in styles.items():
        rs = sorted([q for q in allr if q["mesh"] == m and q.get("kind", "wave") == "wave" and q["ok"]],
                    key=lambda q: q["stop"])
        c = next(q for q in rs if q["stop"] == 0)
        pul = [q for q in rs if q["stop"] > 0]
        gains = [q["x_budget"] - c["x_budget"] for q in pul]
        ax.plot([q["stop"] for q in pul], gains, "-", color=col, label=lab, zorder=2)
        for q, g in zip(pul, gains):
            eight = q["nvort_burst"] >= 7.5
            ax.scatter(q["stop"], g, s=230, color=col if eight else BG, edgecolor=col, lw=3, zorder=3)
    ax.scatter([], [], s=230, color=INK2, edgecolor=INK2, lw=3, label="8 cells (filled)")
    ax.scatter([], [], s=230, color=BG, edgecolor=INK2, lw=3, label="6 cells (open)")
    ax.axhline(0, color=INK2, lw=1.5)
    ax.set_xscale("log")
    ax.set_xticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.2])
    ax.set_xticklabels(["0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "1.2"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("stop length [s]")
    ax.set_ylabel("gain over steady rotation")
    ax.set_ylim(0, 0.072)
    ax.legend(loc="lower center", ncol=3, fontsize=18)
    save(fig, "s_mesh_check.png")


def viscosity_gain():
    fig, ax = plt.subplots(figsize=(14, 7))
    rng = np.random.default_rng(0)
    for nt, col, lab in (("nu0p7", BLUE, "0.7× viscosity"), ("nu1", ORANGE, "nominal"), ("nu2", AQUA, "2× viscosity")):
        rs = [r for r in rows if r["nu_tag"] == nt and r["exp"] in ("e1", "e2") and r["stop"] > 0]
        c = const[nt]
        xs = np.array([r["nvort_burst"] - c["nvort_burst"] for r in rs]) + rng.uniform(-0.15, 0.15, len(rs))
        ax.scatter(xs, [r["x_budget"] - c["x_budget"] for r in rs], s=170, color=col, edgecolor=BG, lw=2,
                   label=f"{lab} (steady: {c['nvort_burst']:.0f} cells)", zorder=3)
    ax.axhline(0, color=INK2, lw=1.5)
    ax.set_xticks([0, 2, 4])
    ax.set_xticklabels(["0", "+2", "+4"])
    ax.set_xlabel("Taylor cells added by pulsing")
    ax.set_ylabel("gain over steady rotation")
    ax.legend(loc="upper left", fontsize=19)
    save(fig, "s_cells_added_gain.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    snapshot()
    stop_curve()
    cycle()
    hold_kick()
    stop_level()
    gain_breakdown()
    clocks()
    mesh()
    viscosity_gain()


if __name__ == "__main__":
    main()
