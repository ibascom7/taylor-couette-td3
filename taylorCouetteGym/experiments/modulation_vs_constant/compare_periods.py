#!/usr/bin/env python3
"""Compare the waveform families measured so far on the Fig. 7 axes:

    constant          (D=1)      from results/summary_table.csv
    pulsed T=25 D=20% (Lopez)    from results/summary_table.csv
    pulsed T=10 D=20%            from results_T10/summary_table.csv
    pulsed T=10 D=15%            from results_T10_D15/summary_table.csv

Families whose summary CSV is missing are skipped, so this can run after any
subset of the sweeps. Writes fig7_period_comparison.png +
conversion_vs_power_comparison.png into the NEWEST family's results dir and
prints the reward table.

Window caveat: X = mean conversion over the last FULL period of each run
(25 s vs 10 s windows; constant rows use the 25 s-window analysis). With the
startup ratchet still climbing at 60 s, sub-0.01 differences between families
with different windows are not meaningful.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# (label, summary csv, mode filter, style) -- newest last; output goes to the
# results dir of the newest family present.
FAMILIES = [
    ("constant", os.path.join(HERE, "results", "summary_table.csv"),
     "constant", dict(ls="-", marker="o", color="k")),
    ("pulsed T=25 D=20%", os.path.join(HERE, "results", "summary_table.csv"),
     "pulsed", dict(ls=":", marker="^", color="#d1495b")),
    ("pulsed T=10 D=20%", os.path.join(HERE, "results_T10", "summary_table.csv"),
     "pulsed", dict(ls="--", marker="s", color="#2e6f95")),
    ("pulsed T=10 D=15%", os.path.join(HERE, "results_T10_D15", "summary_table.csv"),
     "pulsed", dict(ls="-.", marker="D", color="#5f8b4c")),
    ("pulsed T=10 D=30%", os.path.join(HERE, "results_T10_D30", "summary_table.csv"),
     "pulsed", dict(ls=(0, (3, 1, 1, 1, 1, 1)), marker="v", color="#e07a5f")),
    ("pulsed T=10 D=50%", os.path.join(HERE, "results_T10_D50", "summary_table.csv"),
     "pulsed", dict(ls=(0, (5, 1)), marker="P", color="#7b5ea7")),
    ("pulsed T=10 D=60%", os.path.join(HERE, "results_T10_D60", "summary_table.csv"),
     "pulsed", dict(ls=(0, (1, 1)), marker="X", color="#b5651d")),
    ("pulsed T=10 D=70%", os.path.join(HERE, "results_T10_D70", "summary_table.csv"),
     "pulsed", dict(ls=(0, (4, 2)), marker="h", color="#3a7ca5")),
    ("pulsed T=10 D=80%", os.path.join(HERE, "results_T10_D80", "summary_table.csv"),
     "pulsed", dict(ls=(0, (2, 1, 4, 1)), marker="*", color="#9a3b5a")),
]


def load(path, mode):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r["mode"] == mode
                and not int(r.get("diverged", "0") or 0)]
    return sorted(rows, key=lambda r: int(r["wb_rpm"])) or None


fam, out_dir = {}, None
for label, path, mode, style in FAMILIES:
    rows = load(path, mode)
    if rows:
        fam[label] = (rows, style)
        out_dir = os.path.dirname(path)

fig, ax = plt.subplots(figsize=(9.5, 5.5))
for label, (rows, style) in fam.items():
    ax.plot([int(r["wb_rpm"]) for r in rows],
            [float(r["X_conv_lastperiod"]) for r in rows],
            lw=1.8, ms=9, mfc="white", label=label, **style)
ax.set_xlabel("Nominal angular speed $w_b$ [rpm]")
ax.set_ylabel("Conversion  (mean over last full period)")
ax.set_title("Waveform families vs constant -- graded Sc=1075, 60 s episodes")
ax.grid(True, alpha=0.3)
# Legend fully outside the axes so it can never cover data.
ax.legend(title="Rotation", loc="upper left", bbox_to_anchor=(1.02, 1.0),
          borderaxespad=0)
fig.tight_layout()
p = os.path.join(out_dir, "fig7_period_comparison.png")
fig.savefig(p, dpi=140)
plt.close(fig)
print(f"wrote {p}")

fig, ax = plt.subplots(figsize=(10, 5.5))
for label, (rows, style) in fam.items():
    ax.plot([float(r["P_motor_W"]) for r in rows],
            [float(r["X_conv_lastperiod"]) for r in rows],
            lw=1.8, ms=8, mfc="white", label=label, **style)
    for r in rows:
        ax.annotate(r["wb_rpm"], (float(r["P_motor_W"]), float(r["X_conv_lastperiod"])),
                    textcoords="offset points", xytext=(5, 5), fontsize=7,
                    color=style["color"])
ax.set_xlabel("episode-average motor electrical power [W]")
ax.set_ylabel("conversion (mean over last full period)")
ax.set_title("Conversion vs power, all waveform families (labels = $w_b$)")
ax.grid(True, alpha=0.3)
# Legend fully outside the axes so it can never cover data.
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
fig.tight_layout()
p = os.path.join(out_dir, "conversion_vs_power_comparison.png")
fig.savefig(p, dpi=140)
plt.close(fig)
print(f"wrote {p}")

wbs = sorted({int(r["wb_rpm"]) for rows, _ in fam.values() for r in rows})
print(f"\n{'w_b':>5s} | {'family':>18s} | {'X':>7s} | {'P [W]':>6s} | "
      f"{'R- = X-P/Pmax':>13s} | {'R+ = X+P/Pmax':>13s}")
print("-" * 78)
for wb in wbs:
    for label, (rows, _) in fam.items():
        r = next((r for r in rows if int(r["wb_rpm"]) == wb), None)
        if r is None:
            print(f"{wb:>5d} | {label:>18s} |      -- (diverged or missing)")
            continue
        print(f"{wb:>5d} | {label:>18s} | {float(r['X_conv_lastperiod']):7.4f} | "
              f"{float(r['P_motor_W']):6.2f} | {float(r['R_minus (X - P/Pmax)']):13.4f} | "
              f"{float(r['R_plus (X + P/Pmax)']):13.4f}")
