"""Mixing index (intensity of segregation) vs time for Yuhe's case.

The repo's RL env scores mixing with the intensity of segregation over radial bins
at a measurement plane. Yuhe's case is flow-through, but we can apply the SAME
metric to a radial concentration profile at each written time. We sample at
MID-HEIGHT (z≈0), where the catalytic outer wall (c=0) sets up a real radial
gradient (high reactant near the inner wall, 0 at the outer wall) that the rotation
must mix against -- so the metric is meaningful. (At the very bottom outlet band the
profile borders the side-outlet, not the catalytic wall, so I_mix≈0 trivially.)

cell ordering (blockMesh): cells 0..14 = bottom side-outlet band; cells 15..1349 =
main section, 89 z-layers of 15 radial cells (radial 0=inner ... 14=outer wall).

I_mix = σ²/σ²_max  (radius-weighted variance of normalized C over the 15 radial cells).
I_mix → 0 radially uniform (well mixed),  → 1 fully segregated.

Run:  python mixing_index_vs_time.py CASE_DIR OUT_DIR
"""
import sys
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Ri, Ro = 0.0254, 0.03175
NR = 15             # radial cells per z-layer
NTOTAL = 1350       # total cells
MIDLAYER = 44       # z-layer in the main section sampled for the radial profile (z≈0)
C0 = 50.0


def read_internal(path, n):
    """Return the first n internalField scalars from an OpenFOAM ascii volScalarField."""
    txt = Path(path).read_text()
    m = re.search(r"internalField\s+nonuniform\s+List<scalar>\s*\n\s*(\d+)\s*\n\(", txt)
    if not m:
        # uniform field
        mu = re.search(r"internalField\s+uniform\s+([-\d.eE+]+)", txt)
        if mu:
            return np.full(n, float(mu.group(1)))
        raise ValueError(f"cannot parse {path}")
    start = m.end()
    vals = []
    for line in txt[start:].splitlines():
        line = line.strip()
        if not line:
            continue          # skip the blank line right after "("
        if line.startswith(")"):
            break
        try:
            vals.append(float(line))
        except ValueError:
            break
        if len(vals) >= n:
            break
    return np.array(vals[:n])


def midplane_profile(path):
    """Radial profile (15 cells, inner->outer) at the mid-height layer."""
    allc = read_internal(path, NTOTAL)
    if len(allc) < NTOTAL:                      # uniform field (e.g. 0/)
        return np.full(NR, allc[0] if len(allc) else 0.0)
    s = NR + MIDLAYER * NR                       # skip bottom band + MIDLAYER layers
    return allc[s:s + NR]


def mixing_index(C):
    """Env's intensity of segregation on a radial profile C (normalized [0,1])."""
    dr = (Ro - Ri) / len(C)
    r_mids = Ri + (np.arange(len(C)) + 0.5) * dr
    w = r_mids / r_mids.sum()
    Cbar = np.sum(w * C)
    sig2 = np.sum(w * (C - Cbar) ** 2)
    sig2_max = Cbar * (1 - Cbar) + 1e-16
    return sig2 / sig2_max, Cbar


def main():
    case = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
    times = sorted(
        (float(p.name) for p in case.iterdir()
         if p.is_dir() and re.fullmatch(r"\d+(\.\d+)?", p.name) and (p / "c").exists()),
    )
    ts, Im, Cb = [], [], []
    for t in times:
        cfile = case / (f"{int(t)}" if t.is_integer() else f"{t:g}") / "c"
        C = midplane_profile(cfile) / C0
        I, cbar = mixing_index(C)
        ts.append(t); Im.append(I); Cb.append(cbar)
    ts, Im, Cb = np.array(ts), np.array(Im), np.array(Cb)

    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(ts, Im, "o-", ms=3, color="C4", label="mixing index $I_{mix}$")
    ax.set_xlabel("time  [s]"); ax.set_ylabel("$I_{mix}=\\sigma^2/\\sigma^2_{max}$", color="C4")
    ax.tick_params(axis="y", labelcolor="C4"); ax.grid(alpha=0.3); ax.set_ylim(bottom=0)
    ax2 = ax.twinx()
    ax2.plot(ts, Cb, "s-", ms=2.5, color="C2", alpha=0.6, label="mean conc. $\\bar C/c_0$")
    ax2.set_ylabel("$\\bar C / c_0$", color="C2"); ax2.tick_params(axis="y", labelcolor="C2")
    ax.set_title("Mid-height radial mixing index (intensity of segregation) vs time "
                 "— rotation mixes reactant against the catalytic outer wall")
    # mark the bursts so the modulation effect on mixing is visible
    for k in range(int(ts.max() // 20) + 1):
        ax.axvspan(k * 20, k * 20 + 4, color="grey", alpha=0.08)
    fig.tight_layout(); fig.savefig(out / "mixing_index_vs_time.png", dpi=140); plt.close(fig)

    print(f"I_mix (mid-height): start {Im[0]:.4f}, min {Im.min():.4f}, "
          f"max {Im.max():.4f}, end {Im[-1]:.4f}")
    print(f"mid-height mean C/c0: start {Cb[0]:.3f}, end {Cb[-1]:.3f}")
    print(f"wrote {out/'mixing_index_vs_time.png'}")


if __name__ == "__main__":
    main()
