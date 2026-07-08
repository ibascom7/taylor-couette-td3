"""Plot mixing index I and energy E per episode from a constant-sweep .out log.

Each episode of the constant-omega env is one independent constant-omega trial,
so the logged I (final mixing index) and E (total energy) are one value per
episode -- i.e. "over time" here means over training episodes.

Usage: python plot_IE_over_time.py <logfile.out> [out.png]
"""
import re
import sys

import matplotlib.pyplot as plt

LINE = re.compile(
    r"ep=(\d+).*?omega=([+-]?[\d.]+)\s+I=([\d.eE+-]+)\s+E=([\d.eE+-]+)"
)


def parse(path):
    eps, omega, I, E = [], [], [], []
    for line in open(path):
        m = LINE.search(line)
        if not m:
            continue
        eps.append(int(m.group(1)))
        omega.append(float(m.group(2)))
        I.append(float(m.group(3)))
        E.append(float(m.group(4)))
    return eps, omega, I, E


# Same normalization as the env: E_norm = E / (E_max_per_step * episode_duration),
# so a "reference" run sits near 1 and E_norm lands in roughly the same 0..1 band as I.
E_MAX_PER_STEP = 0.0011017031875434
EPISODE_DURATION = 60.0
E_REF = E_MAX_PER_STEP * EPISODE_DURATION


def scatter_IvsE(eps, omega, I, E, out):
    """I vs normalized energy, each point one episode, colored by omega."""
    E_norm = [e / E_REF for e in E]

    fig, ax = plt.subplots(figsize=(8.5, 7))

    # Constant-reward lines: reward = -(I + E_norm) -> I + E_norm = const.
    lim = max(max(I), max(E_norm)) * 1.05
    for c in [0.05, 0.1, 0.2, 0.3, 0.4]:
        ax.plot([0, c], [c, 0], color="gray", lw=0.7, ls="--", alpha=0.5)
        if c <= lim:
            ax.text(c * 0.5, c * 0.5, f"r=-{c:g}", color="gray", fontsize=7,
                    rotation=-45, ha="center", va="center", alpha=0.7)

    sc = ax.scatter(E_norm, I, c=omega, cmap="coolwarm", s=45,
                    edgecolor="k", linewidth=0.3, vmin=-300, vmax=300, zorder=3)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("omega (rpm; sign = spin direction)")

    ax.set_xlabel("normalized energy  E_norm = E / (E_max_per_step * dur)")
    ax.set_ylabel("mixing index  I  (0 = fully mixed)")
    ax.set_title("const_base (seed 0): I vs E_norm tradeoff, colored by omega\n"
                 "dashed lines = equal reward; lower-left = better")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/td3/const_base/tc_td3_const_7104782_0.out"
    out = sys.argv[2] if len(sys.argv) > 2 else "IE_over_time_const_base.png"
    eps, omega, I, E = parse(path)
    scatter_IvsE(eps, omega, I, E, out.replace(".png", "_scatter.png"))

    # Both quantities on one shared linear axis in the same ~0..1 band:
    # E_norm = E / (E_max_per_step * episode_duration), matching the env's reward.
    E_norm = [e / E_REF for e in E]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(eps, I, "-o", ms=3, lw=1.2, color="tab:blue", label="I  (mixing index)")
    ax.plot(eps, E_norm, "-s", ms=3, lw=1.2, color="tab:red", alpha=0.85,
            label="E_norm  (= E / (E_max_per_step * dur))")
    ax.set_xlabel("episode")
    ax.set_ylabel("I  and  E_norm   (both ~0..1; 0 = mixed / no energy)")
    ax.set_ylim(0, max(max(I), max(E_norm)) * 1.05)
    ax.set_title("const_base (seed 0): mixing index I and normalized energy E_norm per episode")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}  ({len(eps)} episodes)")

    # Quick numeric summary of the last 20 episodes (converged region).
    tail = slice(-20, None)
    import statistics as st
    print(f"last 20 ep: I mean={st.mean(I[tail]):.4f}  E mean={st.mean(E[tail]):.2e}  "
          f"omega mean={st.mean(omega[tail]):.1f} rpm")


if __name__ == "__main__":
    main()