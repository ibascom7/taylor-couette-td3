# Hysteresis / bistability — *why* modulation can beat constant

Goal: find **where conversion is bistable or hysteretic in the conversion-vs-power
plane**, and use that to explain the mechanism by which a **time-modulated** inner
speed beats a **constant** one at the same mean power.

## The logic (why this proves the modulation claim)

A constant speed sits at one point on the steady **conversion-vs-power** curve
`C(P)`. A time-modulation between two speeds reaches the **convex hull** of that
curve: holding speed-A a fraction `f` of the time and speed-B the rest gives mean
power `f·P_A+(1−f)·P_B` and (quasi-statically) mean conversion `f·C_A+(1−f)·C_B` —
a straight line between the two points. So:

- where `C(P)` is **concave** (diminishing returns), the curve is already above its
  chords → **constant wins**, modulation cannot help;
- where `C(P)` is **non-convex** (a steep jump / an S-shape / a hysteresis loop),
  a chord rises **above** the curve → **modulation wins at equal power**, by exactly
  the hull-minus-curve gap.

**Hysteresis is a sufficient cause of non-convexity**: a loop is non-convex, and if
the high-transport state *persists* when you lower the speed (memory), a brief burst
buys high conversion at low average power. That is the strongest form of the
mechanism. So: *find the loop / the non-convex band → that is where, and how much,
modulation beats constant.* `hysteresis_sweep.py` computes this gap (the green
"modulation frontier" in `fig_conv_vs_power.png`) automatically.

## Three protocols (each proves more than the last)

### 1. Up/down quasi-static loop — *is there a loop, and where?*
Ramp omega UP a Reynolds ladder and back DOWN, settling at each level. If the up
and down branches differ, conversion is **path-dependent** in that Re band. Plots:
`fig_conv_vs_Re_loop.png`, `fig_conv_vs_power.png` (with the convex hull +
modulation tie-line + predicted gain), `fig_wf_vs_Re_loop.png`,
`fig_mechP_vs_Re_loop.png`.

Two metrics are recorded because they resolve the loop at different speeds:
- **conv** (outlet conversion) — the reported figure of merit, but it lags omega by
  the residence time `τ_res = H/u_ax ≈ 26 s`, so it needs a long settle to be steady.
- **wf_norm** (wall consumption / feed) — a conversion-equivalent that tracks the
  **flow state** on the boundary-layer timescale (seconds); the loop shows here with
  a much shorter settle. (This is the same fast signal the RL reward uses.)
- **mech_P** (viscous power `ρ|Mz|ω`) depends on the flow state, so a loop in
  `fig_mechP_vs_Re_loop.png` is the clearest fingerprint of genuine flow-state
  bistability (the motor-power model is a function of omega alone and can't show it).

### 2. Rate dependence — *is it TRUE hysteresis or just transient lag?*
A finite ramp rate **always** makes a loop (conversion lagging omega). To tell a
genuine **equilibrium** loop from a **dynamic** one, run protocol 1 at several
settle times (the slurm arrays over `SETTLE_TIMES=(15 40 90)`), then:

```
python analyze_hysteresis.py results/h15 results/h40 results/h90
```

`fig_loop_area_vs_rate.png` plots loop area vs settle time:
- area → a **nonzero plateau** ⇒ **equilibrium hysteresis / bistability** (strong claim);
- area → **0** ⇒ **dynamic hysteresis** (transient lag only).

Both still let modulation beat constant (dynamic hysteresis is the paper's
boundary-layer-thinning mechanism); only the plateau is genuine bistability. Stating
which one you found is the honest result.

### 3. Fixed-Re two-IC probe — *the rigorous bistability test*
At a fixed target speed, settle from a **cold** start and from a **hot** (pre-spun)
start. Two different settled conversions at the **same** Re ⇒ two coexisting
attractors ⇒ genuine bistability. Enabled with `--probe-rpm` (the slurm runs it on
the longest-settle task): `bistability_probe.csv`, `fig_bistability_probe.png`.

## ⚠️ Will the wedge show it?

The first Couette→Taylor transition at η=0.8 is **supercritical** (no equilibrium
hysteresis), and the wedge is **2D-axisymmetric** (1 azimuthal cell), so it cannot
represent the **wavy-vortex** transitions where TC bistability classically lives.
The most likely wedge outcome is **dynamic hysteresis + a non-convex jump at the
Taylor-vortex onset** — which still demonstrates the modulation advantage, just via
dynamics/convexity rather than equilibrium multistability.

For the genuinely bistable (wavy/turbulent) regimes, run the **same scripts** on the
full 360° annulus `taylor_couette_mixing/cases/full_tc_mixing_case`: configure it
with `make_case.py --geometry full3d`, and pass `--azimuthal-fraction 1.0` plus that
case's `--r-in/--r-out`. It is ~30× costlier per simulated second.

## Run it

```bash
sbatch experiments/hysteresis/run_carya_hysteresis.slurm        # array: settle 15/40/90 s
python experiments/hysteresis/analyze_hysteresis.py experiments/hysteresis/results/h*
```

Single up/down loop, manual:

```bash
python experiments/hysteresis/hysteresis_sweep.py \
    --case <configured_catalysis_case> --out results/h40 \
    --rpm 40,60,80,100,120,160,220,300,500,800,1200,1800,2500 \
    --settle-seconds 40 --probe-rpm 100,160,300,500
```

Cost: up/down is `2 × n_levels × settle` simulated seconds; the residence lag means
conv only fully settles at `settle ≳ 2–3·τ_res ≈ 60–80 s`, so the long-settle task
dominates wall time (24 h walltime is set for headroom). The `wf_norm` metric is
trustworthy at much shorter settle, so read the loop there first.

## Validating the prediction

`hysteresis_sweep.py` writes `modulation_estimate.txt`: the predicted max conversion
gain, the power it occurs at, and the optimal two-speed time-share (duty + speeds).
**Confirm it with an actual square wave** at that mean power — the existing
`experiments/catalysis_rl/compare_catalysis.py` already produces the
pulsating-vs-constant power-vs-conversion figure (Fig. 11 style); this experiment
*explains* that gap by locating the non-convex / hysteretic band that creates it.
The duty/period the hull predicts is also a strong CMA-ES / TD3 warm-start.

## Outputs (per run, under `results/<TAG>/`)

| file | what |
|------|------|
| `hysteresis_branches.csv` | per-level up & down: rpm, Re, Ta, Re_axial, conv, wf_norm, Mz_kin, mech_P, motor_P, time window |
| `fig_conv_vs_power.png` | **the figure** — conv vs motor power, up/down, convex hull + modulation tie-line + predicted gain |
| `fig_conv_vs_Re_loop.png` | conversion loop in the control variable |
| `fig_wf_vs_Re_loop.png` | fast flow-state loop (wall flux) |
| `fig_mechP_vs_Re_loop.png` | viscous-power loop (flow-state bistability fingerprint) |
| `fig_Re_vs_rpm.png` | Re vs speed with Couette / Taylor-vortex regime bands + catalysis landmarks |
| `fig_Re_staircase.png` | Re vs time — the up-then-down protocol |
| `modulation_estimate.txt` | predicted gain + optimal duty/speeds |
| `bistability_probe.csv`, `fig_bistability_probe.png` | two-IC test (if `--probe-rpm`) |
| `frames_loop/` | the **whole up→down trajectory** as a ParaView animation (1 frame/s) |
| `frames_manifest.csv` | maps each `frames_loop/` time → branch + Re (so the time slider is interpretable) |
| `frames_by_level/<branch>_Re#####_rpm####/` | one labeled **settled still per (branch, Re)** — for montages + matched-Re up/down stills |
| `frames_probe/rpm####_{cold,hot}/` | each two-IC settling trajectory (if `--probe-rpm`) — load the pair to *see* bistability |
| `fig_loop_area_vs_rate.png`, `fig_branches_overlay.png` | from `analyze_hysteresis.py` across runs |

## Visualizing the frames

Everything the OpenFOAM ParaView reader needs is in each `*.foam` folder (mesh + `U`,
`p`, `C`, `phi`). Three views, in order of payoff:

1. **The loop animation** — open `frames_loop/frames_loop.foam`, color by **U → Z**
   (axial velocity: Taylor cells = alternating up/down bands) or compute **vorticity**
   / **Q-criterion** (Filters → Gradient of U). Play it: vortices switch on during the
   up-ramp; whether they *persist* / sit differently on the way down at the same time-
   mirrored Re is the hysteresis, seen directly. `frames_manifest.csv` tells you which
   time is which Re/branch.
2. **Matched-Re up-vs-down still** — open the two `frames_by_level/up_Re00471_*` and
   `down_Re00471_*` folders together (same Re, both branches) and compare the fields
   side by side. A visible difference = path dependence at that Re.
3. **Bistability money shot** — open `frames_probe/rpm0300_cold` and
   `frames_probe/rpm0300_hot` together: same final speed, two initial conditions. If
   they relax to **different** vortex states / dye fields, that Re is genuinely bistable.

Tip: in ParaView use a fixed color-range and the same camera across folders so stills
are comparable; `C` (the catalytic scalar) also makes the vortices and the wall
boundary layer pop.

### Just want the regime visualization?

Run `hysteresis_sweep.py --direction up` (no down branch ⇒ ~half the cost). You still
get `frames_loop/`, the labeled `frames_by_level/` stills, `fig_Re_vs_rpm.png` (regime
bands + landmarks) and `fig_Re_staircase.png` — i.e. the clean Couette→Taylor montage —
without paying for the hysteresis loop. (This subsumes the former standalone
`reynolds_sweep` experiment.)
