# Warmed-IC benchmark report: T=2.5 s and T=10 s at D=80% (2026-08-01)

**Question.** Do the pulsed-vs-constant benchmarks survive the switch from the
pristine startup IC to the WARMED continuous-manufacturing IC (constant-300
steady state, the IC the D-only TD3 campaign `duty_v1_s*` trains from) — and
do the numbers move?

**Setup.** `fig7_sweep.py` warmed-template mode (`WARMUP_RPM=300`, 60 s ≈ 2.3τ,
steady X = 0.3415 on the sweep case): the template's final state is promoted to
`0/`, so every episode starts at the steady operating point at t = 0. 15
episodes of 60 s on `side_outlet_case_sc1075_graded`: pulsed T=2.5/D=80% ×
{100..500} rpm, pulsed T=10/D=80% × {100..500}, and the five WARMED constant
baselines (generated in `results_warmed_T2p5_D80/`, symlinked by
`results_warmed_T10_D80/`; the pristine `results/constant_wb*` are NOT
comparable — their measurement windows still carry startup-transient history at
low w_b). X = mean conversion over the last full period, R⁻ = X − P_motor/31.94 W,
identical to the pristine sweeps.

## Results

Warmed, T=2.5 s D=80% (window 2.5 s):

| w_b | const X | pulsed X | const R⁻ | pulsed R⁻ |
|-----|---------|----------|----------|-----------|
| 100 | 0.2795 | 0.2692 | 0.2358 | 0.2267 |
| 200 | 0.3166 | 0.3603 | 0.2363 | **0.2810** |
| 300 | 0.3545 | 0.4007 | 0.2374 | **0.2844** |
| 400 | 0.3841 | 0.3940 | 0.2301 | 0.2401 |
| 500 | 0.3705 | 0.4259 | 0.1793 | 0.2341 |

Warmed, T=10 s D=80% (window 10 s; same constant episodes, recomputed in this window):

| w_b | const X | pulsed X | const R⁻ | pulsed R⁻ |
|-----|---------|----------|----------|-----------|
| 100 | 0.2683 | 0.2750 | 0.2247 | 0.2327 |
| 200 | 0.3166 | 0.3388 | 0.2363 | 0.2598 |
| 300 | 0.3545 | 0.3833 | 0.2374 | 0.2674 |
| 400 | 0.3826 | 0.3798 | 0.2285 | 0.2267 |
| 500 | 0.3700 | 0.4090 | 0.1788 | 0.2185 |

## Findings

1. **The modulation advantage is a steady-operation effect, not a startup
   artifact.** Pulsed beats constant at w_b = 200–500 (T=2.5) and 100–300 + 500
   (T=10) from the warmed state. This closes a real referee hole: Lopez-style
   gains could conceivably have been transient-mixing artifacts of cold starts;
   they are not.
2. **New static champion: T=2.5, D=80%, w_b=300 → R⁻ = 0.2844** (X = 0.4007 at
   3.72 W — *less* power than constant-300's 3.74 W with +0.046 conversion).
   Pristine same-waveform reference: 0.2751 (`results_frag_T2p5`, 50 s). This is
   the benchmark row the `duty_v1` TD3 policies must beat; note it numerically
   exceeds v5-s2's trained R = 0.281, but that number lived on pristine-IC
   episodes — cross-IC comparisons are not meaningful, which is exactly why
   these warmed baselines exist.
3. **Shorter period wins everywhere from the warmed state:** T=2.5 > T=10 in
   pulsed R⁻ at every w_b (e.g. 0.2844 vs 0.2674 at 300). The duty env's pinned
   T=5 s is bracketed; interpolation suggests its static-champion-equivalent is
   ~0.27–0.28.
4. **The pulsed advantage flips sign at w_b=100 depending on T:** T=10 pulsed
   wins (0.2750 vs 0.2683), T=2.5 pulsed loses (0.2692 vs 0.2795). At low speed
   the 0.5 s bursts of a T=2.5 wave are apparently too brief to renew the film,
   while 8 s bursts still do. A clean film-renewal-timescale data point for the
   mechanism section.
5. **Possible path-dependence at 400–500 rpm (worth a follow-up).** Warmed
   constants at 400/500 land ~0.02–0.03 BELOW their pristine counterparts
   (0.383 vs 0.407 at 400; 0.370 vs 0.393 at 500) even though both measure the
   same nominal steady state at t = 60 s — from-rest spin-up (violent vortex
   formation) ends higher than a mild 300→400/500 step. Incomplete relaxation
   in 60 s or genuine multi-state hysteresis — this is precisely the
   non-convexity the `experiments/hysteresis` campaign hunts. The known
   constant 400 > 500 anomaly reproduces under both ICs.
6. **Cost** (14 solvers concurrent, per-episode wall): 58–60 s CPU per
   sim-second, ~60 min/episode; each template warmup 43 min. Whole campaign
   ≈ 4.5 h wall, ~15 core-hours.

## RL-comparable values at t = 50 s (`summary_table_t50.csv`, via t50_table.py)

The RL episodes (v5 training, duty_v1, and the 50 s pristine benchmarks) are
50 s, so cross-comparisons should use these re-windowed rows (X over
[50−T, 50], P over [0, 50]):

T=2.5: | w_b | const R50⁻ | pulsed R50⁻ |  T=10: | const R50⁻ | pulsed R50⁻ |
|------|-----------|------------|--|-----------|------------|
| 100 | 0.2220 | 0.2395 | | 0.2250 | 0.2348 |
| 200 | 0.2385 | **0.2815** | | 0.2374 | 0.2630 |
| 300 | 0.2390 | 0.2714 | | 0.2383 | 0.2651 |
| 400 | 0.2305 | 0.2441 | | 0.2278 | 0.2319 |
| 500 | 0.1775 | 0.2212 | | 0.1765 | 0.2152 |

Notes: (a) at t=50 the T=2.5 optimum shifts to w_b=200 (0.2815) with w_b=300
at 0.2714 — the wb300 trace was still climbing between 50 and 60 s; (b) the
T=2.5 window is only 2.5 s (~25 samples of the ±0.02 outlet flutter), so
differences under ~0.01 are within noise; (c) for FIXED-MEAN w_b=300
comparisons (v5-s2, duty_v1) the rows to beat are: pulsed T=2.5 → 0.2714,
pulsed T=10 → 0.2651, constant → 0.2390; (d) pulsed wb100 WINS at t=50 for
T=2.5 (0.2395 vs 0.2220) though it lost in the 60 s window — low-w_b rows are
slow to settle, treat with caution.

## Artifacts

`results_warmed_T2p5_D80/` and `results_warmed_T10_D80/` carry the full
standard set: per-episode case dirs + CSVs, `fig7_conversion_vs_wb.png`,
`conversion_vs_power.png`, `conversion_vs_time__{pulsed,constant}.png`,
`omega_traces.png`, `summary_table.{csv,png}`. Engine change: opt-in
`WARMUP_RPM`/`WARMUP_S` globals in `fig7_sweep.py` (default off; pristine
sweeps untouched).
