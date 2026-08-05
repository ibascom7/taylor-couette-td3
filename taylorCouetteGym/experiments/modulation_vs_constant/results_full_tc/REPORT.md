# Full-height reactor (Γ = 30) — pulsed vs constant, final report

Run finished 2026-07-18 (~23 h wall, 218 core-hours, 10 episodes × 300 s in
parallel). All 10 episodes completed to t = 300 s with no divergences, and the
omega-trace guard confirms the wall followed the commanded tables (measured
peaks = 1.25·w_b for pulsed, e.g. 375 rpm at w_b = 300, 625 at w_b = 500 —
no freeze bug). Data: `summary_table.{csv,png}`, per-episode
`*_timeseries.csv` / `*_power.csv`, figures listed at the end.

## Verdict

**In the paper's amortization regime (τ/T ≈ 13), pulsed modulation
(D = 80 %, T = 10 s) beats constant rotation at EVERY nominal speed — in
conversion, in reward, and at equal power.** This is the qualitative Fig. 7
picture Lopez-Guajardo report, and it is much stronger here than in the short
(Γ = 6, τ ≈ 26 s) reactor where the same waveform's gain at w_b = 300 was
+0.034: the tall reactor multiplies the pulsed advantage ~2.6× (+0.09) and
extends it to all w_b, including w_b = 100 where the short reactor showed
little to no gain.

| w_b [rpm] | constant X | pulsed X | ΔX | constant P [W] | pulsed P [W] | const R− | pulsed R− |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.6890 | 0.7110 | +0.022 | 1.394 | 1.351 | 0.6453 | 0.6687 |
| 200 | 0.7617 | 0.8532 | **+0.092** | 2.565 | 2.524 | 0.6814 | **0.7742** |
| 300 | 0.7869 | 0.8764 | **+0.090** | 3.740 | 3.704 | 0.6698 | 0.7605 |
| 400 | 0.8116 | 0.8744 | +0.063 | 4.921 | 4.893 | 0.6575 | 0.7212 |
| 500 | 0.8281 | 0.8769 | +0.049 | 6.108 | 6.091 | 0.6369 | 0.6862 |

(X = mean outlet conversion over the last full period [290, 300] s; P = episode-
average motor power on the commanded ω(t); R− = X − P/31.94 W, the trainer's
reward convention.)

## Headlines

1. **Pulsed at 2.5 W out-converts constant at 6.1 W.** pulsed_wb200
   (X = 0.8532, 2.52 W) beats constant_wb500 (X = 0.8281, 6.11 W): more
   conversion at 2.4× less power. That is the paper's "more conversion for
   less energy" claim, reproduced with resolved Sc = 1075 film physics.
2. **Pulsed always costs slightly LESS than constant at the same w_b** (equal
   commanded mean + motor model regen ⇒ P_pulsed ≲ P_constant), so every ΔX
   in the table is a free lunch — the equal-power comparison is built in.
3. **Pulsed conversion saturates ≈ 0.877 from w_b ≥ 300** (0.8764 / 0.8744 /
   0.8769 at 300/400/500 — differences are inside the ±0.01 noise floor),
   while constant is still climbing at 500 rpm (0.828). Modulation reaches the
   reactor's practical ceiling at ~half the nominal speed.
4. **Best reward is an interior optimum at w_b = 200** (pulsed R− = 0.774),
   the same location as the short-reactor campaign's best — the
   conversion-per-watt sweet spot survives the geometry change.
5. Amortization mechanism confirmed: with τ/T ≈ 13 each fluid parcel averages
   ~13 pulse periods, so idle phases can't starve the outlet — hence pulsed
   wins even at w_b = 100, unlike the short reactor (τ/T ≈ 2.6) where the
   low-w_b gain vanished.

## Context vs the short (Γ = 6) reactor, same waveform family

| quantity | short Γ=6 (50 s eps, [40,50] s) | tall Γ=30 (300 s eps, [290,300] s) |
|---|---:|---:|
| constant-300 X | 0.3402 | 0.7869 |
| pulsed D80/T10 X @300 | 0.3744 | 0.8764 |
| pulsed gain ΔX @300 | +0.034 | +0.090 |
| pulsed gain @100 | ~0 / negative | +0.022 |
| best pulsed R− | 0.258 @300 (T10; 0.270 with T2.5) | 0.774 @200 |

Note the tall-reactor conversions are not directly comparable in magnitude to
the short reactor's (5× residence time ⇒ much higher X overall); the
comparison that matters is pulsed-vs-constant within each geometry.

## Implications for the RL program

- The Γ = 30 case is where modulation-vs-constant is most discriminable
  (ΔR− ≈ +0.09, ~9× the ±0.01 noise floor, vs ~+0.03–0.05 on the short case).
  A future block-modulation TD3 run on `full_tc_cat_case` (spec'd in
  `../modulation_rl/README.md` as the tall-reactor variant, ~300 s episodes)
  would need ~6× the CFD budget per episode (~250–275 s CPU per sim-s
  measured here, ~21–23 h per 300 s episode per core).
- These 10 episodes double as benchmark anchors for that future run, exactly
  like the td3_prep grid does for the current w_b = 300 short-case run.

## Cost (measured)

~247–276 s CPU per sim-second (episode-dependent), ~21–23 h per 300 s episode
on one 7800X3D core; 218 core-hours total, 23 h wall at 10-wide. 17 GB of 1 s
time folders under the episode dirs (ParaView-ready).

## Figures

- `fig7_conversion_vs_wb.png` — the Fig. 7 recreation (the money plot)
- `conversion_vs_power.png` — X vs motor W, both families
- `conversion_vs_time__{pulsed,constant}.png` — full 300 s traces
- `omega_traces.png` — commanded vs measured wall omega (freeze-bug guard)
- `summary_table.png` / `summary_table.csv` — the numbers above
