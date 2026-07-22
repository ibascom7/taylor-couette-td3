# Surrogate test of the RL setup — report (2026-07-21, overnight)

**Question asked:** on the fixed-mean w_b=300 case, is it the RL setup that's
failing or the env? Does the RL system still go to minimum D?

**Answer: it's the RL setup.** The env's physics is clean and learnable; the
learner, at the Carya recipe's tiny gradient budget, fails on a faithful
emulator of that physics in exactly the way it failed on Carya — one seed
reproduced the real run's endpoint to four decimal places. Learner-only
changes (more gradient steps per sample, BC regularization) rescue it on the
same data, which an env problem could not explain. Min-D pinning still occurs,
but it is one face of a seed-dependent lottery, not a systematic preference.

## What the surrogate is

`surrogate_env.py`: same class interface and the *inherited* real decode
(trough w_low, D, T → w_hi via the fixed-mean constraint), with the CFD
replaced by inverse-distance k-NN over **6,600 real block-samples** from the
s0 (1,012 eps) and v3 (308 eps) runs, in physical wave coordinates
(w_hi, w_low, logT, X_prev) so the degenerate constant-300 faces collapse to
one point like the physics. Noise σ_X = 0.005/block (the measured floor).

Fidelity probes (episode return, surrogate vs real CFD):

| probe | surrogate | real |
| --- | ---: | ---: |
| s0 corner, D=0.6 w_low=300 T=5 (constant-300) | 0.8516 | 0.851 |
| v3 corner, D=0.6 w_low=0 T=5 | 0.8611 | 0.843 |
| champion D=0.8 w_low=0 T=2.5 (final-block R) | 0.262 | 0.270 |
| constant-300 via the D=1 face | 0.899 | 0.851 (known +0.05 flaw) |

Ground truth by construction (static grid scan): **D=0.88, w_low=0, T=3.15 →
episode return 0.9641**, final-block R 0.2746.

## The test: real TD3.py, Carya-equivalent recipe, 10 seeds

Recipe exactly mirrors `parallel_train.py`: 300 episodes (200 random + 100
policy), expl noise 0.2, batch 64, done=(terminated or truncated) [the fixed
convention], **1 gradient step per env step** (the Carya pacing).

Result: **10/10 seeds end pinned at an action bound.** Mean return 0.914,
best 0.958, none reaches the optimum region interior. By decoded wave:
~6/10 collapse to constant-300 (via w_low=300 or D=1 — different raw corners,
same degenerate wave), **2/10 go to minimum D** (deep pulse, D=0.60), the
rest hug other faces. Seed 3 = the literal Carya run: D=0.6, w_low=300, T=5,
return **0.8506** (real: 0.8506).

So: "does it still go to min D?" — yes, in 2 of 10 seeds. Which bound a seed
pins is decided by the infant critic's initialization noise, not by physics;
min-D is not special, it's just where seed-0-family arrows point.

## Why (mechanism, now demonstrated rather than argued)

At ~500 gradient steps a 256×256 critic has not fit even 1,500 samples; its
action-gradient is initialization noise; the deterministic actor rides that
noise to a bound and tanh saturation locks it. The env's reward differences
(~0.03–0.11 between corners and optimum) are real and present in the data —
the learner just never develops the resolution to see them at this budget.

## Learner-only fixes, same env, same data budget

| recipe | passes (within 0.01 of optimum or above) | hard corner-pins | mean return |
| --- | ---: | ---: | ---: |
| Carya (UTD=1) | 0/10 | 10/10 | 0.914 |
| UTD=8 | 0/3 | 3/3 | 0.928 |
| **UTD=32** | 5–6/10 | 2/10 | 0.952 |
| **UTD=32 + BC α=0.05** | 6/10 | 1/10 | 0.945 |

UTD = gradient steps per env step (UTD=32 ≈ 48k gradient steps per run —
still seconds on the surrogate, and on Carya the learner thread is idle while
CFD runs, so it is FREE wall-clock there). Several UTD=32 seeds *beat* the
static optimum with block-varying policies (up to 0.987) — the block-varying >
static hypothesis is alive. BC removes almost all hard pinning but drags 3/10
seeds to mediocre shallow-modulation interiors (it regularizes toward the
collected data, which is corner-heavy after collapse begins).

Neither recipe alone passes the README's ≥8/10 gate yet. Obvious next probes
(NOT run tonight, per scope): UTD 64–128, UTD=32 + smaller nets, BC α≈0.02,
longer random phase. The instrument to iterate is now in place and each probe
costs ~1 minute.

## Caveats

- The surrogate over-values the D=1 constant face by ~+0.05 (k-NN smoothing);
  rankings near that face are soft.
- Block-varying returns above the static optimum partially rest on the
  1st-order (X_prev) memory model — treat the margin, not the sign, cautiously.
- GPU nondeterminism makes repeat runs vary by ~±0.01–0.02 in return.

## Files

- `build_dataset.py` → `dataset.npz` (6,600 samples)
- `surrogate_env.py` (emulator + `scan_static` ground truth)
- `probe_points.py` (fidelity checks)
- `run_recipes.py` (the harness; `--utd`, `--bc`, `--seed ...`)

Torch venv: `~/research/taylor-couette/.venv/bin/python`.
