# Meeting brief — Yuhe's side-outlet flow-through reactor (July 1)

Covers the Wed checklist. Yuhe's OpenFOAM set (`taylor_couette_reactor_short.zip`)
is a **5° wedge** of a **shortened (1/5-length) Taylor–Couette reactor** run as a
**continuous flow-through catalytic reactor**: liquid enters at the top annular
inlet, the inner cylinder rotates with the Lopez **modulated** waveform, the
**outer wall is catalytic** (consumes the reactant), and fluid leaves through a
**1-cell side outlet at the bottom of the outer wall**. Solver: `pimpleFoam`
(transient, laminar) + a `scalarTransport` function object for the reactant `c`.

This is a **different modeling choice** from the existing RL env in this repo,
which is a *closed* cell scored by an *intensity-of-segregation* mixing index.
Here performance = **outlet conversion** `X = 1 − c_out/c0`. (Flag this on the
flow chart — see §7.)

---

## 1. Flow chart of the current working flow  (whiteboard)

```
                         ┌─────────────────────────────────────────────┐
                         │   PHYSICAL TARGET (Lopez et al. 2024, CEJ)   │
                         │  TC reactor, silicone oil, modulated rotation │
                         │  claim: modulation beats constant @ equal X   │
                         └───────────────────────┬─────────────────────┘
                                                 │ calibrate geometry, fluid, waveform
                                                 ▼
   ┌───────────────────────────────────────────────────────────────────────────┐
   │  OPENFOAM CASE  (this zip: 5° wedge, shortened H, flow-through)            │
   │  • blockMesh  → 1350-cell wedge (15 r × 90 z × 1 θ)                        │
   │  • 0/U  inner_wall = Lopez modulated rotation (codedFixedValue)            │
   │         inlet = fully-developed annular inflow (codedFixedValue)           │
   │  • 0/c  inlet c0=50, outer_wall c=0 (catalytic), side_outlet zeroGradient  │
   │  • pimpleFoam + scalarTransport(c)                                         │
   │  • coded functionObjects → conversion(t),  rotational power P(t)           │
   └───────────────┬───────────────────────────────────────────┬───────────────┘
                   │ (A) PHYSICS VERIFICATION (no RL — this week)│ (B) RL CONTROL LOOP (existing)
                   ▼                                             ▼
   ┌─────────────────────────────────┐        ┌────────────────────────────────────────┐
   │ • dye reaches bottom? (yes, §3)  │        │  Helpers (helpers.py)                  │
   │ • warm-up / steady state (§7)    │        │   pause → set ω → run Δt → parse metrics │
   │ • ParaView field movie (§4)      │        │            ▲                    │        │
   │ • parameter table (§5)           │        │            │ ω (action)        │ obs     │
   │ • conversion, power, ω vs t (§9) │        │   ┌────────┴─────────┐   ┌─────▼──────┐ │
   └─────────────────────────────────┘        │   │  TD3 / DDPG agent │◀──│ Gym env     │ │
                                              │   │  (TD3.py/DDPG.py)│   │ reward =     │ │
                                              │   └──────────────────┘   │ −αI − βE     │ │
                                              │                          └─────────────┘ │
                                              └────────────────────────────────────────┘
```

**Parts that need further examination (with the criterion to judge them):**

| # | Part of the flow | What to examine | Criterion / test |
|---|---|---|---|
| a | **Mesh resolution** | 15×90 wedge, 1 θ-cell, at Re≈3900 (2500 rpm) | mesh-independence: refine ×2, conversion change < ~2 %? `checkMesh` non-orthogonality OK |
| b | **Turbulence model** | runs `laminar` but Re≈3900 → wavy/turbulent Taylor vortices | is laminar wedge valid, or need transitional model / DNS-like refinement? compare drag to a refined run |
| c | **Diffusivity / Schmidt #** | `D=2.8e-8` ⇒ Sc = ν/D = **384**, but comment claims "Sc ≈ 1e5 from paper" | reconcile Sc; conversion is sensitive to D. Use paper's true value |
| d | **Catalytic BC** | reaction modeled as `outer_wall c=0` (infinitely fast wall sink) | is a fixed-0 wall the right surface-reaction model, or finite-rate (Robin) BC? |
| e | **Wedge vs full 360°** | axisymmetric 5° wedge cannot represent azimuthal (wavy) vortex modes | does the full-annulus case give the same conversion? (×72 power scaling assumes axisymmetry) |
| f | **Steady-state / warm-up** | when is the reactor at (periodic) steady state? (§7) | conversion per-cycle peak/trough drift < threshold |
| g | **Performance metric mismatch** | RL env scores *intensity of segregation* (closed cell); Yuhe's case scores *outlet conversion* (flow-through) | decide which metric the RL agent should optimize on the flow-through geometry |
| h | **Energy model** | CFD gives viscous-drag power only; paper's claim needs the **motor** model (regen + bearing friction) | use motor model (Eqs 18–23) for the energy index, not drag-only |

---

## 5. Table of current simulation setup parameters

**Geometry (shortened reactor, 5° wedge)**

| Parameter | Symbol | Value |
|---|---|---|
| Inner radius | R_i | 0.0254 m (25.4 mm) |
| Outer radius | R_o | 0.03175 m (31.75 mm) |
| Gap width | d = R_o−R_i | 0.00635 m (6.35 mm) |
| Radius ratio | η = R_i/R_o | 0.80 |
| Height (shortened) | H | 0.0381 m (= 1/5 full length) |
| Aspect ratio | Γ = H/d | 6.0 |
| Wedge angle | — | 5° |
| Reactor volume (full 360°, shortened) | V | 43.4 mL (wedge: 0.60 mL) |

**Mesh**

| Parameter | Value |
|---|---|
| Total cells | 1350 |
| Radial × axial × azimuthal | 15 × 90 × 1 |
| Axial split | 1-cell side-outlet band + 89-cell main section |
| Patches (7) | inner_wall, outer_wall, side_outlet (1 face), inlet (15), bottom, front (wedge), back (wedge) |

**Fluid — Lopez silicone oil**

| Parameter | Symbol | Value |
|---|---|---|
| Density | ρ | 930 kg/m³ |
| Dynamic viscosity | μ | 0.01 Pa·s |
| Kinematic viscosity | ν = μ/ρ | 1.075×10⁻⁵ m²/s |
| Reactant diffusivity | D | 2.8×10⁻⁸ m²/s |
| Schmidt number | Sc = ν/D | **384**  *(⚠ comment says "≈1e5 from paper" — reconcile)* |

**Operating conditions**

| Parameter | Symbol | Value |
|---|---|---|
| Through-flow rate (full 360°) | Q₀ | 40 mL/min = 6.67×10⁻⁷ m³/s |
| Mean axial velocity | U = Q₀/A_ann | 5.85×10⁻⁴ m/s (0.585 mm/s) |
| Mean residence time | τ = H/U | 65 s |
| Inlet reactant conc. | c₀ | 50 mmol/m³ |
| Rotation waveform | — | **Lopez modulated** |
| Mean speed | ω_b | 500 rpm (52.4 rad/s) |
| Duty cycle | D | 0.2 (set 1.0 → constant 500 rpm) |
| Period | T | 20 s |
| Burst speed (high phase) | ω_b/D | 2500 rpm (261.8 rad/s) → on for D·T = 4 s, off 16 s |
| Transition smoothing | — | 0.05 s |
| Rotational Reynolds | Re = ωR_i d/ν | 785 @500 rpm, **3927 @2500 rpm** (→ Taylor-vortex / wavy regime) |

**Numerics (pimpleFoam)**

| Parameter | Value |
|---|---|
| Time scheme (ddt) | backward (2nd order) |
| div(phi,U) | bounded Gauss LUST |
| div(phi,c) | bounded Gauss MUSCL |
| laplacian | Gauss linear corrected |
| PIMPLE | 2 outer, 2 inner correctors, 0 non-orth |
| p solver / U,c solver | GAMG / PBiCGStab+DILU |
| Time step | adaptive, max Co = 1.8, ΔT_max = 0.05 s, ΔT₀ = 0.01 s |
| End time / write interval | 200 s / 4 s (adjustableRunTime) |
| Turbulence | laminar |

---

## 8. Formulas for the energy index and the mixing index (for slides)

### Mixing index — intensity of segregation (closed-cell RL env)
Over `N = 20` radial bins at the measurement plane, bin radii `r_i`, area-type
weights `w_i = r_i / Σ_j r_j`, normalized concentrations `C_i ∈ [0,1]`:

```
 C̄  = Σ_i w_i C_i                       (mean concentration)
 σ²  = Σ_i w_i (C_i − C̄)²               (segregation variance)
 σ²_max = C̄ (1 − C̄)                     (max variance, fully segregated)

           σ²
 I_mix =  ─────        I_mix = 1 → unmixed,   I_mix = 0 → perfectly mixed
          σ²_max
```

> On Yuhe's **flow-through** case the reactor-performance analog is **outlet
> conversion**:  `X = 1 − c_out/c0`, with `c_out` the flux-weighted side-outlet
> concentration (`sideOutletConversion` function object).

### Energy index — motor electrical power (Lopez et al. 2024, Eqs 18–23)
A pure function of the commanded `ω(t)` (`motor_power.py`):

```
 T_mot = J·dω/dt + T₀·sign(ω) + (β₂·ω²·sign(ω) + β₁·ω)      (18)  inertia+bearing+drag
 i_mot = T_mot / K_t                                          (20)
 e_mot = L·di_mot/dt + R·i_mot + K_e·ω                        (21)
 P_mot = e_mot · i_mot                                        (22)
 P_e   = P_mot/η      (P_mot ≥ 0, motoring)                   (23)
       = P_mot·η      (P_mot < 0, regenerative braking)
 E = ∫ P_e dt          E_norm = E / E_max          (energy index ∈ [0,1])
```

Constants: J=1.66e-4, T₀=0.1, β₂=7.19e-8, β₁=1.51e-5, K_t=K_e=0.0931,
L=0.28e-3, R=0.178, η=0.90.

> The CFD viscous-drag power (what `pimpleFoam` measures on the inner wall,
> `P = M_z·Ω`, scaled ×72 for full 360°) is *convex* in ω and so makes bursts
> look expensive; the **motor** model — with linear bearing friction and regen —
> is what reproduces the paper's "modulation is cheaper at equal conversion".
> Use the motor model for the energy index. (See §1-h.)

### Reward (RL env)
```
 r = −α·I_mix − β·E_norm          (α = β = 1 by default)
```

---

## 2–4, 6–7, 9 — results (filled in from the run)

See `RESULTS.md` and the figures:
- `omega_vs_time.png`, `energy_vs_time.png`, `conversion_vs_time.png`, `mixing_index_vs_time.png`
- `frames/` (ParaView field snapshots), `montage.png`

**Headline numbers for the meeting**
- Dye reaches the bottom outlet **by t≈4 s** (first 2500 rpm burst) — ~16× faster than
  the 65 s mean residence time; Taylor-vortex axial transport, not bulk flow.
- (Periodic) **steady state at t≈110–130 s** (~7 cycles); steady conversion **X≈0.45**.
- Compute cost: **~19 min wall** for 200 s physical, 1 core, 1350-cell wedge.
- Energy over 200 s: **1284 J, avg 6.42 W** (motor model). Peak burst speed 2500 rpm.
- ⚠ RL warm-up of 10 s is **too short** for this case — needs ≈130 s.
- ⚠ Sc=ν/D=**384** in the case vs the "≈1e5" claimed in the comment — reconcile.
