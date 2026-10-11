#!/usr/bin/env python3
"""Mechanism experiments 1-3 + e5 (see PLAN.md): what sets the best stop length?

Built 2026-10-10 for GCURS. One engine, three run groups, all on the short
reactor (side_outlet_case_sc1075_graded) at maxCo 0.9, every run starting from
a state warmed at constant 300 rpm for 60 s AT ITS OWN VISCOSITY:

  e1  stop-length curve at nominal viscosity: run 4.5 s, mean 300 rpm, stop
      level 0, stops 0 (constant) .. 2.0 s, plus jittered replicates (+0.5 rpm)
      of the two points nearest the endgame-A peak.
  e2  the same curve at 2x and 0.7x viscosity, D held at 1e-8 (nu and mu change
      in constant/transportProperties only). Primary arm: 2x.
  e3  stop-level sweep at nominal viscosity: stop 0.5 s at w_lo 30/60/100/150
      rpm (w_lo 0 is the e1 0.5 s point); run speed solved for the 300 mean.
  e5  vortex-state tests at nominal viscosity (2 runs): 'hold' pulses (stop
      0.5 s) until 60 s then holds constant 300; 'kick' is constant 300 with a
      single 0.5 s stop at 60 s. If either keeps the pulse's conversion, the
      gain is a Taylor-cell state change rather than repeated stopping.

WAVEFORM. One continuous wave per run (no block restarts), burst-first
periods T = 4.5 s + stop, 0.05 s ramps centred on the switch times, phased so
a period boundary falls exactly at t = 130 s. With symmetric ramps the mean
over any whole number of periods is exactly (run*w_hi + stop*w_lo)/T, so
w_hi = (300*T - stop*w_lo)/4.5 gives a 300 rpm mean in the scoring window.

SCORING. Over the last whole number of periods ending at 130 s, n = round(26/T)
(constant runs: last 26 s). Per run:
  X_out     mean outlet conversion (cup-mixed outflow, the production metric)
  X_budget  (integral of wall consumption + inventory change) / integral of
            reactant inflow -- immune to the outlet-cell jet. The wall flux is
            -D*snGrad(c) on the catalytic wall, so the face-to-cell distance is
            the solver's own (the chord-face trap cannot occur).
  closure   inventory-balance residual (bookkeeping check, should be ~0)
  P, r      motor-model power on the commanded omega over the same window,
            r = X_out - P/31.94
  clocks    per stop: e-folding times of mean swirl and secondary-flow energy,
            the wall-flux drop, the restart overshoot and the burst fade.

A coded function object (mechClocks) logs all of this every 0.02 s to
postProcessing/mechClocks/0/mech.dat. Only the last 3000 solver lines are
kept in log.pimpleFoam (full logs are ~350 MB per run).

USAGE (from taylorCouetteGym/, OpenFOAM on PATH)
    python3 experiments/mechanism/mechanism_sweep.py --list
    python3 experiments/mechanism/mechanism_sweep.py --smoke          # ~20 min pipeline test
    python3 experiments/mechanism/mechanism_sweep.py --resume         # the real thing (Carya)
    python3 experiments/mechanism/mechanism_sweep.py --analyze-only   # any time, partial OK
Options: --exp e1,e2  --only TAG,TAG  --workers N
"""

import argparse
import csv
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
GYM_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))

# fig7_sweep as a library: foamDictionary, square-wave builder, motor power.
_F_PATH = os.path.join(GYM_ROOT, "experiments", "modulation_vs_constant", "fig7_sweep.py")
_spec = importlib.util.spec_from_file_location("fig7_sweep", _F_PATH)
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)

BASE_CASE = os.path.join(GYM_ROOT, "taylor_couette_mixing", "cases", "side_outlet_case_sc1075_graded")
RESULTS_ROOT = os.path.join(HERE, "results")

# ---- physics (case values) -------------------------------------------------
NU0 = 1.0752688172043011e-05     # m^2/s (case transportProperties)
RHO = 930.0
D_MOL = 1e-8                     # held fixed in every run
GAP = 0.00635
C0 = 50.0
RPM = 2.0 * math.pi / 60.0
NU = {"nu1": 1.0, "nu2": 2.0, "nu0p7": 0.7}

# ---- protocol ---------------------------------------------------------------
FULL = dict(episode=130.0, warm=60.0, run_s=4.5, window=26.0, write=10.0, clock_t0=30.0, e3_stop=0.5,
            e5_switch=60.0)
SMOKE = dict(episode=6.0, warm=1.0, run_s=1.0, window=2.0, write=2.0, clock_t0=1.0, e3_stop=0.4,
             e5_switch=3.0)
MEAN_RPM = 300.0
MAXCO = 0.9
RAMP = 0.05
P_MAX = 31.94
CLOCK_DT = 0.02
SIGMA_DEFAULT = 0.002            # endgame-A replicate noise in X / r
SIGMA_FLOOR = 0.001
N_BOOT = 2000

E1_STOPS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.85, 1.2, 2.0]
E1_REPS = [0.45, 0.5]
REP_JITTER_RPM = 0.5
E2_STOPS = {"nu2":   [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.3, 1.8],
            "nu0p7": [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0, 1.3]}
E3_STOP = 0.5
E3_WLO = [30.0, 60.0, 100.0, 150.0]

SMOKE_STOPS = [0.0, 0.2, 0.4, 1.0]
SMOKE_WLO = [60.0]

# Pre-registered exponents for t_stop* ~ nu^a (PLAN.md, experiment 2).
A_MOMENTUM = -1.0
A_FILM_ETW = 0.69
A_FILM_LAM = 0.33


# --------------------------------------------------------------------------- #
# Run list
# --------------------------------------------------------------------------- #
def stag(x):
    return f"{x:.2f}".replace(".", "p")


def make_run(exp, nu_tag, stop, w_lo=0.0, rep=0, cfg=FULL, kind="wave"):
    """kind: 'wave' (the stop-length family; stop 0 = constant), or the optional
    vortex-state tests 'hold' (pulse with this stop until e5_switch, then
    constant) and 'kick' (constant with ONE stop at e5_switch)."""
    mean = MEAN_RPM + (REP_JITTER_RPM if rep else 0.0)
    T = cfg["run_s"] + stop
    w_hi = mean if stop == 0 else (mean * T - stop * w_lo) / cfg["run_s"]
    if kind != "wave":
        return dict(exp=exp, nu_tag=nu_tag, nu_f=NU[nu_tag], stop=0.0, w_lo=0.0, w_hi=mean,
                    mean=mean, rep=rep, run_s=cfg["run_s"], period=T, tag=f"{exp}_{nu_tag}_{kind}",
                    kind=kind, kick_stop=stop, pulse_w_hi=w_hi)
    tag = f"{exp}_{nu_tag}_s{stag(stop)}"
    if w_lo:
        tag += f"_lo{int(w_lo)}"
    if rep:
        tag += f"_r{rep}"
    return dict(exp=exp, nu_tag=nu_tag, nu_f=NU[nu_tag], stop=stop, w_lo=w_lo, w_hi=w_hi,
                mean=mean, rep=rep, run_s=cfg["run_s"], period=T, tag=tag, kind="wave")


def e5_runs(cfg):
    s = cfg["e3_stop"]
    return [make_run("e5", "nu1", s, cfg=cfg, kind="hold"), make_run("e5", "nu1", s, cfg=cfg, kind="kick")]


def all_runs(smoke=False):
    cfg = SMOKE if smoke else FULL
    runs = e5_runs(cfg)
    if smoke:
        for s in SMOKE_STOPS:
            runs.append(make_run("e1", "nu1", s, cfg=cfg))
        runs.append(make_run("e1", "nu1", cfg["e3_stop"], rep=1, cfg=cfg))
        for nt in ("nu2", "nu0p7"):
            for s in SMOKE_STOPS:
                runs.append(make_run("e2", nt, s, cfg=cfg))
        for wl in SMOKE_WLO:
            runs.append(make_run("e3", "nu1", cfg["e3_stop"], w_lo=wl, cfg=cfg))
        return runs
    for s in E1_STOPS:
        runs.append(make_run("e1", "nu1", s, cfg=cfg))
    for s in E1_REPS:
        runs.append(make_run("e1", "nu1", s, rep=1, cfg=cfg))
    for nt, stops in E2_STOPS.items():
        for s in stops:
            runs.append(make_run("e2", nt, s, cfg=cfg))
    for wl in E3_WLO:
        runs.append(make_run("e3", "nu1", E3_STOP, w_lo=wl, cfg=cfg))
    return runs


# --------------------------------------------------------------------------- #
# Waveform
# --------------------------------------------------------------------------- #
def wave_points(run, cfg):
    """(t, omega rad/s) table. Burst-first periods aligned so t = episode is a
    period boundary; starts before t=0 and runs one period past the end."""
    ep = cfg["episode"]
    w = run["mean"] * RPM
    kind = run.get("kind", "wave")
    if kind != "wave":
        T, h = run["period"], 0.5 * RAMP
        ts = ep - T * round((ep - cfg["e5_switch"]) / T)      # a period boundary
        if kind == "kick":
            s = run["kick_stop"]
            return [(-1.0, w), (ts - h, w), (ts + h, 0.0), (ts + s - h, 0.0), (ts + s + h, w), (ep + 1.0, w)]
        pulse = dict(run, kind="wave", stop=run["kick_stop"], w_hi=run["pulse_w_hi"])
        pts = [p for p in wave_points(pulse, cfg) if p[0] <= ts - h + 1e-9]
        return pts + [(ts + h, w), (ep + 1.0, w)]
    if run["stop"] == 0:
        return [(-1.0, w), (ep + 1.0, w)]
    T = run["period"]
    n = int(math.ceil(ep / T)) + 2
    t0 = ep + T - n * T
    pts, _ = F.square_wave_points(t0, n * T, run["w_hi"] * RPM, run["w_lo"] * RPM, T,
                                  run["run_s"] / T, RAMP)
    return pts


def score_window(run, cfg):
    ep = cfg["episode"]
    if run["stop"] == 0:
        return ep - cfg["window"], ep
    T = run["period"]
    n = max(1, int(round(cfg["window"] / T)))
    return ep - n * T, ep


def stop_intervals(run, cfg, t_from):
    """[(stop_start, stop_end)] switch times of every stop after t_from."""
    if run["stop"] == 0:
        return []
    T, ep = run["period"], cfg["episode"]
    out = []
    k = 0
    while True:
        b = ep - (k + 1) * T             # burst start of the k-th period before the end
        if b < t_from:
            break
        out.append((b + run["run_s"], b + T))
        k += 1
    return sorted(out)


# --------------------------------------------------------------------------- #
# Case setup
# --------------------------------------------------------------------------- #
MECH_FO = r"""
    mechClocks
    {
        type            coded;
        libs            ( utilityFunctionObjects );
        name            mechClocks;
        executeControl  adjustableRunTime;
        executeInterval __CLOCK_DT__;
        writeControl    adjustableRunTime;
        writeInterval   __CLOCK_DT__;
        codeInclude     #{
            #include "fvCFD.H"
            #include <fstream>
            #include <iomanip>
            #include <vector>
            #include <algorithm>
        #};
        codeOptions     #{
            -I$(LIB_SRC)/finiteVolume/lnInclude
        #};
        codeLibs        #{
            -lfiniteVolume
        #};
        codeExecute     #{
            // Mechanism-experiment diagnostics (experiments/mechanism). All
            // fluxes are per wedge and divided by c0 [m^3/s]; inventory [m^3].
            const fvMesh& mesh = refCast<const fvMesh>(obr_);
            const volScalarField& c = mesh.lookupObject<volScalarField>("c");
            const volVectorField& U = mesh.lookupObject<volVectorField>("U");
            const surfaceScalarField& phi = mesh.lookupObject<surfaceScalarField>("phi");
            const scalar Dmol = __DMOL__;
            const scalar c0 = __C0__;

            // ---- volume averages: swirl, secondary-flow energy, inventory --
            const scalarField& V = mesh.V();
            const vectorField& C = mesh.C();
            scalar sV = 0, sUt = 0, sE = 0, sInv = 0;
            forAll(C, i)
            {
                const scalar x = C[i].x();
                const scalar y = C[i].y();
                const scalar r = sqrt(sqr(x) + sqr(y));
                const vector eT(-y/r, x/r, 0.0);
                const vector eR(x/r, y/r, 0.0);
                const scalar ut = U[i] & eT;
                const scalar ur = U[i] & eR;
                const scalar uz = U[i].z();
                sV   += V[i];
                sUt  += ut*V[i];
                sE   += (sqr(ur) + sqr(uz))*V[i];
                sInv += c[i]*V[i];
            }
            reduce(sV, sumOp<scalar>());
            reduce(sUt, sumOp<scalar>());
            reduce(sE, sumOp<scalar>());
            reduce(sInv, sumOp<scalar>());

            // ---- catalytic outer wall: resolved diffusive consumption ------
            scalar wall = 0;
            {
                const label pI = mesh.boundaryMesh().findPatchID("outer_wall");
                const fvPatchScalarField& cw = c.boundaryField()[pI];
                const scalarField snG(cw.snGrad());
                const scalarField& A = mesh.boundary()[pI].magSf();
                forAll(snG, f) { wall -= Dmol*snG[f]*A[f]; }
            }
            reduce(wall, sumOp<scalar>());

            // ---- inlet: advective + diffusive reactant inflow ---------------
            scalar inAdv = 0, inDiff = 0, Q = 0;
            {
                const label pI = mesh.boundaryMesh().findPatchID("inlet");
                const fvPatchScalarField& cp = c.boundaryField()[pI];
                const fvsPatchScalarField& ph = phi.boundaryField()[pI];
                const scalarField snG(cp.snGrad());
                const scalarField& A = mesh.boundary()[pI].magSf();
                forAll(cp, f)
                {
                    inAdv  -= ph[f]*cp[f];
                    inDiff += Dmol*snG[f]*A[f];
                    Q      -= ph[f];
                }
            }
            reduce(inAdv, sumOp<scalar>());
            reduce(inDiff, sumOp<scalar>());
            reduce(Q, sumOp<scalar>());

            // ---- side outlet: net advective outflow + cup-mixed conversion --
            scalar outAdv = 0, outPhi = 0, outPhiC = 0;
            {
                const label pI = mesh.boundaryMesh().findPatchID("side_outlet");
                const fvPatchScalarField& cp = c.boundaryField()[pI];
                const fvsPatchScalarField& ph = phi.boundaryField()[pI];
                forAll(cp, f)
                {
                    outAdv += ph[f]*cp[f];
                    if (ph[f] > 0) { outPhi += ph[f]; outPhiC += ph[f]*cp[f]; }
                }
            }
            reduce(outAdv, sumOp<scalar>());
            reduce(outPhi, sumOp<scalar>());
            reduce(outPhiC, sumOp<scalar>());
            const scalar xOut = 1.0 - (outPhiC/max(outPhi, VSMALL))/c0;

            // ---- Taylor-cell count: sign changes of u_r along z in the cell
            // column nearest mid-gap (vortex centres sit where u_r = 0). The
            // graded blocks share one radial distribution, so that column has
            // one cell per axial level. Serial runs only (no gather).
            scalar nVort = 0, urMax = 0;
            {
                const scalar rMid = 0.5*(0.0254 + 0.03175);
                scalar dMin = GREAT;
                forAll(C, i)
                {
                    dMin = min(dMin, mag(sqrt(sqr(C[i].x()) + sqr(C[i].y())) - rMid));
                }
                std::vector<std::pair<scalar, scalar>> col;
                forAll(C, i)
                {
                    const scalar r = sqrt(sqr(C[i].x()) + sqr(C[i].y()));
                    if (mag(r - rMid) < dMin + 1e-7)
                    {
                        const vector eR(C[i].x()/r, C[i].y()/r, 0.0);
                        col.push_back(std::make_pair(C[i].z(), U[i] & eR));
                        urMax = max(urMax, mag(U[i] & eR));
                    }
                }
                std::sort(col.begin(), col.end());
                int lastSign = 0;
                for (size_t k = 0; k < col.size(); ++k)
                {
                    const scalar ur = col[k].second;
                    if (mag(ur) < 0.02*urMax) continue;
                    const int sg = (ur > 0) ? 1 : -1;
                    if (lastSign != 0 && sg != lastSign) nVort += 1;
                    lastSign = sg;
                }
            }

            // ---- inner wall angular speed (freeze-bug guard) ----------------
            scalar sOmA = 0, sA = 0;
            {
                const label pI = mesh.boundaryMesh().findPatchID("inner_wall");
                const fvPatchVectorField& Uw = U.boundaryField()[pI];
                const vectorField& Cf = mesh.boundary()[pI].Cf();
                const scalarField& A = mesh.boundary()[pI].magSf();
                forAll(Cf, f)
                {
                    const scalar r = sqrt(sqr(Cf[f].x()) + sqr(Cf[f].y()));
                    const vector eT(-Cf[f].y()/r, Cf[f].x()/r, 0.0);
                    sOmA += ((Uw[f] & eT)/r)*A[f];
                    sA   += A[f];
                }
            }
            reduce(sOmA, sumOp<scalar>());
            reduce(sA, sumOp<scalar>());

            if (Pstream::master())
            {
                fileName outDir = mesh.time().path()/"postProcessing"/"mechClocks"/"0";
                mkDir(outDir);
                const bool fresh = (mesh.time().timeIndex() <= 1);
                std::ofstream os((outDir/"mech.dat").c_str(),
                                 fresh ? std::ios::out : std::ios::app);
                if (fresh)
                {
                    os << "# t Omega Q inAdv inDiff outAdv xOut wall inv V swirl Esec nVort urMax" << std::endl;
                }
                os << std::setprecision(10)
                   << mesh.time().value() << ' ' << sOmA/max(sA, VSMALL) << ' '
                   << Q << ' ' << inAdv/c0 << ' ' << inDiff/c0 << ' '
                   << outAdv/c0 << ' ' << xOut << ' ' << wall/c0 << ' '
                   << sInv/c0 << ' ' << sV << ' ' << sUt/sV << ' ' << sE/sV << ' '
                   << nVort << ' ' << urMax << std::endl;
            }
        #};
    }
"""


def foam_get(case, entry, dictfile):
    r = subprocess.run(["foamDictionary", "-entry", entry, "-value", dictfile], cwd=case,
                       capture_output=True, text=True, check=True)
    return r.stdout.strip()


def run_foam(case, log_name="log.pimpleFoam"):
    """pimpleFoam keeping only the last 3000 stdout lines (full logs ~350 MB)."""
    env = dict(os.environ, OMP_NUM_THREADS="1")
    cmd = f"set -o pipefail; pimpleFoam 2> log.err | tail -n 3000 > {log_name}"
    return subprocess.run(["bash", "-c", cmd], cwd=case, env=env).returncode


def _time_dirs(case):
    return sorted((float(n), n) for n in os.listdir(case)
                  if os.path.isdir(os.path.join(case, n)) and F._is_float(n))


def insert_mech_fo(case):
    cd = os.path.join(case, "system", "controlDict")
    text = open(cd).read()
    if "mechClocks" in text:
        return
    key = "functions\n{"
    i = text.find(key)
    if i < 0:
        raise RuntimeError("controlDict has no 'functions\\n{' block")
    fo = (MECH_FO.replace("__CLOCK_DT__", repr(CLOCK_DT))
          .replace("__DMOL__", f"{D_MOL:.6e}").replace("__C0__", repr(C0)))
    text = text[:i + len(key)] + fo + text[i + len(key):]
    with open(cd, "w") as fh:
        fh.write(text)


def prepare_base(tpl_root):
    """Copy the case, normalise controlDict, add mechClocks, compile coded code."""
    base = os.path.join(tpl_root, "base")
    if os.path.isfile(os.path.join(base, "COMPILED_OK")):
        return base
    if os.path.isdir(base):
        shutil.rmtree(base)
    shutil.copytree(BASE_CASE, base)
    F.clean_run_artifacts(base)
    for k, v in (("startFrom", "startTime"), ("startTime", "0"),
                 ("writeControl", "adjustableRunTime"), ("purgeWrite", "0"),
                 ("runTimeModifiable", "no"), ("maxCo", repr(MAXCO))):
        F.foam_set(base, k, v, "system/controlDict")
    for fo in ("sideOutletConversion", "rotationalPower_innerWall"):
        F.foam_set(base, f"functions.{fo}.executeInterval", "0.1", "system/controlDict")
        F.foam_set(base, f"functions.{fo}.writeInterval", "0.1", "system/controlDict")
    # D held fixed in BOTH places the case reads it.
    F.foam_set(base, "D", f"D [0 2 -1 0 0 0 0] {D_MOL:.6e}", "constant/transportProperties")
    F.foam_set(base, "functions.scalarTransport.D", f"{D_MOL:.6e}", "system/controlDict")
    insert_mech_fo(base)
    F.foam_set(base, "boundaryField.inner_wall.omega", "0", "0/U")
    F.foam_set(base, "endTime", "0.05", "system/controlDict")
    F.foam_set(base, "writeInterval", "1", "system/controlDict")
    rc = run_foam(base, "log.compile")
    if rc != 0 or not os.path.isfile(os.path.join(base, "postProcessing", "mechClocks", "0", "mech.dat")):
        raise RuntimeError(f"template compile failed (rc={rc}); see {base}/log.err, log.compile")
    F.clean_run_artifacts(base)
    open(os.path.join(base, "COMPILED_OK"), "w").write("ok\n")
    return base


def set_viscosity(case, nu_f):
    nu = NU0 * nu_f
    F.foam_set(case, "nu", f"nu [0 2 -1 0 0 0 0] {nu:.10e}", "constant/transportProperties")
    F.foam_set(case, "mu", f"mu [1 -1 -1 0 0 0 0] {RHO * nu:.10e}", "constant/transportProperties")
    got_nu = float(foam_get(case, "nu", "constant/transportProperties").split()[-1])
    got_d = float(foam_get(case, "D", "constant/transportProperties").split()[-1])
    got_d2 = float(foam_get(case, "functions.scalarTransport.D", "system/controlDict"))
    # foamDictionary prints 6 significant digits
    if abs(got_nu / nu - 1) > 1e-4 or abs(got_d / D_MOL - 1) > 1e-4 or abs(got_d2 / D_MOL - 1) > 1e-4:
        raise RuntimeError(f"property check failed in {case}: nu={got_nu} D={got_d}/{got_d2}")


def warm_template(base, tpl_root, nu_tag, cfg):
    """Copy base, set viscosity, spin at constant 300 rpm for cfg['warm'] s at
    maxCo 0.9, promote the final state to 0/ (episodes start warmed at t=0)."""
    dst = os.path.join(tpl_root, nu_tag)
    okf = os.path.join(dst, "WARM_OK.json")
    if os.path.isfile(okf):
        return dst
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(base, dst)
    os.remove(os.path.join(dst, "COMPILED_OK"))
    set_viscosity(dst, NU[nu_tag])
    F.foam_set(dst, "boundaryField.inner_wall.omega", repr(MEAN_RPM * RPM), "0/U")
    F.foam_set(dst, "endTime", repr(cfg["warm"]), "system/controlDict")
    F.foam_set(dst, "writeInterval", repr(cfg["warm"]), "system/controlDict")
    t0 = time.time()
    rc = run_foam(dst, "log.warmup")
    tds = [td for td in _time_dirs(dst) if td[0] > 0]
    if rc != 0 or not tds or abs(tds[-1][0] - cfg["warm"]) > 1e-6:
        raise RuntimeError(f"warm-up failed for {nu_tag} (rc={rc}, times={tds[-1:]}); see {dst}/log.err")
    m = read_mech(dst)
    shutil.rmtree(os.path.join(dst, "0"))
    os.rename(os.path.join(dst, tds[-1][1]), os.path.join(dst, "0"))
    shutil.rmtree(os.path.join(dst, "0", "uniform"), ignore_errors=True)
    F.clean_run_artifacts(dst)
    info = dict(nu_tag=nu_tag, nu=NU0 * NU[nu_tag], warm_s=cfg["warm"],
                wall_min=(time.time() - t0) / 60.0,
                x_out_end=float(m["xOut"][-1]) if m else float("nan"),
                swirl_end=float(m["swirl"][-1]) if m else float("nan"))
    json.dump(info, open(okf, "w"), indent=2)
    print(f"  warm {nu_tag}: {info['wall_min']:.1f} min, X_out(end) = {info['x_out_end']:.4f}", flush=True)
    return dst


# --------------------------------------------------------------------------- #
# One episode
# --------------------------------------------------------------------------- #
SYNC_FILES = ("run.json", "waveform_points.csv", "log.pimpleFoam", "log.err",
              "system/controlDict", "constant/transportProperties")


def sync_back(workdir, final):
    """Copy a run's outputs (not its dynamicCode or every time dir) from the
    node-local work dir to the results tree."""
    os.makedirs(os.path.join(final, "system"), exist_ok=True)
    os.makedirs(os.path.join(final, "constant"), exist_ok=True)
    for f in SYNC_FILES:
        if os.path.isfile(os.path.join(workdir, f)):
            shutil.copy2(os.path.join(workdir, f), os.path.join(final, f))
    pp = os.path.join(workdir, "postProcessing")
    if os.path.isdir(pp):
        shutil.copytree(pp, os.path.join(final, "postProcessing"), dirs_exist_ok=True)
    tds = _time_dirs(workdir)
    if tds and tds[-1][0] > 0:
        shutil.copytree(os.path.join(workdir, tds[-1][1]), os.path.join(final, tds[-1][1]),
                        dirs_exist_ok=True)


def run_episode(run, cfg, tpls, results_dir, work_root, resume):
    final = os.path.join(results_dir, "runs", run["tag"])
    workdir = os.path.join(work_root, "runs", run["tag"])
    if resume and os.path.isfile(os.path.join(final, "COMPLETE")):
        res = metrics(run, cfg, final)
        print(f"  [DONE   ] {run['tag']:24s} X_out={res['x_out']:.4f}", flush=True)
        return res
    pts = wave_points(run, cfg)
    table = "table (" + " ".join(f"({t:.6f} {w:.6f})" for t, w in pts) + ")"
    ep = cfg["episode"]
    latest_t, latest = 0.0, "0"
    if resume and os.path.isdir(workdir):
        tds = _time_dirs(workdir)
        if tds:
            latest_t, latest = tds[-1]
    t0 = time.time()
    if resume and latest_t >= ep - 1e-6:
        rc = 0
        print(f"  [DONE   ] {run['tag']}", flush=True)
    elif resume and latest_t > 0.0:
        print(f"  [RESUME ] {run['tag']} from t={latest_t:g}", flush=True)
        F.foam_set(workdir, "boundaryField.inner_wall.omega", table, f"{latest}/U")
        subprocess.run(["foamDictionary", "-entry", "boundaryField.inner_wall.omegaCoeffs",
                        "-remove", f"{latest}/U"], cwd=workdir, check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        F.foam_set(workdir, "startFrom", "latestTime", "system/controlDict")
        F.foam_set(workdir, "endTime", repr(ep), "system/controlDict")
        json.dump(dict(run, cfg=cfg), open(os.path.join(workdir, "run.json"), "w"), indent=2)
        rc = run_foam(workdir)
    else:
        if os.path.isdir(workdir):
            shutil.rmtree(workdir)
        shutil.copytree(tpls[run["nu_tag"]], workdir)
        os.remove(os.path.join(workdir, "WARM_OK.json"))
        F.clean_run_artifacts(workdir)
        F.foam_set(workdir, "boundaryField.inner_wall.omega", table, "0/U")
        F.foam_set(workdir, "endTime", repr(ep), "system/controlDict")
        F.foam_set(workdir, "writeInterval", repr(cfg["write"]), "system/controlDict")
        json.dump(dict(run, cfg=cfg), open(os.path.join(workdir, "run.json"), "w"), indent=2)
        with open(os.path.join(workdir, "waveform_points.csv"), "w") as fh:
            fh.write("time_s,omega_cmd_rad_s\n" + "".join(f"{t:.6f},{w:.6f}\n" for t, w in pts))
        rc = run_foam(workdir)
    if os.path.abspath(workdir) != os.path.abspath(final):
        sync_back(workdir, final)
    res = metrics(run, cfg, final)
    if res["complete"] and rc == 0:
        open(os.path.join(final, "COMPLETE"), "w").write("ok\n")
    res["ok"] = res["ok"] and rc == 0
    res["wall_h"] = (time.time() - t0) / 3600.0
    st = "OK" if res["ok"] else ("FAILED" if rc else "INCOMPLETE")
    print(f"  [{st:8s}] {run['tag']:24s} X_out={res['x_out']:.4f} X_bud={res['x_budget']:.4f} "
          f"clos={res['closure']:+.1e} r={res['r']:.4f} |dOm|={res['omega_err_rpm']:.2f}rpm "
          f"{res['wall_h']:.2f} h", flush=True)
    return res


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
MECH_COLS = ["t", "Omega", "Q", "inAdv", "inDiff", "outAdv", "xOut", "wall", "inv", "V", "swirl", "Esec",
             "nVort", "urMax"]


def read_mech(case):
    path = os.path.join(case, "postProcessing", "mechClocks", "0", "mech.dat")
    if not os.path.isfile(path):
        return None
    rows = []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) != len(MECH_COLS):
            continue
        try:
            v = [float(p) for p in parts]
        except ValueError:
            continue
        # a restart re-runs from the last time dir: drop rows it supersedes
        while rows and rows[-1][0] >= v[0] - 1e-9:
            rows.pop()
        rows.append(v)
    if len(rows) < 2:
        return None
    a = np.array(rows)
    return {k: a[:, i] for i, k in enumerate(MECH_COLS)}


def _integral(t, y, a, b):
    """Trapezoid integral of samples y(t) over [a, b], with interpolated ends."""
    if b <= a or t[0] > a + 1e-9 or t[-1] < b - 1e-9:
        return float("nan")
    sel = (t > a) & (t < b)
    tt = np.concatenate([[a], t[sel], [b]])
    yy = np.concatenate([[np.interp(a, t, y)], y[sel], [np.interp(b, t, y)]])
    return float(np.sum(0.5 * (yy[1:] + yy[:-1]) * np.diff(tt)))


def _mean(t, y, a, b):
    return _integral(t, y, a, b) / (b - a)


def window_power(run, cfg, a, b):
    pts = wave_points(run, cfg)
    tt = np.array([p[0] for p in pts])
    ww = np.array([p[1] for p in pts])
    grid = np.arange(a, b + 1e-9, 1e-3)
    w = np.interp(grid, tt, ww)
    p = F.motor_power.electrical_power(grid, w)
    return float(np.mean(p)), float(np.mean(w) / RPM)


def _fit_tau(t, y, decades=None):
    """e-folding time from a log-linear fit; nan if not decaying/positive.
    decades: fit only until y first falls below y[0]*10^-decades (keeps a
    noise floor out of the fit)."""
    if decades is not None and len(y):
        below = np.nonzero(y < y[0] * 10.0 ** (-decades))[0]
        if len(below):
            t, y = t[:below[0]], y[:below[0]]
    ok = np.isfinite(y) & (y > 0)
    if ok.sum() < 4:
        return float("nan")
    k = np.polyfit(t[ok] - t[ok][0], np.log(y[ok]), 1)[0]
    return float(-1.0 / k) if k < 0 else float("nan")


def stop_clocks(m, run, cfg):
    """Per-stop decay/overshoot measurements, medians over the stops after clock_t0."""
    t = m["t"]
    out = dict(n_stops=0, tau_swirl=np.nan, tau_esec=np.nan, tau_wall=np.nan,
               swirl_left=np.nan, esec_left=np.nan, wall_left=np.nan,
               overshoot=np.nan, tau_fade=np.nan)
    rows = []
    for s0, s1 in stop_intervals(run, cfg, cfg["clock_t0"]):
        a, b = s0 + 0.5 * RAMP, s1 - 0.5 * RAMP
        if b - a < 3 * CLOCK_DT or s0 - 0.5 < t[0] or s1 + min(run["run_s"], 2.0) > t[-1]:
            continue
        pre = (t >= s0 - 0.5) & (t <= s0 - RAMP)
        ins = (t >= a) & (t <= b)
        if pre.sum() < 3 or ins.sum() < 4:
            continue
        r = {}
        for k in ("swirl", "Esec", "wall"):
            ref = float(np.mean(m[k][pre]))
            r[f"{k}_left"] = float(m[k][ins][-1] / ref) if ref > 0 else np.nan
            # Esec decays onto a through-flow floor (~1e-4 m2/s2): fit 2 decades
            r[f"tau_{k}"] = _fit_tau(t[ins], m[k][ins], decades=2 if k == "Esec" else None)
        # restart: overshoot over the burst plateau (last 1 s of the burst), and
        # the e-folding time of the excess decaying back to it.
        burst_end = s1 + run["run_s"]
        post = (t >= s1) & (t <= s1 + min(run["run_s"], 2.0))
        plat = (t >= burst_end - min(1.0, 0.5 * run["run_s"])) & (t <= burst_end - RAMP)
        if post.sum() >= 4 and plat.sum() >= 3:
            p = float(np.mean(m["wall"][plat]))
            wp, tp = m["wall"][post], t[post]
            i = int(np.argmax(wp))
            r["overshoot"] = float(wp[i] / p - 1.0) if p > 0 else np.nan
            # fade: fit the excess over the plateau from the peak until it first
            # reaches the plateau
            ex = wp[i:] - p
            n_pos = int(np.argmax(ex <= 0)) if (ex <= 0).any() else len(ex)
            r["tau_fade"] = _fit_tau(tp[i:i + n_pos], ex[:n_pos])
        rows.append(r)
    if not rows:
        return out
    out["n_stops"] = len(rows)
    for key, src in (("tau_swirl", "tau_swirl"), ("tau_esec", "tau_Esec"), ("tau_wall", "tau_wall"),
                     ("swirl_left", "swirl_left"), ("esec_left", "Esec_left"), ("wall_left", "wall_left"),
                     ("overshoot", "overshoot"), ("tau_fade", "tau_fade")):
        vals = np.array([r.get(src, np.nan) for r in rows], dtype=float)
        out[key] = float(np.nanmedian(vals)) if np.isfinite(vals).any() else np.nan
    return out


def metrics(run, cfg, workdir):
    ep = cfg["episode"]
    a, b = score_window(run, cfg)
    res = dict(run, workdir=workdir, win_a=a, win_b=b, reached_t=0.0, complete=False, ok=False,
               x_out=np.nan, x_prev=np.nan, drift=np.nan, x_budget=np.nan, x_wallonly=np.nan,
               x_outnet=np.nan, closure=np.nan, p_motor=np.nan, mean_rpm_win=np.nan, r=np.nan,
               omega_err_rpm=np.nan, nvort_burst=np.nan, wall_burst=np.nan, wall_mean=np.nan)
    m = read_mech(workdir)
    if m is None:
        return res
    t = m["t"]
    res["reached_t"] = float(t[-1])
    res["complete"] = bool(t[-1] >= ep - 1e-6)
    pts = wave_points(run, cfg)
    w_cmd = np.interp(t, [p[0] for p in pts], [p[1] for p in pts])
    g = t > 0.1                       # the t=0 sample can still hold the warm-up BC
    res["omega_err_rpm"] = float(np.max(np.abs(m["Omega"][g] - w_cmd[g])) / RPM) if g.any() else np.nan
    finite = np.all(np.isfinite(m["xOut"])) and np.all((m["xOut"] > -0.02) & (m["xOut"] < 1.02))
    if res["complete"]:
        res["x_out"] = _mean(t, m["xOut"], a, b)
        L = b - a
        if a - L >= t[0]:
            res["x_prev"] = _mean(t, m["xOut"], a - L, a)
            res["drift"] = res["x_out"] - res["x_prev"]
        inflow = _integral(t, m["inAdv"] + m["inDiff"], a, b)
        wall = _integral(t, m["wall"], a, b)
        outn = _integral(t, m["outAdv"], a, b)
        dinv = float(np.interp(b, t, m["inv"]) - np.interp(a, t, m["inv"]))
        res["x_budget"] = (wall + dinv) / inflow
        res["x_wallonly"] = wall / inflow
        res["x_outnet"] = 1.0 - outn / inflow
        res["closure"] = (dinv - (inflow - outn - wall)) / inflow
        res["p_motor"], res["mean_rpm_win"] = window_power(run, cfg, a, b)
        win = (t >= a) & (t <= b)
        burst = win & (w_cmd >= 0.98 * run["w_hi"] * RPM)
        res["nvort_burst"] = float(np.median(m["nVort"][burst])) if burst.any() else np.nan
        res["wall_burst"] = float(np.mean(m["wall"][burst])) if burst.any() else np.nan
        res["wall_mean"] = _mean(t, m["wall"], a, b)
        res["r"] = res["x_out"] - res["p_motor"] / P_MAX
        res["ok"] = bool(finite) and res["omega_err_rpm"] < 1.0
    res.update(stop_clocks(m, run, cfg))
    return res


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def locate_peak(stops, xs):
    """Peak stop length: vertex of a quadratic in log(stop) through the max and
    its neighbours (falls back to the argmax). Returns (t_star, at_edge)."""
    k = int(np.argmax(xs))
    n = len(xs)
    if n < 3 or k == 0 or k == n - 1:
        return float(stops[k]), True
    u = np.log(stops[k - 1:k + 2])
    c2, c1, _ = np.polyfit(u, xs[k - 1:k + 2], 2)
    if c2 >= 0:
        return float(stops[k]), False
    uv = float(np.clip(-c1 / (2 * c2), u[0], u[-1]))
    return float(np.exp(uv)), False


def locate_cliff(stops, xs, sigma):
    """Largest drop between adjacent stops at or after the peak. Returns
    (edge, half_gap, drop) or None when the drop is under 3*sqrt(2)*sigma."""
    k = int(np.argmax(xs))
    d = np.diff(xs)
    if k >= len(d):
        return None
    i = k + int(np.argmin(d[k:]))
    drop = -float(d[i])
    if drop < 3.0 * math.sqrt(2.0) * sigma:
        return None
    return 0.5 * (stops[i] + stops[i + 1]), 0.5 * (stops[i + 1] - stops[i]), drop


def curves(rows):
    """Per viscosity: sorted stops, mean X (replicates averaged), constant X."""
    out = {}
    for nt in NU:
        rs = [r for r in rows if r["nu_tag"] == nt and r["exp"] in ("e1", "e2") and r["ok"]]
        if not rs:
            continue
        const = [r["x_out"] for r in rs if r["stop"] == 0]
        by = {}
        for r in rs:
            if r["stop"] > 0:
                by.setdefault(r["stop"], []).append(r)
        stops = np.array(sorted(by))
        out[nt] = dict(stops=stops,
                       x=np.array([np.mean([r["x_out"] for r in by[s]]) for s in stops]),
                       xb=np.array([np.mean([r["x_budget"] for r in by[s]]) for s in stops]),
                       r=np.array([np.mean([r["r"] for r in by[s]]) for s in stops]),
                       x_const=float(np.mean(const)) if const else np.nan,
                       xb_const=float(np.mean([r["x_budget"] for r in rs if r["stop"] == 0]))
                       if const else np.nan)
    return out


def replicate_sigma(rows):
    by = {}
    for r in rows:
        if r["ok"] and r["exp"] == "e1":
            by.setdefault((r["nu_tag"], r["stop"], r["w_lo"]), []).append(r["x_out"])
    diffs = [abs(v[0] - v[1]) / math.sqrt(2) for v in by.values() if len(v) == 2]
    if not diffs:
        return SIGMA_DEFAULT, 0
    return max(SIGMA_FLOOR, float(np.sqrt(np.mean(np.square(diffs))))), len(diffs)


def bootstrap(cv, sigma, key="x", seed=0):
    """Parametric bootstrap of peak / cliff per viscosity and of the exponent a."""
    rng = np.random.default_rng(seed)
    nts = [nt for nt in NU if nt in cv and len(cv[nt]["stops"]) >= 3]
    rep = {}
    peaks = {nt: np.empty(N_BOOT) for nt in nts}
    cliffs = {nt: np.full(N_BOOT, np.nan) for nt in nts}
    for nt in nts:
        s, x = cv[nt]["stops"], cv[nt][key]
        t0, edge = locate_peak(s, x)
        cl = locate_cliff(s, x, sigma)
        for i in range(N_BOOT):
            xb = x + rng.normal(0.0, sigma, size=len(x))
            peaks[nt][i] = locate_peak(s, xb)[0]
            c = locate_cliff(s, xb, sigma)
            if c:
                cliffs[nt][i] = c[0]
        rep[nt] = dict(nu_f=NU[nt], t_star=t0, at_edge=edge,
                       t_star_ci=[float(np.percentile(peaks[nt], 2.5)),
                                  float(np.percentile(peaks[nt], 97.5))],
                       cliff=None if cl is None else dict(edge=cl[0], half_gap=cl[1], drop=cl[2],
                                                          detect_frac=float(np.isfinite(cliffs[nt]).mean())),
                       x_peak=float(np.max(x)), x_const=cv[nt]["x_const"] if key == "x" else cv[nt]["xb_const"],
                       gain=float(np.max(x) - (cv[nt]["x_const"] if key == "x" else cv[nt]["xb_const"])))
    expo = None
    if len(nts) >= 2:
        lnu = np.log([NU[nt] for nt in nts])
        a_s = np.array([np.polyfit(lnu, np.log([peaks[nt][i] for nt in nts]), 1)[0]
                        for i in range(N_BOOT)])
        a0 = float(np.polyfit(lnu, np.log([rep[nt]["t_star"] for nt in nts]), 1)[0])
        expo = dict(viscosities=nts, a=a0, a_ci=[float(np.percentile(a_s, 2.5)),
                                                 float(np.percentile(a_s, 97.5))],
                    p_a_negative=float(np.mean(a_s < 0)),
                    momentum_excluded=bool(np.percentile(a_s, 97.5) < A_MOMENTUM or
                                           np.percentile(a_s, 2.5) > A_MOMENTUM),
                    film_excluded=bool(np.percentile(a_s, 97.5) < A_FILM_LAM or
                                       np.percentile(a_s, 2.5) > A_FILM_ETW))
        # the cliff edge, where every viscosity shows one
        if all(rep[nt]["cliff"] for nt in nts):
            ce = np.array([[cliffs[nt][i] for nt in nts] for i in range(N_BOOT)])
            good = np.all(np.isfinite(ce), axis=1)
            if good.sum() > 100:
                ac = np.array([np.polyfit(lnu, np.log(row), 1)[0] for row in ce[good]])
                expo["a_cliff"] = float(np.polyfit(lnu, np.log([rep[nt]["cliff"]["edge"] for nt in nts]), 1)[0])
                expo["a_cliff_ci"] = [float(np.percentile(ac, 2.5)), float(np.percentile(ac, 97.5))]
    return rep, expo


def clock_table(rows):
    out = {}
    for nt in NU:
        rs = [r for r in rows if r["nu_tag"] == nt and r["exp"] in ("e1", "e2") and r["ok"]
              and r["stop"] >= 0.4 and r.get("n_stops", 0) > 0]
        if not rs:
            continue
        def med(k):
            v = np.array([r[k] for r in rs], dtype=float)
            return float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan
        out[nt] = dict(nu_f=NU[nt], n_runs=len(rs),
                       tau_swirl=med("tau_swirl"), tau_esec=med("tau_esec"), tau_wall=med("tau_wall"),
                       overshoot=med("overshoot"), tau_fade=med("tau_fade"),
                       pred_swirl=GAP ** 2 / (math.pi ** 2 * NU0 * NU[nt]),
                       pred_film_etw=0.8 * NU[nt] ** A_FILM_ETW)
    return out


def load_all(results_dir, smoke):
    """Metrics for every run on disk, using the cfg each run was launched with."""
    rows = []
    for run in all_runs(smoke):
        wd = os.path.join(results_dir, "runs", run["tag"])
        rj = os.path.join(wd, "run.json")
        if os.path.isfile(rj):
            cfg = json.load(open(rj))["cfg"]
            kind = run.get("kind", "wave")
            stop = run["kick_stop"] if kind != "wave" else run["stop"]
            rows.append(metrics(make_run(run["exp"], run["nu_tag"], stop, run["w_lo"],
                                         run["rep"], cfg, kind), cfg, wd))
    return rows


SUMMARY_COLS = ["tag", "exp", "nu_tag", "stop", "w_lo", "w_hi", "mean", "rep", "period", "win_a", "win_b",
                "reached_t", "complete", "ok", "x_out", "x_prev", "drift", "x_budget", "x_wallonly",
                "x_outnet", "closure", "p_motor", "mean_rpm_win", "r", "omega_err_rpm",
                "nvort_burst", "wall_burst", "wall_mean", "n_stops",
                "tau_swirl", "tau_esec", "tau_wall", "swirl_left", "esec_left", "wall_left",
                "overshoot", "tau_fade"]


def analyze(results_dir, smoke):
    rows = load_all(results_dir, smoke)
    if not rows:
        print("No runs yet.")
        return
    with open(os.path.join(results_dir, "summary_table.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(SUMMARY_COLS)
        for r in rows:
            w.writerow([(f"{r[k]:.6g}" if isinstance(r[k], float) else r[k]) for k in SUMMARY_COLS])
    sigma, n_rep = replicate_sigma(rows)
    cv = curves(rows)
    clo = np.array([r["closure"] for r in rows], dtype=float)
    rep_x, expo_x = bootstrap(cv, sigma, "x")
    rep_b, expo_b = bootstrap(cv, sigma, "xb")
    clocks = clock_table(rows)
    e3_stop = (SMOKE if smoke else FULL)["e3_stop"]
    e3 = sorted([r for r in rows if r["ok"] and r["nu_tag"] == "nu1" and
                 (r["exp"] == "e3" or (r["exp"] == "e1" and abs(r["stop"] - e3_stop) < 1e-9 and not r["rep"]))],
                key=lambda r: r["w_lo"])
    report = dict(sigma=sigma, n_replicate_pairs=n_rep, primary_metric="X_out", e3_stop=e3_stop,
                  peaks_x_out=rep_x, exponent_x_out=expo_x,
                  peaks_x_budget=rep_b, exponent_x_budget=expo_b, clocks=clocks,
                  e3=[dict(w_lo=r["w_lo"], x_out=r["x_out"], x_budget=r["x_budget"],
                           esec_left=r["esec_left"], swirl_left=r["swirl_left"]) for r in e3],
                  e5=[dict(tag=r["tag"], x_out=r["x_out"], x_budget=r["x_budget"],
                           nvort_burst=r["nvort_burst"]) for r in rows if r["exp"] == "e5" and r["ok"]],
                  n_ok=sum(r["ok"] for r in rows), n_total=len(rows),
                  max_abs_closure=float(np.nanmax(np.abs(clo))) if np.isfinite(clo).any() else float("nan"))
    json.dump(report, open(os.path.join(results_dir, "mechanism_report.json"), "w"), indent=2, default=float)
    print_report(report, rows)
    plot_all(results_dir, rows, cv, rep_x, expo_x, clocks, e3, sigma)


def print_report(rep, rows):
    print("\n" + "=" * 90)
    print(f"MECHANISM: {rep['n_ok']}/{rep['n_total']} runs OK, sigma = {rep['sigma']:.4f} "
          f"({rep['n_replicate_pairs']} replicate pairs), max |closure| = {rep['max_abs_closure']:.2e}")
    for name, peaks, expo in (("X_out", rep["peaks_x_out"], rep["exponent_x_out"]),
                              ("X_budget", rep["peaks_x_budget"], rep["exponent_x_budget"])):
        print(f"-- {name}")
        for nt, p in peaks.items():
            cl = p["cliff"]
            cls = (f"cliff at {cl['edge']:.3f}+-{cl['half_gap']:.3f} s (drop {cl['drop']:.4f})"
                   if cl else "no cliff")
            print(f"   {nt:6s} t* = {p['t_star']:.3f} s [{p['t_star_ci'][0]:.3f}, {p['t_star_ci'][1]:.3f}]"
                  f"{' (EDGE)' if p['at_edge'] else ''}  gain over constant {p['gain']:+.4f}  {cls}")
        if expo:
            s = (f"   a = {expo['a']:+.2f} [{expo['a_ci'][0]:+.2f}, {expo['a_ci'][1]:+.2f}], "
                 f"P(a<0) = {expo['p_a_negative']:.2f}; momentum (a=-1) "
                 f"{'EXCLUDED' if expo['momentum_excluded'] else 'not excluded'}, film (a=+0.33..0.69) "
                 f"{'EXCLUDED' if expo['film_excluded'] else 'not excluded'}")
            print(s)
            if "a_cliff" in expo:
                print(f"   cliff edge: a = {expo['a_cliff']:+.2f} [{expo['a_cliff_ci'][0]:+.2f}, "
                      f"{expo['a_cliff_ci'][1]:+.2f}]")
    if rep["clocks"]:
        print("-- measured clocks (median over runs with stop >= 0.4 s)")
        for nt, c in rep["clocks"].items():
            print(f"   {nt:6s} tau_swirl {c['tau_swirl']:.3f} s (pred {c['pred_swirl']:.3f})  "
                  f"tau_Esec {c['tau_esec']:.3f}  tau_wall {c['tau_wall']:.3f}  "
                  f"overshoot {c['overshoot']:+.3f}  tau_fade {c['tau_fade']:.3f}")
    if rep["e3"]:
        print(f"-- e3 stop level (stop {rep['e3_stop']:g} s, nu1)")
        for e in rep["e3"]:
            print(f"   w_lo {e['w_lo']:5.0f} rpm  X_out {e['x_out']:.4f}  X_budget {e['x_budget']:.4f}  "
                  f"Esec left {e['esec_left']:.3f}  swirl left {e['swirl_left']:.3f}")
    if rep["e5"]:
        base = {r["stop"]: r for r in rows if r["exp"] == "e1" and r["ok"] and not r["rep"]}
        print("-- e5 vortex-state tests (nu1): compare with constant and the pulse")
        for nm, r in (("constant", base.get(0.0)), ("pulse", base.get(rep["e3_stop"]))):
            if r:
                print(f"   {nm:10s} X_out {r['x_out']:.4f}  Taylor cells (burst) {r['nvort_burst']:.0f}")
        for e in rep["e5"]:
            print(f"   {e['tag']:10s} X_out {e['x_out']:.4f}  X_budget {e['x_budget']:.4f}  "
                  f"Taylor cells {e['nvort_burst']:.0f}")
    bad = [r["tag"] for r in rows if not r["ok"]]
    if bad:
        print(f"-- not OK ({len(bad)}): {', '.join(bad)}")
    print("=" * 90 + "\n")


NU_COL = {"nu0p7": "#1f77b4", "nu1": "#222222", "nu2": "#c8102e"}
NU_LAB = {"nu0p7": "0.7x viscosity", "nu1": "nominal viscosity", "nu2": "2x viscosity"}


def plot_all(results_dir, rows, cv, peaks, expo, clocks, e3, sigma):
    plt.rcParams.update({"font.size": 12})
    # 1) conversion vs stop length, per viscosity (+ vortex count, burst wall flux)
    fig, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes2[0]
    for nt in NU:
        rs = sorted([r for r in rows if r["nu_tag"] == nt and r["exp"] in ("e1", "e2") and r["ok"]],
                    key=lambda r: r["stop"])
        const = [r for r in rs if r["stop"] == 0]
        pul = [r for r in rs if r["stop"] > 0]
        if not pul:
            continue
        axes2[1, 0].plot([r["stop"] for r in pul], [r["nvort_burst"] for r in pul], "o-",
                         color=NU_COL[nt], label=NU_LAB[nt])
        if const:
            axes2[1, 0].axhline(const[0]["nvort_burst"], color=NU_COL[nt], ls=":", lw=1)
            axes2[1, 1].plot([r["stop"] for r in pul],
                             [r["wall_burst"] / const[0]["wall_mean"] for r in pul], "o-",
                             color=NU_COL[nt], label=NU_LAB[nt])
    axes2[1, 0].set_ylabel("Taylor cells during bursts (median)\n(dotted: constant 300)")
    axes2[1, 1].set_ylabel("burst wall flux / constant-300 wall flux")
    for ax in axes2[1]:
        ax.set_xscale("log")
        ax.set_xlabel("stop length [s]")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=10)
    for nt, c in cv.items():
        for ax, key, kc in ((axes[0], "x", "x_const"), (axes[1], "xb", "xb_const")):
            ax.errorbar(c["stops"], c[key] - c[kc], yerr=sigma, fmt="-o", color=NU_COL[nt],
                        capsize=3, label=NU_LAB[nt])
            if key == "x" and nt in peaks:
                p = peaks[nt]
                ax.axvspan(p["t_star_ci"][0], p["t_star_ci"][1], color=NU_COL[nt], alpha=0.12)
    for ax, ttl in ((axes[0], "outlet conversion"), (axes[1], "wall-flux budget conversion")):
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("stop length [s]  (run 4.5 s, mean 300 rpm)")
        ax.set_ylabel("gain over constant 300 rpm")
        ax.set_title(ttl)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "stop_curves.png"), dpi=200)
    plt.close(fig)

    # 2) best stop vs viscosity with the pre-registered predictions
    if peaks:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for nt, p in peaks.items():
            ax.errorbar(p["nu_f"], p["t_star"],
                        yerr=[[p["t_star"] - p["t_star_ci"][0]], [p["t_star_ci"][1] - p["t_star"]]],
                        fmt="o", color=NU_COL[nt], ms=9, capsize=4,
                        label=f"best stop, {NU_LAB[nt]}")
            if p["cliff"]:
                ax.errorbar(p["nu_f"] * 1.03, p["cliff"]["edge"], yerr=p["cliff"]["half_gap"], fmt="s",
                            mfc="white", color=NU_COL[nt], ms=7, capsize=3)
        if "nu1" in peaks:
            t1 = peaks["nu1"]["t_star"]
            nf = np.linspace(0.6, 2.2, 50)
            ax.plot(nf, t1 * nf ** A_MOMENTUM, "--", color="#6b8fb3", label="swirl decay (a = -1)")
            ax.fill_between(nf, t1 * nf ** A_FILM_LAM, t1 * nf ** A_FILM_ETW, color="#e6a141",
                            alpha=0.3, label="wall layer (a = +0.33 .. +0.69)")
        if expo:
            ax.set_title(f"fitted a = {expo['a']:+.2f} [{expo['a_ci'][0]:+.2f}, {expo['a_ci'][1]:+.2f}]")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("viscosity / nominal  (D fixed)")
        ax.set_ylabel("best stop length [s]")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, "best_stop_vs_viscosity.png"), dpi=200)
        plt.close(fig)

    # 3) e3: conversion vs stop level
    if e3:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
        ax[0].plot([e["w_lo"] for e in e3], [e["x_out"] for e in e3], "-o", color="k")
        ax[0].axvline(61, color="#c8102e", ls=":", label="Taylor onset ~61 rpm")
        ax[0].set_xlabel("speed during the low period [rpm]")
        ax[0].set_ylabel("outlet conversion")
        ax[0].legend()
        ax[1].plot([e["esec_left"] for e in e3], [e["x_out"] for e in e3], "o", color="k")
        for e in e3:
            ax[1].annotate(f"{e['w_lo']:.0f}", (e["esec_left"], e["x_out"]), xytext=(4, 4),
                           textcoords="offset points", fontsize=9)
        ax[1].set_xlabel("secondary-flow energy left at end of low period (fraction)")
        ax[1].set_ylabel("outlet conversion")
        for a_ in ax:
            a_.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, "e3_stop_level.png"), dpi=200)
        plt.close(fig)

    # 4) cycle traces: last two periods of a few runs per viscosity
    sel = [r for r in rows if r["complete"] and r["exp"] in ("e1", "e2") and not r["rep"]
           and r["stop"] in (0.2, 0.5, 1.0)]
    if sel:
        fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
        for r in sel:
            m = read_mech(r["workdir"])
            if m is None:
                continue
            T = r["period"] if r["stop"] else 5.0
            sl = m["t"] >= r["win_b"] - 2 * T
            tt = m["t"][sl] - (r["win_b"] - 2 * T)
            lab = f"{r['nu_tag']} stop {r['stop']:g}"
            ls = {0.2: ":", 0.5: "-", 1.0: "--"}[r["stop"]]
            axes[0].plot(tt, m["Omega"][sl] / RPM, ls, color=NU_COL[r["nu_tag"]], lw=1, label=lab)
            axes[1].plot(tt, m["swirl"][sl], ls, color=NU_COL[r["nu_tag"]], lw=1)
            axes[2].plot(tt, m["Esec"][sl], ls, color=NU_COL[r["nu_tag"]], lw=1)
            axes[3].plot(tt, m["wall"][sl], ls, color=NU_COL[r["nu_tag"]], lw=1)
        for a_, yl in zip(axes, ("wall speed [rpm]", "mean swirl [m/s]",
                                 "secondary flow <ur^2+uz^2> [m2/s2]", "wall consumption [m3/s per c0]")):
            a_.set_ylabel(yl, fontsize=9)
            a_.grid(True, alpha=0.3)
        axes[2].set_yscale("log")
        axes[0].legend(fontsize=7, ncol=3)
        axes[-1].set_xlabel("time within the last two periods [s]")
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, "cycle_traces.png"), dpi=170)
        plt.close(fig)

    # 5) measured clocks vs viscosity
    if clocks:
        fig, ax = plt.subplots(figsize=(7, 5))
        nts = list(clocks)
        nf = [clocks[n]["nu_f"] for n in nts]
        for key, mk, lab in (("tau_swirl", "o", "swirl e-folding"), ("tau_esec", "s", "secondary-flow energy"),
                             ("tau_wall", "^", "wall flux during stop"), ("tau_fade", "v", "restart fade")):
            ax.plot(nf, [clocks[n][key] for n in nts], mk + "-", label=lab)
        ax.plot(nf, [clocks[n]["pred_swirl"] for n in nts], "k--", lw=1, label="d^2/(pi^2 nu)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("viscosity / nominal")
        ax.set_ylabel("time scale [s]")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, "measured_clocks.png"), dpi=200)
        plt.close(fig)
    print(f"Figures + tables in {results_dir}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="short pipeline test (6 s episodes)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--exp", default=None, help="comma list of e1,e2,e3,e5")
    ap.add_argument("--only", default=None, help="comma list of run tags")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--episode", type=float, default=None, help="override episode length (testing)")
    ap.add_argument("--results_dir", default=None)
    ap.add_argument("--work_root", default=None,
                    help="run the cases here (node-local disk) and copy outputs back to results_dir")
    args = ap.parse_args()

    cfg = dict(SMOKE if args.smoke else FULL)
    if args.episode:
        cfg["episode"] = args.episode
    results_dir = args.results_dir or os.path.join(RESULTS_ROOT, "smoke" if args.smoke else "main")
    runs = all_runs(args.smoke)
    if args.exp:
        keep = {e.strip() for e in args.exp.split(",")}
        runs = [r for r in runs if r["exp"] in keep]
    if args.only:
        keep = {s.strip() for s in args.only.split(",")}
        runs = [r for r in runs if r["tag"] in keep]

    if args.list:
        for r in runs:
            a, b = score_window(r, cfg)
            print(f"{r['tag']:26s} nu x{r['nu_f']:<4g} stop {r['stop']:.2f} s  T {r['period']:.2f} s  "
                  f"{r['w_lo']:.0f}<->{r['w_hi']:.1f} rpm (mean {r['mean']:.1f})  window [{a:.2f}, {b:.0f}]")
        print(f"{len(runs)} runs")
        return
    os.makedirs(results_dir, exist_ok=True)
    if args.analyze_only:
        analyze(results_dir, args.smoke)
        return
    for app in ("pimpleFoam", "foamDictionary"):
        if shutil.which(app) is None:
            sys.exit(f"ERROR: {app} not on PATH (source the OpenFOAM bashrc).")

    print(f"mechanism sweep -> {results_dir}\n  {len(runs)} runs x {cfg['episode']:g} s, "
          f"warm {cfg['warm']:g} s, maxCo {MAXCO}\n", flush=True)
    work_root = args.work_root or results_dir
    tpl_root = os.path.join(work_root, "_templates")
    os.makedirs(tpl_root, exist_ok=True)
    t0 = time.time()
    base = prepare_base(tpl_root)
    print(f"  base template compiled ({time.time() - t0:.0f} s)", flush=True)
    todo = [r for r in runs if not (args.resume and os.path.isfile(
        os.path.join(results_dir, "runs", r["tag"], "COMPLETE")))]
    need = sorted({r["nu_tag"] for r in todo})       # skip warm-ups nobody needs
    with ThreadPoolExecutor(max_workers=max(1, len(need))) as ex:
        tpls = dict(zip(need, ex.map(lambda nt: warm_template(base, tpl_root, nt, cfg), need)))
    for nt, d in tpls.items():                      # keep a record next to the results
        rec = os.path.join(results_dir, "_templates", nt)
        os.makedirs(rec, exist_ok=True)
        for f in ("WARM_OK.json", "log.warmup"):
            if os.path.isfile(os.path.join(d, f)) and os.path.abspath(d) != os.path.abspath(rec):
                shutil.copy2(os.path.join(d, f), os.path.join(rec, f))
    print(f"  warm templates ready ({(time.time() - t0) / 60:.1f} min)\n", flush=True)

    # slowest first so the tail is short: the 2x-viscosity arm ran ~1.7x slower
    # per simulated second in the smoke test; then highest peak speed (smallest dt)
    runs.sort(key=lambda r: (-r["nu_f"], -r["w_hi"]))
    workers = args.workers or len(runs)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_episode, r, cfg, tpls, results_dir, work_root, args.resume) for r in runs]
        for fut in as_completed(futs):
            fut.result()
    print(f"\nall runs done in {(time.time() - t0) / 3600:.2f} h", flush=True)
    analyze(results_dir, args.smoke)


if __name__ == "__main__":
    main()
