#!/usr/bin/env python3
"""
Configure a tc_mixing_case copy for the "oscillating vs constant omega" study.

This does NOT use the RL env / train.py. It prescribes omega(t) directly on the
inner-cylinder rotatingWallVelocity BC and lets pimpleFoam run straight through,
which is both cheaper and a faithful analogue of Lopez-Guajardo et al. (CEJ 489
(2024) 151174), who solved a 2D-axisymmetric model -- exactly what this wedge is.

Two modes (same mean speed, so the comparison is apples-to-apples):
  constant    omega = mean (default 500 rpm) for the whole run.
  squarewave  omega = 0 during the idle fraction, omega_high during the active
              fraction, with mean == the constant case. Per the paper's best
              modulated case: duty D=0.2, period T=30 s  ->  omega_high = mean/D
              = 2500 rpm, idle = 0 rpm. Transitions are linear ramps of width
              --ramp (default 0.05 s) centred on each switch, so the time-mean is
              preserved exactly and there is no infinite angular acceleration.

It edits an existing case copy IN PLACE:
  * seeds 0/ from 0.orig (uniform fields -> mesh-agnostic, clean cold start),
  * writes 0/U with the requested omega spec on inner_wall,
  * writes system/controlDict with adjustTimeStep (stable at the 2500 rpm peak)
    and the scalarTransport + rlMetrics function objects that emit the outlet
    radial dye profile we score mixing on.

Usage:
  python make_case.py --case <run_case_dir> --mode squarewave \
        --mean-rpm 500 --duty 0.2 --period 30 --run-seconds 120
"""
import argparse
import math
import os
import shutil
import sys

RPM = 2.0 * math.pi / 60.0   # rpm -> rad/s


# --------------------------------------------------------------------------- #
# omega(t) specifications
# --------------------------------------------------------------------------- #
def constant_omega_entry(mean_rad):
    return f"        omega           {mean_rad:.6f};   // {mean_rad/RPM:.1f} rpm, constant\n"


def squarewave_points(mean_rad, duty, period, run_seconds, ramp):
    """Piecewise-linear omega(t): idle 0, active mean/duty, ramps width `ramp`
    centred on each switch (mean preserved). Active phase leads each period."""
    w_high = mean_rad / duty            # mean = duty*w_high + (1-duty)*0
    half = 0.5 * ramp
    pts = [(0.0, w_high)]               # cold start straight into the active phase
    n_periods = int(math.ceil(run_seconds / period)) + 1
    for k in range(n_periods):
        t0 = k * period
        t_active_end = t0 + duty * period
        t_period_end = t0 + period
        # ramp down at the end of the active phase
        pts.append((t_active_end - half, w_high))
        pts.append((t_active_end + half, 0.0))
        # ramp up into the next active phase
        pts.append((t_period_end - half, 0.0))
        pts.append((t_period_end + half, w_high))
    # keep strictly increasing, drop anything well past the run
    out = []
    for t, w in pts:
        t = max(t, 0.0)
        if out and t <= out[-1][0]:
            t = out[-1][0] + 1e-6
        out.append((t, w))
        if t > run_seconds + period:
            break
    return w_high, out


def squarewave_omega_entry(mean_rad, duty, period, run_seconds, ramp):
    w_high, pts = squarewave_points(mean_rad, duty, period, run_seconds, ramp)
    lines = [
        f"        // square wave: idle 0 rpm / active {w_high/RPM:.0f} rpm, "
        f"D={duty}, T={period}s, mean {mean_rad/RPM:.0f} rpm\n",
        "        omega           table\n",
        "        (\n",
    ]
    lines += [f"            ({t:.5f} {w:.5f})\n" for t, w in pts]
    lines += ["        );\n"]
    return "".join(lines)


# --------------------------------------------------------------------------- #
# file writers
# --------------------------------------------------------------------------- #
def write_U(case, omega_entry, geometry):
    # Patches differ by geometry: the 2D wedge has wedge front/back and a
    # rotating (omega 0) outer wall; the full 3D annulus has a no-slip outer
    # wall and no wedge patches. Everything else (inner_wall rotation, top
    # inlet velocity, bottom outflow) is identical.
    if geometry == "wedge":
        outer = '''    "outer_wall"
    {
        type            rotatingWallVelocity;
        origin          ( 0 0 0 );
        axis            ( 0 0 1 );
        omega           0;
    }'''
        wedge = '''    "front"         { type wedge; }
    "back"          { type wedge; }
'''
    else:  # full3d
        outer = '''    "outer_wall"
    {
        type            noSlip;
    }'''
        wedge = ""

    path = os.path.join(case, "0", "U")
    with open(path, "w") as f:
        f.write(f"""\
FoamFile
{{
    version         2;
    format          ascii;
    class           volVectorField;
    object          U;
}}

dimensions      [ 0 1 -1 0 0 0 0 ];

internalField   uniform ( 0 0 0 );

boundaryField
{{
    "inner_wall"
    {{
        type            rotatingWallVelocity;
        origin          ( 0 0 0 );
        axis            ( 0 0 1 );
{omega_entry}    }}
{outer}
    "top"
    {{
        type            fixedValue;
        value           uniform ( 0 0 -0.001462 );   // u_ax = Q0/A (100 mL/min)
    }}
    "bottom"
    {{
        type            inletOutlet;
        inletValue      uniform ( 0 0 0 );
        value           uniform ( 0 0 0 );
    }}
{wedge}}}
""")


# scalarTransport + rlMetrics, copied verbatim from the master controlDict so the
# outlet radial dye profile (C0..C19) is computed identically. rlMetrics is rate-
# limited to executeInterval below so the METRICS log stays small under tiny dt.
FUNCTIONS_BLOCK = r"""
functions
{
    scalarTransport
    {
        type            scalarTransport;
        libs            ( solverFunctionObjects );
        field           C;
        resetOnStartUp  false;
        schemesField    C;
        D               __D__;
        nCorr           1;
        executeControl  timeStep;
        executeInterval 1;
        writeControl    writeTime;
    }
    rlMetrics
    {
        type            coded;
        libs            ( utilityFunctionObjects );
        name            rlMetrics;
        executeControl  timeStep;
        executeInterval __SAMPLE__;
        codeOptions     #{
            -I$(LIB_SRC)/TurbulenceModels/turbulenceModels/lnInclude \
            -I$(LIB_SRC)/TurbulenceModels/incompressible/lnInclude \
            -I$(LIB_SRC)/transportModels \
            -I$(LIB_SRC)/transportModels/incompressible/lnInclude
        #};
        codeLibs        #{
            -lincompressibleTurbulenceModels \
            -lturbulenceModels \
            -lincompressibleTransportModels
        #};
        codeInclude     #{
            #include "turbulentTransportModel.H"
        #};
        codeExecute     #{
            const fvMesh& mesh = dynamic_cast<const fvMesh&>(this->mesh());
            const Time&   runTime = mesh.time();

            const volScalarField& C = mesh.lookupObject<volScalarField>("C");
            const volVectorField& U = mesh.lookupObject<volVectorField>("U");

            // ---------- 20 radial bins at the bottom outlet ----------
            const label bottomID = mesh.boundaryMesh().findPatchID("bottom");
            const scalar Rin  = 0.0254;
            const scalar Rout = 0.03175;
            const label  nBins = 20;
            scalarField  binC(nBins, 0.0);
            scalarField  binVz(nBins, 0.0);
            scalarField  binW(nBins, 0.0);

            if (bottomID >= 0)
            {
                const scalarField& Cb = C.boundaryField()[bottomID];
                const vectorField& Ub = U.boundaryField()[bottomID];
                const vectorField& Cf = mesh.Cf().boundaryField()[bottomID];
                const scalarField  magSf(mag(mesh.Sf().boundaryField()[bottomID]));

                forAll(Cb, i)
                {
                    scalar r = sqrt(sqr(Cf[i].x()) + sqr(Cf[i].y()));
                    label  b = label((r - Rin) / (Rout - Rin) * nBins);
                    if (b < 0) b = 0;
                    if (b >= nBins) b = nBins - 1;
                    binC[b]  += Cb[i] * magSf[i];
                    binVz[b] += Ub[i].z() * magSf[i];
                    binW[b]  += magSf[i];
                }
            }
            reduce(binC,  sumOp<scalarField>());
            reduce(binVz, sumOp<scalarField>());
            reduce(binW,  sumOp<scalarField>());
            forAll(binC, b)
            {
                if (binW[b] > SMALL)
                {
                    binC[b]  /= binW[b];
                    binVz[b] /= binW[b];
                }
            }

            // ---------- Viscous torque on the inner cylinder ----------
            const label innerID = mesh.boundaryMesh().findPatchID("inner_wall");

            const incompressible::turbulenceModel& turb =
                mesh.lookupObject<incompressible::turbulenceModel>
                ("turbulenceProperties");

            tmp<volSymmTensorField> tdevReff = turb.devReff();
            const volSymmTensorField& devReff = tdevReff();

            vector M(vector::zero);
            if (innerID >= 0)
            {
                const vectorField& Sf  = mesh.Sf().boundaryField()[innerID];
                const vectorField& Cf  = mesh.Cf().boundaryField()[innerID];
                const symmTensorField& tauB = devReff.boundaryField()[innerID];

                forAll(Sf, i)
                {
                    M += (Cf[i] ^ (tauB[i] & Sf[i]));
                }
            }
            reduce(M, sumOp<vector>());

            // ---------- Log: t, Mz (kinematic), C and Vz values ----------
            Info<< "METRICS t=" << runTime.value()
                << " Mz_kin=" << M.z();
            forAll(binC, b)
            {
                Info<< " C" << b << "=" << binC[b];
            }
            forAll(binVz, b)
            {
                Info<< " Vz" << b << "=" << binVz[b];
            }
            Info<< endl;
        #};
    }
}
"""


# Catalysis variant: same torque + outlet radial bins, PLUS the two quantities
# the paper actually reports -- cup-mixing conversion at the outlet and the
# diffusive flux consumed at the catalytic (outer) wall.
FUNCTIONS_BLOCK_CATALYSIS = r"""
functions
{
    scalarTransport
    {
        type            scalarTransport;
        libs            ( solverFunctionObjects );
        field           C;
        resetOnStartUp  false;
        schemesField    C;
        D               __D__;
        nCorr           1;
        executeControl  timeStep;
        executeInterval 1;
        writeControl    writeTime;
    }
    catalysis
    {
        type            coded;
        libs            ( utilityFunctionObjects );
        name            catalysis;
        executeControl  timeStep;
        executeInterval __SAMPLE__;
        codeOptions     #{
            -I$(LIB_SRC)/TurbulenceModels/turbulenceModels/lnInclude \
            -I$(LIB_SRC)/TurbulenceModels/incompressible/lnInclude \
            -I$(LIB_SRC)/transportModels \
            -I$(LIB_SRC)/transportModels/incompressible/lnInclude
        #};
        codeLibs        #{
            -lincompressibleTurbulenceModels \
            -lturbulenceModels \
            -lincompressibleTransportModels
        #};
        codeInclude     #{
            #include "turbulentTransportModel.H"
        #};
        codeExecute     #{
            const fvMesh& mesh = dynamic_cast<const fvMesh&>(this->mesh());
            const Time&   runTime = mesh.time();

            const volScalarField& C = mesh.lookupObject<volScalarField>("C");
            const volVectorField& U = mesh.lookupObject<volVectorField>("U");

            const scalar Dmol = __D__;          // species diffusivity (matches scalarTransport)
            const scalar Rin  = 0.0254;
            const scalar Rout = 0.03175;
            const label  nBins = 20;
            scalarField  binC(nBins, 0.0);
            scalarField  binVz(nBins, 0.0);
            scalarField  binW(nBins, 0.0);

            // ---------- bottom outlet: radial bins + cup-mixing conversion ----------
            const label bottomID = mesh.boundaryMesh().findPatchID("bottom");
            scalar cupNum = 0.0, cupDen = 0.0;   // flux-weighted (cup-mixing) average C
            if (bottomID >= 0)
            {
                const scalarField& Cb = C.boundaryField()[bottomID];
                const vectorField& Ub = U.boundaryField()[bottomID];
                const vectorField& Cf = mesh.Cf().boundaryField()[bottomID];
                const scalarField  magSf(mag(mesh.Sf().boundaryField()[bottomID]));
                forAll(Cb, i)
                {
                    scalar r = sqrt(sqr(Cf[i].x()) + sqr(Cf[i].y()));
                    label  b = label((r - Rin) / (Rout - Rin) * nBins);
                    if (b < 0) b = 0;
                    if (b >= nBins) b = nBins - 1;
                    binC[b]  += Cb[i] * magSf[i];
                    binVz[b] += Ub[i].z() * magSf[i];
                    binW[b]  += magSf[i];
                    scalar f = mag(Ub[i].z()) * magSf[i];   // |axial volumetric flux|
                    cupNum += Cb[i] * f;
                    cupDen += f;
                }
            }
            reduce(binC,  sumOp<scalarField>());
            reduce(binVz, sumOp<scalarField>());
            reduce(binW,  sumOp<scalarField>());
            reduce(cupNum, sumOp<scalar>());
            reduce(cupDen, sumOp<scalar>());
            forAll(binC, b)
                if (binW[b] > SMALL) { binC[b] /= binW[b]; binVz[b] /= binW[b]; }
            scalar cupC = (cupDen > SMALL) ? cupNum / cupDen : 0.0;
            scalar conv = 1.0 - cupC;            // c0 = 1, so conversion = 1 - cup outlet C

            // ---------- catalytic (outer) wall: diffusive consumption flux ----------
            const label outerID = mesh.boundaryMesh().findPatchID("outer_wall");
            scalar wallFlux = 0.0;               // mol/s (per unit c0): -D * dC/dn integrated
            if (outerID >= 0)
            {
                const scalarField snGradC(C.boundaryField()[outerID].snGrad());
                const scalarField magSf(mag(mesh.Sf().boundaryField()[outerID]));
                forAll(snGradC, i) wallFlux += -Dmol * snGradC[i] * magSf[i];
            }
            reduce(wallFlux, sumOp<scalar>());

            // ---------- inner-cylinder torque (for power/energy) ----------
            const label innerID = mesh.boundaryMesh().findPatchID("inner_wall");
            const incompressible::turbulenceModel& turb =
                mesh.lookupObject<incompressible::turbulenceModel>("turbulenceProperties");
            tmp<volSymmTensorField> tdevReff = turb.devReff();
            const volSymmTensorField& devReff = tdevReff();
            vector M(vector::zero);
            if (innerID >= 0)
            {
                const vectorField& Sf  = mesh.Sf().boundaryField()[innerID];
                const vectorField& Cf  = mesh.Cf().boundaryField()[innerID];
                const symmTensorField& tauB = devReff.boundaryField()[innerID];
                forAll(Sf, i) M += (Cf[i] ^ (tauB[i] & Sf[i]));
            }
            reduce(M, sumOp<vector>());

            Info<< "METRICS t=" << runTime.value()
                << " Mz_kin=" << M.z()
                << " conv=" << conv
                << " cupC=" << cupC
                << " wallFlux=" << wallFlux;
            forAll(binC, b)  Info<< " C"  << b << "=" << binC[b];
            forAll(binVz, b) Info<< " Vz" << b << "=" << binVz[b];
            Info<< endl;
        #};
    }
}
"""


def write_C_catalysis(case, geometry):
    # Reactant fed at c0=1, consumed at the catalytic outer wall (C=0 sink).
    # Inner (rotating) wall is inert (zeroGradient). Conversion = 1 - cup outlet C.
    wedge = ('    "front"         { type wedge; }\n'
             '    "back"          { type wedge; }\n') if geometry == "wedge" else ""
    with open(os.path.join(case, "0", "C"), "w") as f:
        f.write(f"""\
FoamFile
{{
    version         2;
    format          ascii;
    class           volScalarField;
    object          C;
}}

dimensions      [0 0 0 0 0 0 0];

internalField   uniform 1;                 // reactor initially full of feed (c0=1)

boundaryField
{{
    "inner_wall"    {{ type zeroGradient; }}             // inert rotating wall
    "outer_wall"    {{ type fixedValue; value uniform 0; }}   // CATALYTIC wall: fast reaction sink
    "top"           {{ type fixedValue; value uniform 1; }}   // reactant feed c0=1
    "bottom"
    {{
        type            inletOutlet;
        inletValue      uniform 0;
        value           uniform 1;
    }}
{wedge}}}
""")


def write_controlDict(case, run_seconds, max_co, max_dt, sample_interval,
                      funcs_block=FUNCTIONS_BLOCK, scalar_d="1e-09"):
    header = f"""\
FoamFile
{{
    version         2;
    format          ascii;
    class           dictionary;
    location        "system";
    object          controlDict;
}}

application     pimpleFoam;

startFrom       startTime;
startTime       0;

stopAt          endTime;
endTime         {run_seconds};

// Stable at the 2500 rpm modulation peak (azimuthal Courant on the inner wall
// drives dt down to ~3e-4 s there; it relaxes during the idle phase).
deltaT          1e-4;
adjustTimeStep  yes;
maxCo           {max_co};
maxDeltaT       {max_dt};

writeControl    adjustableRunTime;
writeInterval   1;        // one field snapshot per simulated second

purgeWrite      0;
writeFormat     ascii;
writePrecision  9;
writeCompression no;
timeFormat      general;
timePrecision   9;
runTimeModifiable no;
"""
    funcs = (funcs_block.replace("__SAMPLE__", f"{sample_interval}")
                        .replace("__D__", f"{scalar_d}"))
    with open(os.path.join(case, "system", "controlDict"), "w") as f:
        f.write(header + funcs)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="case copy to configure in place")
    ap.add_argument("--geometry", default="wedge", choices=["wedge", "full3d"],
                    help="wedge (2D axisymmetric) or full3d (360 annulus) patch layout")
    ap.add_argument("--mode", required=True, choices=["constant", "squarewave"])
    ap.add_argument("--mean-rpm", type=float, default=500.0)
    ap.add_argument("--duty", type=float, default=0.20)
    ap.add_argument("--period", type=float, default=30.0)
    ap.add_argument("--ramp", type=float, default=0.05)
    ap.add_argument("--run-seconds", type=float, default=120.0)
    ap.add_argument("--max-co", type=float, default=0.8)
    ap.add_argument("--max-dt", type=float, default=1e-3)
    ap.add_argument("--sample-interval", type=int, default=20,
                    help="emit a METRICS line every N timesteps (analyze.py uses "
                         "the logged t= values, so irregular dt spacing is fine)")
    ap.add_argument("--catalysis", action="store_true",
                    help="catalytic-wall mode: feed reactant c0=1, outer wall = C=0 "
                         "sink, log conversion + wall flux (the paper's actual metric)")
    ap.add_argument("--scalar-D", default="1e-8",
                    help="species diffusivity [m^2/s] in catalysis mode (paper's is "
                         "~1e-10 / Sc~1e5; 1e-8 keeps the wall boundary layer resolvable)")
    args = ap.parse_args()

    case = os.path.abspath(args.case)
    orig = os.path.join(case, "0.orig")
    zero = os.path.join(case, "0")

    # Clean cold start. The wedge ships a uniform 0.orig template; the full 3D
    # case has no 0.orig but its 0/ fields are already uniform (mesh-agnostic),
    # so we keep 0/ and just overwrite U below. Either way internalField is
    # uniform, so the start is identical for both omega modes.
    if os.path.isdir(orig):
        if os.path.exists(zero):
            shutil.rmtree(zero)
        shutil.copytree(orig, zero)
    elif not os.path.isdir(zero):
        sys.exit(f"{case} has neither 0.orig/ nor 0/ -- not a usable case copy")

    mean_rad = args.mean_rpm * RPM
    if args.mode == "constant":
        omega_entry = constant_omega_entry(mean_rad)
    else:
        omega_entry = squarewave_omega_entry(
            mean_rad, args.duty, args.period, args.run_seconds, args.ramp)

    write_U(case, omega_entry, args.geometry)
    if args.catalysis:
        write_C_catalysis(case, args.geometry)
        write_controlDict(case, args.run_seconds, args.max_co, args.max_dt,
                          args.sample_interval,
                          funcs_block=FUNCTIONS_BLOCK_CATALYSIS,
                          scalar_d=args.scalar_D)
    else:
        write_controlDict(case, args.run_seconds, args.max_co, args.max_dt,
                          args.sample_interval)

    print(f"[make_case] {args.mode}{' CATALYSIS' if args.catalysis else ''}: "
          f"mean={args.mean_rpm:.0f} rpm "
          f"({mean_rad:.4f} rad/s), run={args.run_seconds:.0f}s -> {case}")
    if args.catalysis:
        print(f"[make_case]   feed c0=1, outer wall=C0 sink, D={args.scalar_D} m^2/s")
    if args.mode == "squarewave":
        w_high, _ = squarewave_points(mean_rad, args.duty, args.period,
                                      args.run_seconds, args.ramp)
        print(f"[make_case]   active={w_high/RPM:.0f} rpm for {args.duty*args.period:.1f}s "
              f"every {args.period:.0f}s (idle 0 rpm), ramp {args.ramp}s")


if __name__ == "__main__":
    main()
