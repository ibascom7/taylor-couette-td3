/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2019-2021 OpenCFD Ltd.
    Copyright (C) YEAR AUTHOR, AFFILIATION
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "functionObjectTemplate.H"
#define namespaceFoam  // Suppress <using namespace Foam;>
#include "fvCFD.H"
#include "unitConversion.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

defineTypeNameAndDebug(rlMetricsFunctionObject, 0);

addRemovableToRunTimeSelectionTable
(
    functionObject,
    rlMetricsFunctionObject,
    dictionary
);


// * * * * * * * * * * * * * * * Global Functions  * * * * * * * * * * * * * //

// dynamicCode:
// SHA1 = 68499454f7dfc132d82f8fa5a80863ef9bcd0e68
//
// unique function name that can be checked if the correct library version
// has been loaded
extern "C" void rlMetrics_68499454f7dfc132d82f8fa5a80863ef9bcd0e68(bool load)
{
    if (load)
    {
        // Code that can be explicitly executed after loading
    }
    else
    {
        // Code that can be explicitly executed before unloading
    }
}


// * * * * * * * * * * * * * * * Local Functions * * * * * * * * * * * * * * //

//{{{ begin localCode

//}}} end localCode

} // End namespace Foam


// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

const Foam::fvMesh&
Foam::rlMetricsFunctionObject::mesh() const
{
    return refCast<const fvMesh>(obr_);
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::
rlMetricsFunctionObject::
rlMetricsFunctionObject
(
    const word& name,
    const Time& runTime,
    const dictionary& dict
)
:
    functionObjects::regionFunctionObject(name, runTime, dict)
{
    read(dict);
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::
rlMetricsFunctionObject::
~rlMetricsFunctionObject()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

bool
Foam::
rlMetricsFunctionObject::read(const dictionary& dict)
{
    if (false)
    {
        printMessage("read rlMetrics");
    }

//{{{ begin code
    
//}}} end code

    return true;
}


bool
Foam::
rlMetricsFunctionObject::execute()
{
    if (false)
    {
        printMessage("execute rlMetrics");
    }

//{{{ begin code
    #line 86 "/home/ibascom/research/taylor-couette-td3/taylorCouetteGym/experiments/modulation_rl/results/full_tc_eval/s2_wrap/case/system/controlDict/functions/rlMetrics"
const fvMesh& mesh = refCast<const fvMesh>(obr_);
            const Time& runTime = mesh.time();

            const volScalarField& c = mesh.lookupObject<volScalarField>("c");
            const volVectorField& U = mesh.lookupObject<volVectorField>("U");
            const surfaceScalarField& phi =
                mesh.lookupObject<surfaceScalarField>("phi");

            // transport properties (nu for the torque, D for the wall flux)
            IOdictionary transportProperties
            (
                IOobject
                (
                    "transportProperties",
                    runTime.constant(),
                    mesh,
                    IOobject::MUST_READ_IF_MODIFIED,
                    IOobject::NO_WRITE
                )
            );
            const scalar nu = dimensionedScalar("nu", dimViscosity, transportProperties).value();
            const scalar Dmol = dimensionedScalar("D", dimViscosity, transportProperties).value();

            const scalar c0 = 50.0;   // feed concentration [mmol/m3] (0/c inlet)

            // ---------- inner-wall kinematic torque about z ----------
            scalar Mz = 0.0;
            const label innerID = mesh.boundaryMesh().findPatchID("inner_wall");
            if (innerID >= 0)
            {
                const volTensorField gradU(fvc::grad(U));
                const fvPatchTensorField& gradUp = gradU.boundaryField()[innerID];
                const fvPatch& pIn = mesh.boundary()[innerID];
                const vectorField& Cf = pIn.Cf();
                const vectorField& Sf = pIn.Sf();
                forAll(Cf, i)
                {
                    const tensor g = gradUp[i];
                    // kinematic viscous traction * area  =  nu*(gradU + gradU^T) . Sf
                    const vector tr = nu*((g + g.T()) & Sf[i]);
                    Mz += (Cf[i] ^ tr).z();
                }
            }
            reduce(Mz, sumOp<scalar>());

            // ---------- side-outlet cup-mixing conversion (phi weighted) ----------
            scalar num = 0.0, den = 0.0;
            const label outID = mesh.boundaryMesh().findPatchID("side_outlet");
            if (outID >= 0)
            {
                const fvPatchScalarField& cp = c.boundaryField()[outID];
                const fvsPatchScalarField& phip = phi.boundaryField()[outID];
                forAll(cp, i)
                {
                    const scalar f = phip[i];
                    if (f > 0.0) { den += f; num += f*cp[i]; }
                }
            }
            reduce(num, sumOp<scalar>());
            reduce(den, sumOp<scalar>());
            const scalar cupC = (den > VSMALL) ? num/den : 0.0;
            const scalar conv = 1.0 - cupC/c0;

            // ---------- catalytic outer-wall consumption flux (RESOLVED) ----------
            // The c=0 wall consumes reactant by molecular diffusion across the
            // graded near-wall cells: flux into the wall = -D * dc/dn * A summed
            // over the patch (snGrad < 0 since c drops to 0 at the wall, so the
            // sum is positive). Divided by c0 -> [m^3/s per unit c0], the exact
            // convention of side_outlet_cat_case's modeled flux, so wf_index =
            // wallFlux/wallflux_max and the Q*c0 ceiling work unchanged.
            scalar wallFlux = 0.0;
            const label outerID = mesh.boundaryMesh().findPatchID("outer_wall");
            if (outerID >= 0)
            {
                const fvPatchScalarField& cw = c.boundaryField()[outerID];
                const scalarField snG(cw.snGrad());
                const scalarField& magSf = mesh.boundary()[outerID].magSf();
                forAll(snG, i)
                {
                    wallFlux -= Dmol*snG[i]*magSf[i];
                }
            }
            reduce(wallFlux, sumOp<scalar>());
            wallFlux /= c0;

            Info<< "METRICS t=" << runTime.value()
                << " Mz_kin=" << Mz
                << " conv=" << conv
                << " cupC=" << cupC
                << " wallFlux=" << wallFlux
                << endl;
//}}} end code

    return true;
}


bool
Foam::
rlMetricsFunctionObject::write()
{
    if (false)
    {
        printMessage("write rlMetrics");
    }

//{{{ begin code
    
//}}} end code

    return true;
}


bool
Foam::
rlMetricsFunctionObject::end()
{
    if (false)
    {
        printMessage("end rlMetrics");
    }

//{{{ begin code
    
//}}} end code

    return true;
}


// ************************************************************************* //

