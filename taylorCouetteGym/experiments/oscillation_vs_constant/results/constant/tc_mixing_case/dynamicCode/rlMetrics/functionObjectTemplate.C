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
// SHA1 = 1567184ff445ec1435b07213d50b74498fb08ccc
//
// unique function name that can be checked if the correct library version
// has been loaded
extern "C" void rlMetrics_1567184ff445ec1435b07213d50b74498fb08ccc(bool load)
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
    #line 73 "/project/mwang/ibascom/taylor-couette-td3/taylorCouetteGym/experiments/oscillation_vs_constant/paraview/constant/tc_mixing_case/system/controlDict/functions/rlMetrics"
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

