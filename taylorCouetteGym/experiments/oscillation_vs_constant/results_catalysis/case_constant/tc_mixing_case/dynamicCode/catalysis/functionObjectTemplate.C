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

defineTypeNameAndDebug(catalysisFunctionObject, 0);

addRemovableToRunTimeSelectionTable
(
    functionObject,
    catalysisFunctionObject,
    dictionary
);


// * * * * * * * * * * * * * * * Global Functions  * * * * * * * * * * * * * //

// dynamicCode:
// SHA1 = ec602091c42c16faf62a6584cdc5fc9ae4ccf17b
//
// unique function name that can be checked if the correct library version
// has been loaded
extern "C" void catalysis_ec602091c42c16faf62a6584cdc5fc9ae4ccf17b(bool load)
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
Foam::catalysisFunctionObject::mesh() const
{
    return refCast<const fvMesh>(obr_);
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::
catalysisFunctionObject::
catalysisFunctionObject
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
catalysisFunctionObject::
~catalysisFunctionObject()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

bool
Foam::
catalysisFunctionObject::read(const dictionary& dict)
{
    if (false)
    {
        printMessage("read catalysis");
    }

//{{{ begin code
    
//}}} end code

    return true;
}


bool
Foam::
catalysisFunctionObject::execute()
{
    if (false)
    {
        printMessage("execute catalysis");
    }

//{{{ begin code
    #line 73 "/project/mwang/ibascom/taylor-couette-td3/taylorCouetteGym/experiments/oscillation_vs_constant/results_catalysis/case_constant/tc_mixing_case/system/controlDict/functions/catalysis"
const fvMesh& mesh = dynamic_cast<const fvMesh&>(this->mesh());
            const Time&   runTime = mesh.time();

            const volScalarField& C = mesh.lookupObject<volScalarField>("C");
            const volVectorField& U = mesh.lookupObject<volVectorField>("U");

            const scalar Dmol = 1e-8;          // species diffusivity (matches scalarTransport)
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
//}}} end code

    return true;
}


bool
Foam::
catalysisFunctionObject::write()
{
    if (false)
    {
        printMessage("write catalysis");
    }

//{{{ begin code
    
//}}} end code

    return true;
}


bool
Foam::
catalysisFunctionObject::end()
{
    if (false)
    {
        printMessage("end catalysis");
    }

//{{{ begin code
    
//}}} end code

    return true;
}


// ************************************************************************* //

