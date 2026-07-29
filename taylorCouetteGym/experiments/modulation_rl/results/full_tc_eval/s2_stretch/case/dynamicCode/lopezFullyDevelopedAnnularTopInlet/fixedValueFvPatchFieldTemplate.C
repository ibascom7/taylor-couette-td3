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

#include "fixedValueFvPatchFieldTemplate.H"
#include "addToRunTimeSelectionTable.H"
#include "fvPatchFieldMapper.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "unitConversion.H"
#include "PatchFunction1.H"

//{{{ begin codeInclude

//}}} end codeInclude


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * Local Functions * * * * * * * * * * * * * * //

//{{{ begin localCode

//}}} end localCode


// * * * * * * * * * * * * * * * Global Functions  * * * * * * * * * * * * * //

// dynamicCode:
// SHA1 = e8151fb15a6c9c8b76f52595667eddb48d414ae9
//
// unique function name that can be checked if the correct library version
// has been loaded
extern "C" void lopezFullyDevelopedAnnularTopInlet_e8151fb15a6c9c8b76f52595667eddb48d414ae9(bool load)
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

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

makeRemovablePatchTypeField
(
    fvPatchVectorField,
    lopezFullyDevelopedAnnularTopInletFixedValueFvPatchVectorField
);

} // End namespace Foam


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::
lopezFullyDevelopedAnnularTopInletFixedValueFvPatchVectorField::
lopezFullyDevelopedAnnularTopInletFixedValueFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF
)
:
    parent_bctype(p, iF)
{
    if (false)
    {
        printMessage("Construct lopezFullyDevelopedAnnularTopInlet : patch/DimensionedField");
    }
}


Foam::
lopezFullyDevelopedAnnularTopInletFixedValueFvPatchVectorField::
lopezFullyDevelopedAnnularTopInletFixedValueFvPatchVectorField
(
    const lopezFullyDevelopedAnnularTopInletFixedValueFvPatchVectorField& rhs,
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const fvPatchFieldMapper& mapper
)
:
    parent_bctype(rhs, p, iF, mapper)
{
    if (false)
    {
        printMessage("Construct lopezFullyDevelopedAnnularTopInlet : patch/DimensionedField/mapper");
    }
}


Foam::
lopezFullyDevelopedAnnularTopInletFixedValueFvPatchVectorField::
lopezFullyDevelopedAnnularTopInletFixedValueFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const dictionary& dict
)
:
    parent_bctype(p, iF, dict)
{
    if (false)
    {
        printMessage("Construct lopezFullyDevelopedAnnularTopInlet : patch/dictionary");
    }
}


Foam::
lopezFullyDevelopedAnnularTopInletFixedValueFvPatchVectorField::
lopezFullyDevelopedAnnularTopInletFixedValueFvPatchVectorField
(
    const lopezFullyDevelopedAnnularTopInletFixedValueFvPatchVectorField& rhs
)
:
    parent_bctype(rhs),
    dictionaryContent(rhs)
{
    if (false)
    {
        printMessage("Copy construct lopezFullyDevelopedAnnularTopInlet");
    }
}


Foam::
lopezFullyDevelopedAnnularTopInletFixedValueFvPatchVectorField::
lopezFullyDevelopedAnnularTopInletFixedValueFvPatchVectorField
(
    const lopezFullyDevelopedAnnularTopInletFixedValueFvPatchVectorField& rhs,
    const DimensionedField<vector, volMesh>& iF
)
:
    parent_bctype(rhs, iF)
{
    if (false)
    {
        printMessage("Construct lopezFullyDevelopedAnnularTopInlet : copy/DimensionedField");
    }
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::
lopezFullyDevelopedAnnularTopInletFixedValueFvPatchVectorField::
~lopezFullyDevelopedAnnularTopInletFixedValueFvPatchVectorField()
{
    if (false)
    {
        printMessage("Destroy lopezFullyDevelopedAnnularTopInlet");
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void
Foam::
lopezFullyDevelopedAnnularTopInletFixedValueFvPatchVectorField::updateCoeffs()
{
    if (this->updated())
    {
        return;
    }

    if (false)
    {
        printMessage("updateCoeffs lopezFullyDevelopedAnnularTopInlet");
    }

//{{{ begin code
    #line 45 "/home/ibascom/research/taylor-couette-td3/taylorCouetteGym/experiments/modulation_rl/results/full_tc_eval/s2_stretch/case/0/U/boundaryField/inlet"
const fvPatch& p = patch();
            vectorField Uin(p.size(), vector::zero);

            const vectorField& Cf = p.Cf();
            const vectorField& Sf = p.Sf();

            const scalar Ri = 0.0254;
            const scalar Ro = 0.03175;
            const scalar Q0 = 100e-6/60.0;  // full 360-degree flow rate [m3/s]
            // UNIFIED to 100 mL/min to match side_outlet_cat_wallmodel (was Yuhe's
            // 40 mL/min). This makes the outer-wall catalytic BC the ONLY difference
            // between the two cases (single controlled variable), and shrinks the
            // residence time (tau ~ 26 s) so the episode reaches steady state well.
            const scalar pi = 3.1415926535897932384626433832795;

            const scalar Aann = pi*(sqr(Ro) - sqr(Ri));
            const scalar Umean = Q0/Aann;

            scalarField shape(p.size(), scalar(0));
            scalar sumShapeA = scalar(0);
            scalar sumA = scalar(0);

            forAll(Cf, faceI)
            {
                const scalar r = sqrt(sqr(Cf[faceI].x()) + sqr(Cf[faceI].y()));

                scalar f =
                    sqr(Ro) - sqr(r)
                  - (sqr(Ro) - sqr(Ri))/log(Ro/Ri)*log(Ro/r);

                shape[faceI] = max(f, scalar(0));

                const scalar Af = mag(Sf[faceI]);
                sumShapeA += shape[faceI]*Af;
                sumA      += Af;
            }

            reduce(sumShapeA, sumOp<scalar>());
            reduce(sumA, sumOp<scalar>());

            const scalar meanShape = sumShapeA/max(sumA, VSMALL);
            const scalar scale = Umean/max(meanShape, VSMALL);

            forAll(Cf, faceI)
            {
                const vector nHat = Sf[faceI]/mag(Sf[faceI]);
                // Top inlet outward normal points +z; inflow is opposite to it.
                Uin[faceI] = -scale*shape[faceI]*nHat;
            }

            operator==(Uin);
//}}} end code

    this->parent_bctype::updateCoeffs();
}


// ************************************************************************* //

