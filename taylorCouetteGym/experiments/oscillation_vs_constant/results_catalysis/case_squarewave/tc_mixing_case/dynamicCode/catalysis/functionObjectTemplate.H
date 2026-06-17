/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2019 OpenCFD Ltd.
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

Description
    Template for use with dynamic code generation of a functionObject.

SourceFiles
    functionObjectTemplate.C

\*---------------------------------------------------------------------------*/

#ifndef coded_functionObjectTemplate_H
#define coded_functionObjectTemplate_H

#include "regionFunctionObject.H"
#include "dictionaryContent.H"

//{{{ begin codeInclude
#line 70 "/project/mwang/ibascom/taylor-couette-td3/taylorCouetteGym/experiments/oscillation_vs_constant/results_catalysis/case_squarewave/tc_mixing_case/system/controlDict/functions/catalysis"
#include "turbulentTransportModel.H"
//}}} end codeInclude

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Forward Declarations
class fvMesh;

/*---------------------------------------------------------------------------*\
                         A templated functionObject
\*---------------------------------------------------------------------------*/

class catalysisFunctionObject
:
    public functionObjects::regionFunctionObject,
    public dictionaryContent
{
    // Private Data

//{{{ begin codeData
        
//}}} end codeData


    // Private Member Functions

        //- Report a message with the SHA1sum
        inline static void printMessage(const char* message)
        {
            Info<< message << " sha1: " << SHA1sum << '\n';
        }

        //- Cast reference of objectRegistry to fvMesh
        const fvMesh& mesh() const;

        //- No copy construct
        catalysisFunctionObject
        (
            const catalysisFunctionObject&
        ) = delete;

        //- No copy assignment
        void operator=
        (
            const catalysisFunctionObject&
        ) = delete;

public:

    //- SHA1 representation of the code content
    static constexpr const char* const SHA1sum = "ec602091c42c16faf62a6584cdc5fc9ae4ccf17b";

    //- Runtime type information
    TypeName("catalysis");


    // Constructors

        //- Construct from Time and dictionary
        catalysisFunctionObject
        (
            const word& name,
            const Time& runTime,
            const dictionary& dict
        );


    //- Destructor
    virtual ~catalysisFunctionObject();


    // Member Functions

        //- Code context as a dictionary
        const dictionary& codeContext() const noexcept
        {
            return dictionaryContent::dict();
        }

        //- Read optional controls
        virtual bool read(const dictionary& dict);

        //- Execute (at time-step)
        virtual bool execute();

        //- Write (at write interval)
        virtual bool write();

        //- Executed at the final time-loop
        virtual bool end();
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //

