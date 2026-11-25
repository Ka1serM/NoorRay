#pragma once

#include "NoorRHI/Shaders/ShaderTypes.h"

namespace GridLayout {

using namespace NoorRHI;

static const uint shBasisCount = 9;
static const uint shCount = 3 * shBasisCount;
static const uint brickWidth = 8;
static const uint brickSize = brickWidth * brickWidth * brickWidth;

//! A sparse voxel grid over [-1, 1]^3: links maps each voxel to its data index, or -1 for an empty voxel.
struct Grid {
    Ptr<int> links;
    Ptr<float> density;
    //! shCount values for each voxel: the nine coefficients of red, then of green, then of blue.
    Ptr<float> sh;
    //! One flag for each brick of cells; a cell lies between the voxel centres floor(u) and floor(u) + 1, shifted by one.
    Ptr<uint> occupancy;
    uint resolution;
    uint occupancyResolution;
};

}
