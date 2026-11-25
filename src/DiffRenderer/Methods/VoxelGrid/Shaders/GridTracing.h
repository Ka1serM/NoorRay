#pragma once

#include "GridLayout.h"

namespace GridTracing {

using namespace NoorRHI;

struct Arguments {
    //! The rows of the world-to-camera transform.
    float4 rows[3];
    float4 intrinsics;
    float4 origin;
    //! The centre of the grid in xyz and the scale from world to grid units in w.
    float4 placement;
    float4 background;
    GridLayout::Grid grid;
    Ptr<float> image;
    Ptr<float> imageGradients;
    AtomicPtr<float> densityGradients;
    AtomicPtr<float> shGradients;
    AtomicPtr<uint> maxWeights;
    uint width;
    uint height;
    uint contracted;
    uint maxSamples;
    //! The distance between samples in grid units, half a voxel.
    float stepSize;
};

}
