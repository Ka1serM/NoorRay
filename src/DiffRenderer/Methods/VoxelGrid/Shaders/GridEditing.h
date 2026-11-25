#pragma once

#include "GridLayout.h"

namespace GridEditing {

using namespace NoorRHI;

struct FillArguments {
    Ptr<uint> values;
    uint count;
    uint threads;
    uint value;
};

struct VariationArguments {
    GridLayout::Grid grid;
    AtomicPtr<float> densityGradients;
    AtomicPtr<float> shGradients;
    float densityWeight;
    float shWeight;
    //! The share of the voxels that the term covers in this step, chosen by a hash of the voxel and the seed.
    float fraction;
    uint seed;
};

struct KeepArguments {
    Ptr<int> links;
    Ptr<uint> maxWeights;
    Ptr<uint> keep;
    uint resolution;
    float threshold;
};

struct DilateArguments {
    Ptr<uint> source;
    Ptr<uint> destination;
    uint resolution;
    uint axis;
    uint radius;
};

struct LinkArguments {
    //! The keep flags of the old grid, or null to keep every voxel.
    Ptr<uint> keep;
    Ptr<uint> brickCounts;
    Ptr<uint> brickOffsets;
    Ptr<int> links;
    uint oldResolution;
    uint resolution;
};

struct ResampleArguments {
    //! The old grid, or null links for the first grid.
    GridLayout::Grid source;
    GridLayout::Grid destination;
    float initialDensity;
};

struct OccupancyArguments {
    GridLayout::Grid grid;
};

}
